import dataclasses
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import yaml
from absl import logging
from clu import metric_writers, periodic_actions
from flax import nnx
from jax.sharding import Mesh

from ..configs import default  # noqa: TID252
from ..input_pipeline import input_pipeline_interface  # noqa: TID252
from . import checkpointing, optimizers
from . import utils as train_utils
from .checkpointing import save_checkpoint
from .metrics import RelativeError
from .multihost_dataloading import MultiHostDataLoadIterator

Batch = dict[str, jax.Array | np.ndarray]
Model = nnx.Module


def loss_fn(model: nnx.Module, batch: Batch) -> jax.Array:
    """Loss function used for training."""
    labels = batch["label"]
    predictions = model(batch)  # type: ignore[attr-defined]
    return jnp.mean((predictions - labels) ** 2)


@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch: Batch
) -> None:
    """Perform a single training step."""
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
    optimizer.update(grads)
    metrics.update(loss=loss, mean_squared_labels=jnp.mean(batch["label"] ** 2))


def accumulate_gradent(micro_steps: int, global_batch_size: int) -> Callable[..., None]:
    """Accumulate gradients over multiple micro-steps.

    Args:
        micro_steps: Number of micro-steps to accumulate gradients over.
        global_batch_size: The global batch size.

    Returns:
        A function that performs the accumulated training step.

    """
    if not micro_steps or micro_steps < 0:
        return train_step

    batch_size_per_device = global_batch_size // jax.device_count()
    if batch_size_per_device % micro_steps != 0:
        error_message = "batch_size_per_device must be divisible by micro_steps"
        raise ValueError(error_message)

    def accumulated_train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        batch: Batch,
    ) -> None:
        batch = jax.tree.map(
            lambda x: x.reshape((-1, micro_steps) + x.shape[1:]), batch
        )
        for i in range(micro_steps):
            micro_batch = jax.tree.map(lambda x, i=i: x[:, i], batch)
            train_step(model, optimizer, metrics, micro_batch)

    return accumulated_train_step


@nnx.jit
def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch: Batch) -> None:
    """Calculate evaluation metrics on a batch."""
    loss = loss_fn(model, batch)
    metrics.update(loss=loss, mean_squared_labels=jnp.mean(batch["label"] ** 2))


def evaluate(
    model: nnx.Module, metrics: nnx.MultiMetric, eval_iter: MultiHostDataLoadIterator
) -> None:
    """Evaluate the target an return a dictionary with the metrics."""
    logging.info("Gathering evaluation metrics.")
    for eval_batch in eval_iter:
        eval_step(model, metrics, eval_batch)


def train_and_evaluate(config: default.Config, workdir: str) -> None:  # noqa: C901, PLR0915
    """Run a training and evaluation loop.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.

    """
    tf.io.gfile.makedirs(workdir)

    init_rng = jax.random.key(config.seed)

    start_step = 0

    # Mesh definition
    # ---------------------------------------------------------------------------
    logging.info("Initializing mesh.")

    devices_array = train_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    # Build model constructor, optimizer and checkpoint manager
    # ---------------------------------------------------------------------------
    logging.info("Initializing optimizer, model and checkpointer.")

    lr_schedule = optimizers.create_learning_rate_schedule(config)
    tx = optimizers.create_optimizer(config, lr_schedule)
    tx = optax.MultiSteps(tx, every_k_schedule=config.micro_steps)

    accumulated_train_step = accumulate_gradent(
        config.micro_steps, config.global_batch_size
    )

    ckpt_mngr = checkpointing.create_orbax_checkpoint_manager(
        workdir,
        enable_checkpointing=config.save_checkpoints,
        use_async=config.async_checkpointing,
        save_interval_steps=config.checkpoint_every_steps,
        dataset_type=config.dataset_type,
    )

    # Setup Metrics
    # ---------------------------------------------------------------------------
    metrics: nnx.MultiMetric = nnx.MultiMetric(
        mse=nnx.metrics.Average("loss"),
        rmse=RelativeError("loss", "mean_squared_labels"),
    )

    # Create metric writers
    writer = metric_writers.create_default_writer(
        workdir, just_logging=jax.process_index() > 0
    )

    # Load Dataset
    # ---------------------------------------------------------------------------
    logging.info("Initializing dataset.")
    data_iter = input_pipeline_interface.create_data_iterator(config, mesh)
    if data_iter is None:
        error_message = "Data iterator creation failed, received None."
        raise ValueError(error_message)
    (train_iter, eval_iter), data_sharding = data_iter

    # Initialize train state
    # ---------------------------------------------------------------------------
    logging.info("Initializing train state.")
    model, optimizer, train_iter = train_utils.setup_training_state(
        model_class=Model,
        config=config,
        rng=init_rng,
        tx=tx,
        mesh=mesh,
        data_iterator=train_iter,
        checkpoint_manager=ckpt_mngr,
    )
    num_params = train_utils.calculate_num_params_from_pytree(nnx.state(model))
    logging.info(f"Number of model params={num_params}")

    start_step = optimizer.step.value // config.micro_steps
    if start_step == 0:
        with Path(f"{workdir}/config.yaml").open("w") as f:
            yaml.dump(dataclasses.asdict(config), f)

    # Main Train Loop
    # ---------------------------------------------------------------------------
    logging.info("Starting training loop.")
    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_train_steps, writer=writer
    )
    if jax.process_index() == 0:
        hooks += [
            report_progress,
            periodic_actions.Profile(logdir=workdir, num_profile_steps=5),
        ]
    with metric_writers.ensure_flushes(writer):
        for step in range(start_step, config.num_train_steps):
            is_last_step = step == config.num_train_steps - 1

            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                batch = next(train_iter)
                accumulated_train_step(model, optimizer, metrics, batch)

            # Quick indication that training is happening.
            logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
            for h in hooks:
                h(step)

            # Periodic metric handling.
            if step % config.log_every_steps == 0 or is_last_step:
                with report_progress.timed("training_metrics"):
                    logging.info("Gathering training metrics.")
                    train_metrics = {}
                    for metric, value in metrics.compute().items():
                        train_metrics[metric] = float(value)  # type: ignore[assignment]
                    writer.write_scalars(step, train_metrics)
                metrics.reset()

            if (eval_iter and step % config.eval_every_steps == 0) or is_last_step:
                with report_progress.timed("eval"):
                    evaluate(model, metrics, eval_iter)  # type: ignore[arg-type]
                    eval_metrics = {}
                    for eval_metric, value in metrics.compute().items():
                        eval_metrics[eval_metric] = float(value)  # type: ignore[assignment]
                    writer.write_scalars(step, eval_metrics)
                metrics.reset()

            if ckpt_mngr and config.save_checkpoints:
                with report_progress.timed("checkpoint"):
                    save_checkpoint(
                        ckpt_mngr,
                        step,
                        nnx.state(optimizer),
                        config.dataset_type,
                        train_iter,
                    )
        if ckpt_mngr:
            ckpt_mngr.wait_until_finished()
