import dataclasses

import jax
import optax
import yaml
from absl import app, flags, logging
from etils import epath
from flax import nnx
from jax.sharding import Mesh

from {{ cookiecutter.module_name }}.configs import default
from {{ cookiecutter.module_name }}.model.model import Model, ModelConfig
from {{ cookiecutter.module_name }}.train_lib import checkpointing, optimizers
from {{ cookiecutter.module_name }}.train_lib import utils as train_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config", None, "File path to the training hyperparameter configuration."
)
flags.DEFINE_string("checkpoint_dir", None, "Directory to store model params.")
flags.mark_flags_as_required(["config", "checkpoint_dir"])


def _read_train_checkpoint(
    config: default.Config, mesh: Mesh
) -> tuple[nnx.State, nnx.State]:
    """Read training checkpoint at path defined by load_full_state_path."""
    # Model and Optimizer definition
    rng = jax.random.key(0)
    lr_schedule = optimizers.create_learning_rate_schedule(config)
    tx = optimizers.create_optimizer(config, lr_schedule)
    tx = optax.MultiSteps(tx, every_k_schedule=config.micro_steps)

    model, optimizer, _ = train_utils.setup_training_state(
        model_class=Model,  # type: ignore[assignment]
        config=config,
        rng=rng,
        tx=tx,
        mesh=mesh,
        data_iterator=None,
        checkpoint_manager=None,
    )
    num_params = train_utils.calculate_num_params_from_pytree(nnx.state(model))
    logging.info(f"In input checkpoint Number of model params={num_params}.")
    return nnx.state(model), nnx.state(optimizer)


def generate_infer_checkpoint(config: default.Config, checkpoint_dir: str) -> bool:
    """Generate a params checkpoint from a given training checkpoint.

    - Training checkpoint is loaded from config.load_full_state_path.
    - Params checkpoint will be saved at the checkpoint directory under params folder.
    """
    devices_array = train_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    if not checkpoint_dir:
        error_message = "checkpoint_dir not configured"
        raise ValueError(error_message)
    # Remove any old checkpoint
    path = epath.Path(checkpoint_dir)
    if path.exists() and jax.process_index() == 0:
        path.rmtree()

    # Read training state from config.load_full_state_path
    logging.info(f"Read training checkpoint from: {config.load_full_state_path}")
    model_state, train_state = _read_train_checkpoint(config, mesh)
    if train_state.opt_state == {}:
        error_message = "missing opt_state in training checkpoint"
        raise ValueError(error_message)

    # Save params to checkpoint directory under params folder
    logging.info(f"Save infer checkpoint at: {checkpoint_dir}")
    checkpointing.save_params_to_path(f"{checkpoint_dir}/params", model_state)
    logging.info(
        f"Successfully generated params checkpoint at: {checkpoint_dir}/params"
    )

    # Save config file to checkpoint directory
    model_config = ModelConfig(
        in_dim=config.in_dim,
        out_dim=config.out_dim,
        num_mlp_layers=config.num_mlp_layers,
        mlp_dim=config.mlp_dim,
    )
    config_dict = dataclasses.asdict(model_config)
    config_dict["load_full_state_path"] = config.load_full_state_path
    with epath.Path(checkpoint_dir).joinpath("config.yaml").open("w") as f:
        yaml.dump(config_dict, f)
    logging.info(
        f"Successfully save model config file at: {checkpoint_dir}/config.yaml"
    )

    return True


def main(argv: list[str]) -> None:
    """Run the script.

    Args:
        argv (list[str]): List of command-line arguments.

    """
    if len(argv) > 1:
        error_message = "Too many command-line arguments."
        raise app.UsageError(error_message)

    config = default.get_config(FLAGS.config)
    generate_infer_checkpoint(config, FLAGS.checkpoint_dir)


if __name__ == "__main__":
    app.run(main)
