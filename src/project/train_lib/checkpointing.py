"""Create an Orbax CheckpointManager with specified (Async or not) Checkpointer."""

import typing as tp

import grain.python as grain
import orbax.checkpoint as ocp
from absl import logging
from etils import epath
from flax import nnx
from orbax.checkpoint.logging import abstract_logger

from .multihost_dataloading import MultiHostDataLoadIterator


def create_orbax_checkpoint_manager(  # noqa: PLR0913
    checkpoint_dir: str,
    *,
    enable_checkpointing: bool,
    use_async: bool = True,
    save_interval_steps: int = 1,
    dataset_type: str | None = "tfds",
    orbax_logger: abstract_logger.AbstractLogger | None = None,
) -> ocp.CheckpointManager | None:
    """Create an Orbax CheckpointManager.

    Creates either an async or synchronous CheckpointManager, or returns None if
    checkpointing is disabled.
    """
    if not enable_checkpointing:
        logging.info("Checkpointing disabled, not creating checkpoint manager.")
        return None
    logging.info("Creating checkpoint manager...")
    p = epath.Path(checkpoint_dir)

    if dataset_type == "grain":
        item_names = ("train_state", "data_iter")
    else:
        item_names = ("train_state",)

    options = ocp.CheckpointManagerOptions(
        save_interval_steps=save_interval_steps, enable_async_checkpointing=use_async
    )
    mngr = ocp.CheckpointManager(
        p, item_names=item_names, options=options, logger=orbax_logger
    )
    logging.info("Checkpoint manager created!")

    return mngr


def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    step: int,
    train_state: nnx.State,
    dataset_type: str = "tfds",
    data_iterator: MultiHostDataLoadIterator | None = None,
) -> bool:
    """Save checkpoint using the checkpoint manager.

    Args:
        checkpoint_manager: The Orbax checkpoint manager.
        step: Current training step.
        train_state: The training state to save.
        dataset_type: Type of dataset being used ('tfds' or 'grain').
        data_iterator: Data iterator to save for grain datasets.

    Returns:
        The result of the checkpoint save operation.

    """
    if dataset_type == "grain":
        return checkpoint_manager.save(
            step,
            args=ocp.args.Composite(
                train_state=ocp.args.StandardSave(train_state),  # type: ignore[assignment]
                data_iter=grain.PyGrainCheckpointSave(data_iterator.local_iterator),  # type: ignore[arg-type]
            ),
        )
    return checkpoint_manager.save(
        step,
        args=ocp.args.Composite(train_state=ocp.args.StandardSave(train_state)),  # type: ignore[arg-type]
    )


def load_state_if_possible(  # noqa: PLR0913
    checkpoint_manager: ocp.CheckpointManager | None,
    data_iterator: MultiHostDataLoadIterator | None,
    load_parameters_from_path: str,
    load_full_state_from_path: str,
    abstract_train_state: nnx.State,
    dataset_type: str | None = "tfds",
) -> tuple[tp.Any | None, tp.Any | None]:
    """Load training state from checkpoints if available.

    Args:
        checkpoint_manager: The Orbax checkpoint manager.
        data_iterator: Data iterator for grain datasets.
        load_parameters_from_path: Path to load model parameters from.
        load_full_state_from_path: Path to load full training state from.
        abstract_train_state: Abstract training state to restore into.
        dataset_type: Type of dataset being used ('tfds' or 'grain').

    Returns:
        A tuple of (restored_state, restored_params), where each can be None.

    """
    if checkpoint_manager is not None:
        logging.info(
            "checkpoint manager exists so trying to load this run's existing checkpoint"
        )

        latest_step = checkpoint_manager.latest_step()
        if latest_step is not None:
            logging.info(
                f"restoring from this run's directory latest step {latest_step}"
            )

            if dataset_type == "grain" and data_iterator is not None:
                return (
                    checkpoint_manager.restore(
                        latest_step,
                        args=ocp.args.Composite(
                            train_state=ocp.args.StandardRestore(abstract_train_state),  # type: ignore[assignment]
                            data_iter=grain.PyGrainCheckpointRestore(  # type: ignore[arg-type]
                                data_iterator.local_iterator  # type: ignore[arg-type]
                            ),
                        ),
                    ),
                    None,
                )
            return (
                checkpoint_manager.restore(
                    latest_step,
                    args=ocp.args.Composite(
                        train_state=ocp.args.StandardRestore(abstract_train_state)  # type: ignore[assignment]
                    ),
                ),
                None,
            )

    if load_parameters_from_path != "":
        restored_params = load_params_from_path(
            load_parameters_from_path, abstract_train_state.model
        )
        return None, restored_params
    if load_full_state_from_path != "":
        logging.info(f"restoring full state from {load_full_state_from_path=}")
        p = epath.Path(load_full_state_from_path)
        ckptr = ocp.StandardCheckpointer()
        restored_train_state = ckptr.restore(p, abstract_train_state)
        return {"train_state": restored_train_state}, None

    logging.info("No existing checkpoints found, not restoring checkpoint.")
    return None, None


def load_params_from_path(
    load_parameters_from_path: str, abstract_model_state: nnx.State
) -> nnx.State:
    """Load inference params from checkpoint at specified path."""
    if not load_parameters_from_path:
        msg = "load_parameters_from_path is not defined"
        raise ValueError(msg)
    logging.info(f"restoring params from {load_parameters_from_path}")
    ckpt = epath.Path(load_parameters_from_path)
    ckptr = ocp.StandardCheckpointer()
    return ckptr.restore(ckpt, target=abstract_model_state)


def save_params_to_path(checkpoint_dir: str, model_state: nnx.State) -> None:
    """Save params in checkpoint at specified path."""
    if not checkpoint_dir:
        msg = "checkpoint_dir is not defined."
        raise ValueError(msg)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(checkpoint_dir, model_state)
    ckptr.wait_until_finished()
