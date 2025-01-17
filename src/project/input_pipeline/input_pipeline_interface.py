import typing as tp

import jax
from jax.sharding import PartitionSpec as P

from ..configs import default  # noqa: TID252
from .grain_data_processing import make_grain_iterator

Mesh = jax.sharding.Mesh
Sharding = jax.sharding.NamedSharding


def get_process_loading_real_data(
    config: default.Config, mesh: Mesh
) -> tuple[list[int], jax.sharding.NamedSharding]:
    """Get list of processes loading data when expansion_factor_real_data != -1."""
    sharding = jax.sharding.NamedSharding(mesh, P(*config.data_sharding))
    devices_indices_map = sharding.devices_indices_map((config.global_batch_size,))
    batch_cutoff = config.global_batch_size
    process_loading_real_data = set()
    for p, indices in devices_indices_map.items():
        stop = indices[0].stop if indices[0].stop is not None else batch_cutoff
        if stop <= batch_cutoff:
            process_loading_real_data.add(p.process_index)
    return list(process_loading_real_data), sharding


def make_mixed_train_iterator(
    config: default.Config, mesh: Mesh
) -> tuple[tuple[tp.Iterator, tp.Iterator | None], Sharding | None]:
    """Return iterators according to dataset_type."""
    process_indices, sharding = get_process_loading_real_data(config, mesh)
    if jax.process_index() in process_indices:
        if config.dataset_type == "grain":
            return make_grain_iterator(config, mesh, process_indices), sharding
        error_message = "Unknown dataset_type"
        raise ValueError(error_message)
    error_message = "Process not loading real data"
    raise ValueError(error_message)


def create_data_iterator(
    config: default.Config, mesh: Mesh
) -> tuple[tuple[tp.Iterator, tp.Iterator | None], Sharding | None]:
    """Create data iterator based on the dataset type in the configuration.

    Args:
        config (default.Config): Configuration object containing
        dataset type and other settings.
        mesh (Mesh): JAX sharding mesh object.

    Returns:
        tuple: A tuple containing iterators and sharding information.

    """
    if config.dataset_type in ("tfds", "grain"):
        return make_mixed_train_iterator(config, mesh)
    error_message = (
        f"Unknown dataset_type {config.dataset_type}, "
        "dataset_type must be synthetic, tfds"
    )
    raise ValueError(error_message)
