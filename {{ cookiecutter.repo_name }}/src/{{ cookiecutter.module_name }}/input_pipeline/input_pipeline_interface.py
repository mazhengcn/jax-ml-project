import typing as tp

import jax

from ..configs import default  # noqa: TID252
from .grain_data_processing import make_grain_iterator

Mesh = jax.sharding.Mesh
Sharding = jax.sharding.NamedSharding


def create_data_iterator(
    config: default.Config, mesh: Mesh
) -> tuple[tp.Iterator, tp.Iterator | None]:
    """Create data iterator based on the dataset type in the configuration.

    Args:
        config (default.Config): Configuration object containing
        dataset type and other settings.
        mesh (Mesh): JAX sharding mesh object.

    Returns:
        tuple: A tuple containing iterators and sharding information.

    """
    if config.dataset_type == "grain":
        return make_grain_iterator(config, mesh)
    error_message = (
        f"Unknown dataset_type {config.dataset_type}, dataset_type must be grain."
    )
    raise ValueError(error_message)
