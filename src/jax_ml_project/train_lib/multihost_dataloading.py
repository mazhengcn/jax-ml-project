import time
import typing as tp
from collections.abc import Iterable, Iterator
from functools import partial

import grain.python as grain
import jax
import jax.tree_util as jtu
import numpy as np
import tensorflow as tf
from absl import logging
from jax.sharding import Mesh, NamedSharding, PartitionSpec

PyTree = tp.Any

Dtype = tp.Any
Shape = tuple[int, ...]

SLEEP_TIME = 10
MAX_DATA_LOAD_ATTEMPTS = 30

_DATALOADER_TYPE_ERROR = "dataloader should be either tf.data.Dataset or Iterable"


def _build_global_shape_and_sharding(
    local_shape: tuple[int, ...], global_mesh: Mesh, data_pspec: PartitionSpec
) -> tuple[tuple[int, ...], NamedSharding]:
    sharding = NamedSharding(global_mesh, data_pspec)
    global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]

    return global_shape, sharding


def _form_global_array(
    path: tuple[int | str, ...],
    array: np.ndarray,
    global_mesh: Mesh,
    data_pspec: PartitionSpec,
) -> jax.Array:
    """Put local sharded array into local devices."""
    global_shape, sharding = _build_global_shape_and_sharding(
        np.shape(array), global_mesh, data_pspec
    )

    try:
        local_device_arrays = np.split(array, len(global_mesh.local_devices), axis=0)
    except ValueError as array_split_error:
        error_msg = (
            f"Unable to put to devices shape {array.shape} with "
            f"local device count {len(global_mesh.local_devices)} "
            f"at {jtu.keystr(path)}"
        )
        raise ValueError(error_msg) from array_split_error

    local_device_buffers = jax.device_put(
        local_device_arrays, global_mesh.local_devices
    )
    return jax.make_array_from_single_device_arrays(
        global_shape, sharding, local_device_buffers
    )


def get_next_batch_sharded(
    local_iterator: Iterator, global_mesh: Mesh, data_pspec: PartitionSpec
) -> jax.Array:
    """Split the host loaded data equally over all devices."""
    data_load_attempts = 0
    loaded_data_success = False
    local_data = None
    while not loaded_data_success and data_load_attempts < MAX_DATA_LOAD_ATTEMPTS:
        data_load_attempts += 1
        try:
            local_data = next(local_iterator)
            loaded_data_success = True
        except tf.errors.FailedPreconditionError:
            logging.info("Failed to get next data batch, retrying")
            time.sleep(SLEEP_TIME)

    # Try one last time, if this fails we will see the full stack trace.
    if not loaded_data_success:
        local_data = next(local_iterator)

    return jtu.tree_map_with_path(
        partial(_form_global_array, global_mesh=global_mesh, data_pspec=data_pspec),
        local_data,
    )


class MultiHostDataLoadIterator:
    """Folds get_next_batch_sharded into a iterator class."""

    def __init__(
        self,
        dataloader: tf.data.Dataset | grain.DataLoader,
        global_mesh: Mesh,
        data_pspec: PartitionSpec | None = None,
    ) -> None:
        """Initialize the MultiHostDataLoadIterator.

        Args:
            dataloader: The dataset to iterate over
            global_mesh: The mesh to use for sharding
            data_pspec: Optional partition spec for data sharding

        """
        self.global_mesh = global_mesh
        self.dataloader = dataloader
        if data_pspec:
            self.data_pspec = data_pspec
        else:
            self.data_pspec = PartitionSpec(global_mesh.axis_names)
        if isinstance(self.dataloader, tf.data.Dataset):
            self.local_iterator = self.dataloader.as_numpy_iterator()
        elif isinstance(self.dataloader, Iterable):
            self.local_iterator = iter(self.dataloader)
        else:
            raise TypeError(_DATALOADER_TYPE_ERROR)

    def reset(self) -> None:
        """Reset the iterator to its initial state."""
        if isinstance(self.dataloader, tf.data.Dataset):
            self.local_iterator = self.dataloader.as_numpy_iterator()
        elif isinstance(self.dataloader, Iterable):
            self.local_iterator = iter(self.dataloader)
        else:
            raise TypeError(_DATALOADER_TYPE_ERROR)

    def __iter__(self) -> tp.Iterator:
        """Return an iterator over the dataset."""
        self.reset()
        return self

    def __next__(self) -> jax.Array:
        """Return the next batch of sharded data from the iterator.

        Returns:
            A JAX array containing the next batch of sharded data.

        """
        return get_next_batch_sharded(
            self.local_iterator, self.global_mesh, self.data_pspec
        )
