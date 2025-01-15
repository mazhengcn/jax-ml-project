import typing as tp

import grain.python as grain
import jax
from jax.sharding import Mesh, PartitionSpec

from ..configs import default  # noqa: TID252
from ..train_lib import multihost_dataloading  # noqa: TID252
from .splits import get_split_instruction

Data = tp.Any


class Dataset(grain.RandomAccessDataSource):
    """A dataset class that wraps raw data for random access."""

    def __init__(self, raw_data: Data) -> None:
        """Initialize the Dataset with raw data.

        Args:
            raw_data (Data): The raw data to be used in the dataset.

        """
        self.data = raw_data

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, index: tp.SupportsIndex) -> Data:
        """Retrieve an item from the dataset by index.

        Args:
            index (tp.SupportsIndex): The index of the item to retrieve.

        Returns:
            Data: The data item at the specified index.

        """
        return self.data[index]


def load_dataset(filename: str, data_dir: str) -> Data:
    """Load a dataset from a file."""


def get_datasets(dataset_name: str, data_dir: str, data_split: str) -> Dataset:
    """Load a dataset as grain datasource."""
    raw_data = load_dataset(dataset_name, data_dir)
    num_examples = raw_data[0].shape[0]
    split_instr = get_split_instruction(data_split, num_examples)
    raw_data = jax.tree.map(lambda x: x[split_instr.from_ : split_instr.to], raw_data)
    return Dataset(raw_data)


def preprocessing_pipeline(  # noqa: PLR0913
    dataset: grain.RandomAccessDataSource,
    *,
    global_batch_size: int,
    global_mesh: Mesh,
    data_pspec: PartitionSpec,
    worker_count: int | None = 0,
    worker_buffer_size: int = 1,
    dataloading_host_index: int,
    dataloading_host_count: int,
    shuffle: bool = False,
    data_shuffle_seed: int = 0,
    num_epochs: int | None = 1,
    drop_remainder: bool = True,
) -> multihost_dataloading.MultiHostDataLoadIterator:
    """Use grain to pre-process the dataset and return iterators."""
    if global_batch_size % global_mesh.size != 0:
        error_message = (
            "Batch size should be divisible by the number of global devices."
        )
        raise ValueError(error_message)

    # Batch examples.
    batch_size_per_process = global_batch_size // jax.process_count()

    ops = []
    ops.append(grain.Batch(batch_size_per_process, drop_remainder=drop_remainder))

    index_sampler = grain.IndexSampler(
        num_records=len(dataset),
        num_epochs=num_epochs,
        shard_options=grain.ShardOptions(
            shard_index=dataloading_host_index,
            shard_count=dataloading_host_count,
            drop_remainder=drop_remainder,
        ),
        shuffle=shuffle,
        seed=data_shuffle_seed,
    )
    dataloader = grain.DataLoader(
        data_source=dataset,
        operations=ops,
        sampler=index_sampler,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
    )
    # Return multi-host jax.Array prep iterator
    return multihost_dataloading.MultiHostDataLoadIterator(
        dataloader, global_mesh, data_pspec
    )


def make_grain_iterator(
    config: default.Config, global_mesh: Mesh, process_indices: list[int]
) -> tuple[tp.Iterator, tp.Iterator | None]:
    """Load dataset, preprocess and return iterators."""
    train_ds = get_datasets(
        dataset_name=config.dataset_name,
        data_dir=config.data_dir,
        data_split=config.train_split,
    )

    train_iter = preprocessing_pipeline(
        dataset=train_ds,
        global_batch_size=config.global_batch_size,
        global_mesh=global_mesh,
        data_pspec=PartitionSpec(*config.data_sharding),
        worker_count=config.grain_worker_count,
        worker_buffer_size=config.grain_worker_buffer_size,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        shuffle=config.enable_data_shuffling,
        num_epochs=None,
        data_shuffle_seed=config.data_shuffle_seed,
    )

    if config.eval_every_steps > 0:
        eval_ds = get_datasets(
            dataset_name=config.dataset_name,
            data_dir=config.data_dir,
            data_split=config.eval_split,
        )

        eval_iter = preprocessing_pipeline(
            dataset=eval_ds,
            global_batch_size=config.eval_batch_size,
            global_mesh=global_mesh,
            data_pspec=PartitionSpec(*config.data_sharding),
            worker_count=config.grain_worker_count,
            worker_buffer_size=config.grain_worker_buffer_size,
            dataloading_host_index=process_indices.index(jax.process_index()),
            dataloading_host_count=len(process_indices),
            shuffle=False,
            data_shuffle_seed=config.data_shuffle_seed,
        )
    else:
        eval_iter = None

    return train_iter, eval_iter
