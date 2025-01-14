import os
import socket
import time
import typing as tp
from collections.abc import Callable, Sequence

import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from absl import logging
from etils import epath
from flax import nnx
from jax.experimental import mesh_utils

from ..configs import default  # noqa: TID252
from . import checkpointing

Mesh = jax.sharding.Mesh
PyTree = tp.Any
PartitionSpec = jax.sharding.PartitionSpec
Sharding = jax.sharding.Sharding

Dtype = tp.Any
Shape = tuple[int, ...]
PROXY = object()

JAX_INIT_INFO_FILE = "jax-init-info.txt"


# Tree utils.
def _expand_axes(
    axes: PyTree | int, values: PyTree, name: str = "collect_pytrees"
) -> PyTree:
    values_tree_def = jax.tree.flatten(values)[1]
    flat_axes = jax.api_util.flatten_axes(name, values_tree_def, axes)
    # Replace None's with PROXY
    flat_axes = [PROXY if x is None else x for x in flat_axes]
    return jax.tree.unflatten(values_tree_def, flat_axes)


def collect_pytrees(
    pytrees: Sequence[PyTree],
    axes: PyTree | int = 0,
    collective_fn: Callable[[Sequence, int], PyTree] | None = None,
) -> PyTree:
    """Collect pytrees along specified axes using a collective function.

    Args:
        pytrees: A sequence of pytrees to be collected.
        axes: The axes along which to collect the pytrees.
        collective_fn: A function to apply to the collected pytrees.

    Returns:
        A pytree with the collected values.

    """
    axes_ = _expand_axes(axes, pytrees[0])
    if collective_fn:
        collect_args = lambda *args: collective_fn(args[:-1], args[-1])  # noqa: E731
    else:
        collect_args = lambda *args: list(args[:-1])  # noqa: E731
    return jax.tree.map(collect_args, *pytrees, axes_)


# Mesh utils.
# -----------------------------------------------------------------------------
def create_device_mesh(
    config: default.Config, devices: Sequence[jax.Device] | None = None
) -> np.ndarray:
    """Create a device mesh with each slice in its own data parallel group.

    If there is only one slice, use two replicas.
    """
    if devices is None:
        devices = jax.devices()
    num_devices = len(devices)
    try:
        num_slices = 1 + max([d.slice_index for d in devices])
    except:  # noqa: E722
        num_slices = 1
    num_devices_per_slice = num_devices // num_slices
    logging.info(f"Devices: {devices}")
    logging.info(f"Number of devices: {num_devices}")

    multi_slice_env = hasattr(jax.devices()[0], "slice_index")

    dcn_parallelism = [
        config.dcn_data_parallelism,
        config.dcn_fsdp_parallelism,
        config.dcn_tensor_parallelism,
    ]
    ici_parallelism = [
        config.ici_data_parallelism,
        config.ici_fsdp_parallelism,
        config.ici_tensor_parallelism,
    ]

    # Find possible unspecified parallelisms
    dcn_parallelism = fill_unspecified_mesh_axes(dcn_parallelism, num_slices, "DCN")
    ici_parallelism = fill_unspecified_mesh_axes(
        ici_parallelism, num_devices_per_slice, "ICI"
    )

    if multi_slice_env:
        mesh = mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism)
    else:
        mesh = mesh_utils.create_device_mesh(ici_parallelism)

    logging.info(f"Decided on mesh: {mesh}")
    logging.info(f"Mesh shape: {mesh.shape}")

    return mesh


def fill_unspecified_mesh_axes(
    parallelism_vals: list[int], target_product: int, parallelism_type: str
) -> Sequence[int]:
    """Evaluate unspecified DCN/ICI parallelism values."""
    if -1 in parallelism_vals:
        if parallelism_vals.count(-1) != 1:
            error_message = (
                f"Found unspecified values (-1) for more than one {parallelism_type} "
                "parallelism axis. At most one axis can be unspecified."
            )
            raise ValueError(error_message)

        determined_val = target_product / np.prod(parallelism_vals) * -1

        if not (determined_val >= 1 and determined_val.is_integer):
            error_message = (
                "Unspecified value unable to be determined with the given "
                f"{parallelism_type} parallelism values"
            )
            raise ValueError(error_message)

        parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

    target_type = "slices" if parallelism_type == "DCN" else "devices per slice"

    if np.prod(parallelism_vals) != target_product:
        error_message = (
            f"Number of {target_type} {target_product} does not match the product"
            f" of the {parallelism_type} parallelism {np.prod(parallelism_vals)}"
        )
        raise ValueError(error_message)

    return parallelism_vals


# State initialization utils.
# -----------------------------------------------------------------------------
def calculate_num_params_from_pytree(params: PyTree) -> int:
    """Calculate the total number of parameters from a pytree.

    Args:
        params: A pytree of parameters.

    Returns:
        The total number of parameters as an integer.

    """
    params_sizes = jax.tree.map(jax.numpy.size, params)
    total_parameters = jax.tree.reduce(lambda x, y: x + y, params_sizes)
    if total_parameters < 0:
        error_message = "Total parameters is negative."
        raise ValueError(error_message)
    return total_parameters


def create_init_fn(
    model_class: nnx.Module,
    config: default.Config,
    rng: int | jax.Array,
    tx: optax.GradientTransformation | optax.MultiSteps | None,
    *,
    sharded: bool = True,
) -> Callable[[], nnx.Module | tuple[nnx.Module, nnx.Optimizer]]:
    """Create an initialization function for the model.

    Args:
        model_class: The class of the model to be initialized.
        config: Configuration for the model.
        rng: Random key for initialization.
        tx: Optimizer or None.
        sharded: Whether to use sharded initialization.

    Returns:
        A callable that initializes the model and optionally the optimizer.

    """

    # Initialization
    def init_fn() -> tuple[nnx.Module, nnx.Optimizer] | nnx.Module:
        model = model_class(config, rngs=nnx.Rngs(rng))  # type: ignore[arg-type]
        if sharded:
            state = nnx.state(model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(model, sharded_state)
        if tx:
            optimizer = nnx.Optimizer(model, tx)  # type: ignore[arg-type]
            return model, optimizer
        return model

    if sharded:
        return nnx.jit(init_fn)
    return init_fn


def setup_training_state(  # noqa: PLR0913
    model_class: nnx.Module,
    config: default.Config,
    rng: int | jax.Array,
    tx: optax.GradientTransformation | optax.MultiSteps | None,
    mesh: Mesh,
    data_iterator: tp.Iterable | None,
    checkpoint_manager: ocp.CheckpointManager | None,
) -> tuple[nnx.Module, nnx.Optimizer, tp.Iterator]:
    """Set up the training state for the model.

    Args:
        model_class: The class of the model to be initialized.
        config: Configuration for the model.
        rng: Random key for initialization.
        tx: Optimizer.
        mesh: Device mesh for sharding.
        data_iterator: Iterator for the training data.
        data_iterator: Iterator for the training data.
        checkpoint_manager: Manager for handling checkpoints.

    Returns:
        A tuple containing the model, optimizer, and data iterator.

    """
    if tx is None:
        error_message = "Optimizer must be provided for training."
        raise ValueError(error_message)

    model, optimizer = nnx.eval_shape(
        create_init_fn(model_class, config, rng, tx, sharded=False)
    )  # type: ignore[call-arg]
    abstract_train_state = nnx.state(optimizer)
    abstract_sharded_train_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
        abstract_train_state,
        nnx.get_named_sharding(abstract_train_state, mesh),
    )

    # Initialization
    restored_train_state, restored_model_state = checkpointing.load_state_if_possible(
        checkpoint_manager,
        data_iterator,
        config.load_parameters_path,
        config.load_full_state_path,
        abstract_sharded_train_state,
        config.dataset_type,
    )

    if restored_train_state:
        if (
            "data_iter" in restored_train_state
            and restored_train_state["data_iter"] is not None
        ):
            data_iterator.local_iterator = restored_train_state["data_iter"]  # type: ignore[attr-defined]
        nnx.update(optimizer, restored_train_state["train_state"])
    elif restored_model_state:
        nnx.update(model, restored_model_state)
        optimizer = nnx.Optimizer(model, tx)  # type: ignore[arg-type]
    else:
        init_model_and_opt = create_init_fn(model_class, config, rng, tx)
        with mesh:
            model, optimizer = init_model_and_opt()  # type: ignore[return-value]

    return model, optimizer, data_iterator  # type: ignore[return-value]


def setup_infer_state(
    model_class: nnx.Module, config: default.Config, rng: int | jax.Array, mesh: Mesh
) -> nnx.Module:
    """Set up the inference state for the model.

    Args:
        model_class: The class of the model to be initialized.
        config: Configuration for the model.
        rng: Random key for initialization.
        mesh: Device mesh for sharding.

    Returns:
        The initialized model.

    """
    if not config.load_parameters_path:
        # generate random params
        logging.info("No infer checkpoint specified - generating random weights.")
        init_model = create_init_fn(model_class, config, rng, None)
        with mesh:
            model = init_model()
    else:
        # Load params from checkpoint
        logging.info(f"Loading decode params from {config.load_parameters_path}")
        model = nnx.eval_shape(
            create_init_fn(model_class, config, rng, None, sharded=False)
        )
        abstract_model_state = nnx.state(model)
        abstract_sharded_model_state = jax.tree.map(
            lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
            abstract_model_state,
            nnx.get_named_sharding(abstract_model_state, mesh),
        )
        model_state = checkpointing.load_params_from_path(
            config.load_parameters_path, abstract_sharded_model_state
        )
        nnx.update(model, model_state)

    return model  # type: ignore[return-value]


# Distributed system initialization.
# -----------------------------------------------------------------------------
def maybe_initialize_jax_distributed_system(config: default.Config) -> None:
    """Provide the best recipe to initialize the Jax Distributed System.

    The best recipe to initialize the Jax Distributed System has varied over time.
    We keep a layer of indirection to avoid breaking the call sites unnecessarily.

    Currently jax.distributed.initialize() fully works as expected!

    For CPUs, we call jax.distributed.initialize() explicitly, with the specified
    arguments.
    """
    if is_gpu_backend(config):
        logging.info(
            "Attempting to initialize the jax distributed system for GPU backend..."
        )
        initialize_jax_for_gpu()
        logging.info("Jax distributed system initialized on GPU!")
    elif is_cpu_backend(config):
        logging.info(
            "Attempting to initialize the jax distributed system for CPU backend..."
        )
        initialize_jax_for_cpu()
        logging.info("Jax distributed system initialized on CPUs!")
    elif (
        config.save_checkpoints and config.async_checkpointing
    ) or config.hardware == "gpu_multiprocess":
        logging.info("Attempting to initialize the jax distributed system...")
        initialize_jax_for_tpu_with_emergency_checkpointing(config)
        logging.info("Jax distributed system initialized!")


def initialize_jax_for_gpu() -> None:
    """Jax distributed initialize for GPUs."""
    if os.environ.get("JAX_COORDINATOR_IP") is not None:
        coordinator_ip = str(os.getenv("JAX_COORDINATOR_IP"))
        coordinator_port = str(os.getenv("JAX_COORDINATOR_PORT"))
        jax.distributed.initialize(
            coordinator_address=f"{coordinator_ip}:{coordinator_port}",
            num_processes=int(os.getenv("NNODES")),  # type: ignore[arg-type]
            process_id=int(os.getenv("NODE_RANK")),  # type: ignore[arg-type]
        )
        logging.info(f"JAX global devices: {jax.devices()}")


def initialize_jax_for_cpu() -> None:
    """Jax distributed initialize for CPUs.

    Includes retries until the coordinator is ready.
    """
    coordinator_ip_address = get_coordinator_ip_address()
    coordinator_address = (
        coordinator_ip_address + ":1234"
    )  # JAX coordinator port used in XPK
    # Env variables to be set in XPK or otherwise
    job_index = int(os.environ.get("JOB_INDEX"))  # type: ignore[arg-type]
    job_completion_index = int(os.environ.get("JOB_COMPLETION_INDEX"))  # type: ignore[arg-type]
    processes_in_job = int(os.environ.get("PROCESSES_IN_JOB"))  # type: ignore[arg-type]
    pid = job_index * processes_in_job + job_completion_index
    logging.info(f" Jax process id is {pid} ")
    # Explicit initialize is needed only for CPUs
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        process_id=pid,
        num_processes=int(os.environ.get("JAX_PROCESS_COUNT")),  # type: ignore[arg-type]
    )


def initialize_jax_for_tpu_with_emergency_checkpointing(config: default.Config) -> None:
    """Initialize JAX distributed runtime for TPUs when emergency checkpointing is used.

    The information required to initialize JAX distributed runtime will be written by
    GKE to the local checkpoint directory. This function retrieves that information and
    initializes JAX distributed runtime.
    """
    process_id, coordinator_address = _retrieve_jax_init_info(config)

    if process_id != "" and coordinator_address != "":
        logging.info(
            f"Using {process_id} as the process_id and {coordinator_address} as the"
            " coordinator_address to initialize JAX distributed runtime..."
        )
        jax.distributed.initialize(
            coordinator_address=coordinator_address, process_id=int(process_id)
        )
    else:
        logging.info(
            "Initializing JAX distributed runtime without args when emergency "
            "checkpointing is enabled."
            "This should not happen and your workload may have unexpected behavior."
        )
        jax.distributed.initialize()

    ocp.multihost.initialize_runtime_to_distributed_ids()


def _retrieve_jax_init_info(config: default.Config) -> tuple[str, str] | list[str]:
    """Retrieve JAX init info from a local file."""
    local_jax_init_info_file = (
        epath.Path(config.local_checkpoint_directory) / JAX_INIT_INFO_FILE
    )
    # Allow time for the JAX init info file to be populated by GKE. This is needed
    # because the file is only populated when the worker with process id of 0 is
    # determined. After a disruption, although some workers might be up and running,
    # the init info file won't be populated until the node with process id is known
    # and this could take time. Using 900 seconds for now and it needs to be increased
    # if the "repair" time is longer.
    for i in range(900):
        if local_jax_init_info_file.exists():
            return local_jax_init_info_file.read_text().split("\n")[:2]
        logging.info(
            f"Unable to locate {JAX_INIT_INFO_FILE} after {i} seconds, "
            "sleeping for 1 second before retrying..."
        )
        time.sleep(1)
    logging.info(
        f"Unable to locate {JAX_INIT_INFO_FILE} after 900 seconds,"
        "returning empty process id and coordinator address."
    )
    return "", ""


def is_cpu_backend(config: default.Config) -> bool:
    """Determine whether Maxtext is intended to run on a CPU backend."""
    return config.hardware == "cpu"


def is_gpu_backend(config: default.Config) -> bool:
    """Determine whether Maxtext is intended to run on a GPU backend."""
    return config.hardware == "gpu"


def get_coordinator_ip_address() -> str:
    """Get coordinator IP Address with retries."""
    coordinator_address = ""
    coordinator_ip_address = ""
    if os.environ.get("JAX_COORDINATOR_ADDRESS") is not None:
        coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
        coordinator_found = False
        lookup_attempt = 1
        max_coordinator_lookups = 50
        while not coordinator_found and lookup_attempt <= max_coordinator_lookups:
            try:
                coordinator_ip_address = socket.gethostbyname(coordinator_address)  # type: ignore[arg-type]
                coordinator_found = True
            except socket.gaierror:
                logging.info(
                    f"Failed to recognize coordinator address {coordinator_address} "
                    f"on attempt {lookup_attempt}, retrying..."
                )
                lookup_attempt += 1
                time.sleep(5)
    logging.info(f"Coordinator IP address: {coordinator_ip_address}")
    return coordinator_ip_address
