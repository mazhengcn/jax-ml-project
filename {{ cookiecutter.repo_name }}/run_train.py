"""The file is intentionally kept short.

The majority for logic is in libraries
that can be easily tested and imported in Colab.
"""

import jax
from absl import app, flags, logging
from clu import platform

from {{ cookiecutter.module_name }}.configs import default
from {{ cookiecutter.module_name }}.train_lib import train

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_string(
    "config", None, "File path to the training hyperparameter configuration."
)
flags.mark_flags_as_required(["config", "workdir"])


def main(argv: list[str]) -> None:
    """Run the training script.

    Args:
        argv (list[str]): Command-line arguments.

    """
    if len(argv) > 1:
        error_message = "Too many command-line arguments."
        raise app.UsageError(error_message)

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    # tf.config.experimental.set_visible_devices([], "GPU")  # noqa: ERA001

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, process_count: {jax.process_count()}"
    )
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir"
    )

    # Load the configuration.
    config = default.get_config(FLAGS.config)
    # Train and evaluate
    train.train_and_evaluate(config, FLAGS.workdir)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
