import jax
import jax.numpy as jnp
from flax import nnx


class RelativeError(nnx.Metric):
    """A metric that computes the relative error between two values.

    This metric accumulates the sum of errors and true values separately,
    then computes their ratio to get the relative error.
    """

    def __init__(self, argname_1: str = "loss", argname_2: str = "true_value") -> None:
        """Initialize the RelativeError metric.

        Args:
            argname_1: Name of the first argument (default: "loss")
            argname_2: Name of the second argument (default: "true_value")

        """
        self.argname_1, self.argname_2 = argname_1, argname_2
        self.error = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        self.true = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))

    def reset(self) -> None:
        """Reset the metric state by setting error and true values to zero."""
        self.error.value = jnp.array(0, dtype=jnp.float32)
        self.true.value = jnp.array(0, dtype=jnp.float32)

    def update(self, **kwargs) -> None:  # noqa: ANN003
        """Update the metric state with new values.

        Args:
            **kwargs: Keyword arguments containing the values to update the metric.
                     Must include keys matching argname_1 and argname_2.

        Raises:
            TypeError: If required keyword arguments are missing.

        """
        if self.argname_1 not in kwargs:
            msg = f"Expected keyword argument '{self.argname_1}'"
            raise TypeError(msg)
        if self.argname_2 not in kwargs:
            msg = f"Expected keyword argument '{self.argname_2}'"
            raise TypeError(msg)

        error: int | float | jax.Array = kwargs[self.argname_1]
        self.error.value += error if isinstance(error, int | float) else error.mean()

        true_value: int | float | jax.Array = kwargs[self.argname_2]
        self.true.value += (
            true_value if isinstance(true_value, int | float) else true_value.mean()
        )

    def compute(self) -> jax.Array:
        """Compute the relative error by dividing accumulated error by true value.

        Returns:
            jax.Array: The computed relative error value.

        """
        return self.error.value / self.true.value
