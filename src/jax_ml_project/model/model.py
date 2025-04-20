import dataclasses

import jax
from flax import nnx

from .layers import Mlp


@dataclasses.dataclass(unsafe_hash=True)
class ModelConfig:
    """Configuration for the Model."""

    # Input dimension.
    in_dim: int = 2
    #  Output dimension.
    out_dim: int = 1
    # Number of MLP layers.
    num_mlp_layers: int = 4
    # MLP dimension.
    mlp_dim: int = 128


class Model(nnx.Module):
    """A neural network model with an MLP backbone."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize the Model with the given configuration.

        Args:
            config (ModelConfig): Configuration for the model.
            rngs (nnx.Rngs): Random number generators.

        """
        self.config = config

        self.mlp = Mlp(
            in_features=config.in_dim,
            out_features=config.out_dim,
            hidden_features=config.mlp_dim,
            num_layers=config.num_mlp_layers,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the model to the input data.

        Args:
            x (jax.Array): Input data.

        Returns:
            jax.Array: Output of the model.

        """
        return self.mlp(x)
