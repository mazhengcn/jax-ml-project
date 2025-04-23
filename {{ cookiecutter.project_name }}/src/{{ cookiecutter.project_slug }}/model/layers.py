import jax
from flax import nnx

kernel_init = nnx.initializers.glorot_uniform()
bias_init = nnx.initializers.zeros_init()
activation_fn = nnx.relu


class Mlp(nnx.Module):
    """A multi-layer perceptron (MLP) module.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        hidden_features (int): Number of hidden features.
        num_layers (int): Number of layers in the MLP.
        rngs (nnx.Rngs): Random number generators for initialization.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        num_layers: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the MLP module.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            hidden_features (int): Number of hidden features.
            num_layers (int): Number of layers in the MLP.
            rngs (nnx.Rngs): Random number generators for initialization.

        """
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers

        layers = []
        for idx in range(num_layers):
            if idx == 0:
                layers.append(
                    nnx.Linear(
                        in_features=in_features,
                        out_features=hidden_features,
                        kernel_init=kernel_init,
                        bias_init=bias_init,
                        rngs=rngs,
                    )
                )
                layers.append(activation_fn)
            elif idx == num_layers - 1:
                layers.append(
                    nnx.Linear(
                        in_features=hidden_features,
                        out_features=out_features,
                        kernel_init=kernel_init,
                        bias_init=bias_init,
                        rngs=rngs,
                    )
                )
            else:
                layers.append(
                    nnx.Linear(
                        in_features=hidden_features,
                        out_features=hidden_features,
                        kernel_init=kernel_init,
                        bias_init=bias_init,
                        rngs=rngs,
                    )
                )
                layers.append(activation_fn)

        self.layers = nnx.Sequential(*layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the MLP to the input array.

        Args:
            x (jax.Array): Input array.

        Returns:
            jax.Array: Output array after applying the MLP.

        """
        return self.layers(x)
