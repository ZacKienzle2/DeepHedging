"""Network building blocks shared by the policies and the BSDE solvers."""

from itertools import pairwise

from torch import nn


def mlp(in_features: int, hidden_sizes: tuple[int, ...], out_features: int) -> nn.Sequential:
    """Builds a multilayer perceptron with SiLU activations.

    Args:
        in_features: Width of the input.
        hidden_sizes: Widths of the hidden layers, in order.
        out_features: Width of the output.

    Returns:
        Linear layers of the given widths joined by SiLU, with no activation
        after the last.
    """
    layers: list[nn.Module] = []
    for width, size in pairwise((in_features, *hidden_sizes, out_features)):
        layers += (nn.Linear(width, size), nn.SiLU())
    return nn.Sequential(*layers[:-1])
