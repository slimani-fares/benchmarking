"""DP-SGD-compatible MNIST CNN built with PyTorch.

declearn's DP-SGD path computes per-sample gradients via `torch.func.vmap`,
which strips the batch dimension before each forward call. The default
`torch.nn.Flatten` and `torch.nn.Unflatten(dim=0, ...)` both assume that
dim 0 is the batch dim, so they break under vmap. `FlexibleFlatten` works
under both regular forward (4D input) and vmap forward (3D input).

This model is used only when `dp=True`; eval still runs in regular forward
(4D), training runs under vmap (3D).
"""

import torch

from declearn.model.api import Model
from declearn.model.torch import TorchModel

__all__ = ["build_model"]


class FlexibleFlatten(torch.nn.Module):
    """Flatten that tolerates the missing batch dim introduced by vmap."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x.flatten(start_dim=1)
        return x.flatten()


def build_model() -> Model:
    """Return a `TorchModel` shaped like `torch_cnn` but vmap-safe."""
    network = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, 1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Dropout(0.25),
        FlexibleFlatten(),
        torch.nn.Linear(1352, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(64, 10),
        torch.nn.Softmax(dim=-1),
    )
    return TorchModel(network, loss=torch.nn.CrossEntropyLoss())
