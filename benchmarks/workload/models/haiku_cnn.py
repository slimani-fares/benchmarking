"""Haiku/JAX MNIST CNN.

Currently a deferred stub. Haiku's `hk.Conv2D` defaults to NHWC and the
declearn `HaikuModel` requires a purely-functional model definition plus
a sample-wise loss; building this end-to-end alongside reliable smoke
tests was de-scoped in the v1 build. See `benchmarks/NOTES.md` for the
specifics and the proposed shape (HWC layout, sparse-categorical-style
loss derived from log-softmax outputs).
"""

from declearn.model.api import Model

__all__ = ["build_model"]


def build_model() -> Model:
    """Raise an informative error pointing at the deferred state."""
    raise NotImplementedError(
        "The haiku backend is not implemented in v1 of the benchmark "
        "suite. See benchmarks/NOTES.md for the deferral rationale and "
        "for the sketch of what a working build_model() should look "
        "like."
    )
