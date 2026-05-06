"""Linear MNIST classifier built with scikit-learn's SGDClassifier.

scikit-learn does not support CNNs, so this baseline uses a flat 784-d
input and a logistic-regression-style linear model with l2 penalty.
The model is architecturally different from the CNN-backed backends:
it is roughly an order of magnitude faster but reaches lower accuracy,
so accuracy comparisons across backends are not meaningful — runtime
comparisons still are, since timing is what ASV measures.
"""

from declearn.model.api import Model
from declearn.model.sklearn import SklearnSGDModel

__all__ = ["build_model"]


def build_model() -> Model:
    """Return a `SklearnSGDModel` configured as a multinomial classifier."""
    return SklearnSGDModel.from_parameters(
        kind="classifier",
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
    )
