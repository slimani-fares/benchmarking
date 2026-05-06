"""ASV benchmark classes for declearn.

Each class is a thin wrapper over `build_benchmark(...)` +
`run_benchmark(...)`. ASV discovers them automatically. Heavy lifting
(parameter interpretation, data preparation, network/model setup) lives
in `benchmarks.workload`. The classes here only declare which slice of
the parameter space each benchmark exercises.

Every category is crossed with `n_clients` so the suite captures how
each feature scales with client count. To narrow a category to a
single client count, edit its `params` / `param_names` to drop the
`n_clients` axis.
"""

from typing import List

from benchmarks.workload import build_benchmark, run_benchmark
from benchmarks.workload.data import ensure_data_for_n_clients

__all__ = [
    "BackendsBenchmark",
    "DPBenchmark",
    "RegularizersBenchmark",
    "ScaffoldBenchmark",
    "SecAggBenchmark",
    "SklearnBenchmark",
]


_BACKEND_LAYOUT = {
    "torch": "chw",
    "tensorflow": "hwc",
    "sklearn": "flat",
    "haiku": "hwc",
}

# Single source of truth for the n_clients sweep across every category.
# Trim per-class if a particular cross becomes too slow: override
# `params[0]` in that class. Two points (small + medium) are the
# minimum that still produces a scaling line; widen to e.g.
# `[2, 5, 10, 100]` to surface intermediate behavior.
N_CLIENTS_AXIS: List[int] = [5, 20]

# sklearn is much slower per round than the other backends — it iterates
# more steps per epoch on the same dataset — so its axis tracks the
# global one but n=100 should remain off-limits (one such run is
# ~30–60 min on its own).
N_CLIENTS_AXIS_SKLEARN: List[int] = [5, 20]


class BackendsBenchmark:
    """Sweep fast model backends and client count on the FedAvg baseline.

    sklearn is excluded from this class — see `SklearnBenchmark` — to
    keep the cross-product runtime bounded.
    """

    timeout = 900.0
    params = (N_CLIENTS_AXIS, ["torch", "tensorflow", "haiku"])
    param_names = ["n_clients", "backend"]

    def setup(self, n_clients: int, backend: str) -> None:
        ensure_data_for_n_clients(n_clients, _BACKEND_LAYOUT[backend])

    def time_run(self, n_clients: int, backend: str) -> None:
        spec = build_benchmark(backend=backend, n_clients=n_clients)
        run_benchmark(spec)


class SklearnBenchmark:
    """sklearn FedAvg over a trimmed client-count axis.

    Split out of `BackendsBenchmark` because sklearn runs are much
    slower per round (a single FL round at the baseline takes ~3 min
    where torch/TF take ~20s). Restricting to `n_clients in [2, 5]`
    keeps this from dominating per-version runtime.
    """

    timeout = 1200.0
    params = N_CLIENTS_AXIS_SKLEARN
    param_names = ["n_clients"]

    def setup(self, n_clients: int) -> None:
        ensure_data_for_n_clients(n_clients, "flat")

    def time_run(self, n_clients: int) -> None:
        spec = build_benchmark(backend="sklearn", n_clients=n_clients)
        run_benchmark(spec)


class RegularizersBenchmark:
    """Sweep client-side loss regularizers and client count (torch FedAvg)."""

    timeout = 600.0
    params = (N_CLIENTS_AXIS, ["lasso", "ridge", "fedprox"])
    param_names = ["n_clients", "regularizer"]

    def setup(self, n_clients: int, regularizer: str) -> None:
        ensure_data_for_n_clients(n_clients, "chw")

    def time_run(self, n_clients: int, regularizer: str) -> None:
        spec = build_benchmark(
            backend="torch", regularizer=regularizer, n_clients=n_clients
        )
        run_benchmark(spec)


class DPBenchmark:
    """Sweep client count for DP-SGD on torch."""

    timeout = 900.0
    params = N_CLIENTS_AXIS
    param_names = ["n_clients"]

    def setup(self, n_clients: int) -> None:
        ensure_data_for_n_clients(n_clients, "chw")

    def time_run(self, n_clients: int) -> None:
        spec = build_benchmark(backend="torch", dp=True, n_clients=n_clients)
        run_benchmark(spec)


class ScaffoldBenchmark:
    """Sweep client count for SCAFFOLD on torch."""

    timeout = 600.0
    params = N_CLIENTS_AXIS
    param_names = ["n_clients"]

    def setup(self, n_clients: int) -> None:
        ensure_data_for_n_clients(n_clients, "chw")

    def time_run(self, n_clients: int) -> None:
        spec = build_benchmark(
            backend="torch", scaffold=True, n_clients=n_clients
        )
        run_benchmark(spec)


class SecAggBenchmark:
    """SecAgg masking sweep over client count on torch.

    Joye-Libert is intentionally left out: its modular-exponentiation
    cost scales with the model parameter count, and a single 3-client
    run on the CNN baseline did not finish in 5 min during smoke
    testing. To re-enable it once a tractable configuration is
    settled (smaller `bitsize`, a smaller model, or a more efficient
    declearn implementation), turn `params` back into a 2D tuple
    `(N_CLIENTS_AXIS, ["masking", "joye-libert"])` and bump `timeout`.
    """

    timeout = 1200.0
    params = (N_CLIENTS_AXIS, ["masking"])
    param_names = ["n_clients", "secagg"]

    def setup(self, n_clients: int, secagg: str) -> None:
        ensure_data_for_n_clients(n_clients, "chw")

    def time_run(self, n_clients: int, secagg: str) -> None:
        spec = build_benchmark(
            backend="torch", secagg=secagg, n_clients=n_clients
        )
        run_benchmark(spec)


