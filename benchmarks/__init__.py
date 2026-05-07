"""ASV benchmark classes for declearn.

Each class is a thin wrapper over `build_benchmark(...)` +
`run_benchmark(...)`. ASV discovers them automatically. Heavy lifting
(parameter interpretation, data preparation, network/model setup) lives
in `benchmarks.workload`. The classes here only declare which slice of
the parameter space each benchmark exercises.

The `n_clients` axis is currently a single point (`[5]`) to keep the
per-version sweep under ~2 hours. To restore the scaling story
(n=5 vs n=20), set `N_CLIENTS_AXIS = [5, 20]` below — it is the
single source of truth shared across every category. SecAgg is
deliberately pinned to `[5]` regardless of `N_CLIENTS_AXIS`: the n=20
masking cell timed out in earlier sweeps and has not been
diagnosed; do not re-enable n=20 there until that is investigated.
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
]


_BACKEND_LAYOUT = {
    "torch": "chw",
    "tensorflow": "hwc",
    "sklearn": "flat",
    "haiku": "hwc",
}

# Single source of truth for the n_clients sweep across every category.
# Trimmed to a single point to keep the per-version runtime bounded.
N_CLIENTS_AXIS: List[int] = [5]


class BackendsBenchmark:
    """Sweep fast model backends and client count on the FedAvg baseline.

    sklearn used to be its own class (`SklearnBenchmark`) but was
    dropped from the suite because a single FL round on the baseline
    takes ~10 min and it dominated per-version runtime. haiku was
    dropped because its `build_model()` is still a stub raising
    `NotImplementedError`; reintroduce both by adding the backend back
    to the params tuple once their respective issues are resolved.
    """

    timeout = 900.0
    params = (N_CLIENTS_AXIS, ["torch", "tensorflow"])
    param_names = ["n_clients", "backend"]

    def setup(self, n_clients: int, backend: str) -> None:
        ensure_data_for_n_clients(n_clients, _BACKEND_LAYOUT[backend])

    def time_run(self, n_clients: int, backend: str) -> None:
        spec = build_benchmark(backend=backend, n_clients=n_clients)
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

    Pinned to `n_clients=[5]` independently of `N_CLIENTS_AXIS`: the
    n=20 masking cell timed out in earlier sweeps (root cause not yet
    diagnosed). Do not widen until that is resolved.

    Joye-Libert is intentionally left out: its modular-exponentiation
    cost scales with the model parameter count, and a single 3-client
    run on the CNN baseline did not finish in 5 min during smoke
    testing. To re-enable it once a tractable configuration is
    settled (smaller `bitsize`, a smaller model, or a more efficient
    declearn implementation), turn `params` back into a 2D tuple
    `(N_CLIENTS_AXIS, ["masking", "joye-libert"])` and bump `timeout`.
    """

    timeout = 1200.0
    params = ([5], ["masking"])
    param_names = ["n_clients", "secagg"]

    def setup(self, n_clients: int, secagg: str) -> None:
        ensure_data_for_n_clients(n_clients, "chw")

    def time_run(self, n_clients: int, secagg: str) -> None:
        spec = build_benchmark(
            backend="torch", secagg=secagg, n_clients=n_clients
        )
        run_benchmark(spec)
