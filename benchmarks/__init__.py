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

Memory tracking: `time_run` measures peak host-RSS-delta and peak
GPU bytes as side effects (microseconds; invisible against the
~70-180s workload) and writes both to a `/tmp` cache. The
`track_peakmem_run` and `track_peakgpu_run` methods only read from
that cache — no extra workload runs, so the sweep stays the same
length. The delta-RSS form sidesteps the "setup dominates peak"
blind spot; the GPU number sidesteps "ru_maxrss misses VRAM".
"""

import json
import os
import resource
import tempfile
from typing import List, Tuple

import torch

from benchmarks.workload import build_benchmark, run_benchmark
from benchmarks.workload.build import BACKEND_LAYOUT
from benchmarks.workload.data import ensure_data_for_n_clients


def _cuda() -> bool:
    return torch.cuda.is_available()


def _cache_path(cls_name: str, params: Tuple) -> str:
    key = "_".join(str(p) for p in params)
    return os.path.join(
        tempfile.gettempdir(), f"declearn_mem_{cls_name}_{key}.json"
    )


def _mem_capture_start() -> int:
    if _cuda():
        torch.cuda.reset_peak_memory_stats()
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _mem_capture_end(cls_name: str, params: Tuple, baseline_kb: int) -> None:
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    host_delta_bytes = max(peak_kb - baseline_kb, 0) * 1024
    gpu_bytes = torch.cuda.max_memory_allocated() if _cuda() else 0
    with open(_cache_path(cls_name, params), "w") as f:
        json.dump(
            {"host_delta_bytes": host_delta_bytes, "gpu_bytes": gpu_bytes}, f
        )


def _read_cached(cls_name: str, params: Tuple, key: str) -> int:
    try:
        with open(_cache_path(cls_name, params)) as f:
            return int(json.load(f).get(key, 0))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


__all__ = [
    "BackendsBenchmark",
    "RegularizersBenchmark",
    "ScaffoldBenchmark",
    "SecAggBenchmark",
]


# Single source of truth for the n_clients sweep across every category.
# Trimmed to a single point to keep the per-version runtime bounded.
N_CLIENTS_AXIS: List[int] = [5]


class BackendsBenchmark:
    """Sweep model backends and client count on the FedAvg baseline."""

    timeout = 900.0
    params = (N_CLIENTS_AXIS, ["torch", "tensorflow"])
    param_names = ["n_clients", "backend"]

    def setup(self, n_clients: int, backend: str) -> None:
        ensure_data_for_n_clients(n_clients, BACKEND_LAYOUT[backend])

    def time_run(self, n_clients: int, backend: str) -> None:
        baseline = _mem_capture_start()
        spec = build_benchmark(backend=backend, n_clients=n_clients)
        run_benchmark(spec)
        _mem_capture_end("BackendsBenchmark", (n_clients, backend), baseline)

    def track_peakmem_run(self, n_clients: int, backend: str) -> int:
        return _read_cached(
            "BackendsBenchmark", (n_clients, backend), "host_delta_bytes"
        )
    track_peakmem_run.unit = "bytes"  # type: ignore[attr-defined]

    def track_peakgpu_run(self, n_clients: int, backend: str) -> int:
        # Only torch tensors are tracked by torch.cuda. TF would need its
        # own probe; skip the cell rather than report a misleading 0.
        if backend != "torch":
            raise NotImplementedError("GPU mem only tracked for torch backend")
        return _read_cached(
            "BackendsBenchmark", (n_clients, backend), "gpu_bytes"
        )
    track_peakgpu_run.unit = "bytes"  # type: ignore[attr-defined]


class RegularizersBenchmark:
    """Sweep client-side loss regularizers and client count (torch FedAvg)."""

    timeout = 600.0
    params = (N_CLIENTS_AXIS, ["ridge", "fedprox"])
    param_names = ["n_clients", "regularizer"]

    def setup(self, n_clients: int, regularizer: str) -> None:
        ensure_data_for_n_clients(n_clients, "chw")

    def time_run(self, n_clients: int, regularizer: str) -> None:
        baseline = _mem_capture_start()
        spec = build_benchmark(
            backend="torch", regularizer=regularizer, n_clients=n_clients
        )
        run_benchmark(spec)
        _mem_capture_end(
            "RegularizersBenchmark", (n_clients, regularizer), baseline
        )

    def track_peakmem_run(self, n_clients: int, regularizer: str) -> int:
        return _read_cached(
            "RegularizersBenchmark",
            (n_clients, regularizer),
            "host_delta_bytes",
        )
    track_peakmem_run.unit = "bytes"  # type: ignore[attr-defined]

    def track_peakgpu_run(self, n_clients: int, regularizer: str) -> int:
        return _read_cached(
            "RegularizersBenchmark", (n_clients, regularizer), "gpu_bytes"
        )
    track_peakgpu_run.unit = "bytes"  # type: ignore[attr-defined]


class ScaffoldBenchmark:
    """Sweep client count for SCAFFOLD on torch."""

    timeout = 600.0
    params = N_CLIENTS_AXIS
    param_names = ["n_clients"]

    def setup(self, n_clients: int) -> None:
        ensure_data_for_n_clients(n_clients, "chw")

    def time_run(self, n_clients: int) -> None:
        baseline = _mem_capture_start()
        spec = build_benchmark(
            backend="torch", scaffold=True, n_clients=n_clients
        )
        run_benchmark(spec)
        _mem_capture_end("ScaffoldBenchmark", (n_clients,), baseline)

    def track_peakmem_run(self, n_clients: int) -> int:
        return _read_cached(
            "ScaffoldBenchmark", (n_clients,), "host_delta_bytes"
        )
    track_peakmem_run.unit = "bytes"  # type: ignore[attr-defined]

    def track_peakgpu_run(self, n_clients: int) -> int:
        return _read_cached("ScaffoldBenchmark", (n_clients,), "gpu_bytes")
    track_peakgpu_run.unit = "bytes"  # type: ignore[attr-defined]


class SecAggBenchmark:
    """SecAgg masking sweep over client count on torch.

    Pinned to `n_clients=[5]` independently of `N_CLIENTS_AXIS`: the
    n=20 masking cell timed out in earlier sweeps (root cause not yet
    diagnosed). Do not widen until that is resolved.
    """

    timeout = 1200.0
    params = ([5], ["masking"])
    param_names = ["n_clients", "secagg"]

    def setup(self, n_clients: int, secagg: str) -> None:
        ensure_data_for_n_clients(n_clients, "chw")

    def time_run(self, n_clients: int, secagg: str) -> None:
        baseline = _mem_capture_start()
        spec = build_benchmark(
            backend="torch", secagg=secagg, n_clients=n_clients
        )
        run_benchmark(spec)
        _mem_capture_end("SecAggBenchmark", (n_clients, secagg), baseline)

    def track_peakmem_run(self, n_clients: int, secagg: str) -> int:
        return _read_cached(
            "SecAggBenchmark", (n_clients, secagg), "host_delta_bytes"
        )
    track_peakmem_run.unit = "bytes"  # type: ignore[attr-defined]

    def track_peakgpu_run(self, n_clients: int, secagg: str) -> int:
        return _read_cached(
            "SecAggBenchmark", (n_clients, secagg), "gpu_bytes"
        )
    track_peakgpu_run.unit = "bytes"  # type: ignore[attr-defined]
