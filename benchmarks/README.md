# declearn benchmark suite

This folder is a self-contained ASV benchmark suite for declearn. It
replaces the previous ad-hoc `quickrun`-based benchmarking, which had
several hard-coded constraints that made it unsuitable for systematic
sweeps:

- `secagg=None` was hardcoded in `quickrun/_run.py`
- `InMemoryDataset` was wired in where `FairnessInMemoryDataset` is
  needed for fairness experiments
- the client count was tied to the number of subdirectories in the
  data folder, so changing it required regenerating data manually

The suite drives `FederatedServer` and `FederatedClient` directly via
declearn's main API and bypasses `quickrun` entirely.

## Architecture

Three layers, each in its own subfolder/file:

1. **Workload builder** (`workload/`): turns a small set of high-level
   parameters (backend, n_clients, scaffold, secagg, …) into a
   fully-instantiated `BenchmarkSpec`. This is where parameter
   interpretation happens.
2. **Runner** (`workload/runner.py`): given a `BenchmarkSpec`, spawns
   the server and N clients on `asyncio` and awaits completion.
3. **ASV classes** (`__init__.py`): thin wrappers that declare which
   parameter slice each benchmark exercises. ASV times each `time_*`
   method.

## Adding a new benchmark category

1. Add the new axis (or axis combination) to `workload/build.py`'s
   `build_benchmark(...)` signature, and update its validation logic
   to reject combinations declearn cannot honor.
2. Add a class to `__init__.py` declaring the axis values via
   `params` / `param_names`, with `setup` triggering data preparation
   and `time_run` calling `build_benchmark` + `run_benchmark`.

## Running

Single category, single Python:
```
cd benchmarks
asv run --python=same --quick -b BackendsBenchmark
```

Cross-version sweep (uses the venv pointed to by `$BENCH_VENV`,
defaulting to `~/.venvs/declearn311`):
```
./run_benchmarks.sh           # default versions
./run_benchmarks.sh 2.7.0 2.8.0
```

Then publish/visualize with `asv publish` / `asv preview`.

## Cluster setup

To bootstrap a benchmark-ready environment on a fresh cluster node
(e.g. Magnet), use the included script:

```
./bootstrap_cluster.sh
```

It creates `~/.venvs/declearn-bench-gpu` (override with `BENCH_VENV=`),
installs PyTorch with CUDA 12.1 wheels (compatible with 12.6+ drivers),
declearn + extras, and ASV. It finishes by verifying torch and
tensorflow can both see the GPU. Idempotent.

## GPU enforcement

declearn's default device policy is `gpu=True` (auto-detect), so the
suite uses any available CUDA GPU automatically. To **fail loudly**
when GPU isn't available — useful in CI / on a cluster, where a
silent CPU fallback would corrupt comparison graphs with a fake
10× regression — export:

```
DECLEARN_BENCH_FORCE_GPU=1
```

before invoking `asv run`. With that set, every `run_benchmark` call
verifies `torch.cuda.is_available()` and pins
`set_device_policy(gpu=True)`, raising `RuntimeError` on any CPU
fallback.

## Baseline configuration

Every benchmark category varies exactly one axis from the fixed
baseline declared in [`workload/baseline.py`](workload/baseline.py):
torch + Adam (lrate 1e-3) + averaging aggregator + 3 clients +
2 rounds + batch size 48, on websockets at 127.0.0.1:8765.

## Findings, blockers, deferred work

See [`NOTES.md`](NOTES.md).
