# declearn-benchmarks

This is the benchmarking folder for declearn, developed
as a standalone repository. The end goal is for this suite to live at
`declearn/benchmarks/` inside the declearn repo once the work is ready
to be merged upstream.

## Layout

```
benchmarks/                # the ASV suite
├── asv.conf.json
├── bootstrap_cluster.sh
├── run_benchmarks.sh
├── __init__.py            # ASV benchmark classes
├── data/                  # generated MNIST shards (gitignored)
└── workload/
    ├── baseline.py
    ├── build.py
    ├── data.py
    ├── runner.py
    ├── spec.py
    └── models/
        ├── torch_cnn.py
        └── tensorflow_cnn.py
```

`benchmarks/asv.conf.json` resolves declearn at `../../declearn` (for now ).

## Architecture

```
ASV layer  ─────►  build  ─────►  runner
```

1. **ASV** ([benchmarks/__init__.py](benchmarks/__init__.py)) — declares benchmark classes; each picks one parameter slice.
2. **Build** ([workload/build.py](benchmarks/workload/build.py)) — turns toggles into a `BenchmarkSpec`.
3. **Runner** ([workload/runner.py](benchmarks/workload/runner.py)) — runs the spec as a local asyncio FL experiment.

## File-by-file

| File | Role |
|---|---|
| [benchmarks/__init__.py](benchmarks/__init__.py) | ASV entry: four benchmark classes + memory-cache helpers |
| [workload/baseline.py](benchmarks/workload/baseline.py) | Single source of truth for every baseline value (lrate, batch size, rounds, dataset fraction, ports, etc.) |
| [workload/build.py](benchmarks/workload/build.py) | `build_benchmark(...)` and per-aspect helpers (`_build_model`, `_build_optim`, `_build_run_config`, `_build_secagg`, `_build_clients`) |
| [workload/data.py](benchmarks/workload/data.py) | Idempotent MNIST split → layout-specific NumPy shards on disk |
| [workload/runner.py](benchmarks/workload/runner.py) | `run_benchmark(spec)`: device-policy enforcement + asyncio FL loop |
| [workload/spec.py](benchmarks/workload/spec.py) | `BenchmarkSpec` and `ClientSpec` dataclasses |
| [workload/models/torch_cnn.py](benchmarks/workload/models/torch_cnn.py) | `build_model() -> TorchModel` (CHW input) |
| [workload/models/tensorflow_cnn.py](benchmarks/workload/models/tensorflow_cnn.py) | `build_model() -> TensorflowModel` (HWC input) |

## The four benchmark classes

| Class | Axes swept | Notes |
|---|---|---|
| `BackendsBenchmark` | `n_clients × backend` | torch / tensorflow on FedAvg |
| `RegularizersBenchmark` | `n_clients × regularizer` | torch FedAvg + ridge / fedprox |
| `ScaffoldBenchmark` | `n_clients` | torch SCAFFOLD |
| `SecAggBenchmark` | `n_clients × method` | masking only; pinned to `n_clients=[5]` (bigger client count takes much longer)|



## Running

### 1. Prerequisites

- Python 3.11
- A clone of declearn (this suite lives at `declearn/benchmarks/`).

### 2. Create and activate a venv

```bash
cd declearn
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Install dependencies

```bash
pip install -e ".[torch,tensorflow,websockets]"   # declearn itself
pip install asv cryptography "websockets<14.0"    # suite-specific
```

The `websockets<14.0` pin is required — declearn breaks on 14.x.

### 4. Smoke test (optional, no ASV)

Runs one benchmark end-to-end to confirm the suite is wired up. Run from the declearn root (the parent of `benchmarks/`), not from inside `benchmarks/`:

```bash
python -c "from benchmarks.workload import build_benchmark, run_benchmark; run_benchmark(build_benchmark())"
```

### 5. Run the sweep

```bash
cd benchmarks
asv check --python=same                       # multiple samples per cell to avoid noise
asv run   --python=same --quick --show-stderr # one sample per cell
```

`--quick` runs each benchmark once; drop it for multi-sample timings (slower but tighter).

### 6. View results

```bash
asv publish    # renders HTML into .asv/html/
asv preview    # serves it at http://localhost:8080
```

