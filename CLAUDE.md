# CLAUDE.md

Repo: standalone ASV benchmark suite for [declearn](https://gitlab.inria.fr/magnet/declearn/declearn).

## What this repo is

The actual ASV suite lives under `benchmarks/`. The repo is designed
to live as a **sibling** of a declearn checkout:

```
<parent>/
├── declearn/                  # declearn source repo (read-only here)
└── declearn-benchmarks/       # this repo
    └── benchmarks/            # the ASV suite
```

`benchmarks/asv.conf.json` references declearn at `../../declearn`.

The user operates declearn from CPU locally and runs benchmarks on a
GPU cluster (Magnet at Inria). Per-version runtime is what's
budgeted, not aggregate cross-version time — see the auto-memory
note `project_benchmarks_usage_model.md`.

## Architecture (three layers)

1. `workload/` — turns a small set of high-level toggles
   (`backend`, `n_clients`, `scaffold`, `secagg`, …) into a
   fully-instantiated `BenchmarkSpec`. `build.py` is the only entry
   point. Validation rejects parameter combinations declearn can't
   honor (e.g. SCAFFOLD on non-torch in the v1 suite).
2. `workload/runner.py` — given a `BenchmarkSpec`, spins up
   `FederatedServer` + N `FederatedClient`s on `asyncio` and awaits
   completion. ~50 lines.
3. `benchmarks/__init__.py` — ASV benchmark classes (thin wrappers
   over `build_benchmark` + `run_benchmark`). `params` and
   `param_names` declare which slice of the parameter space each
   class exercises.

## Constraint: do not modify declearn

The original brief was strict: `benchmarks/` is the only place we
write. If a benchmark hits a declearn limitation, document it in
this CLAUDE.md, don't patch declearn.

## Running

Local (CPU dev):
```
cd benchmarks
python -c "from workload.build import build_benchmark; from workload.runner import run_benchmark; run_benchmark(build_benchmark())"
asv check --python=same     # discovery validation, no timing
```

Cluster (GPU production):
```
./bootstrap_cluster.sh                                  # idempotent
source ~/.venvs/declearn-bench-gpu/bin/activate
export DECLEARN_BENCH_FORCE_GPU=1                       # fail loudly on CPU fallback
asv run --python=same --quick --show-stderr
```

## Production env

`bootstrap_cluster.sh` installs the suite into a Python 3.11 venv at
`$BENCH_VENV` (default `~/.venvs/declearn-bench-gpu`). It does *not*
create the venv, install Python, or manage CUDA — the host is expected
to provide a Python 3.11 interpreter and a working CUDA toolchain on
the standard search path (driver + cuDNN libraries).

Two env vars are exported, mirroring declearn's `tox -e py311-ci`:
`TF_FORCE_GPU_ALLOW_GROWTH=true` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
Without these, TF or JAX pre-allocate the entire GPU and starve torch
in side-by-side cells.

`websockets<14.0` is pinned explicitly because pip's resolver
occasionally lets 14.x slip past declearn 2.8.0's `<14.0` constraint;
14.x renamed `extra_headers` → `additional_headers`, breaking
declearn's adapter.

## Benchmarks at a glance

| Class | Cross | Notes |
|---|---|---|
| `BackendsBenchmark` | `n_clients × backend` | torch / TF |
| `RegularizersBenchmark` | `n_clients × regularizer` | torch FedAvg + ridge/fedprox |
| `ScaffoldBenchmark` | `n_clients` | torch SCAFFOLD |
| `SecAggBenchmark` | `n_clients × method` | masking only |

`N_CLIENTS_AXIS = [5]` (in `benchmarks/__init__.py`). Widen to e.g.
`[5, 20]` to add a scaling story; trim per class by overriding
`params` in that class.

## Known gotchas / deferrals

- **n_clients=100** was never smoke-tested in the v1 build; if
  re-enabled, raise `ulimit -n` first (websockets fd usage).

## Setting up Claude Code on a fresh cluster node

The cluster (e.g. Magnet 8) ships with no Node.js. Easiest path is
`nvm`, no admin required:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
exec bash
nvm install 20      # Node >=20; Claude Code's postinstall uses ?. syntax
npm install -g @anthropic-ai/claude-code
cd ~/declearn-benchmarks
claude
```

On first launch, `claude` will prompt to authenticate. SSH-only
nodes can use the device-flow login (a URL + code prompt).

## Conventions

- All new code lives under `benchmarks/`. Don't touch the declearn
  checkout next door.
- Each model file in `models/` exposes a single `build_model() -> Model`.
- `baseline.py` is the source of truth for any value the brief
  declared baseline. New params go to `build.py` signatures, not
  there.
- Prefer existing files over creating new ones; prefer editing the
  ASV class over creating a parallel one.
- The auto-memory at `~/.claude/projects/-home-fslimani-work-declearn/memory/`
  has additional context on the user's role and the usage model.
  Do read it before making design changes.
