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
`benchmarks/NOTES.md`, don't patch declearn.

## Running

Local (CPU dev):
```
cd benchmarks
python -c "from workload.build import build_benchmark; from workload.runner import run_benchmark; run_benchmark(build_benchmark())"
asv check --python=same     # discovery validation, no timing
```

Cluster (GPU production):
```
./bootstrap_cluster.sh                          # idempotent, see below
source ~/.venvs/declearn-bench-gpu/bin/activate
export DECLEARN_BENCH_FORCE_GPU=1               # fail loudly on CPU fallback
asv run --python=same --quick --show-stderr
```

## bootstrap_cluster.sh quirks

- **Pinned to Python 3.11** (`PY_VERSION=3.11`). PyTorch CUDA wheels
  lag the latest Python; conda's base ships 3.13, which breaks
  `pip install torch`. Don't change this without verifying torch
  wheels still exist for the new pin.
- **Uses conda only to obtain a python3.11 binary**, then seeds a
  regular venv from it. ASV never touches conda.
- **conda-forge channel** to dodge Anaconda's default-channel ToS
  prompt that newer Miniconda installs require.
- **Torch installed *after* declearn** with `--force-reinstall
  --no-deps` from `--index-url cu121`. declearn[torch] pulls a PyPI
  torch as a side effect; we override it. Without this you end up
  with a CUDA 13 build that the driver can't load.
- **`websockets<14.0` explicit pin**. websockets 14 renamed
  `extra_headers` to `additional_headers`; declearn 2.8.0's
  websockets adapter still uses the old name. declearn pins
  `<14.0` in pyproject but pip's resolver sometimes lets newer
  versions through; the explicit pin defends against this.

## Benchmarks at a glance

| Class | Cross | Notes |
|---|---|---|
| `BackendsBenchmark` | `n_clients × backend` | torch / TF / haiku |
| `SklearnBenchmark` | `n_clients` (trimmed axis) | split out — sklearn is ~20× slower |
| `RegularizersBenchmark` | `n_clients × regularizer` | torch FedAvg + lasso/ridge/fedprox |
| `ScaffoldBenchmark` | `n_clients` | torch SCAFFOLD |
| `SecAggBenchmark` | `n_clients × method` | masking only — joye-libert dropped, see NOTES |

`N_CLIENTS_AXIS = [5, 20]` (in `benchmarks/__init__.py`). Trim per
class by overriding `params` in that class.

## Known gotchas / deferrals

- **haiku backend** is a stub raising `NotImplementedError`. Sketch
  for a working impl is in `benchmarks/NOTES.md`.
- **joye-libert SecAgg** dropped from the suite — modular
  exponentiation × CNN parameter count exceeds practical timeouts.
- **n_clients=100** was never smoke-tested in the v1 build; if
  re-enabled, raise `ulimit -n` first (websockets fd usage).

## Setting up Claude Code on a fresh cluster node

The cluster (e.g. Magnet 8) ships with no Node.js, no npm, no conda
out of the box. The recipe below assumes you've already run
`benchmarks/bootstrap_cluster.sh` once (which installs Miniconda
under `~/miniconda3` if it wasn't there). After that, Claude Code
itself is two commands:

```bash
# 1. Install Node.js >=20 into the conda base env. The version pin
#    matters: Claude Code's postinstall script uses optional-chaining
#    syntax (`?.`) that breaks on Node 12, which is what an unpinned
#    `conda install nodejs` sometimes gives you on conda-forge.
conda install -y -c conda-forge --override-channels 'nodejs>=20'
node --version       # sanity check: should be v20.x or higher

# 2. Install Claude Code globally via the conda env's npm.
npm install -g @anthropic-ai/claude-code

# 3. Run from the repo.
cd ~/benchmarking      # or wherever you cloned this repo on the cluster
claude
```

If conda isn't yet available on the cluster (no Miniconda installed):

```bash
# One-shot Miniconda install (no admin required, ~500 MB under $HOME)
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
$HOME/miniconda3/bin/conda init bash
exec bash      # restart shell so conda is on PATH
# Then proceed with the two-command Claude Code install above.
```

If you'd rather not use conda for Node.js, the no-conda alternative
is `nvm`:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
exec bash
nvm install 20
npm install -g @anthropic-ai/claude-code
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
