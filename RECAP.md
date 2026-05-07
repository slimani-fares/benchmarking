# Benchmarking System — Full Recap

A dump of everything that's currently in place: the hardware, the
suite, the benchmarks we run, the environment quirks we worked
around, the configs, the patches we benchmark against, and the
state on disk. No structure for presentation — context first.

---

## 1. Hardware & cluster

- Cluster node: **Magnet 8** at Inria Lille (`fslimani@magnet8`).
- GPU: **NVIDIA A10**, 23 GB, compute capability 8.6.
- CUDA driver: 570.169, ships CUDA 12.8.
- Other than the GPU, this is a shared Linux node — no admin rights,
  everything lives under `$HOME`.
- Local dev box: CPU only. The user (Fares) develops declearn locally
  on CPU and runs the benchmark sweeps on Magnet for GPU coverage.

Cluster filesystem layout used by the suite:

```
/home/fslimani/
├── declearn/                      # declearn source (read-only, git checkout)
├── declearn-patched/              # local fork of declearn with 4 perf patches
├── benchmarking/                  # this repo (the ASV benchmark suite)
│   ├── benchmarks/
│   │   ├── workload/              # builder + runner
│   │   ├── data/                  # MNIST shards (gitignored)
│   │   └── .asv/                  # ASV's working dir; only results/ is tracked
│   └── ...
├── internship-writeups/           # the 4 perf-patch writeups (compression, dp, secagg, vector)
├── miniconda3/                    # used only to obtain a Python 3.11 binary
└── .venvs/declearn-bench-gpu/     # the actual venv all benchmarking runs inside
```

The benchmarking repo expects to live next to `declearn/` because
`asv.conf.json:5` resolves the project repo via `repo: "../../declearn"`.

---

## 2. Why we built this (and why it isn't quickrun)

declearn historically benchmarked itself via `quickrun` (a config-file-
driven launcher that orchestrates server + clients on asyncio). For
systematic sweeps it had three hardcoded constraints that made it
unusable:

1. `secagg=None` was hardcoded in `quickrun/_run.py`, so SecAgg
   (one of the more interesting benchmark axes) was unreachable.
2. `InMemoryDataset` was wired in where `FairnessInMemoryDataset` is
   needed for fairness experiments.
3. The number of clients was tied to the number of subdirectories
   in the data folder, so changing N required regenerating data
   manually.

This repo bypasses quickrun entirely. It drives `FederatedServer` +
N `FederatedClient`s directly via declearn's main API. The original
brief that produced the v1 of the suite is preserved at
`build_benchmarks_prompt.md`. The constraint from that brief that's
still in force: **we don't modify any file outside `benchmarks/`**;
declearn is treated as read-only.

The exception is the patched fork (see §10) — that's a deliberate
side-by-side comparison, not an upstream change.

---

## 3. Architecture (three layers)

```
benchmarks/
├── __init__.py                   # ASV class definitions (Layer 3)
├── asv.conf.json
├── bootstrap_cluster.sh          # one-shot env setup
├── run_benchmarks.sh             # cross-version sweep launcher
├── workload/
│   ├── baseline.py               # the fixed baseline config (Layer 1 constants)
│   ├── build.py                  # build_benchmark(...) -> BenchmarkSpec (Layer 1)
│   ├── runner.py                 # run_benchmark(spec) (Layer 2, ~50 lines)
│   ├── data.py                   # MNIST split + per-layout caching
│   ├── spec.py                   # BenchmarkSpec dataclass
│   └── models/                   # one file per backend (torch, tf, sklearn, haiku)
├── NOTES.md                      # decisions/blockers log from the v1 build
└── README.md
```

**Layer 1 — workload builder (`workload/build.py`).** A single
function `build_benchmark(...)` translates a small set of toggles
(`backend`, `n_clients`, `dp`, `scaffold`, `secagg`, `regularizer`,
`rounds`, `batch_size`) into a fully-instantiated `BenchmarkSpec`
(server `Model` + `FLOptimConfig` + `FLRunConfig` + per-client
`Dataset`s + optional SecAgg configs + network config). Validation
rejects parameter combinations declearn can't honor (e.g. DP on
non-torch backends).

**Layer 2 — runner (`workload/runner.py`).** ~50 lines. Spawns a
`FederatedServer` and N `FederatedClient`s on asyncio, awaits
`asyncio.gather(server, *clients)` to completion. This is the
quickrun replacement.

**Layer 3 — ASV classes (`benchmarks/__init__.py`).** Thin wrappers
that declare which slice of the parameter space each benchmark
exercises. ASV discovers these. Each class declares `params` and
`param_names`, has a `setup(...)` that triggers data prep, and a
`time_run(...)` that calls `build_benchmark + run_benchmark`.

The architecture matters because it puts everything that varies
(parameters, axis selection) in one place (`__init__.py` and
`build.py`) and everything that's fixed (the baseline, declearn
internals) somewhere else.

---

## 4. Benchmark suite (the actual matrix)

Currently shipped, all in `benchmarks/__init__.py`:

| Class | Cells (per commit) | What's varied | Notes |
|---|---|---|---|
| `BackendsBenchmark` | n=5 × {torch, tensorflow} = 2 | model backend | sklearn split out, haiku dropped |
| `RegularizersBenchmark` | n=5 × {lasso, ridge, fedprox} = 3 | client-side L1/L2/proximal regularizer | torch FedAvg |
| `DPBenchmark` | n=5 = 1 | client count for DP-SGD on torch | budget=[5.0, 1e-5], rdp accountant |
| `ScaffoldBenchmark` | n=5 = 1 | client count for SCAFFOLD on torch | |
| `SecAggBenchmark` | n=5 × {masking} = 1 | secagg method | joye-libert dropped (timeout); pinned to n=5 (n=20 timed out before fixes) |

**8 cells per commit.** Multi-axis matrices reduced to single-point
n_clients to keep wall time bounded.

What was removed and why:

- `SklearnBenchmark` (sklearn FedAvg) — was the single biggest time
  cost, ~78 min/version with full data. sklearn iterates per-step
  over a much larger effective batch count, no GPU acceleration
  helps, and we don't actually need sklearn timings for the
  declearn-plumbing comparison. Removed entirely.
- `BackendsBenchmark[haiku]` — `models/haiku_cnn.py` is a stub that
  raises `NotImplementedError`. Permanent failure in the matrix is
  noise. Removed from the params list (file is still there with a
  sketch of a working impl in NOTES.md).
- `n_clients=20` from every category except SecAgg — the n=20 cells
  were where overhead amplified disproportionately with the original
  full-MNIST shard sizes. With dataset fraction 0.1 and the patched
  declearn (esp. the secagg-batched mask draw), n=20 should be
  re-enableable; not done yet, just to keep the matrix simple.
- `SecAggBenchmark` is pinned to `n_clients=[5]` independently of
  the global `N_CLIENTS_AXIS`, on purpose. Even after the secagg
  patch, n=20 hasn't been re-validated.

To restore any of the above: edit `benchmarks/__init__.py`. The
`N_CLIENTS_AXIS = [5]` constant at the top is the single source of
truth for the n_clients sweep.

---

## 5. Fixed baseline (`workload/baseline.py`)

Every benchmark category varies exactly one axis from this:

```python
BASELINE_BACKEND          = "torch"
BASELINE_AGGREGATOR       = "averaging"
BASELINE_CLIENT_LRATE     = 0.001
BASELINE_CLIENT_MODULES   = ["adam"]
BASELINE_SERVER_LRATE     = 1.0
BASELINE_ROUNDS           = 2
BASELINE_BATCH_SIZE       = 48
BASELINE_EVAL_BATCH_SIZE  = 128
BASELINE_N_CLIENTS        = 3            # only used as a default; classes override
BASELINE_REGISTRATION_TIMEOUT = 60       # seconds
BASELINE_NETWORK_HOST     = "127.0.0.1"
BASELINE_NETWORK_PORT     = 8765
BASELINE_NETWORK_PROTOCOL = "websockets"
BASELINE_METRICS          = [["multi-classif", {"labels": [0,1,2,3,4,5,6,7,8,9]}]]
BASELINE_DATASET_FRACTION = 0.1          # see §6
```

`BASELINE_DATASET_FRACTION = 0.1` is the per-cell-time biggest lever
we have. Each cell trains on **10 % of each client's MNIST shard**
instead of the full ~9600 samples. The full FL pipeline
(registration, aggregation, eval, encryption, DP accounting) is
exercised identically; only the inner training loop sees less data.
We're benchmarking declearn plumbing, not MNIST training, so the
sample count is irrelevant to the cross-version differential.

Set to `1.0` to restore the full split. Cached layout dirs are keyed
on this fraction (`chw_5_f010` for fraction 0.1 with 5 clients),
so multiple fractions can coexist on disk.

`BASELINE_ROUNDS = 2` is the minimum that still exercises the round-
handshake code; lower means almost all the time is setup/teardown,
higher means more data points but linear time growth. We left it at
the brief's value.

---

## 6. Dataset prep (`workload/data.py`)

One canonical IID MNIST split per `n_clients` value, then re-shaped
on demand into the layout each backend expects. All derived layouts
come from the same canonical split so benchmark runs at the same N
see identical sample assignment across backends.

Layouts:
- `"raw"`: `(N, 28, 28)` — straight from `declearn-split`
- `"chw"`: `(N, 1, 28, 28)`, **int64 targets** — for torch (vmap+gather
  inside DP-SGD rejects uint8)
- `"hwc"`: `(N, 28, 28, 1)` — for tensorflow / haiku (channels-last)
- `"flat"`: `(N, 784)` — for sklearn (kept around even though sklearn
  benchmark is now disabled)

`ensure_data_for_n_clients(n, layout, fraction)` is idempotent: it
checks whether the cache dir exists and is complete, and if not,
re-derives from the canonical source. Seed is fixed at 42.

---

## 7. The venv and bootstrap (`benchmarks/bootstrap_cluster.sh`)

The cluster doesn't ship a usable Python 3.11 by default and pip's
torch resolver is unhelpful here. The bootstrap script handles all
of it idempotently in five steps:

1. **Use conda only to obtain a Python 3.11 binary.**
   Why conda: cluster's system Python is 3.10/3.13 (depending on
   node); PyTorch's CUDA wheels lag the latest Python; we need 3.11
   specifically. Why not use conda for everything else: ASV doesn't
   integrate with conda environments cleanly, and the conda env adds
   a layer of indirection.
2. **Seed a regular venv from that python3.11 binary** at
   `~/.venvs/declearn-bench-gpu`. Everything else lives in this
   venv; ASV only ever sees this Python.
3. **`pip install declearn[torch,tensorflow,haiku,websockets]==2.8.0
   websockets<14.0 asv opacus cryptography gmpy2`.** The explicit
   `websockets<14.0` pin is critical — see below.
4. **Reinstall torch from `--index-url cu121`** with
   `--force-reinstall --no-deps` overriding the PyPI torch that
   declearn[torch] pulled in. PyPI ships cu13-bundled torch which
   the older driver can't load.
4b. **Re-pin `nvidia-cudnn-cu12==9.3.0.75`** with `--force-reinstall
   --no-deps`. This is the single most subtle environment fix —
   see below.
5. **Verify GPU access** including a real `Conv2d` to actually
   exercise cuDNN, not just `torch.cuda.is_available()`.

### 7.1 The `websockets<14.0` pin

declearn 2.8.0's websockets adapter still uses `extra_headers=`,
which `websockets` 14 renamed to `additional_headers=`. With
websockets ≥ 14, the client immediately fails:
`TypeError: BaseEventLoop.create_connection() got an unexpected
keyword argument 'extra_headers'`. declearn's pyproject pins
`<14.0` but pip's resolver sometimes lets newer versions through
with extras involved, hence the explicit pin.

### 7.2 The cu12/cu13 cuDNN clash

The two trickiest hours of this build went into this one.

- torch 2.5.1+cu121 was built against cuDNN 9.1.0 cu12.
- `declearn[tensorflow,haiku]` transitively installs `tensorflow 2.21`
  and that pulls in `nvidia-cudnn-cu13` 9.19 + `cuda-toolkit==13.0.2`.
- Both `nvidia-cudnn-cu12` and `nvidia-cudnn-cu13` write into the
  same Python namespace package: `site-packages/nvidia/cudnn/lib/`.
  Whoever ran last wins, and `nvidia-cudnn-cu13` was running last.
- Result: torch tries to load cuDNN at first `Conv2d` and throws
  `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`.
- Reinstalling `nvidia-cudnn-cu12==9.1.0.70` (last write wins again)
  fixed torch — but TF 2.21 then refused to load with
  `Loaded runtime CuDNN library: 9.1.0 but source was compiled with: 9.3.0`.
- The compromise: `nvidia-cudnn-cu12==9.3.0.75`. cuDNN 9.x is
  forward-compatible at the major version level, so torch (which
  expected 9.1.0) is happy with 9.3.0, and TF (which requires
  ≥ 9.3.0) is happy too.

Step 4b in the bootstrap pins this. Step 4b in `run_benchmarks.sh`
also re-pins it before each `asv run` invocation as a defense
against the next time someone reinstalls `tensorflow` and
inadvertently bumps cuDNN out from under torch.

The corresponding memory file is at
`~/.claude/projects/-home-fslimani-benchmarking/memory/project_cuda_cudnn_pin.md`
so future debugging sessions don't re-derive this from scratch.

### 7.3 Other notes

- Python pinned to **3.11** because PyTorch's CUDA wheels lag newer
  Python releases; conda's base ships 3.13 which breaks
  `pip install torch`.
- conda-forge channel only — Anaconda's default-channel ToS gate
  on newer Miniconda installs would otherwise prompt interactively.
- declearn is installed *before* torch is overridden because
  `declearn[torch]` would otherwise refuse to install if torch
  weren't yet present.

---

## 8. The run script (`benchmarks/run_benchmarks.sh`)

What it does at a high level: for each "version" passed to it,
install that declearn version, verify the install took, re-pin
cuDNN, look up the matching commit SHA, then `asv run --python=same
--set-commit-hash=<sha> [--quick]`. Each iteration runs in a
subshell so a failure on one version doesn't kill the loop. After
the loop, `asv publish` regenerates the comparison HTML.

Modes:

```bash
./run_benchmarks.sh                              # default versions, --quick
./run_benchmarks.sh 2.7.0 2.8.0                  # explicit released versions
./run_benchmarks.sh path:/abs/path/to/declearn   # local checkout (a fork)
FULL=1 ./run_benchmarks.sh                       # multi-sample timings, slow
```

`path:` mode is the one we added when introducing the patched fork.
It does `pip install <fork-path> --no-deps` and tags the asv result
with `git -C <path> rev-parse HEAD`. To make the SHA resolvable
from ASV's configured repo (`../../declearn`), the script also
fetches the fork's HEAD into `refs/benchmarks/<basename>-<sha8>` of
the upstream declearn checkout — local-only, no remote interaction.

Defenses baked into the script:

- **Version verify**: `pip show declearn | awk '/^Version:/'` is
  compared to the requested version, and the loop iteration aborts
  if they don't match. Without this, a silent pip-install failure
  would benchmark the previously-installed declearn while tagging
  the result with the new version's SHA.
- **cuDNN re-pin per iteration**: defends against the cu12/cu13
  reshuffle described in §7.2.
- **`(...) || echo WARN`**: asv's forkserver child sometimes crashes
  with a cosmetic `KeyboardInterrupt` during shutdown cleanup on
  Python 3.11+, after results have already been written to disk.
  Without the `|| echo WARN`, `set -euo pipefail` would abort the
  loop and skip the remaining versions.
- **`source $VENV/bin/activate` before `asv publish`**: the per-
  version subshells activate the venv themselves, but the publish
  call at the end runs in the parent shell where the venv isn't
  active.
- **`DECLEARN_BENCH_FORCE_GPU=1` exported by default**: a silent
  CPU fallback would corrupt the comparison graph with a phantom
  10× slowdown.

---

## 9. ASV configuration (`benchmarks/asv.conf.json`)

```json
{
    "version": 1,
    "project": "declearn",
    "project_url": "https://gitlab.inria.fr/magnet/declearn/declearn2",
    "repo": "../../declearn",
    "branches": ["develop"],
    "environment_type": "existing",
    "matrix": {},
    "benchmark_dir": ".",
    "env_dir": ".asv/env",
    "results_dir": ".asv/results",
    "html_dir": ".asv/html"
}
```

Three things to call out:

- **`environment_type: "existing"`.** ASV will not create or manage
  Python environments. It runs benchmarks inside whatever venv is
  active when you call `asv run`. Code-under-test is whatever
  `pip install` put on disk; the swap between versions is the
  responsibility of `run_benchmarks.sh`, not of ASV.
- **`branches: ["develop"]`.** declearn's default branch is
  `develop`, not `main`. This was a small surprise mid-sweep — ASV
  refuses to start with `Unknown branch main in configuration`.
- **`repo: "../../declearn"`.** ASV uses this for commit-SHA
  resolution. Fork commits get fetched into this repo's object DB
  (see §8 path: mode) so they're resolvable.

---

## 10. The patched fork (`/home/fslimani/declearn-patched`)

Local clone of `/home/fslimani/declearn`, branch `patched-2.8.0`,
based on `v2.8.0`. Single squash commit (`9fbdf79`) carrying four
performance patches lifted from `~/internship-writeups/`:

| Patch | Source files modified | Reported speedup |
|---|---|---|
| **compression** | `communication/websockets/{_server,_client}.py` | ~40 % at N=100 (FedAvg-torch) |
| **DP-H3** | `training/dp/_manager.py` | ~3× on DP-SGD |
| **secagg-batched** | `secagg/api/_encrypt.py`, `secagg/masking/_encrypt.py` | 16× at N=10 (masking SecAgg) |
| **vector-foreach** | `model/torch/_vector.py` | 4–7 % on big models, ~noise on small |

Each patch is independent (different files, no overlap). All four
were squashed into one commit on purpose — the user wanted exactly
three checkpoints on the timeline (2.7.0, 2.8.0 vanilla, 2.8.0
patched), not three plus four sub-commits.

The fork is purely local. No `git remote`, no pushing. Whoever
re-runs the sweep on a different machine would have to reproduce
the patch application step (the originals are in
`internship-writeups/<topic>/reproducing/`). I haven't scripted the
patch-application step yet — this could be a follow-up if the fork
becomes a recurring artifact.

What each patch does, briefly:

- **compression**: declearn's websockets adapter never passes
  `compression=` to `ws.serve()` / `ws.connect()`, so per-message
  DEFLATE is on by default and accounts for ~38 % of wall time at
  N=100. The patch passes `compression=None` on both ends.
- **DP-H3**: declearn's DP-SGD calls `accountant.get_epsilon()`
  per training step to enforce the privacy budget. opacus's RDP
  accountant walks the full history each call, so the cost is
  O(N²) in step count. H3 binary-searches the max steps that keep
  ε ≤ budget at round-start, then per-step is a single integer
  compare. Mid-round detection is preserved exactly.
- **secagg-batched**: masking SecAgg used to draw one mask number
  per scalar (one `numpy` PRNG call per parameter element). The
  patch draws a vector of masks in one numpy call.
- **vector-foreach**: `Vector._apply_operation` walks the parameter
  dict and calls the underlying torch op once per tensor in Python.
  `TorchVector` now overrides this to use `torch._foreach_*`
  batched ops where available, falling through to the canonical
  per-tensor loop for unsupported funcs.

---

## 11. The three checkpoints currently on disk

Result files live at `benchmarks/.asv/results/magnet8/<sha>-<env>.json`,
one per (commit, env) pair, **append-only** by design. ASV's model is
that you accumulate results over time and republish on demand.

```
benchmarks/.asv/results/
├── benchmarks.json                       # suite catalog (regenerated by asv)
└── magnet8/
    ├── machine.json                      # host metadata
    ├── 1e5de98c-existing-py_*.json       # declearn 2.7.0 (full + sklearn cells)
    ├── 943ce1cd-existing-py_*.json       # declearn 2.8.0 vanilla (trimmed matrix)
    └── 9fbdf792-existing-py_*.json       # declearn 2.8.0 patched (trimmed matrix)
```

Headline timings (single-sample, `--quick`, dataset fraction 0.1,
n=5 except where noted, all on Magnet 8 / A10):

| Cell | 2.7.0 | 2.8.0 | Patched | Δ patched vs 2.8.0 |
|---|---|---|---|---|
| BackendsBenchmark[torch] | 64.22 s | 64.61 s | 62.82 s | -2.8 % |
| BackendsBenchmark[tensorflow] | 70.19 s | 71.00 s | 68.78 s | -3.1 % |
| DPBenchmark | 87.70 s | 97.40 s | 92.68 s | -4.8 % |
| RegularizersBenchmark[lasso] | 64.26 s | 64.84 s | 62.76 s | -3.2 % |
| RegularizersBenchmark[ridge] | 64.26 s | 64.81 s | 63.67 s | -1.8 % |
| RegularizersBenchmark[fedprox] | 64.32 s | 64.67 s | 63.07 s | -2.5 % |
| ScaffoldBenchmark | 65.24 s | 65.31 s | 63.29 s | -3.1 % |
| **SecAggBenchmark[masking]** | **99.81 s** | **105.67 s** | **65.32 s** | **-38 %** |

The 2.7.0 file also has `SklearnBenchmark[5]=587 s` and `[20]=581 s`
left over from yesterday's untrimmed run — preserved by ASV because
`SklearnBenchmark` was in the suite when those were recorded.

The SecAgg-batched patch is the standout. The other patches show
small (1–5 %) wins, which matches the writeups' caveats: compression
benefits scale with N, vector-foreach benefits scale with parameter
count, and at n=5 / 0.1 dataset fraction those costs aren't
exercised hard.

---

## 12. Workflow / how to do things

**Run the full sweep on the cluster** (re-runs all default versions,
~10–15 min total with current settings):

```bash
cd ~/benchmarking/benchmarks
./run_benchmarks.sh                     # 2.7.0 + 2.8.0
./run_benchmarks.sh path:~/declearn-patched   # patched
asv preview                              # http://localhost:8080
```

**Benchmark just one version (the steady-state CI cost)**:

```bash
./run_benchmarks.sh 2.8.0
# or:
./run_benchmarks.sh path:/abs/path/to/some/declearn-checkout
```

Either appends one result file. `asv publish` at the end of the
sweep regenerates the HTML.

**Local Layer-2 smoke (no ASV)**:

```bash
cd ~/benchmarking
python -c "
from benchmarks.workload.build import build_benchmark
from benchmarks.workload.runner import run_benchmark
run_benchmark(build_benchmark())
"
```

Useful when you want to know whether the FL pipeline runs at all
on a freshly-installed declearn, before paying for the full ASV
measurement loop.

**View results on another machine**:

```bash
git pull
cd benchmarks
asv publish      # regenerates HTML from result JSONs
asv preview      # serves at :8080
```

The `.asv/results/` tree is committed and pushed; `.asv/html/` is
regeneratable, so it's gitignored.

---

## 13. Known gotchas / deferred items

- **`SecAggBenchmark` n=20** still pinned off in `__init__.py` — was
  timing out before the secagg patch; almost certainly works with
  the patch but not yet re-validated.
- **`BackendsBenchmark[haiku]`** removed; `models/haiku_cnn.py` is
  a stub. NOTES.md has the sketch.
- **n_clients=100** never smoke-tested; would need `ulimit -n` raised
  for websockets fd usage.
- **Joye-Libert SecAgg** dropped from the suite; modular
  exponentiation × CNN parameter count exceeds practical timeouts.
  Reproducible only with a smaller bitsize / smaller model.
- **The patched fork is unscripted.** Re-applying the four patches
  on a fresh machine means manually copying files from
  `~/internship-writeups/<topic>/reproducing/` into a fresh
  `git clone` of declearn. A `apply_patches.sh` would help; I
  haven't written one yet.
- **ASV publish warning** `Couldn't find 9fbdf792 in branches
  (develop)` — informational. The patched commit isn't on
  declearn's `develop`, so ASV doesn't apply its built-in
  regression/progression detector against it linearly. The numbers
  and graphs are still correct. Adding `patched-2.8.0` to
  `branches` in `asv.conf.json` would make ASV light up the
  progress arrow.
- **Forkserver `KeyboardInterrupt` traceback** at the end of every
  asv run is cosmetic and absorbed by `(...) || echo WARN`. It
  isn't an error.
- **`ptxas` warning** from TF (`Failed to compile generated PTX
  with ptxas. Falling back to compilation by driver.`) is
  cosmetic — the driver fallback works.

---

## 14. How to extend the suite

- **New benchmark category**: add a class to
  `benchmarks/__init__.py` with `params`, `param_names`, `setup`,
  `time_run`. Make sure the underlying parameter combination is
  rejected or accepted by `_validate` in `workload/build.py`.
- **New backend**: add a model file under `workload/models/`,
  exposing `build_model() -> Model`. Wire it into
  `_BACKEND_LAYOUT` and `_BACKEND_MODEL_MODULE` in `build.py`.
- **New axis**: add the kwarg to `build_benchmark`'s signature in
  `build.py`, validate it, plumb it into the constructed objects.
  Don't put new axes in `baseline.py` — that's for fixed values
  shared across all benchmarks.
- **Bigger sweep**: bump `N_CLIENTS_AXIS = [5, 20]` (or wider)
  in `__init__.py`. Bump `BASELINE_DATASET_FRACTION = 1.0` if you
  want full MNIST. Run `FULL=1 ./run_benchmarks.sh` for noise-aware
  multi-sample timings instead of `--quick`.

---

## 15. Steady state

The intended long-term shape: each declearn release / commit gets
benchmarked once on Magnet, the result file gets appended to
`.asv/results/magnet8/`, and `asv publish` regenerates the
comparison HTML. **The cross-version backfill we did manually is a
one-time cost.** Going forward, per-release cost is ~5–10 minutes,
runnable from CI.

The interesting comparison the next CI run can already produce is
"declearn HEAD vs the last release" — same suite, same configs,
two adjacent points on the timeline. ASV's regression detector
will flag any cell that's slower by a configurable threshold (5 %
by default).
