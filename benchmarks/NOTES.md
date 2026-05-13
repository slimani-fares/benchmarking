# NOTES — declearn benchmark suite build log

This document is a living log of decisions, blockers, deferrals, and
findings encountered while building the suite. The "Final report"
section at the top is updated last and summarizes the as-shipped
state for human review.

## Final report

Built against declearn 2.8.0 in `~/.venvs/declearn311` on 2026-05-06.

### Verified end-to-end (smoke tests pass)

- `BackendsBenchmark[torch]` — Layer 2 smoke (build + 2-round FedAvg + 3
  clients) completes in ~20s. Wired through ASV discovery (`asv check`
  finds the class without errors).
- `BackendsBenchmark[tensorflow]` — Layer 2 smoke completes; runtime
  comparable to torch. ASV discovery clean.
- `BackendsBenchmark[sklearn]` — Layer 2 smoke completes but takes
  ~3:15 (much slower than torch/TF in our setup; sklearn iterates
  per-step over a much larger effective batch count). The class
  `timeout` is set to 600s so ASV won't kill it.
- `RegularizersBenchmark[lasso|ridge|fedprox]` — all three pass
  Layer 2 smoke.
- `ScaffoldBenchmark` — passes Layer 2 smoke.
- `SecAggBenchmark[masking]` — passes Layer 2 smoke. Confirms the
  rebuild (no longer hardcoding `secagg=None` like quickrun did) is
  correct end-to-end.

### Known issues / not verified

- `BackendsBenchmark[haiku]` — backend stubbed out, raises
  `NotImplementedError`. ASV slice fails as designed. To finish:
  drop the sketch in NOTES → `models/haiku_cnn.py`, switch
  `_BACKEND_LAYOUT["haiku"]` to `"hwc"` (already done in build.py).
- **Joye-Libert dropped from the suite** (was: option 3 from the
  earlier list). Its modular-exponentiation cost scales with the
  model parameter count, and a 3-client run on the CNN baseline did
  not finish in 5 min during smoke testing. `SecAggBenchmark` now
  sweeps only `["masking"]`. To re-enable it once a tractable
  configuration is settled (smaller bitsize, smaller model, or a more
  efficient declearn implementation), turn the class's `params` back
  into `(N_CLIENTS_AXIS, ["masking", "joye-libert"])` and bump the
  timeout.
- **sklearn split into its own class with a trimmed n_clients axis**
  (`SklearnBenchmark`, sweeps n=[2, 5] only). sklearn runs are an
  order of magnitude slower than torch/TF on the same dataset, so
  including it in `BackendsBenchmark` × `[2, 5, 10, 100]` would have
  dominated per-version runtime. Trim/extend by editing
  `N_CLIENTS_AXIS_SKLEARN`.
- The 100-client slice (`n_clients=100` on any of the categories) was
  not smoke-tested in this build. May also need `ulimit -n` raised
  before running.

### GPU enforcement (CI safety)

declearn defaults to `gpu=True`, but on a misconfigured environment
(wrong CUDA toolchain, missing driver, mismatched PyTorch build) it
silently falls back to CPU. That would corrupt the comparison graph
with a phantom 10× slowdown.

Set `DECLEARN_BENCH_FORCE_GPU=1` before `asv run` to make
`run_benchmark` verify `torch.cuda.is_available()` and explicitly
pin `set_device_policy(gpu=True)`, raising `RuntimeError` if GPU
isn't actually usable. Recommended for any cluster / CI run.

### Deviations from the brief

- `haiku_cnn.py` is a stub raising `NotImplementedError` rather than a
  working CNN, per the brief's explicit allowance ("If you can't get
  this working in the time available, document in NOTES.md and leave
  a stub …"). The data layout for haiku is set to `"hwc"` in
  `_BACKEND_LAYOUT` so a future implementation that uses Haiku's
  default NHWC convention will get the right shape.
- `BackendsBenchmark.timeout` is 900s (was 300s in the brief), with
  similar bumps elsewhere, to accommodate sklearn and large-`n_clients`
  runs.
- `run_benchmarks.sh` reads its venv path from `$BENCH_VENV` (default
  `~/.venvs/declearn311`) so the script doesn't hardcode one user's
  path. Otherwise unchanged from the brief.
- **`n_clients` axis crosses every category**, not just one dedicated
  scaling benchmark. The brief said scaling should be the only one,
  but Fares chose to add it everywhere so cross-product behavior is
  visible (e.g. how SCAFFOLD scales with client count). The brief's
  separate `ScalingBenchmark` was therefore dropped — its parameter
  slice is now exactly `BackendsBenchmark[backend="torch"]`.
  Per-class trimming of the `n_clients` axis is done by overriding
  `params` in the affected class.

### What I'd like Fares to review before committing

1. Whether to keep haiku as a stub or remove it from
   `BackendsBenchmark.params` so ASV doesn't surface a permanent
   failure on every run. I left it as a visible failure on purpose
   — easier to remember to come back to.
2. The `BASELINE_CLIENT_MODULES = ["adam"]` choice (brief specified
   it). If the cross-version comparison should match what users
   reach for by default, vanilla SGD might be a better baseline.



## Why this folder exists

Supervisor's spec, paraphrased:

- 5 benchmark categories (backends, regularizers, SCAFFOLD,
  SecAgg, scaling), no axis crossings beyond what each category
  declares.
- 4 client counts (2, 5, 10, 100) sweep on the FedAvg baseline.
- All categories share one fixed baseline so cross-version
  comparisons are meaningful.

## Fixed baseline

Mirrored from `workload/baseline.py`:

- backend: `torch`
- aggregator: `averaging`
- client optimizer: lrate=1e-3, modules=`["adam"]`
- server optimizer: lrate=1.0
- rounds: 2
- training batch size: 48
- evaluation batch size: 128
- n_clients: 3
- registration timeout: 60s (high enough for 100-client runs)
- network: websockets on 127.0.0.1:8765
- metrics: `multi-classif` over labels `[0..9]`

## Known gotchas

- **sklearn limitations**:
  - SCAFFOLD modules are model-agnostic in declearn but the
    integration with sklearn was not exercised during this build;
    `_validate` rejects `scaffold=True` with `backend="sklearn"` for
    safety. Relax this only after a smoke test passes.

- **websockets at n_clients=100**:
  - Registration alone is non-trivial at this count. The baseline
    registration timeout is set to 60s to leave headroom.
  - May hit the file-descriptor limit (`ulimit -n` on Linux). Bump
    it to 4096+ before running the `n_clients=100` slice on any
    category.

## Deferred items

- **Fairness category**: not implemented in v1. MNIST has no natural
  sensitive attribute, so a fairness story for this dataset would
  need synthetic group labels. Punt until the supervisor settles on
  the fairness data story.
- **Haiku backend**: implemented as a stub that raises
  `NotImplementedError` with a clear message. `hk.Conv2D` defaults
  to NHWC (so the `hwc` data layout fits naturally) and the
  `HaikuModel` API requires a purely-functional `model(inputs)`
  callable plus a sample-wise loss function — the build is
  straightforward but verifying it works end-to-end alongside the
  other smoke tests was de-scoped to keep v1 reliable. Sketch:
  ```python
  def cnn_fn(x):
      return hk.Sequential([
          hk.Conv2D(8, kernel_shape=3, padding="VALID"),
          jax.nn.relu,
          hk.MaxPool(window_shape=(2, 2, 1), strides=(2, 2, 1),
                     padding="VALID"),
          hk.Flatten(),
          hk.Linear(64), jax.nn.relu,
          hk.Linear(10), jax.nn.log_softmax,
      ])(x)
  def loss(y_pred, y_true):
      return -y_pred[jnp.arange(y_pred.shape[0]), y_true.astype(int)]
  build_model = lambda: HaikuModel(cnn_fn, loss=loss)
  ```
  ASV's `BackendsBenchmark` will surface the stub's error for the
  haiku slice, which is what we want until someone wires it up.

## Smoke test outcomes (raw)

| Smoke                              | Result                       |
|------------------------------------|------------------------------|
| Layer 1 (build only, defaults)     | OK, ~instant                 |
| Layer 2 (FedAvg torch, 3 clients)  | OK, ~20s                     |
| Layer 2 with `scaffold=True`       | OK                           |
| Layer 2 with `regularizer=lasso`   | OK                           |
| Layer 2 with `regularizer=ridge`   | OK                           |
| Layer 2 with `regularizer=fedprox` | OK                           |
| Layer 2 with `secagg=masking`      | OK                           |
| Layer 2 with `secagg=joye-libert`  | killed after >5 min          |
| Layer 2 with `backend=tensorflow`  | OK                           |
| Layer 2 with `backend=sklearn`     | OK, ~3:15                    |
| Layer 2 with `backend=haiku`       | NotImplementedError (stub)   |
| `asv check` (discovery validation) | "No problems found"          |

