# Build a Benchmark Suite for declearn — Claude Code Brief

You are working inside a fresh local clone of the declearn repository. Your task is to build a complete, working benchmark suite as a new top-level folder `benchmarks/`. This suite replaces ad-hoc benchmarking via `quickrun`. It is to be **end-to-end functional** — code, configuration, helpers, ASV classes, models, runner — all built and verified on at least one configuration.

This brief is detailed on purpose. Read it in full before starting. Do not skip sections.

---

## 1. Operating constraints

Before anything else, internalize these:

- **Do NOT modify any file outside `benchmarks/`.** The declearn package itself is read-only for this task. If you discover that a benchmark needs a declearn change, document it in `benchmarks/NOTES.md` rather than patching declearn.
- **Do NOT use `quickrun`.** The whole point of this rebuild is to bypass quickrun's limitations (hardcoded `secagg=None`, `InMemoryDataset` used where `FairnessInMemoryDataset` is needed, client count tied to data folder subdirectories). You will use `FederatedServer` and `FederatedClient` directly via declearn's main API.
- **Match declearn's coding style.** Type hints throughout, docstrings on every public function/class, snake_case for functions, PascalCase for classes. Look at existing declearn modules (`declearn/main/_server.py`, `declearn/quickrun/_run.py`) for conventions.
- **Keep dependencies minimal.** Use what's already in declearn's environment: `torch`, `tensorflow`, `scikit-learn`, `dm-haiku`, `numpy`, `opacus`, `cryptography`, `gmpy2`. Do not add new dependencies.
- **Test what you build.** After implementation, run a smoke test on at least one configuration and verify it completes without errors. Document any failures in `benchmarks/NOTES.md`.

---

## 2. Architectural overview

The suite has **three layers**, each in its own subfolder. Understand these before writing code; the coupling between layers determines what each layer can assume.

### Layer 1: Workload builder (`benchmarks/workload/`)

A parameterized program that, given a configuration, produces a fully-instantiated `BenchmarkSpec` ready to run. This is where parameter interpretation happens — translating "torch + DP + 5 clients" into concrete `Model`, `FLOptimConfig`, `FLRunConfig`, dataset list, network config, and optional SecAgg configs.

**Key entry point**: `build_benchmark(...) -> BenchmarkSpec`

### Layer 2: Runner (`benchmarks/workload/runner.py`)

Takes a `BenchmarkSpec` and runs it: starts a `WebsocketsServer`, instantiates `FederatedServer`, instantiates N `FederatedClient`s, awaits `asyncio.gather(server, *clients)` to completion. Replaces quickrun's `_run.py` for the benchmark suite. Should be small (~50–80 lines).

**Key entry point**: `run_benchmark(spec: BenchmarkSpec) -> None`

### Layer 3: ASV classes (`benchmarks/__init__.py`)

Thin wrappers that declare which slice of the parameter space each benchmark exercises. ASV discovers these. Each class:
- Declares `params` and `param_names` if it varies on an axis (e.g., backend type)
- Has a `setup(...)` method that triggers data preparation
- Has a `time_*` method that calls `build_benchmark(...)` then `run_benchmark(...)`

ASV times the `time_*` method.

---

## 3. Folder structure to create

Create exactly this structure under `benchmarks/`:

```
benchmarks/
├── __init__.py                          # ASV class definitions
├── README.md                            # Suite overview and usage
├── NOTES.md                             # Findings, blockers, decisions during build
├── asv.conf.json                        # ASV configuration
├── run_benchmarks.sh                    # Cross-version sweep launcher
├── .gitignore                           # Ignore generated data/, .asv/, __pycache__
└── workload/
    ├── __init__.py
    ├── spec.py                          # BenchmarkSpec dataclass + supporting types
    ├── baseline.py                      # The fixed baseline configuration
    ├── build.py                         # build_benchmark(...)
    ├── runner.py                        # run_benchmark(spec)
    ├── data.py                          # Data prep helpers
    └── models/
        ├── __init__.py
        ├── torch_cnn.py                 # Standard torch CNN
        ├── torch_cnn_dp.py              # vmap-compatible torch CNN for DP
        ├── tensorflow_cnn.py            # TF CNN
        ├── sklearn_sgd.py               # sklearn SGDClassifier
        └── haiku_cnn.py                 # Haiku/JAX CNN
```

`benchmarks/data/` will be auto-created at runtime by data helpers; gitignore it.

---

## 4. Layer 1 — Workload builder details

### 4.1 `workload/spec.py`

Define a `BenchmarkSpec` dataclass holding everything the runner needs:

```python
@dataclass
class BenchmarkSpec:
    server_model: Model                     # The Model instance for the server
    optim_config: FLOptimConfig             # Server + client opt + aggregator
    run_config: FLRunConfig                 # Rounds, training params, optional privacy
    network_host: str
    network_port: int
    network_protocol: str                   # "websockets" (only supported value initially)
    clients: List[ClientSpec]               # One per client
    server_secagg: Optional[SecaggConfigServer]
    metrics: List[Any]                      # MetricSet input format
```

And a `ClientSpec` dataclass:

```python
@dataclass
class ClientSpec:
    name: str                               # "client_0", "client_1", ...
    train_data: Dataset
    valid_data: Dataset
    secagg: Optional[SecaggConfigClient]
```

### 4.2 `workload/baseline.py`

A module of constants defining the baseline configuration. Every benchmark category varies *one* axis from this baseline; everything else stays fixed.

```python
# Fixed baseline — every category varies exactly one of these
BASELINE_BACKEND = "torch"
BASELINE_AGGREGATOR = "averaging"
BASELINE_CLIENT_LRATE = 0.001
BASELINE_CLIENT_MODULES = ["adam"]
BASELINE_SERVER_LRATE = 1.0
BASELINE_ROUNDS = 2
BASELINE_BATCH_SIZE = 48
BASELINE_EVAL_BATCH_SIZE = 128
BASELINE_N_CLIENTS = 3
BASELINE_REGISTRATION_TIMEOUT = 60       # seconds; high enough for 100 clients
BASELINE_NETWORK_HOST = "127.0.0.1"
BASELINE_NETWORK_PORT = 8765
BASELINE_NETWORK_PROTOCOL = "websockets"
BASELINE_METRICS = [
    ["multi-classif", {"labels": [0,1,2,3,4,5,6,7,8,9]}]
]
```

### 4.3 `workload/build.py`

Main entry point: `build_benchmark(...)`. Signature:

```python
def build_benchmark(
    backend: str = BASELINE_BACKEND,         # "torch" | "tensorflow" | "sklearn" | "haiku"
    n_clients: int = BASELINE_N_CLIENTS,
    regularizer: Optional[str] = None,       # None | "lasso" | "ridge" | "fedprox"
    dp: bool = False,
    scaffold: bool = False,
    secagg: Optional[str] = None,            # None | "masking" | "joye-libert"
    rounds: int = BASELINE_ROUNDS,
    batch_size: int = BASELINE_BATCH_SIZE,
) -> BenchmarkSpec:
    """Build a fully-instantiated benchmark specification.
    
    Validates parameter combinations; raises ValueError on invalid combinations
    (e.g., backend='sklearn' + dp=True, since opacus is torch-only).
    """
```

**Note**: fairness is NOT in the v1 signature. Document in NOTES.md that it's deferred.

**Logic**:

1. **Validate parameter combinations.** Build a small validation function up front. Examples of what to reject:
   - `dp=True` with `backend != "torch"` (opacus is torch-only)
   - `scaffold=True` with `backend != "torch"` (SCAFFOLD modules in declearn are model-agnostic but only commonly used with torch; reject non-torch for safety initially)
   - `secagg is not None` with anything that produces non-numeric updates (shouldn't happen with the supported configs, but document the validation)
   - `regularizer` set to a value other than `None`, `"lasso"`, `"ridge"`, `"fedprox"`
   - `secagg` set to a value other than `None`, `"masking"`, `"joye-libert"`

2. **Build the model** by importing from `workload/models/<backend>_cnn.py` (or `sklearn_sgd.py`). Each model module exposes a `build_model() -> Model` function. For DP+torch, import from `torch_cnn_dp.py` instead of `torch_cnn.py`.

3. **Build the optimizer config (`FLOptimConfig`)**:
   - Aggregator: `"averaging"`
   - Client opt: `lrate=BASELINE_CLIENT_LRATE`, `modules=[...]`. Add `"scaffold-client"` to modules if `scaffold=True`. Add regularizer to `regularizers` list if `regularizer` is set.
   - Server opt: `lrate=BASELINE_SERVER_LRATE`. Add `"scaffold-server"` to modules if `scaffold=True`.

4. **Build the run config (`FLRunConfig`)**:
   - `rounds=rounds`
   - `register={"min_clients": n_clients, "timeout": BASELINE_REGISTRATION_TIMEOUT}`
   - `training={"batch_size": batch_size}`
   - `evaluate={"batch_size": BASELINE_EVAL_BATCH_SIZE}`
   - If `dp=True`: add `privacy={"budget": [5.0, 1e-5], "sclip_norm": 1.0, "accountant": "rdp"}`. Note that this auto-flips Poisson sampling to True; don't fight it.

5. **Prepare data**: call `data.ensure_data_for_n_clients(n_clients, layout=...)`. The `layout` argument selects from:
   - `"chw"` for torch (channels-first, int64 targets — use this for both regular torch and DP torch)
   - `"hwc"` for tensorflow (channels-last)
   - `"flat"` for sklearn (784-d flat)
   - `"chw"` for haiku (same as torch initially; adjust if haiku model needs different shape)

6. **Build per-client `Dataset` instances**: for each client folder produced by data prep, create an `InMemoryDataset` pointing at `train_data.npy` + `train_target.npy`, and another for valid. Wrap each pair in a `ClientSpec`.

7. **Build SecAgg configs** (if `secagg` is set):
   - For `"masking"`: generate `n_clients` Ed25519 private keys; build `IdentityKeys` for each (with all public keys as trusted). Server config = `MaskingSecaggConfigServer(bitsize=64, clipval=1e8)`. Each client gets a `MaskingSecaggConfigClient(id_keys=...)`.
   - For `"joye-libert"`: same Ed25519 setup; server config = `JoyeLibertSecaggConfigServer(bitsize=64, clipval=1e8)`; each client gets `JoyeLibertSecaggConfigClient(id_keys=...)`.
   - Reference: `test/functional/test_toy_clf_secagg.py` for the masking pattern.

8. **Return** the assembled `BenchmarkSpec`.

### 4.4 `workload/data.py`

Idempotent data preparation. One source MNIST split, multiple derived layouts.

Functions to implement:

```python
def ensure_source_data(n_clients: int) -> Path:
    """Run declearn-split for the given client count if not already done.
    
    Output: benchmarks/data/source_<n>/  (one subdirectory per client, each
    containing train_data.npy, train_target.npy, valid_data.npy, valid_target.npy)
    
    Idempotent. Uses fixed seed=42 for reproducibility.
    """

def ensure_data_for_n_clients(n_clients: int, layout: str) -> Path:
    """Ensure data exists in the requested layout for the given client count.
    
    Layouts:
      - "raw":  (N, 28, 28), uint8 targets — straight from declearn-split
      - "chw":  (N, 1, 28, 28), int64 targets — for torch/haiku, DP-compatible
      - "hwc":  (N, 28, 28, 1), uint8 targets — for tensorflow
      - "flat": (N, 784), uint8 targets — for sklearn
    
    Output: benchmarks/data/<layout>_<n>/
    
    Idempotent. Derives from source_<n>/ on first call.
    """
```

**Design notes**:
- All derived layouts come from the same canonical source split. This guarantees identical sample assignment across backends for the same `n_clients`.
- Targets are cast to `int64` for `chw` (vmap+gather requirement); other layouts can keep `uint8`.
- Use `np.save` / `np.load` and `shutil.copy` as appropriate.
- `pathlib.Path` throughout; resolve everything against `BENCH_ROOT = Path(__file__).resolve().parent.parent`.

### 4.5 `workload/models/`

One file per backend, each exposing a `build_model() -> Model` function.

#### `torch_cnn.py`
Standard MNIST CNN, expects `(B, 1, 28, 28)` input directly (no `Unflatten`).

```
Conv2d(1, 8, 3, 1) → ReLU → MaxPool2d(2) → Dropout(0.25)
→ Flatten() → Linear(1352, 64) → ReLU → Dropout(0.5)
→ Linear(64, 10) → Softmax(dim=-1)
```

Wrap in `TorchModel(network, loss=torch.nn.CrossEntropyLoss())`.

#### `torch_cnn_dp.py`
Same as `torch_cnn.py` but with a `FlexibleFlatten` module replacing `nn.Flatten()`. The custom module:

```python
class FlexibleFlatten(torch.nn.Module):
    """Flatten that works under both torch.func.vmap (no batch dim, 3D input)
    and regular forward (batch dim present, 4D input).
    
    Required because declearn's DP-SGD uses torch.func.vmap for per-sample
    gradients, which hides the batch dim from the model. Default nn.Flatten
    assumes dim 0 is batch and breaks under vmap.
    """
    def forward(self, x):
        if x.dim() == 4:
            return x.flatten(start_dim=1)
        return x.flatten()
```

Document in the docstring: this model is used only when `dp=True`. Eval still uses regular forward (4D input), training uses vmap (3D input). FlexibleFlatten handles both.

#### `tensorflow_cnn.py`
Equivalent CNN using `tf_keras.layers`. Input shape `(28, 28, 1)`. Wrap in `TensorflowModel(network, loss="sparse_categorical_crossentropy")`.

#### `sklearn_sgd.py`
Linear classifier (sklearn doesn't support CNNs):

```python
SklearnSGDModel.from_parameters(
    kind="classifier",
    loss="log_loss",
    penalty="l2",
    alpha=1e-4,
)
```

Note in the docstring: this is architecturally different from the CNN-based backends. It's a flat-input linear model, ~100x faster but not directly comparable in accuracy.

#### `haiku_cnn.py`
Equivalent CNN using Haiku. Reference: `examples/` if any haiku example exists, else build from declearn's HaikuModel API. If you can't get this working in the time available, document in NOTES.md and leave a stub that raises `NotImplementedError` with a clear message. **Do not block the rest of the suite on haiku.**

---

## 5. Layer 2 — Runner details (`workload/runner.py`)

```python
async def _run_async(spec: BenchmarkSpec) -> None:
    """Async core: spawn server + N clients, await completion."""
    # 1. Build NetworkServerConfig and NetworkClientConfig instances
    # 2. Construct FederatedServer with model, optim, metrics, secagg, network
    # 3. For each ClientSpec, construct FederatedClient with its dataset and secagg
    # 4. asyncio.gather(server.async_run(spec.run_config), *[c.async_run() for c in clients])

def run_benchmark(spec: BenchmarkSpec) -> None:
    """Synchronous entry point. Wraps _run_async in asyncio.run()."""
    asyncio.run(_run_async(spec))
```

**Notes**:
- The server takes `run_config` as an argument to `async_run`, not at construction time. The `FLRunConfig` includes `privacy`, which the server reads at runtime to trigger the DP handshake automatically.
- Clients don't need to know about DP at construction — the server tells them via `InitRequest`.
- For SecAgg, the server gets `secagg=spec.server_secagg`; each client gets its own `secagg=client_spec.secagg` from its `ClientSpec`.
- Reference: `quickrun/_run.py` for the orchestration pattern, but rebuild it from scratch — don't copy it wholesale, because quickrun has the `secagg=None` hardcoding you're trying to escape.

---

## 6. Layer 3 — ASV classes (`benchmarks/__init__.py`)

Implement these classes, in this order. Each is a thin wrapper around `build_benchmark` + `run_benchmark`.

### `BackendsBenchmark`
Vary backend across `["torch", "tensorflow", "sklearn", "haiku"]`. Fixed: `n_clients=3`, all other axes off.

```python
class BackendsBenchmark:
    timeout = 300
    params = ["torch", "tensorflow", "sklearn", "haiku"]
    param_names = ["backend"]
    
    def setup(self, backend):
        # Trigger data prep for whatever layout this backend needs
        ...
    
    def time_run(self, backend):
        spec = build_benchmark(backend=backend)
        run_benchmark(spec)
```

### `RegularizersBenchmark`
Vary regularizer across `["lasso", "ridge", "fedprox"]`. Fixed: `backend="torch"`, `n_clients=3`.

### `DPBenchmark`
Single configuration. Fixed: `backend="torch"`, `dp=True`, `n_clients=3`.

### `ScaffoldBenchmark`
Single configuration. Fixed: `backend="torch"`, `scaffold=True`, `n_clients=3`.

### `SecAggBenchmark`
Vary `secagg` across `["masking", "joye-libert"]`. Fixed: `backend="torch"`, `n_clients=3`.

### `ScalingBenchmark`
Vary `n_clients` across `[2, 5, 10, 100]`. Fixed: `backend="torch"`, all features off (pure FedAvg). This is the only benchmark that crosses with client count.

### Notes on ASV class design
- `setup` must complete before timing begins. Put data preparation there.
- Set `timeout` generously. 100-client runs take time to register clients alone; 300s is reasonable for most, may need 600s for `ScalingBenchmark` at `n_clients=100`.
- Don't put `print` statements in `time_*` methods — ASV captures stdout and it pollutes output.

---

## 7. Configuration files

### `benchmarks/asv.conf.json`

Standard ASV config. Key fields:
- `"project": "declearn"`
- `"project_url"` and `"repo"`: point to the repo root (use `".."` since `asv.conf.json` is one level deep)
- `"branches": ["main"]`
- `"environment_type": "existing"` — use the active venv, don't have ASV manage environments
- `"matrix"`: empty `{}` since the existing-venv mode handles dependencies
- `"benchmark_dir": "."`
- `"env_dir": ".asv/env"`
- `"results_dir": ".asv/results"`
- `"html_dir": ".asv/html"`

### `benchmarks/run_benchmarks.sh`

Cross-version sweep launcher. Adapted from the existing `run_benchmarks.sh` but simplified — only one venv now (`declearn311`), versions starting from `2.7.0`. Skeleton:

```bash
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

VENV="$HOME/.venvs/declearn311"
DEFAULT_VERSIONS=("2.7.0" "2.8.0")

VERSIONS=("${@:-${DEFAULT_VERSIONS[@]}}")

for VERSION in "${VERSIONS[@]}"; do
    echo "=== Benchmarking declearn ${VERSION} ==="
    (
        source "$VENV/bin/activate"
        pip install "declearn==${VERSION}" --no-deps --quiet
        REAL_SHA=$(git -C .. rev-parse "v${VERSION}^{commit}")
        asv run --python=same --set-commit-hash="$REAL_SHA" --quick --show-stderr
    )
done

asv publish
```

### `benchmarks/.gitignore`

```
data/
.asv/
__pycache__/
*.pyc
```

### `benchmarks/README.md`

Explain:
- What this folder is and why it exists (replaces quickrun-based benchmarking)
- The three-layer architecture in one paragraph
- How to add a new benchmark category (modify `build.py` + add an ASV class)
- How to run: `cd benchmarks && asv run --quick -b BackendsBenchmark` for one class, or `./run_benchmarks.sh` for the full sweep
- The fixed baseline configuration (link to `workload/baseline.py`)

### `benchmarks/NOTES.md`

Living document of decisions, blockers, and findings during the build. Initialize with:
- Why this folder exists (verbatim from supervisor's spec: 6 categories, no crossing axes, 4 client counts on baseline)
- The fixed baseline configuration (mirror from `baseline.py`)
- Known gotchas:
  - `Unflatten(dim=0)` and default `nn.Flatten()` break under `torch.func.vmap` — DP needs vmap-compatible layers
  - `uint8` targets break `torch.gather()` inside vmap — DP needs `int64` targets
  - sklearn cannot run DP, SCAFFOLD, or SecAgg meaningfully (no per-sample gradients API in the same way; SCAFFOLD modules are model-agnostic but the integration was untested with sklearn during build)
  - `n_clients=100` with websockets requires generous registration timeout (60+ seconds) and may hit file-descriptor limits (`ulimit -n`)
- Deferred items:
  - Fairness category not yet implemented; needs a sensitive-attribute story for MNIST that isn't natural in the data
  - Anything else you discover during build

---

## 8. Smoke testing requirements

After implementation, **you must verify the suite works** before declaring completion. Do these in order:

1. **Layer 1 smoke**: `python -c "from benchmarks.workload.build import build_benchmark; spec = build_benchmark(); print(type(spec))"` — should print `<class '...BenchmarkSpec'>`.

2. **Layer 2 smoke**: `python -c "from benchmarks.workload.build import build_benchmark; from benchmarks.workload.runner import run_benchmark; run_benchmark(build_benchmark())"` — should run a full FedAvg+torch+3-client benchmark to completion.

3. **Layer 3 smoke** (ASV discovery): `cd benchmarks && asv run --python=same --quick --show-stderr -b BackendsBenchmark` — should run the torch case at minimum. If TF/sklearn/haiku error, document in NOTES.md and proceed.

4. **DP smoke**: `python -c "from benchmarks.workload.build import build_benchmark; from benchmarks.workload.runner import run_benchmark; run_benchmark(build_benchmark(dp=True))"` — confirms the vmap + int64 fixes carried over correctly.

5. **SecAgg smoke**: `python -c "from benchmarks.workload.build import build_benchmark; from benchmarks.workload.runner import run_benchmark; run_benchmark(build_benchmark(secagg='masking'))"` — confirms SecAgg integration works since you're bypassing quickrun.

If any of these fail, **fix what you can without modifying declearn itself**, document the rest in `NOTES.md`, and report which categories work and which don't in your final summary.

---

## 9. Final report

When done, write a summary at the top of `NOTES.md` covering:
- Which ASV classes work end-to-end (verified by smoke test)
- Which classes have known issues, with specific error reproductions
- Any deviations from this brief and their rationale
- Anything you want the human (Fares) to review or test before he commits the work

---

## 10. Key API references

These are the declearn APIs you'll use most. Read the source for exact signatures:

- `declearn/main/_server.py:FederatedServer` — server instantiation
- `declearn/main/_client.py:FederatedClient` — client instantiation
- `declearn/main/config/_strategy.py:FLOptimConfig` — `from_params(...)` for programmatic build
- `declearn/main/config/_run_config.py:FLRunConfig` — `from_params(...)` for programmatic build
- `declearn/communication/utils/_build.py:NetworkServerConfig`, `NetworkClientConfig`
- `declearn/communication/websockets/_server.py:WebsocketsServer`
- `declearn/dataset/_inmemory.py:InMemoryDataset`
- `declearn/secagg/masking/_setup.py:MaskingSecaggConfigServer`, `MaskingSecaggConfigClient`
- `declearn/secagg/joye_libert/_setup.py:JoyeLibertSecaggConfigServer`, `JoyeLibertSecaggConfigClient`
- `declearn/secagg/utils/_ed25519.py:IdentityKeys`
- `cryptography.hazmat.primitives.asymmetric.ed25519:Ed25519PrivateKey` for SecAgg key generation

Reference test for SecAgg orchestration: `test/functional/test_toy_clf_secagg.py`.

---

## 11. What to do if you get stuck

If you hit a blocker that requires changing declearn itself, **stop and document** in `NOTES.md`:
- What you were trying to do
- The specific declearn limitation that's blocking you
- A proposed minimal patch (just the diff, don't apply it)
- Whether the rest of the suite can proceed without resolving this

Then continue with whatever else is buildable. **Do not modify any file outside `benchmarks/`.** Fares will review blockers and decide which (if any) become actual declearn patches.

---

Begin.
