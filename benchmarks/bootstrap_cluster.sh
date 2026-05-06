#!/bin/bash
# Bootstrap a benchmark-ready Python environment on a fresh cluster
# node (e.g. Magnet 8). Idempotent — safe to re-run.
#
# The benchmark env is always a regular venv at $BENCH_VENV. If no
# system Python 3.11+ is available, the script uses conda *only* to
# install a python3.11 interpreter (in a side conda env named
# $CONDA_BOOTSTRAP_ENV), then uses that interpreter to create the
# real venv. ASV itself never touches conda.
#
# What it does:
#   1. Locates a Python 3.11+ interpreter, falling back to
#      "conda create -n declearn-bench-py311 python=3.11" if needed
#   2. Creates (or reuses) the venv at $BENCH_VENV
#   3. Installs PyTorch with CUDA 12.1 wheels (forward-compatible
#      with driver 12.6 / 12.8)
#   4. Installs declearn from PyPI plus the deps the suite needs
#      (tensorflow, scikit-learn, opacus, cryptography, asv)
#   5. Verifies torch + (optional) tensorflow can see the GPU
#
# Usage:
#   ./bootstrap_cluster.sh
#   BENCH_VENV=/some/path ./bootstrap_cluster.sh
#   PYTHON_BIN=/path/to/python3.11 ./bootstrap_cluster.sh
#
# After it succeeds, run the benchmark suite with:
#   source "$BENCH_VENV/bin/activate"
#   export DECLEARN_BENCH_FORCE_GPU=1
#   cd <repo>/benchmarks
#   asv run --python=same --quick --show-stderr

set -euo pipefail

BENCH_VENV="${BENCH_VENV:-$HOME/.venvs/declearn-bench-gpu}"
CONDA_BOOTSTRAP_ENV="${CONDA_BOOTSTRAP_ENV:-declearn-bench-py311}"
PY_VERSION="${PY_VERSION:-3.11}"
DECLEARN_VERSION="${DECLEARN_VERSION:-2.8.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

# 1. Locate (or install via conda) a Python interpreter at exactly
#    $PY_VERSION (default 3.11). We pin the major.minor version
#    because PyTorch CUDA wheels lag the latest Python release; even
#    if the system has python3.13, `pip install torch` will fail
#    until upstream ships 3.13 wheels. Pinning to 3.11 sidesteps
#    that and keeps benchmark results comparable across machines.
need_install=1
if [ -n "${PYTHON_BIN:-}" ]; then
    need_install=0
elif path=$(command -v "python$PY_VERSION" 2>/dev/null); then
    PYTHON_BIN=$path
    need_install=0
fi

if [ "$need_install" = "1" ]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: no python$PY_VERSION on PATH and conda is unavailable." >&2
        echo "       declearn $DECLEARN_VERSION requires Python $PY_VERSION." >&2
        echo "       Try 'module load python/$PY_VERSION' or 'module load conda'" >&2
        echo "       and re-run, or set PYTHON_BIN=/path/to/python$PY_VERSION." >&2
        exit 1
    fi
    echo "[1/5] no system python$PY_VERSION found — bootstrapping via conda"
    echo "      ($(command -v conda))"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_BOOTSTRAP_ENV"; then
        echo "      creating conda env $CONDA_BOOTSTRAP_ENV (python=$PY_VERSION)"
        conda create -y -n "$CONDA_BOOTSTRAP_ENV" "python=$PY_VERSION" pip >/dev/null
    else
        echo "      reusing existing conda env $CONDA_BOOTSTRAP_ENV"
    fi
    PYTHON_BIN=$(conda run -n "$CONDA_BOOTSTRAP_ENV" --no-capture-output \
                    python -c 'import sys; print(sys.executable)')
fi

# Verify the chosen interpreter matches PY_VERSION (catches a mismatch
# from a stale venv pointing at the wrong python).
actual_ver=$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$actual_ver" != "$PY_VERSION" ]; then
    echo "ERROR: PYTHON_BIN=$PYTHON_BIN reports $actual_ver, expected $PY_VERSION." >&2
    echo "       Either set PYTHON_BIN to a python$PY_VERSION binary or" >&2
    echo "       unset it and let the script find one via conda." >&2
    exit 1
fi
echo "      python → $PYTHON_BIN ($($PYTHON_BIN --version))"

echo "=== bootstrap_cluster.sh ==="
echo "    venv:     $BENCH_VENV"
echo "    declearn: $DECLEARN_VERSION"
echo "    torch:    cu121 wheels (compat with 12.6+ drivers)"
echo

# 2. Create or reuse the venv (always a regular venv, never a conda env).
#    If a venv already exists but was seeded with the wrong Python
#    version, recreate it — pip would later choke on missing torch
#    wheels for that version.
recreate_venv=0
if [ -d "$BENCH_VENV" ]; then
    venv_ver=$("$BENCH_VENV/bin/python" -c \
        'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")
    if [ "$venv_ver" != "$PY_VERSION" ]; then
        echo "[2/5] venv at $BENCH_VENV has python $venv_ver — recreating with $PY_VERSION"
        rm -rf "$BENCH_VENV"
        recreate_venv=1
    fi
fi
if [ ! -d "$BENCH_VENV" ]; then
    [ "$recreate_venv" = "0" ] && echo "[2/5] creating venv at $BENCH_VENV"
    "$PYTHON_BIN" -m venv "$BENCH_VENV"
else
    echo "[2/5] venv exists at $BENCH_VENV — reusing"
fi
# shellcheck disable=SC1091
source "$BENCH_VENV/bin/activate"
pip install --quiet --upgrade pip

# 3. Install torch with CUDA wheels first (so deps don't pull CPU torch).
if ! python -c "import torch" >/dev/null 2>&1; then
    echo "[3/5] installing torch (cu121)"
    pip install --quiet --index-url "$TORCH_INDEX" torch
else
    echo "[3/5] torch already installed — skipping"
fi

# 4. Install declearn + the rest of the benchmark deps.
#    tensorflow is large; install only if not present.
echo "[4/5] installing declearn==$DECLEARN_VERSION + deps"
pip install --quiet \
    "declearn[torch,tensorflow,haiku]==$DECLEARN_VERSION" \
    asv \
    opacus \
    cryptography \
    gmpy2

# 5. Verify GPU visibility.
echo "[5/5] verifying GPU access"
python - <<'PY'
import sys

ok = True

try:
    import torch
    gpu = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if gpu else None
    print(f"  torch.cuda.is_available() = {gpu}")
    print(f"  torch.version.cuda        = {torch.version.cuda}")
    if gpu:
        print(f"  torch.cuda.get_device_name(0) = {name}")
    else:
        print("  WARNING: torch cannot see the GPU.")
        ok = False
except Exception as exc:
    print(f"  torch import failed: {exc!r}")
    ok = False

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  tf.config.list_physical_devices('GPU') = {gpus}")
    if not gpus:
        print("  NOTE: tensorflow cannot see the GPU (the suite still runs"
              " on TF/CPU but BackendsBenchmark[tensorflow] timings will"
              " be slow).")
except Exception as exc:
    print(f"  tensorflow import failed: {exc!r}")

sys.exit(0 if ok else 2)
PY

cat <<EOF

=== bootstrap finished ===
Next steps:
  source "$BENCH_VENV/bin/activate"
  export DECLEARN_BENCH_FORCE_GPU=1
  cd $(dirname "$0")
  asv run --python=same --quick --show-stderr
EOF
