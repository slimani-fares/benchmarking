#!/bin/bash
# Bootstrap a benchmark-ready Python environment on a fresh cluster
# node (e.g. Magnet 8). Idempotent — safe to re-run.
#
# What it does:
#   1. Creates (or reuses) a venv at $BENCH_VENV
#   2. Installs PyTorch with CUDA 12.1 wheels (forward-compatible with
#      driver 12.6 / 12.8)
#   3. Installs declearn from PyPI plus the deps the suite needs
#      (tensorflow, scikit-learn, opacus, cryptography, asv)
#   4. Verifies torch + (optional) tensorflow can see the GPU
#
# Usage:
#   ./bootstrap_cluster.sh                  # uses default venv path
#   BENCH_VENV=/some/path ./bootstrap_cluster.sh
#
# After it succeeds, run the benchmark suite with:
#   source "$BENCH_VENV/bin/activate"
#   export DECLEARN_BENCH_FORCE_GPU=1
#   cd <repo>/benchmarks
#   asv run --python=same --quick --show-stderr

set -euo pipefail

BENCH_VENV="${BENCH_VENV:-$HOME/.venvs/declearn-bench-gpu}"
DECLEARN_VERSION="${DECLEARN_VERSION:-2.8.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

echo "=== bootstrap_cluster.sh ==="
echo "    venv:     $BENCH_VENV"
echo "    declearn: $DECLEARN_VERSION"
echo "    torch:    cu121 wheels (compat with 12.6+ drivers)"
echo

# 1. Locate a working python3.
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is not on PATH on this node." >&2
    exit 1
fi
PYTHON_BIN=$(command -v python3)
echo "[1/5] python3 → $PYTHON_BIN ($($PYTHON_BIN --version))"

# 2. Create or reuse the venv.
if [ ! -d "$BENCH_VENV" ]; then
    echo "[2/5] creating venv at $BENCH_VENV"
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
