#!/bin/bash
# Install the benchmark suite's runtime deps into an existing venv.
#
# Designed to run inside declearn's CI image
# (registry.gitlab.inria.fr/magnet/declearn/declearn2/ci-python311),
# which already provides Python 3.11 at /venv and a system-level
# CUDA 12.8 + cuDNN toolchain. Outside that image, the venv at
# $BENCH_VENV must already exist and the host must have a working
# CUDA toolchain on the standard search path.
#
# Usage:
#   ./bootstrap_cluster.sh
#   BENCH_VENV=/path/to/venv ./bootstrap_cluster.sh

set -euo pipefail

BENCH_VENV="${BENCH_VENV:-/venv}"
DECLEARN_VERSION="${DECLEARN_VERSION:-2.8.0}"

if [ ! -d "$BENCH_VENV" ]; then
    echo "ERROR: no venv at $BENCH_VENV." >&2
    echo "  Expected to run inside declearn's CI image, or with" >&2
    echo "  BENCH_VENV pointing at a Python 3.11 venv you created." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$BENCH_VENV/bin/activate"
pip install --quiet --upgrade pip

# Mirror declearn's `tox -e py311-ci` install path: declearn from PyPI
# plus its torch/tensorflow/websockets extras. Add the suite-specific
# deps on top (asv runs the sweep, cryptography supplies the Ed25519
# keys consumed by SecAgg masking). The explicit websockets pin defends
# against pip resolver edge cases that have let 14.x slip through
# declearn's own constraint in the past.
echo "[1/2] installing declearn==$DECLEARN_VERSION + deps"
pip install --quiet \
    "declearn[torch,tensorflow,websockets]==$DECLEARN_VERSION" \
    "websockets<14.0" \
    asv \
    cryptography

# GPU smoke. Forces cuDNN to initialise via a tiny conv — basic CUDA
# can be live while cuDNN is broken, and that breakage will only
# surface mid-sweep. Better to fail here.
echo "[2/2] verifying GPU access"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python - <<'PY'
import sys

ok = True

try:
    import torch
    gpu = torch.cuda.is_available()
    print(f"  torch.cuda.is_available() = {gpu}")
    print(f"  torch.version.cuda        = {torch.version.cuda}")
    if gpu:
        print(f"  torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
        x = torch.randn(2, 1, 8, 8, device="cuda")
        torch.nn.Conv2d(1, 4, 3).cuda()(x)
        print("  cuDNN conv smoke         = OK")
    else:
        print("  WARNING: torch cannot see the GPU.")
        ok = False
except Exception as exc:
    print(f"  torch GPU smoke failed: {exc!r}")
    ok = False

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  tf.config.list_physical_devices('GPU') = {gpus}")
    if not gpus:
        print("  NOTE: tensorflow cannot see the GPU (the suite still runs"
              " but BackendsBenchmark[tensorflow] timings will be slow).")
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
