#!/bin/bash
# Cross-version benchmark sweep launcher (full timings, no --quick).
#
# Activates the configured venv, pins declearn to each requested
# released version in turn, and runs the full ASV suite with normal
# sampling so results are noise-aware (multiple samples per param).
# Each version's results are tagged with the matching declearn git
# commit so ASV plots them on a version timeline.
#
# Set QUICK=1 to fall back to one-shot --quick timings (useful for
# smoke-testing the sweep itself without spending an hour timing).
#
# Usage:
#   ./run_benchmarks.sh                  # use DEFAULT_VERSIONS
#   ./run_benchmarks.sh 2.7.0 2.8.0      # explicit versions
#   QUICK=1 ./run_benchmarks.sh          # fast smoke pass (one sample)

set -euo pipefail
cd "$(dirname "$0")"

VENV="${BENCH_VENV:-$HOME/.venvs/declearn-bench-gpu}"
DECLEARN_REPO="${DECLEARN_REPO:-$(cd "$(dirname "$0")/../../declearn" 2>/dev/null && pwd)}"
DEFAULT_VERSIONS=("2.7.0" "2.8.0")

if [ -z "${DECLEARN_REPO:-}" ] || [ ! -d "$DECLEARN_REPO/.git" ]; then
    echo "ERROR: cannot find a declearn git checkout." >&2
    echo "Expected sibling layout: <parent>/declearn/ + <parent>/declearn-benchmarks/" >&2
    echo "Override with DECLEARN_REPO=/path/to/declearn ./run_benchmarks.sh ..." >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    VERSIONS=("$@")
else
    VERSIONS=("${DEFAULT_VERSIONS[@]}")
fi

ASV_QUICK_FLAG=()
if [ "${QUICK:-0}" = "1" ]; then
    ASV_QUICK_FLAG=(--quick)
    echo "(QUICK=1 set — using --quick, one-shot timings)"
fi

# Force-loud GPU: a silent CPU fallback would corrupt the comparison.
export DECLEARN_BENCH_FORCE_GPU="${DECLEARN_BENCH_FORCE_GPU:-1}"

for VERSION in "${VERSIONS[@]}"; do
    echo "=== Benchmarking declearn ${VERSION} ==="
    # `|| true`: asv's forkserver child sometimes crashes with a
    # cosmetic KeyboardInterrupt during shutdown cleanup on Python
    # 3.11+, AFTER results have been written to disk. asv inherits
    # that non-zero exit code; without `|| true` the loop's `set -e`
    # would abort before the next version runs.
    (
        # shellcheck disable=SC1091
        source "$VENV/bin/activate"
        pip install "declearn==${VERSION}" --no-deps --quiet
        # Verify the install actually took: a silent pip failure here
        # would otherwise benchmark whatever declearn was previously
        # installed, mis-tagged with the new version's commit SHA.
        INSTALLED=$(pip show declearn | awk '/^Version:/ {print $2}')
        if [ "$INSTALLED" != "$VERSION" ]; then
            echo "ERROR: pip install declearn==${VERSION} did not take (got '${INSTALLED}'). Skipping." >&2
            exit 1
        fi
        # Re-pin cuDNN to the cu12 build torch was linked against.
        # `--no-deps` above blocks transitive cu13 cuDNN reinstalls in
        # principle, but this is a $0.10 safety net; symptom of the
        # alternative is `CUDNN_STATUS_NOT_INITIALIZED` halfway through
        # the sweep with no clear cause. See project_cuda_cudnn_pin.md.
        pip install --quiet --force-reinstall --no-deps "nvidia-cudnn-cu12==9.3.0.75"
        REAL_SHA=$(git -C "$DECLEARN_REPO" rev-parse "v${VERSION}^{commit}")
        asv run \
            --python=same \
            --set-commit-hash="$REAL_SHA" \
            --show-stderr \
            "${ASV_QUICK_FLAG[@]}"
    ) || echo "WARN: ${VERSION} exited non-zero — results on disk may still be valid; continuing."
done

asv publish

cat <<EOF

=== sweep finished ===
Browse the comparison graph with:
  cd $(pwd) && asv preview        # http://localhost:8080
EOF
