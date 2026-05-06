#!/bin/bash
# Cross-version benchmark sweep launcher.
#
# Activates the configured venv, optionally pins declearn to one or more
# released versions, and runs `asv run --quick` against each one. Results
# are tagged with the matching git commit so ASV can plot them on a
# version timeline.
#
# Usage:
#   ./run_benchmarks.sh                  # use DEFAULT_VERSIONS
#   ./run_benchmarks.sh 2.7.0 2.8.0      # explicit versions

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

for VERSION in "${VERSIONS[@]}"; do
    echo "=== Benchmarking declearn ${VERSION} ==="
    (
        # shellcheck disable=SC1091
        source "$VENV/bin/activate"
        pip install "declearn==${VERSION}" --no-deps --quiet
        REAL_SHA=$(git -C "$DECLEARN_REPO" rev-parse "v${VERSION}^{commit}")
        asv run --python=same --set-commit-hash="$REAL_SHA" --quick --show-stderr
    )
done

asv publish
