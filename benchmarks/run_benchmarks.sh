#!/bin/bash
# Cross-version benchmark sweep launcher (--quick by default).
#
# Activates the configured venv, pins declearn to each requested
# released version in turn, and runs the ASV suite with `--quick`
# (one sample per cell). This matches our usage model: declearn FL
# benchmarks are network/Python-bound, not GPU-bound, and we care
# about ~2× regressions, not 5% noise. Set FULL=1 to run the multi-
# sample noise-aware mode instead.
#
# Each version's results are tagged with the matching declearn git
# commit so ASV plots them on a version timeline.
#
# Usage:
#   ./run_benchmarks.sh                              # --quick, default versions
#   ./run_benchmarks.sh 2.7.0 2.8.0                  # explicit released versions
#   ./run_benchmarks.sh path:/abs/path/to/declearn   # local checkout (e.g. a fork)
#   FULL=1 ./run_benchmarks.sh                       # multi-sample, slow but tighter
#
# `path:` mode installs declearn from a local directory (via
# `pip install <path> --no-deps`) and tags the asv result with the
# checkout's `git rev-parse HEAD`. Use this to benchmark a fork or
# in-progress branch alongside released versions; results land in
# .asv/results/<machine>/<sha>-<env>.json same as for released versions.

set -euo pipefail
cd "$(dirname "$0")"

VENV="${BENCH_VENV:-/venv}"
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

# --quick is the default. Opt out with FULL=1 for multi-sample timings.
if [ "${FULL:-0}" = "1" ]; then
    ASV_QUICK_FLAG=()
    echo "(FULL=1 set — multi-sample timings, slow)"
else
    ASV_QUICK_FLAG=(--quick)
fi

# Force-loud GPU: a silent CPU fallback would corrupt the comparison.
export DECLEARN_BENCH_FORCE_GPU="${DECLEARN_BENCH_FORCE_GPU:-1}"

# Mirror declearn's CI env (see their tox.ini): TF and JAX pre-allocate
# the entire GPU by default, which starves torch in a side-by-side run.
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

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
        if [[ "$VERSION" == path:* ]]; then
            FORK_PATH="${VERSION#path:}"
            if [ ! -d "$FORK_PATH/.git" ]; then
                echo "ERROR: $FORK_PATH is not a git checkout." >&2
                exit 1
            fi
            pip install --quiet --no-deps --force-reinstall "$FORK_PATH"
            REAL_SHA=$(git -C "$FORK_PATH" rev-parse HEAD)
            INSTALLED=$(pip show declearn | awk '/^Version:/ {print $2}')
            echo "  installed declearn ${INSTALLED} from ${FORK_PATH} @ ${REAL_SHA:0:8}"
            # ASV's `repo` (asv.conf.json) is the upstream declearn checkout.
            # Our fork commit lives only in $FORK_PATH; without it in the
            # upstream repo, asv's `--set-commit-hash` lookup fails. We
            # fetch the fork's HEAD into upstream as a proper branch
            # named after the fork's current branch, so that branch can
            # also be listed in asv.conf.json's `branches` and the
            # progression detector treats the patched commit as a
            # linear successor instead of a side-ref.
            FORK_BRANCH=$(git -C "$FORK_PATH" rev-parse --abbrev-ref HEAD)
            git -C "$DECLEARN_REPO" fetch --quiet "$FORK_PATH" \
                "+HEAD:refs/heads/${FORK_BRANCH}"
        else
            pip install "declearn==${VERSION}" --no-deps --quiet
            # Verify the install actually took: a silent pip failure here
            # would otherwise benchmark whatever declearn was previously
            # installed, mis-tagged with the new version's commit SHA.
            INSTALLED=$(pip show declearn | awk '/^Version:/ {print $2}')
            if [ "$INSTALLED" != "$VERSION" ]; then
                echo "ERROR: pip install declearn==${VERSION} did not take (got '${INSTALLED}'). Skipping." >&2
                exit 1
            fi
            REAL_SHA=$(git -C "$DECLEARN_REPO" rev-parse "v${VERSION}^{commit}")
        fi
        asv run \
            --python=same \
            --set-commit-hash="$REAL_SHA" \
            --show-stderr \
            "${ASV_QUICK_FLAG[@]}"
    ) || echo "WARN: ${VERSION} exited non-zero — results on disk may still be valid; continuing."
done

# `asv publish` lives in the venv too. The per-version subshells above
# activate it locally; here in the parent shell we need to source it
# explicitly or asv won't be on PATH.
# shellcheck disable=SC1091
source "$VENV/bin/activate"
asv publish

cat <<EOF

=== sweep finished ===
Browse the comparison graph with:
  cd $(pwd) && asv preview        # http://localhost:8080
EOF
