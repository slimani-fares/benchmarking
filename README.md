# declearn-benchmarks

Standalone ASV benchmark suite for [declearn](https://gitlab.inria.fr/magnet/declearn/declearn).

## Layout

This repo expects to live as a sibling of a declearn checkout:

```
<parent>/
├── declearn/                # the declearn source repo
└── declearn-benchmarks/     # this repo
    └── benchmarks/          # the actual ASV suite
        ├── asv.conf.json
        ├── run_benchmarks.sh
        ├── bootstrap_cluster.sh
        └── ...
```

`asv.conf.json` and `run_benchmarks.sh` reference declearn at
`../declearn` (relative) by default. Override with `DECLEARN_REPO=...`
if your layout differs.

## Quick start

```bash
# from anywhere with python3 + git
git clone https://gitlab.inria.fr/magnet/declearn/declearn.git
git clone <this-repo-url> declearn-benchmarks

cd declearn-benchmarks/benchmarks
./bootstrap_cluster.sh        # creates venv, installs declearn + GPU torch
source ~/.venvs/declearn-bench-gpu/bin/activate
export DECLEARN_BENCH_FORCE_GPU=1     # opt-in: fail loud if GPU unavailable
asv run --python=same --quick --show-stderr
```

For a cross-version sweep (released versions):

```bash
./run_benchmarks.sh 2.7.0 2.8.0
asv publish
asv preview        # serves the comparison graph at http://localhost:8080
```

## Background

This suite replaces declearn's older `quickrun`-based benchmarking,
which had hardcoded constraints (`secagg=None`, the wrong dataset
class for fairness, client count tied to data folder structure) that
made systematic sweeps impossible. The full design rationale and the
brief that produced this code live in `build_benchmarks_prompt.md`,
and per-decision documentation lives in `benchmarks/NOTES.md`.

## Usage model

Results live under `benchmarks/.asv/results/<machine>/<sha>-<env>.json`,
one file per (declearn commit, environment) pair. The model is
**append-only**:

- Past releases are benchmarked once and the results stored.
- On a new release / commit / on-demand trigger, the suite runs
  against just that new commit and the result is appended.
- `asv publish` regenerates the comparison graph from all stored
  results — old + new together.

This is what ASV is built for, and the eventual target is CI
integration: a release/commit triggers a benchmark run automatically.
