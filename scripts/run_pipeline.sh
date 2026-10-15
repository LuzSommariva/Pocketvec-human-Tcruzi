#!/usr/bin/env bash
# Run the pipeline stages in order. Paths come from config/config.yaml.
#
# Usage:
#   scripts/run_pipeline.sh                # target selection (01-04), then PocketVec + analysis (05-07)
#   scripts/run_pipeline.sh --selection    # only ChEMBL target selection (01-04)
#   scripts/run_pipeline.sh --pocketvec    # only PocketVec batch + analysis (05-07)
#   scripts/run_pipeline.sh --example      # 05-07 on the bundled MAGL example pair (no docking, no network)
#
# Arguments after "--" are passed to 05_run_pocketvec_batch.py, e.g.:
#   scripts/run_pipeline.sh --pocketvec -- --target-ids 1,3
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python}"

mode="all"
batch_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --selection) mode="selection"; shift ;;
        --pocketvec) mode="pocketvec"; shift ;;
        --example)   mode="example"; shift ;;
        --) shift; batch_args=("$@"); break ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
done

run() {
    echo
    echo "==> $*"
    "$PY" "$@"
}

config_value() {
    "$PY" -c "import sys; sys.path.insert(0, 'src'); from utils import config; print(config.$1)"
}

run_selection() {
    run src/01_extract_chembl.py
    local filtered_pkl
    filtered_pkl="$(config_value CHEMBL_FILTERED_PKL)"
    if [[ ! -f "$filtered_pkl" ]]; then
        echo "Missing $filtered_pkl: run notebooks/chembl37_bioactivity_filtering.ipynb, then re-run with --selection." >&2
        exit 1
    fi
    run src/02_map_targets_to_pdb.py
    run src/03_characterize_pdb_ligands.py
    run src/04_rank_targets.py
}

run_pocketvec() {
    local targets_csv
    targets_csv="$(config_value TARGETS_CSV)"
    if [[ ! -f "$targets_csv" ]]; then
        echo "Missing $targets_csv: build it from the ranked targets and the T. cruzi homolog search (see README)." >&2
        exit 1
    fi
    run src/05_run_pocketvec_batch.py "${batch_args[@]}"
    run src/06_analyze_selectivity.py
    run src/07_qc_plddt_pocket.py "$@"
}

case "$mode" in
    all)       run_selection; run_pocketvec ;;
    selection) run_selection ;;
    pocketvec) run_pocketvec ;;
    example)
        export POCKETVEC_CONFIG=config/example.yaml
        run_pocketvec --no-download
        ;;
esac
