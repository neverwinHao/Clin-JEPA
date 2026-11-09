#!/usr/bin/env bash
# Resumable preparation of the released Clin-JEPA trajectories.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
source .env
set +a
PYTHON="${CLIN_JEPA_PYTHON:?Set CLIN_JEPA_PYTHON to the project Python}"
jobs="${CLIN_JEPA_CPU_JOBS:-2}"
if ! [[ "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'CLIN_JEPA_CPU_JOBS must be a positive integer\n' >&2
    exit 1
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
umask 077
progress="${CLIN_JEPA_DATA}/.progress"
mkdir -p "$progress" "${CLIN_JEPA_OUTPUT}/logs"

"$PYTHON" - <<'PY'
from pathlib import Path
from clin_jepa.utils import load_config
config = load_config("configs/data/paths.yaml")
missing = [str(value) for group in config["concepts"].values() for value in group.values()
           if not Path(value).is_file() or Path(value).stat().st_size == 0]
if missing:
    raise SystemExit("Missing concept tables:\n" + "\n".join(missing))
print("Concept files present:", sum(len(x) for x in config["concepts"].values()), flush=True)
PY

run_step() {
    local key="$1"
    shift
    if [ -f "${progress}/${key}.done" ]; then
        printf '%s: already complete\n' "$key"
        return
    fi
    printf '%s: starting\n' "$key"
    if ! "$PYTHON" -u "$@" > "${CLIN_JEPA_OUTPUT}/logs/${key}.log" 2>&1; then
        printf '%s: failed; see %s/logs/%s.log\n' "$key" "$CLIN_JEPA_OUTPUT" "$key" >&2
        return 1
    fi
    touch "${progress}/${key}.done"
    printf '%s: complete\n' "$key"
}

run_chunks() {
    local stage="$1" module="$2" i pid pending=0 failed=0 limit="$jobs"
    if [ "$stage" = trajectories ] && [ "$limit" -gt 2 ]; then limit=2; fi
    local -a pids=()
    for i in {0..9}; do
        if [ -f "${progress}/${stage}_$i.done" ]; then
            printf "%s_%s: already complete\n" "$stage" "$i"
            continue
        fi
        if [ "$stage" = trajectories ]; then
            run_step "${stage}_$i" -m "$module" --chunk_idx "$i" --n_workers 4 &
        else
            run_step "${stage}_$i" -m "$module" --chunk_idx "$i" --total_chunks 10 &
        fi
        pids[$pending]=$!
        pending=$((pending + 1))
        if [ "$pending" -eq "$limit" ]; then
            for pid in "${pids[@]}"; do
                if ! wait "$pid"; then failed=1; fi
            done
            if [ "$failed" -ne 0 ]; then return 1; fi
            pids=()
            pending=0
        fi
    done
    if [ "$pending" -gt 0 ]; then
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then failed=1; fi
        done
    fi
    return "$failed"
}

run_step cohort -m clin_jepa.data.step01_cohort
run_step split -m clin_jepa.data.step05_split
run_chunks observations clin_jepa.data.step02_observations
run_chunks actions clin_jepa.data.step03_actions
run_chunks discretization clin_jepa.data.step04_discretization
run_chunks trajectories clin_jepa.data.step06_trajectories
run_step validation scripts/validate_trajectories.py --data "$CLIN_JEPA_DATA"
printf 'Trajectory preparation complete.\n'
