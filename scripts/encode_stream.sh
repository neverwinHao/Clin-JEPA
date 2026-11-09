#!/usr/bin/env bash
# Encode completed trajectory shards as they appear, using every idle GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
source .env
set +a
PYTHON="${CLIN_JEPA_PYTHON:?Set CLIN_JEPA_PYTHON to the project Python}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
umask 077
mkdir -p "${CLIN_JEPA_OUTPUT}/logs"
exec 9>"${CLIN_JEPA_OUTPUT}/.encode.lock"
if ! flock -n 9; then
    printf 'An encoding job is already running.\n' >&2
    exit 1
fi
mapfile -t gpus < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); if ($2 < 100 && $3 == 0) print $1}')
if [ "${#gpus[@]}" -eq 0 ]; then
    printf 'No idle GPUs found.\n' >&2
    exit 1
fi
printf 'Streaming encoding on idle GPU indices: %s\n' "${gpus[*]}"
pids=()
cleanup() {
    trap - INT TERM HUP
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 129' HUP
for gpu in "${gpus[@]}"; do
    log="${CLIN_JEPA_OUTPUT}/logs/encode_stream_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_DISABLE_PROGRESS_BARS=1 \
        "$PYTHON" -u -m clin_jepa.encode_stream \
        --complete-marker "$CLIN_JEPA_DATA/.progress/validation.done" > "$log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
if [ "$failed" -ne 0 ]; then
    printf 'An encoder failed; see GPU logs under %s/logs.\n' "$CLIN_JEPA_OUTPUT" >&2
    exit 1
fi
printf 'All trajectory embeddings complete.\n'
