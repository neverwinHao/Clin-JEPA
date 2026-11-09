#!/usr/bin/env bash
# Encode distinct shards concurrently on currently idle GPUs.
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
    printf "An encoding job is already running.\n" >&2
    exit 1
fi
mapfile -t gpus < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); if ($2 < 100 && $3 == 0) print $1}')
count="${#gpus[@]}"
if [ "$count" -eq 0 ]; then
    printf 'No idle GPUs found.\n' >&2
    exit 1
fi
printf 'Encoding on idle GPU indices: %s\n' "${gpus[*]}"
for split in train val test; do
    shopt -s nullglob
    shards=("$CLIN_JEPA_DATA"/trajectories/"$split"/trajectories_*.pt)
    shopt -u nullglob
    if [ "${#shards[@]}" -eq 0 ]; then
        printf 'No %s trajectory shards found.\n' "$split" >&2
        exit 1
    fi
    manifest=$(mktemp "${CLIN_JEPA_OUTPUT}/logs/encode_${split}_manifest.XXXXXX")
    printf "%s\n" "${shards[@]}" > "$manifest"
    workers="$count"
    if [ "${#shards[@]}" -lt "$workers" ]; then workers="${#shards[@]}"; fi
    pids=()
    for ((slot=0; slot<workers; slot++)); do
        gpu="${gpus[$slot]}"
        log="${CLIN_JEPA_OUTPUT}/logs/encode_${split}_gpu${gpu}.log"
        CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_DISABLE_PROGRESS_BARS=1 \
            "$PYTHON" -u -m clin_jepa.encode --config configs/encode_frozen_base.yaml \
            --model qwen3_base --split "$split" --task_id "$slot" --num_tasks "$workers" --manifest "$manifest" \
            > "$log" 2>&1 &
        pids[$slot]=$!
    done
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    if [ "$failed" -ne 0 ]; then
        printf '%s encoding failed; see GPU logs under %s/logs.\n' "$split" "$CLIN_JEPA_OUTPUT" >&2
        exit 1
    fi
    rm "$manifest"
    printf '%s embeddings complete.\n' "$split"
done
