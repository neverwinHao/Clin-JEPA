# Frozen Qwen base and trajectory preparation

This experiment uses the unmodified pretrained Qwen3-8B base, frozen throughout.
It does not use SFT or joint encoder/predictor training, and is not the paper's
SFT baseline. The state predictor and its training objective are unchanged.

Set paths in .env, including QWEN3_BASE (a local Hugging Face model directory)
and CLIN_JEPA_PYTHON (the project's Python interpreter).

## Prepare the MIMIC concepts and trajectories

The raw MIMIC files remain read-only. Point MIMIC_RAW to a staging directory
with hosp and icu links plus a writable concepts directory. The script uses
the upstream mimic-code DuckDB SQL and stores its database outside Git.
On this server the mimic-code snapshot is 303d26c623dcc9c49cc0f204468d4acc2f063797.
The suspicion-of-infection join uses an equivalent hash-join rewrite to avoid
a slow nested loop on DuckDB 1.1; tests compare it with the original SQL.

    set -a; source .env; set +a
    "$CLIN_JEPA_PYTHON" scripts/prepare_mimic_concepts.py \
      --mimic-code "$MIMIC_CODE" --raw "$MIMIC_RAW" --mimic-root "$MIMIC_RAW" \
      --db "$CLIN_JEPA_DATA/../mimic.duckdb" --temp-dir "$CLIN_JEPA_DATA/../tmp"
    scripts/build_trajectories.sh

The concept build defaults to four CPU threads and a 24 GB DuckDB memory
limit. Successful import and concept steps are recorded in the database; the
trajectory script records completed chunks and runs two chunks concurrently by
default. Set CLIN_JEPA_CPU_JOBS to adjust this. Re-run after a failure to resume.
Each stage writes logs under CLIN_JEPA_OUTPUT/logs.

## Encode and train the predictor

    scripts/encode_parallel.sh
    "$CLIN_JEPA_PYTHON" -m clin_jepa.training.train_predictor \
      --config configs/train/predictor_frozen_base.yaml

The script distributes a stable snapshot of shards over all currently idle GPUs.
It skips already encoded shards; rerun it after more trajectories are generated.
The default embedding batch size is 32, with automatic splitting when
padded inputs exceed 4096 tokens. Test GPU memory before increasing these limits. The predictor uses microbatches of 128 and two gradient accumulation steps,
retaining an effective batch size of 256. Its output uses the same cache
format as the original project.

## Small real-data end-to-end check

Before a full run, generate a separate, deterministic smoke dataset from the
first existing train and validation shards. It preserves all windows and used
hourly records for the selected stays. The full pipeline's patient split must
already have been validated; the smoke run inherits that split. These few
stays are for checking connectivity and model updates only, not for estimating
clinical performance or reproducing the paper's results.

    set -a; source .env; set +a
    SMOKE=/media/cuichang/liuhao/clin-jepa/smoke  # choose an isolated writable directory
    "$CLIN_JEPA_PYTHON" scripts/create_smoke_subset.py \
      --source "$CLIN_JEPA_DATA/trajectories" --output "$SMOKE/processed/trajectories"
    "$CLIN_JEPA_PYTHON" scripts/validate_smoke_subset.py \
      --source "$CLIN_JEPA_DATA/trajectories" --subset "$SMOKE/processed/trajectories"
    # Use distinct idle GPUs for train and validation if available.
    CUDA_VISIBLE_DEVICES=0 CLIN_JEPA_DATA="$SMOKE/processed" \
      "$CLIN_JEPA_PYTHON" -m clin_jepa.encode --config configs/encode_frozen_base.yaml --model qwen3_base --split train
    CUDA_VISIBLE_DEVICES=1 CLIN_JEPA_DATA="$SMOKE/processed" \
      "$CLIN_JEPA_PYTHON" -m clin_jepa.encode --config configs/encode_frozen_base.yaml --model qwen3_base --split val
    "$CLIN_JEPA_PYTHON" scripts/validate_smoke_embeddings.py --data "$SMOKE/processed"
    CUDA_VISIBLE_DEVICES=0 CLIN_JEPA_DATA="$SMOKE/processed" CLIN_JEPA_OUTPUT="$SMOKE" \
      "$CLIN_JEPA_PYTHON" -m clin_jepa.training.train_predictor \
      --config configs/train/predictor_frozen_base_smoke.yaml

The smoke configuration uses the original predictor architecture, a small batch
and two epochs. No SFT adapter is loaded and no Qwen parameters are trained.
Results and embeddings stay under the separate smoke directory; neither data
nor model checkpoints should be committed or pushed to Git.

Verify the saved predictor can be reloaded and reproduces its validation loss:

    CUDA_VISIBLE_DEVICES=0 "$CLIN_JEPA_PYTHON" -m scripts.validate_smoke_predictor \
      --checkpoint "$SMOKE/predictor_qwen3_base_smoke/best.pt" \
      --embedding-dir "$SMOKE/processed/embeddings/qwen3_base"

## Patient-sampled experiment

To obtain an exploratory result sooner, sample patients independently within
existing train/validation/test splits. The defaults select 1,000 / 200 / 200
patients with seed 42 and retain all their stays and windows. This produces a
new dataset and refuses to overwrite an existing one. The result is not a full
cohort estimate; compare models on the same held-out patients.

    set -a; source .env; set +a
    EXP=/path/to/new/experiment
    mkdir -p "$EXP/logs"
    "$CLIN_JEPA_PYTHON" scripts/sample_trajectories.py \
      --source "$CLIN_JEPA_DATA" --output "$EXP/processed"
    "$CLIN_JEPA_PYTHON" scripts/validate_trajectories.py --data "$EXP/processed"
    "$CLIN_JEPA_PYTHON" scripts/validate_sample_source.py \
      --source "$CLIN_JEPA_DATA" --sample "$EXP/processed"
    touch "$EXP/processed/.progress/validation.done"
    # Start one process on each idle GPU; all processes share file locks.
    for gpu in 0 1 2; do
      CUDA_VISIBLE_DEVICES="$gpu" CLIN_JEPA_DATA="$EXP/processed" \
        "$CLIN_JEPA_PYTHON" -m clin_jepa.encode_stream \
        --complete-marker "$EXP/processed/.progress/validation.done" &
    done
    wait
    "$CLIN_JEPA_PYTHON" scripts/validate_embedding_caches.py --data "$EXP/processed"
    CUDA_VISIBLE_DEVICES=0 CLIN_JEPA_DATA="$EXP/processed" CLIN_JEPA_OUTPUT="$EXP/output" \
      "$CLIN_JEPA_PYTHON" -m clin_jepa.training.train_predictor \
      --config configs/train/predictor_frozen_base.yaml
    CUDA_VISIBLE_DEVICES=0 CLIN_JEPA_DATA="$EXP/processed" CLIN_JEPA_OUTPUT="$EXP/output" \
      "$CLIN_JEPA_PYTHON" -m clin_jepa.rollout \
      --config configs/rollout_frozen_base.yaml --model qwen3_base --split test
    "$CLIN_JEPA_PYTHON" scripts/summarize_rollout.py \
      --errors "$EXP/output/rollout/qwen3_base/test/errors.npz" --output "$EXP/output/rollout_summary.json"

Rollout conditions on recorded actions, including future actions; it is an
offline diagnostic and does not establish prospective clinical performance.
