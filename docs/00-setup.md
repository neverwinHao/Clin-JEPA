# 00 · Setup

## Hardware

| stage | what we used |
|---|---|
| data (01) | CPU nodes, 8 cores and 64 GB per task |
| encoder training (02) | NVIDIA H200 (141 GB); Clin-JEPA pretraining ran on 8 GPUs for about 54 hours (≈ 430 GPU-hours) |
| embedding caches, predictors, rollout (03–05) | one H200 per job |
| evaluation (06–08) | CPU nodes; one GPU for the stability diagnostics, the sequence baselines and the read-outs |

The training configs fix `effective_batch_size`; gradient accumulation follows from the number of GPUs, which must divide `effective_batch_size / per_gpu_batch_size` (lower `per_gpu_batch_size` to use more GPUs or less memory). Host memory we requested: 512 GB for the 8-GPU curriculum run, 256 GB for encoder SFT on 4 GPUs.

## Environment

Python 3.10 or later, CUDA 12.

```bash
pip install -r requirements.txt
pip install -e .
pip install flash-attn==2.8.3 --no-build-isolation   # on a GPU node with the CUDA toolkit
pip install -e ".[dev]" && pytest                    # tests, a few seconds on CPU
```

## Paths

Four environment variables set the paths; `.env.example` lists them with the two CLMBR-T-base paths of stage 08:

| variable | content |
|---|---|
| `MIMIC_RAW` | MIMIC-IV v3.1 (`hosp/`, `icu/`) and the mimic-code concept tables (`concepts/`) |
| `CLIN_JEPA_DATA` | pipeline outputs: about 25 GB for stage 01, about 96 GB per embedding cache |
| `CLIN_JEPA_OUTPUT` | checkpoints, rollouts, evaluation results, figures and tables |
| `HF_HOME` | Hugging Face cache holding `Qwen/Qwen3-8B` (16 GB) |

```bash
cp .env.example .env    # edit, then:
set -a; source .env; set +a
```

## Data access

1. Obtain credentialed access to [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) on PhysioNet and download v3.1 into `$MIMIC_RAW` (`hosp/` and `icu/`).
2. Build the derived concept tables with [mimic-code](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts) and export each one as `$MIMIC_RAW/concepts/<group>/<table>.csv`. The tables that are read are listed under `concepts:` in `configs/data/paths.yaml`.
3. Download the base model once: `huggingface-cli download Qwen/Qwen3-8B`. The trainers then load it offline (`hf.offline` in the configs).

## SLURM

`slurm/data_array.sh` runs a data step as a ten-task array and `slurm/train.sh` launches a trainer on the GPUs of one node; fill in the partition and resources, and create `logs/`, where both write their output.
