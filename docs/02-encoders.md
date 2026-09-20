# 02 · Encoders

Input: `$CLIN_JEPA_DATA/trajectories`. Output: `$CLIN_JEPA_OUTPUT/<run>/`. The curriculum and the refinement save `<run>/resume.pt` every 50 steps and on SIGUSR1 / SIGTERM, and resume from it when it exists; encoder SFT starts over. `--max_windows N` (`--max_samples N` for SFT) runs any trainer on a small part of the data.

## Encoder SFT (shared initialization of all models)

```bash
torchrun --nproc_per_node=4 -m clin_jepa.training.encoder_sft --config configs/train/encoder_sft.yaml
```

Writes `encoder_sft/{latest,best,final}`; every later stage starts from `encoder_sft/final`. About 10 hours on 4 GPUs.

## Co-training curriculum

```bash
torchrun --nproc_per_node=8 -m clin_jepa.training.curriculum --config configs/train/clin_jepa.yaml
torchrun --nproc_per_node=4 -m clin_jepa.training.curriculum --config configs/train/no_warmup.yaml
torchrun --nproc_per_node=4 -m clin_jepa.training.curriculum --config configs/train/no_alignment.yaml
```

`training.phases` gives the lengths of warmup, co-training, alignment and finalization in optimizer steps, `training.sync` switches the hard sync before finalization, and `training.max_steps`, when set, ends the run at that step. Each run writes `checkpoints/after_<phase>/`, `checkpoints/after_sync/`, `checkpoints/latest/` and, when the schedule completes, `checkpoints/final/`; a checkpoint holds the encoder adapter (`online/`) and the predictor (`predictor.pt`).

| model | config | evaluated checkpoint |
|---|---|---|
| Clin-JEPA | `clin_jepa.yaml` | `final` |
| Clin-JEPA w/o warmup | `no_warmup.yaml` | `after_sync` |
| Clin-JEPA w/o alignment | `no_alignment.yaml` | `after_co_training` |

Clin-JEPA took about 54 hours on 8 H200 GPUs (≈ 430 GPU-hours).

## V-JEPA 2-AC style refinement

```bash
torchrun --nproc_per_node=2 -m clin_jepa.training.refine_encoder_vjepa --config configs/train/refine_encoder_vjepa.yaml
```

Stops when `z_std` falls below `training.collapse_guard`, when the validation loss stops improving (`training.early_stop`) or at the end of `training.epochs`; the evaluated encoder is `vjepa2ac/checkpoints/best/online`. About 24 hours on 2 GPUs.

The SFT baseline w/o JEPA uses `encoder_sft/final` unchanged.
