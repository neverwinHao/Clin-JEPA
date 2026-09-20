# Architecture

Two components are deployed, the encoder and the predictor; a target encoder and a masked-prediction head exist only during training.

## Encoder

Qwen3-8B with frozen base weights and LoRA adapters (r = 16, alpha = 32, on `q_proj`, `k_proj`, `v_proj`, `o_proj`; about 15.3M trainable parameters). Each text, an hour of observations, an hour of interventions or the demographics of a stay, is mapped to a 4096-d embedding by the hidden state of its last token. Adapter handling, batched encoding and the target encoder are in `clin_jepa/training/encoder_ops.py`.

## Predictor

`clin_jepa.model.predictor.ACTransformerPredictor`: a 6-layer pre-norm Transformer encoder (92.5M parameters; hidden size 1024, 8 heads, FFN 4096) that works in the encoder's latent space. States, actions and demographics enter through three linear maps from 4096 to 1024 dimensions. The input is the interleaved sequence `[d, z_1, u_1, ..., z_T, u_T]` of length `2T + 1` under block-causal attention; every state position predicts the next state. At inference the predictor is kept and rolled out autoregressively, each predicted state becoming the next state input (`clin_jepa/rollout.py`).

## Target encoder (training only)

A second set of adapters held in fp32, updated as an exponential moving average of the online adapters (momentum 0.996 in the curriculum, raised linearly from 0.996 to 1 over the V-JEPA 2-AC style refinement). It supplies the stop-gradient targets of the L1 prediction loss and is discarded after training.

## Masked-prediction head (V-JEPA 2-AC style baseline only)

`clin_jepa.model.jepa_predictor.JEPAPredictor`: a 3-layer cross-attention Transformer decoder (16.9M parameters) that predicts the target embeddings of randomly masked hours from the visible ones during the baseline's encoder refinement; discarded afterwards.

## Curriculum

`clin_jepa/training/curriculum.py` trains the encoder adapters and the predictor in one run. The loss is the L1 error of the next-state prediction plus the L1 error of a two-step rollout. Phase lengths are those of `configs/train/clin_jepa.yaml`.

| phase | steps | encoder adapters | predictor | prediction targets |
|---|---|---|---|---|
| warmup | 3,072 | frozen | trained | online encoder |
| co-training | 3,072 | trained | trained | target encoder, updated every step |
| alignment | 1,024 | frozen | trained | target encoder, updated every step |
| hard sync | | target adapters set to the online adapters | | |
| finalization | 4,608 | frozen | trained | online encoder |

The encoder adapters do not change after co-training, so the checkpoints `after_co_training`, `after_alignment` and `final` share one encoder and differ in the predictor.

## The six models

| model | encoder (02) | embedding cache (03) | predictor | rollout (05) | evaluated in |
|---|---|---|---|---|---|
| Clin-JEPA | curriculum, `clin_jepa.yaml`, checkpoint `final` | `clin_jepa` | same run and checkpoint (02) | `clin_jepa` | 06, 07, 08 |
| V-JEPA 2-AC style | masked-prediction refinement, checkpoint `best` | `vjepa2ac` | trained on the cache (04) | `vjepa2ac` | 06, 07, 08 |
| SFT baseline w/o JEPA | `encoder_sft/final` | `sft_baseline` | trained on the cache (04) | `sft_baseline` | 06, 07, 08 |
| Clin-JEPA w/o warmup | curriculum, `no_warmup.yaml`, checkpoint `after_sync` | `no_warmup`, test split | same run and checkpoint (02) | `no_warmup` | 06 |
| Clin-JEPA w/o alignment | curriculum, `no_alignment.yaml`, checkpoint `after_co_training` | `no_alignment`, test split | same run and checkpoint (02) | `no_alignment` | 06 |
| Clin-JEPA w/ separately trained predictor | Clin-JEPA's | `clin_jepa` | three trainings on the cache (04) | `separate_seed_{42,1337,2026}` | 06 |

All models start from `encoder_sft/final`. For stage 06, stage 05 also rolls out Clin-JEPA's predictor as saved after warmup, co-training and alignment, on the `sft_baseline`, `clin_jepa` and `clin_jepa` caches.

## Flow

```
trajectory shards (01)
  -> encoder SFT (02)
       -> curriculum: encoder adapters and predictor in one run (02)
       -> V-JEPA 2-AC style refinement of the encoder (02)
  -> embedding caches (03) -> predictors on frozen encoders (04)
  -> rollout (05) -> stability (06), downstream (08)
     embedding caches (03) -> geometry (07)
```
