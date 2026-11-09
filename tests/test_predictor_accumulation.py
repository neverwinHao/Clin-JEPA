"""Verify gradient accumulation preserves the full-batch objective."""
import copy
import random

import torch

from clin_jepa.model.predictor import ACTransformerPredictor
from clin_jepa.training.predictor_dataset import predictor_collate_fn
from clin_jepa.training.train_predictor import backward_accumulated, rollout_loss, teacher_forcing_loss


def test_accumulation_matches_full_batch_with_variable_lengths():
    torch.manual_seed(7)
    samples = []
    for length in (7, 6, 5, 6):
        samples.append({
            "z_states": torch.randn(length, 4),
            "z_actions": torch.randn(length, 4),
            "z_static": torch.randn(4),
            "length": length,
        })
    full = predictor_collate_fn(samples)
    microbatches = [predictor_collate_fn(samples[:2]), predictor_collate_fn(samples[2:])]
    model = ACTransformerPredictor(state_dim=4, action_dim=4, static_dim=4,
                                   hidden_dim=8, num_layers=1, num_heads=2,
                                   ffn_dim=16, max_timesteps=8,
                                   ffn_dropout=0.0, attn_dropout=0.0)
    accumulated = copy.deepcopy(model)
    random.seed(13)
    predictions = model(full["z_states"], full["z_static"], full["z_actions"], full["lengths"])
    teacher = teacher_forcing_loss(predictions, full["z_states"][:, 1:], full["padding_mask"][:, 1:])
    rollout = rollout_loss(model, full["z_states"], full["z_static"], full["z_actions"], full["lengths"], 2)
    (teacher + rollout).backward()
    random.seed(13)
    tf, roll = backward_accumulated(accumulated, microbatches, torch.device("cpu"), 2, 0.0, "fp32")
    assert abs(tf - teacher.item()) < 1e-5
    assert abs(roll - rollout.item()) < 1e-5
    for (name, p), (other_name, q) in zip(model.named_parameters(), accumulated.named_parameters()):
        assert name == other_name
        assert torch.allclose(p.grad, q.grad, atol=3e-5, rtol=3e-4), name
