"""Synthetic future policy is timed correctly and cannot read recorded future."""
import torch
from scripts.prepare_all_task_rollouts import rollout_features


class ToyPredictor:
    def __init__(self):
        self.seen = []

    def forward_rollout_step(self, z_hist, static, a_hist, next_action, current):
        self.seen.append(next_action.clone())
        return current + next_action * .01


def test_future_is_synthetic_and_uses_absolute_hour():
    torch.manual_seed(7)
    state = torch.randn(2, 30, 4)
    action = torch.randn(2, 30, 4)
    static = torch.randn(2, 4)
    policy = torch.arange(100).float().unsqueeze(1).expand(-1, 4)
    starts = torch.tensor([0, 12])
    first = ToyPredictor()
    result = rollout_features(first, state, action, static, policy, starts, torch.device('cpu'))
    state[:, 24:] = 9999
    action[:, 24:] = -9999
    second = ToyPredictor()
    changed = rollout_features(second, state, action, static, policy, starts, torch.device('cpu'))
    assert torch.equal(result, changed)
    assert torch.equal(first.seen[0], action[:, 23])
    assert torch.equal(first.seen[1], policy[starts + 24])
    assert torch.equal(first.seen[47], policy[starts + 70])
    assert result.shape == (2, 3, 3, 4)
