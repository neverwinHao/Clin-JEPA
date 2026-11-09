"""The hypothetical rollout must not use recorded future actions or states."""
import torch
from scripts.prepare_clinical_rollout import future_features


class ToyPredictor:
    def __init__(self):
        self.actions = []

    def forward_rollout_step(self, z_hist, static, a_hist, next_action, current):
        self.actions.append(next_action.clone())
        return current + next_action


def test_future_records_are_never_used():
    torch.manual_seed(3)
    states = torch.randn(2, 30, 4)
    actions = torch.randn(2, 30, 4)
    static = torch.randn(2, 4)
    synthetic = torch.zeros(23, 4)
    m1 = ToyPredictor()
    result = future_features(m1, states, actions, static, synthetic, torch.device('cpu'))
    later_states = states.clone()
    later_actions = actions.clone()
    later_states[:, 24:] = 9999
    later_actions[:, 24:] = -9999
    m2 = ToyPredictor()
    modified = future_features(m2, later_states, later_actions, static, synthetic, torch.device('cpu'))
    assert torch.equal(result, modified)
    assert torch.equal(m1.actions[0], actions[:, 23])
    assert all(torch.count_nonzero(a) == 0 for a in m1.actions[1:])
    assert result.shape == (2, 3, 4)
