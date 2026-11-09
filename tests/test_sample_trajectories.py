"""Subsampling preserves hours and overlapping windows."""
import numpy as np
from scripts.sample_trajectories import subset_shard


def test_remaps_hours_without_changing_content():
    source = {
        'per_hour': {'stay_ids': np.array([10, 10, 20, 20, 20]),
                     'hours': np.array([0, 1, 0, 1, 2]),
                     'state_texts': ['a', 'b', 'c', 'd', 'e'],
                     'action_texts': ['f', 'g', 'h', 'i', 'j'],
                     'sofa': np.arange(5, dtype=np.float32)},
        'per_stay': {'stay_ids': np.array([10, 20]), 'ages': np.array([40., 70.]),
                     'genders': ['M', 'F'], 'races': ['A', 'B']},
        'windows': [{'stay_index': 0, 'start_step': 0, 'length': 2, 'hour_indices': np.array([0, 1])},
                    {'stay_index': 1, 'start_step': 0, 'length': 2, 'hour_indices': np.array([2, 3])},
                    {'stay_index': 1, 'start_step': 1, 'length': 2, 'hour_indices': np.array([3, 4])}]}
    result = subset_shard(source, {20})
    assert result['per_stay']['stay_ids'].tolist() == [20]
    assert result['per_hour']['state_texts'] == ['c', 'd', 'e']
    assert result['per_hour']['action_texts'] == ['h', 'i', 'j']
    assert [w['stay_index'] for w in result['windows']] == [0, 0]
    assert [w['hour_indices'].tolist() for w in result['windows']] == [[0, 1], [1, 2]]
    assert subset_shard(source, {99}) is None
