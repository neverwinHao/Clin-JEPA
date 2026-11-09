"""Task labels must align to the end of history without crossing splits."""
import pandas as pd
import pytest
from clin_jepa.downstream.tasks import Task
from scripts.evaluate_all_downstream import task_rows


def test_hourly_and_stay_labels_use_correct_time_and_admission():
    indexes = {}
    split_rows = []
    hourly_rows = []
    stay_rows = []
    for i, split in enumerate(('train', 'val', 'test')):
        stay_id = 100 + i
        indexes[split] = pd.DataFrame({'row': [0, 1], 'stay_id': [stay_id, stay_id], 'start_step': [0, 12]})
        split_rows.append({'stay_id': stay_id, 'subject_id': 200 + i, 'split': split})
        hourly_rows += [{'stay_id': stay_id, 'hour': 23, 'event': 0.}, {'stay_id': stay_id, 'hour': 35, 'event': 1.}]
        stay_rows.append({'stay_id': stay_id, 'outcome': float(i == 2)})
    splits = pd.DataFrame(split_rows)
    hourly = pd.DataFrame(hourly_rows)
    stay = pd.DataFrame(stay_rows)
    dynamic = task_rows(Task('event', 'E1', 8, 'Event'), indexes, hourly, stay, splits)
    for split in indexes:
        assert dynamic[split]['hour'].tolist() == [23, 35]
        assert dynamic[split]['event'].tolist() == [0., 1.]
        assert dynamic[split]['row'].tolist() == [0, 1]
    fixed = task_rows(Task('outcome', 'E3', None, 'Outcome'), indexes, hourly, stay, splits)
    for split in indexes:
        assert len(fixed[split]) == 1 and fixed[split].iloc[0]['start_step'] == 0
    bad = splits.copy()
    bad.loc[0, 'split'] = 'test'
    with pytest.raises(ValueError, match='split mismatch'):
        task_rows(Task('event', 'E1', 8, 'Event'), indexes, hourly, stay, bad)
