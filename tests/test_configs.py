from pathlib import Path

import pytest

from clin_jepa.downstream.baselines import configurations
from clin_jepa.downstream.report import ENCODERS
from clin_jepa.downstream.tasks import BY_NAME, TASK_SETS, TASKS, benchmark_tasks, feature_horizon
from clin_jepa.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").rglob("*.yaml"))


def keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from keys(value)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_config_loads_and_has_string_keys(path):
    assert all(isinstance(key, str) for key in keys(load_config(path)))


def test_task_contract():
    assert len(TASKS) == len(BY_NAME) == 34
    assert [len(benchmark_tasks(b)) for b in ("E1", "E2", "E3")] == [8, 18, 8]
    assert sum(task.hourly for task in TASKS) == 18
    assert {feature_horizon(task) for task in TASKS} == {8, 24, 48}
    assert all(feature_horizon(task) == 48 for task in TASKS if not task.hourly)


def test_task_sets_cover_every_task_once():
    named = [name for names in TASK_SETS.values() for name in names if name in BY_NAME]
    assert sorted(named) == sorted(BY_NAME)
    for names in TASK_SETS.values():
        assert len({BY_NAME[name].hourly for name in names if name in BY_NAME}) == 1


def test_baseline_configurations_are_fixed():
    spec = load_config(ROOT / "configs" / "downstream.yaml")["baselines"]
    counts = {name: len(configurations(spec[name], spec["seed"])) for name in ("ridge", "lightgbm", "sequence")}
    assert counts == {"ridge": 6, "lightgbm": 20, "sequence": 16}
    drawn = configurations(spec["lightgbm"], spec["seed"])
    assert drawn == configurations(spec["lightgbm"], spec["seed"])
    assert len({tuple(c.items()) for c in drawn}) == 20


def test_readout_specs_are_complete():
    readout = load_config(ROOT / "configs" / "downstream.yaml")["readout"]
    clmbr = load_config(ROOT / "configs" / "clmbr.yaml")["readout"]
    configurations = [dict(clmbr["defaults"], **c) for c in clmbr["configurations"]]
    assert len(configurations) == 66
    fields = set(readout["E1"])
    assert all(set(readout[b]) == fields for b in ("E2", "E3"))
    assert all(fields <= set(c) for c in configurations)
    cells = {(task.name, variant) for task in TASKS for variant in ("hist", "full")}
    assert {(r["task"], r["variant"]) for r in readout["seeds"]} <= cells
    assert all(r["encoder"] in ENCODERS for r in readout["seeds"])
