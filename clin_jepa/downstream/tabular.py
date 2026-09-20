"""Hourly tabular view of the clinical record the encoders read: 400 columns per stay-hour and 7 static columns."""

import argparse
import functools
import json
import logging
import pickle
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from clin_jepa.data.helpers.action_filter import filter_act_events
from clin_jepa.data.helpers.observation_filter import filter_obs_events
from clin_jepa.utils import ensure_dir, load_config, repo_path, setup_logging

logger = logging.getLogger(__name__)

OBSERVATIONS = [
    "heart_rate", "sbp", "dbp", "mbp", "resp_rate", "temperature", "spo2",
    "so2", "po2", "pco2", "fio2", "ph", "baseexcess", "lactate", "pao2fio2ratio",
    "albumin", "aniongap", "bicarbonate", "bun", "calcium", "chloride", "creatinine", "glucose", "sodium", "potassium",
    "hematocrit", "hemoglobin", "mch", "mchc", "mcv", "platelet", "rbc", "rdw", "wbc",
    "inr", "pt", "ptt",
    "alt", "alp", "ast", "bilirubin_total", "ck_cpk", "ck_mb", "ld_ldh",
    "magnesium",
    "gcs", "gcs_motor", "gcs_verbal", "gcs_eyes",
    "sofa_24hours", "respiration_24hours", "coagulation_24hours", "liver_24hours", "cardiovascular_24hours",
    "cns_24hours", "renal_24hours",
    "urineoutput", "uo",
    "aki_stage",
    "o2_flow",
    "weight",
]
VASOACTIVE = ["norepinephrine", "epinephrine", "dopamine", "phenylephrine", "vasopressin", "dobutamine", "milrinone"]
VENTILATION_STATUS = ["InvasiveVent", "NonInvasiveVent", "HFNC", "SupplementalOxygen", "Tracheostomy"]
VENTILATOR_SETTINGS = [
    "fio2", "peep", "tidal_volume_observed", "tidal_volume_set", "tidal_volume_spontaneous", "minute_volume",
    "respiratory_rate_set", "respiratory_rate_spontaneous", "respiratory_rate_total", "plateau_pressure", "flow_rate",
]
VENTILATOR_MODES = ("ventilator_mode", "ventilator_mode_hamilton")
DIALYSIS_TYPES = {"crrt": ("CRRT", "CVVH", "CVVHD", "CVVHDF", "SCUF"), "ihd": ("IHD",), "peritoneal": ("Peritoneal",)}
CRRT_SETTINGS = [
    "blood_flow", "dialysate_rate", "replacement_rate", "prefilter_replacement_rate", "postfilter_replacement_rate",
    "hourly_patient_fluid_removal", "ultrafiltrate_output", "current_goal", "citrate", "heparin_dose",
    "access_pressure", "effluent_pressure", "filter_pressure", "return_pressure", "system_active", "clots", "clotted",
]
MEDICATION_TABLES = ("inputevents", "emar", "prescriptions", "antibiotic")
ADMINISTRATION_TABLES = ("inputevents", "emar", "antibiotic")
NOT_AN_ITEM_TABLE = {"ventilator_setting", "crrt", "rrt", "vasoactive_agent", "norepinephrine_equivalent_dose",
                     "ventilation", "neuroblock"}
NOT_AN_ITEM_NAME = {"route"}
STATICS = ["admission_age", "is_female", "race_white", "race_black", "race_hispanic", "race_asian", "race_other"]
RATE = re.compile(r"rate:\s*([-+]?\d*\.?\d+)")


def normalise(name) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def sanitise(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def has_value(events: pd.DataFrame) -> np.ndarray:
    text = events["text_value"]
    return (events["numeric_value"].notna() | (text.notna() & ~text.astype(str).isin(["", "nan", "None"]))).to_numpy()


def item_names(events: pd.DataFrame) -> pd.Series:
    """Item of an intervention row: its variable name, or the drug for rows of the antibiotic table."""
    names = events["variable_name"].astype(str)
    drug = (events["source_table"] == "antibiotic") & (events["variable_name"] == "antibiotic")
    return names.where(~drug, events["text_value"].astype(str)).map(normalise)


class MedicationClasses:
    """Ordered regular-expression rules on the item name; the first match wins."""

    def __init__(self, path: str) -> None:
        rules = yaml.safe_load(repo_path(path).read_text())
        self.names, self.fallback = list(rules["classes"]), rules["fallback"]
        self.rules = [(r["class"], re.compile(r["pattern"], re.IGNORECASE)) for r in rules["rules"]]
        self.of = functools.lru_cache(maxsize=None)(self._of)

    def _of(self, item: str) -> str:
        return next((name for name, pattern in self.rules if pattern.search(item)), self.fallback)


class Layout:
    """Column names of the hourly view; the item and ventilator-mode columns come from the training-split census."""

    def __init__(self, classes: MedicationClasses, census: dict) -> None:
        self.classes, self.modes = classes, list(census["ventilator_modes"])
        nan_columns = list(OBSERVATIONS) + [f"vaso_{a}" for a in VASOACTIVE] + ["norepinephrine_equivalent_dose"]
        columns = list(nan_columns) + [f"vent_status_{s}" for s in VENTILATION_STATUS]
        settings = [f"vent_{s}" for s in VENTILATOR_SETTINGS]
        columns += settings + [f"vent_mode_{i}_{sanitise(m)}" for i, m in enumerate(self.modes)] + ["vent_mode_other"]
        dialysis = ["dialysis_active", "dialysis_present"]
        columns += dialysis + [f"dialysis_type_{t}" for t in DIALYSIS_TYPES] + ["crrt_any"]
        crrt = [f"crrt_{s}" for s in CRRT_SETTINGS]
        columns += crrt + ["neuromuscular_blocker_active", "neuroblock_rate", "neuroblock_amount",
                           "antibiotic_active", "antibiotic_n_agents"]
        amounts = [f"class_{c}_amount" for c in classes.names] + [f"class_{c}_rate" for c in classes.names]
        columns += [f"class_{c}" for c in classes.names] + amounts + [f"order_placed_{c}" for c in classes.names]
        columns += [item["column"] for item in census["items"]]
        columns += ["hour_index", "n_obs_statements", "n_act_statements", "no_active_interventions"]
        self.columns = columns
        self.at = {name: i for i, name in enumerate(columns)}
        nan_columns += settings + dialysis + crrt + ["neuroblock_rate", "neuroblock_amount"] + amounts
        self.nan_columns = np.array([self.at[name] for name in nan_columns])
        self.item_column = {(item["source_table"], item["name"]): self.at[item["column"]] for item in census["items"]}


def hourly_mean(events: pd.DataFrame, n_hours: int, names: list[str]) -> np.ndarray:
    out = np.full((n_hours, len(names)), np.nan, dtype=np.float32)
    rows = events[events["numeric_value"].notna() & events["variable_name"].isin(names)]
    if len(rows):
        means = rows.groupby(["step_idx", "variable_name"], sort=False)["numeric_value"].mean()
        steps = means.index.get_level_values(0).to_numpy(dtype=np.int64)
        columns = pd.Index(names).get_indexer(means.index.get_level_values(1))
        ok = (steps >= 0) & (steps < n_hours)
        out[steps[ok], columns[ok]] = means.to_numpy(dtype=np.float32)[ok]
    return out


def add_hourly_sum(x: np.ndarray, steps: np.ndarray, columns: np.ndarray, values: np.ndarray) -> None:
    total = np.zeros(x.shape, dtype=np.float64)
    count = np.zeros(x.shape, dtype=np.int32)
    np.add.at(total, (steps, columns), values.astype(np.float64))
    np.add.at(count, (steps, columns), 1)
    charted = count > 0
    x[charted] = np.where(np.isnan(x[charted]), 0.0, x[charted])
    x += np.where(charted, total.astype(np.float32), 0.0)


def stay_matrix(stay: dict, layout: Layout, state_text: dict) -> np.ndarray:
    """``(n_hours, 400)`` float32; NaN where a numeric column has nothing charted, 0 for indicators and counts."""
    n_hours, at = int(stay["n_steps"]), layout.at
    x = np.zeros((n_hours, len(layout.columns)), dtype=np.float32)
    x[:, layout.nan_columns] = np.nan
    x[:, at["hour_index"]] = np.arange(n_hours, dtype=np.float32)

    obs = stay["obs_events"]
    x[:, : len(OBSERVATIONS)] = hourly_mean(obs, n_hours, OBSERVATIONS)
    if len(obs):
        stated = filter_obs_events(obs, state_text)
        steps = stated["step_idx"].to_numpy(dtype=np.int64)[has_value(stated)]
        x[:, at["n_obs_statements"]] = np.bincount(steps[(steps >= 0) & (steps < n_hours)], minlength=n_hours)

    act = stay["act_events"]
    act = filter_act_events(act) if len(act) else act
    act = act[(act["step_idx"] >= 0) & (act["step_idx"] < n_hours)] if len(act) else act
    if len(act) == 0:
        x[:, at["no_active_interventions"]] = 1.0
        return x

    steps = act["step_idx"].to_numpy(dtype=np.int64)
    table = act["source_table"].to_numpy()
    variable = act["variable_name"].astype(str).to_numpy()
    text = act["text_value"].astype(str).to_numpy()
    number = act["numeric_value"].to_numpy(dtype=np.float64)
    valued = has_value(act)

    def mark(mask: np.ndarray, column: str) -> None:
        x[np.unique(steps[mask]), at[column]] = 1.0

    def means(mask: np.ndarray, names: list[str], columns: list[str]) -> None:
        if mask.any():
            x[:, [at[c] for c in columns]] = hourly_mean(act[mask], n_hours, names)

    n_statements = np.bincount(steps[valued], minlength=n_hours).astype(np.float32)
    x[:, at["n_act_statements"]] = n_statements
    x[:, at["no_active_interventions"]] = n_statements == 0

    means(table == "vasoactive_agent", VASOACTIVE, [f"vaso_{a}" for a in VASOACTIVE])
    dose = "norepinephrine_equivalent_dose"
    means(table == dose, [dose], [dose])
    status = (table == "ventilation") & (variable == "ventilation_status")
    for s in VENTILATION_STATUS:
        mark(status & (text == s), f"vent_status_{s}")
    setting = table == "ventilator_setting"
    means(setting, VENTILATOR_SETTINGS, [f"vent_{s}" for s in VENTILATOR_SETTINGS])
    mode_rows = setting & np.isin(variable, VENTILATOR_MODES) & valued
    modes = np.array([normalise(t) for t in text[mode_rows]])
    known = [normalise(m) for m in layout.modes]
    for i, mode in enumerate(layout.modes):
        x[np.unique(steps[mode_rows][modes == known[i]]), at[f"vent_mode_{i}_{sanitise(mode)}"]] = 1.0
    x[np.unique(steps[mode_rows][~np.isin(modes, known)]), at["vent_mode_other"]] = 1.0

    rrt = table == "rrt"
    means(rrt, ["dialysis_active", "dialysis_present"], ["dialysis_active", "dialysis_present"])
    for kind, values in DIALYSIS_TYPES.items():
        mark(rrt & (variable == "dialysis_type") & valued & np.isin(text, values), f"dialysis_type_{kind}")
    mark(table == "crrt", "crrt_any")
    means(table == "crrt", CRRT_SETTINGS, [f"crrt_{s}" for s in CRRT_SETTINGS])
    mark(table == "neuroblock", "neuromuscular_blocker_active")
    means(table == "neuroblock", ["drug_rate", "drug_amount"], ["neuroblock_rate", "neuroblock_amount"])

    antibiotic = (table == "antibiotic") & (variable == "antibiotic") & valued
    if antibiotic.any():
        mark(antibiotic, "antibiotic_active")
        agents = pd.DataFrame({"step": steps[antibiotic], "agent": [normalise(t) for t in text[antibiotic]]})
        n_agents = agents.drop_duplicates().groupby("step").size()
        x[n_agents.index.to_numpy(), at["antibiotic_n_agents"]] = n_agents.to_numpy(dtype=np.float32)

    medication = np.isin(table, MEDICATION_TABLES) & valued & ~((table == "antibiotic") & (variable != "antibiotic"))
    if medication.any():
        items = item_names(act[medication]).to_numpy()
        m_steps, m_table, m_number, m_text = steps[medication], table[medication], number[medication], text[medication]
        group = np.array([layout.classes.of(item) for item in items])
        group = np.where((m_table == "antibiotic") & (group == layout.classes.fallback), "antibiotic", group)

        def columns(prefix: str, suffix: str, rows: np.ndarray) -> np.ndarray:
            return np.array([at[f"{prefix}{c}{suffix}"] for c in group[rows]], dtype=np.int64)

        given = np.isin(m_table, ADMINISTRATION_TABLES)
        x[m_steps[given], columns("class_", "", given)] = 1.0
        ordered = m_table == "prescriptions"
        x[m_steps[ordered], columns("order_placed_", "", ordered)] = 1.0
        infused = m_table == "inputevents"
        amount = infused & np.isfinite(m_number)
        if amount.any():
            add_hourly_sum(x, m_steps[amount], columns("class_", "_amount", amount), m_number[amount])
        rates = np.array([float(m.group(1)) if infused[i] and (m := RATE.search(m_text[i])) else np.nan
                          for i in range(len(m_text))])
        rated = np.isfinite(rates)
        if rated.any():
            add_hourly_sum(x, m_steps[rated], columns("class_", "_rate", rated), rates[rated])
        for i, item in enumerate(items):
            column = layout.item_column.get((m_table[i], item))
            if column is not None:
                x[m_steps[i], column] = 1.0
    return x


def race_group(race) -> str:
    text = "" if race is None or (isinstance(race, float) and np.isnan(race)) else str(race).upper()
    for group, words in (("white", ["WHITE"]), ("black", ["BLACK", "AFRICAN"]), ("hispanic", ["HISPANIC", "LATINO"]),
                         ("asian", ["ASIAN"])):
        if any(word in text for word in words):
            return group
    return "other"


def statics(stay_ids: np.ndarray, cohort: pd.DataFrame) -> np.ndarray:
    rows = cohort.reindex(stay_ids)
    race = np.array([race_group(r) for r in rows["race"]])
    columns = [rows["admission_age"].to_numpy(dtype=np.float32),
               (rows["gender"].astype(str).str.upper() == "F").to_numpy(dtype=np.float32)]
    columns += [(race == name.removeprefix("race_")).astype(np.float32) for name in STATICS[2:]]
    return np.stack(columns, axis=1)


def census_chunk(paths: list[Path]) -> tuple[Counter, Counter, Counter, Counter]:
    events, stays, mode_events, mode_stays = Counter(), Counter(), Counter(), Counter()
    for path in paths:
        act = pickle.load(open(path, "rb"))["act_events"]
        if len(act) == 0:
            continue
        act = filter_act_events(act)
        act = act[has_value(act)]
        keys = list(zip(act["category"], act["source_table"], item_names(act)))
        events.update(keys)
        stays.update(set(keys))
        rows = (act["source_table"] == "ventilator_setting") & act["variable_name"].isin(VENTILATOR_MODES)
        modes = [normalise(t) for t in act.loc[rows, "text_value"]]
        mode_events.update(modes)
        mode_stays.update(set(modes))
    return events, stays, mode_events, mode_stays


def census(config: dict, output_dir: Path) -> None:
    """Most common medication, fluid and antibiotic items and ventilator modes among the training stays."""
    data_dir, cfg = Path(config["data_dir"]), config["tabular"]
    splits = pd.read_csv(data_dir / "splits" / "splits.csv")
    train = sorted(splits.loc[splits["split"] == "train", "stay_id"])
    paths = [p for p in (data_dir / "discretized" / f"{int(s)}.pkl" for s in train) if p.exists()]
    chunks = [paths[i : i + 200] for i in range(0, len(paths), 200)]
    with Pool(config["n_workers"]) as pool:
        parts = pool.map(census_chunk, chunks)
    events, stays, mode_events, mode_stays = (sum((p[i] for p in parts), Counter()) for i in range(4))

    ranked = sorted(events, key=lambda k: (-stays[k], -events[k], k[1], k[2]))
    items, used = [], set()
    for category, table, name in ranked:
        if table in NOT_AN_ITEM_TABLE or name in NOT_AN_ITEM_NAME:
            continue
        column = base = f"item_{sanitise(table)}__{sanitise(name)}"
        suffix = 2
        while column in used:
            column, suffix = f"{base}_{suffix}", suffix + 1
        used.add(column)
        items.append({"category": category, "source_table": table, "name": name, "column": column,
                      "n_stays": stays[(category, table, name)]})
        if len(items) == cfg["n_items"]:
            break
    modes = sorted(mode_events, key=lambda m: (-mode_stays[m], -mode_events[m], m))[: cfg["n_ventilator_modes"]]
    (output_dir / "census.json").write_text(json.dumps({"items": items, "ventilator_modes": modes}, indent=1))
    logger.info("%d training stays, %d items, ventilator modes %s", len(paths), len(items), modes)


def extract(config: dict, output_dir: Path, task_id: int, num_tasks: int) -> None:
    data_dir = Path(config["data_dir"])
    layout = Layout(MedicationClasses(config["tabular"]["medication_classes"]),
                    json.loads((output_dir / "census.json").read_text()))
    state_text = load_config(config["tabular"]["trajectory_config"])["state_text"]
    cohort = pd.read_csv(data_dir / "cohort" / "cohort.csv",
                         usecols=["stay_id", "gender", "admission_age", "race"]).set_index("stay_id")
    shards = [p for split in ("train", "val", "test") for p in sorted((data_dir / "trajectories" / split).glob("*.pt"))]
    for path in shards[task_id::num_tasks]:
        stay_ids = np.asarray(torch.load(path, map_location="cpu", weights_only=False)["per_stay"]["stay_ids"],
                              dtype=np.int64)
        hourly = [stay_matrix(pickle.load(open(data_dir / "discretized" / f"{int(s)}.pkl", "rb")), layout, state_text)
                  for s in stay_ids]
        output = ensure_dir(output_dir / path.parent.name) / path.name.replace("trajectories_", "tabular_")
        torch.save({"stay_ids": stay_ids, "hourly": hourly, "statics": statics(stay_ids, cohort),
                    "columns": layout.columns, "static_columns": STATICS}, output)
        logger.info("%s: %d stays", output.name, len(stay_ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["census", "extract"])
    parser.add_argument("--config", default="configs/downstream.yaml")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=1)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    output_dir = ensure_dir(Path(config["output_dir"]) / "tabular")
    if args.step == "census":
        census(config, output_dir)
    else:
        extract(config, output_dir, args.task_id, args.num_tasks)


if __name__ == "__main__":
    main()
