"""Patient timelines for CLMBR-T-base: the mapped MIMIC-IV events of every subject before its cutoff."""

import argparse
import logging
import pickle
from pathlib import Path

import msgpack
import numpy as np
import pandas as pd

from clin_jepa.downstream.rows import scan
from clin_jepa.downstream.tasks import CONTEXT
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

TABLES = {
    "labevents": ("hosp", "charttime", "valuenum", "d_labitems_to_loinc.csv"),
    "chartevents": ("icu", "charttime", "valuenum", "meas_chartevents_main.csv"),
    "inputevents": ("icu", "starttime", None, "inputevents_to_rxnorm.csv"),
    "outputevents": ("icu", "charttime", "value", "outputevents_to_loinc.csv"),
    "procedureevents": ("icu", "starttime", None, "proc_itemid.csv"),
}
BIRTH_CODE = "SNOMED/184099003"
CHUNK_ROWS = 2_000_000
EARLIEST_BIRTH_YEAR = 1900


def code_map(mapping_dir: Path, table: str) -> pd.Series:
    """``itemid -> "<vocabulary>/<concept code>"`` of one event table."""
    frame = pd.read_csv(mapping_dir / TABLES[table][3], low_memory=False)
    frame = frame.rename(columns={"itemid (omop_source_code)": "itemid"})
    frame = frame.dropna(subset=["itemid", "omop_vocabulary_id", "omop_concept_code"])
    frame["itemid"] = pd.to_numeric(frame["itemid"], errors="coerce")
    frame = frame.dropna(subset=["itemid"]).drop_duplicates("itemid")
    concept = frame["omop_concept_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    return pd.Series((frame["omop_vocabulary_id"].astype(str) + "/" + concept).to_numpy(),
                     index=frame["itemid"].astype(int).to_numpy())


class Vocabulary:
    """Codes of the model's dictionary: plain codes, and numeric codes with the range their bins cover."""

    def __init__(self, path: Path) -> None:
        with open(path, "rb") as f:
            entries = msgpack.load(f)["vocab"]
        self.plain = {e["code_string"] for e in entries if e["type"] == "code" and e["code_string"]}
        bins = pd.DataFrame([(e["code_string"], e["val_start"], e["val_end"]) for e in entries
                             if e["type"] == "numeric" and e["code_string"]], columns=["code", "start", "end"])
        self.low, self.high = bins.groupby("code")["start"].min(), bins.groupby("code")["end"].max()

    def measurements(self, events: pd.DataFrame) -> pd.DataFrame:
        """Rows the tokenizer keeps: a binned code with its value clipped into range, or a plain code without value."""
        binned = events["code"].isin(self.low.index) & events["value"].notna()
        plain = ~binned & events["code"].isin(self.plain)
        low, high = events["code"].map(self.low), events["code"].map(self.high)
        value = events["value"].clip(lower=low, upper=high).where(binned)
        return events.assign(value=value)[binned | plain]


def evaluation_rows(config: dict, rows: str, split: str) -> pd.DataFrame:
    """Rows to represent and the time each one is read at."""
    data_dir = Path(config["data_dir"])
    cohort = pd.read_csv(data_dir / "cohort" / "cohort.csv", usecols=["stay_id", "subject_id", "icu_intime"],
                         parse_dates=["icu_intime"])
    windows, stays = scan(data_dir / "trajectories", split)
    if rows == "stay":
        stays = stays[["stay_id"]].merge(cohort, on="stay_id")
        out = stays.sort_values(["icu_intime", "stay_id"], kind="stable").groupby("subject_id", as_index=False).first()
        out["target_hour"] = CONTEXT - 1
    else:
        out = windows.merge(cohort, on="stay_id")
    out["cutoff"] = out["icu_intime"] + pd.to_timedelta(out["target_hour"] + 1, unit="h")
    return out[["subject_id", "stay_id", "target_hour", "cutoff"]]


def read_events(config: dict, vocabulary: Vocabulary, cutoff: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept, coverage = [], []
    for table, (directory, time_column, value_column, _) in TABLES.items():
        codes = code_map(Path(config["mapping_dir"]), table)
        columns = ["subject_id", "itemid", time_column] + ([value_column] if value_column else [])
        counts = np.zeros(3, dtype=np.int64)
        for chunk in pd.read_csv(Path(config["mimic_dir"]) / directory / f"{table}.csv.gz", usecols=columns,
                                 chunksize=CHUNK_ROWS, low_memory=False):
            chunk = chunk[chunk["subject_id"].isin(cutoff.index)]
            time = pd.to_datetime(chunk[time_column], errors="coerce")
            chunk = chunk[time.notna() & (time < chunk["subject_id"].map(cutoff))]
            value = pd.to_numeric(chunk[value_column], errors="coerce") if value_column else np.nan
            events = pd.DataFrame({"subject_id": chunk["subject_id"], "time": time[chunk.index],
                                   "code": chunk["itemid"].map(codes), "value": value})
            mapped = events[events["code"].notna()]
            in_vocabulary = vocabulary.measurements(mapped)
            counts += [len(events), len(mapped), len(in_vocabulary)]
            kept.append(in_vocabulary)
        coverage.append({"table": table, "events": counts[0], "mapped": counts[1], "in_vocabulary": counts[2]})
        logger.info("%s: %d events, %d mapped, %d in vocabulary", table, *counts)
    events = pd.concat(kept, ignore_index=True).sort_values(["subject_id", "time"], kind="stable")
    return events, pd.DataFrame(coverage)


def patients(events: pd.DataFrame, vocabulary: Vocabulary, mimic_dir: Path):
    """One MEDS patient per subject: a birth event, then one event per distinct time."""
    people = pd.read_csv(mimic_dir / "hosp" / "patients.csv.gz",
                         usecols=["subject_id", "gender", "anchor_age", "anchor_year"]).set_index("subject_id")
    birth_year = (people["anchor_year"] - people["anchor_age"]).clip(lower=EARLIEST_BIRTH_YEAR)
    subjects, times = events["subject_id"].to_numpy(), events["time"].to_numpy()
    codes, values = events["code"].to_numpy(), events["value"].to_numpy()
    first = np.flatnonzero(np.r_[True, (subjects[1:] != subjects[:-1]) | (times[1:] != times[:-1])])
    patient = None
    for start, stop in zip(first, np.r_[first[1:], len(events)]):
        subject = int(subjects[start])
        if patient is None or patient["patient_id"] != subject:
            if patient is not None:
                yield patient
            birth = [{"code": BIRTH_CODE}]
            gender = f"Gender/{people.at[subject, 'gender']}"
            if gender in vocabulary.plain:
                birth.append({"code": gender})
            born = pd.Timestamp(year=int(birth_year[subject]), month=1, day=1).to_pydatetime()
            patient = {"patient_id": subject, "events": [{"time": born, "measurements": birth}]}
        measurements = [{"code": c} if np.isnan(v) else {"code": c, "numeric_value": float(v)}
                        for c, v in zip(codes[start:stop], values[start:stop])]
        patient["events"].append({"time": pd.Timestamp(times[start]).to_pydatetime(), "measurements": measurements})
    if patient is not None:
        yield patient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/clmbr.yaml")
    parser.add_argument("--rows", required=True, choices=["stay", "hourly"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    vocabulary = Vocabulary(Path(config["model_dir"]) / "dictionary.msgpack")
    rows = evaluation_rows(config, args.rows, args.split)
    events, coverage = read_events(config, vocabulary, rows.groupby("subject_id")["cutoff"].max())

    output_dir = ensure_dir(Path(config["output_dir"]) / "clmbr" / "timelines" / args.rows / args.split)
    rows.to_parquet(output_dir / "rows.parquet", index=False)
    coverage.to_csv(output_dir / "coverage.csv", index=False)
    shard: list[dict] = []
    n_shards = 0
    for patient in patients(events, vocabulary, Path(config["mimic_dir"])):
        shard.append(patient)
        if len(shard) == config["patients_per_shard"]:
            (output_dir / f"patients_{n_shards:04d}.pkl").write_bytes(pickle.dumps(shard))
            shard, n_shards = [], n_shards + 1
    if shard:
        (output_dir / f"patients_{n_shards:04d}.pkl").write_bytes(pickle.dumps(shard))
    logger.info("%d subjects with events", events["subject_id"].nunique())


if __name__ == "__main__":
    main()
