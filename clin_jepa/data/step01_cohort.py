"""Step 01: select the ICU-stay cohort and write ``cohort/cohort.csv``."""

import argparse
import logging

import pandas as pd

from clin_jepa.data.helpers.io_utils import load_concepts_table, resolve_table_path
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "stay_id",
    "subject_id",
    "hadm_id",
    "gender",
    "admission_age",
    "race",
    "icu_intime",
    "icu_outtime",
    "los_icu",
    "icu_los_hours",
    "hospital_expire_flag",
    "sepsis3",
    "sepsis_sofa_time",
]


def _log_filter(name: str, before: int, after: int) -> None:
    logger.info("filter [%s]: %d -> %d", name, before, after)


def _flag(column: pd.Series) -> pd.Series:
    """Boolean column of a concept table, exported as ``t`` / ``f`` or as ``true`` / ``false``."""
    return column.astype(str).str.lower().isin(["t", "true"])


def _stays_with_vitals(paths_config: dict) -> set:
    filepath = resolve_table_path(paths_config, "concepts.measurement.vitalsign")
    stay_ids: set = set()
    for chunk in pd.read_csv(filepath, usecols=["stay_id"], chunksize=1_000_000):
        stay_ids.update(chunk["stay_id"].dropna().astype(int).unique())
    return stay_ids


def build_cohort(paths_config: dict, cohort_config: dict) -> pd.DataFrame:
    """Apply the inclusion criteria to ``icustay_detail`` and join the Sepsis-3 flag."""
    df = load_concepts_table(paths_config, "concepts.demographics.icustay_detail")
    n_initial = len(df)

    df = df.dropna(subset=["stay_id", "subject_id", "hadm_id"])
    df = df.drop_duplicates(subset=["stay_id"], keep="first")

    n = len(df)
    inclusion = cohort_config["inclusion"]
    min_age = inclusion["min_age"]
    df = df[df["admission_age"] >= min_age]
    _log_filter(f"age >= {min_age}", n, len(df))

    n = len(df)
    df = df.dropna(subset=["icu_intime", "icu_outtime"])
    _log_filter("valid ICU times", n, len(df))

    n = len(df)
    df = df[_flag(df["first_icu_stay"])]
    _log_filter("first ICU stay of the admission", n, len(df))

    n = len(df)
    min_los_hours = inclusion["min_icu_los_hours"]
    df = df[(df["los_icu"] * 24) >= min_los_hours]
    _log_filter(f"ICU LOS >= {min_los_hours} h", n, len(df))

    n = len(df)
    df = df[df["stay_id"].isin(_stays_with_vitals(paths_config))]
    _log_filter("at least one vital sign", n, len(df))

    df["icu_los_hours"] = (df["los_icu"] * 24).clip(upper=cohort_config["max_icu_los_hours"])

    sepsis = load_concepts_table(
        paths_config, "concepts.sepsis.sepsis3",
        usecols=["stay_id", "sepsis3", "sofa_time"],
    )
    sepsis["sepsis3"] = _flag(sepsis["sepsis3"])
    sepsis = sepsis.rename(columns={"sofa_time": "sepsis_sofa_time"})
    df = df.merge(sepsis, on="stay_id", how="left")
    df["sepsis3"] = df["sepsis3"].fillna(False).astype(bool)

    df = df[OUTPUT_COLUMNS].sort_values("stay_id").reset_index(drop=True)
    logger.info("cohort: %d of %d ICU stays", len(df), n_initial)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths_config", default="configs/data/paths.yaml")
    parser.add_argument("--cohort_config", default="configs/data/cohort.yaml")
    args = parser.parse_args()

    setup_logging()
    paths_config = load_config(args.paths_config)
    cohort = build_cohort(paths_config, load_config(args.cohort_config))

    output_dir = ensure_dir(resolve_table_path(paths_config, "output.cohort"))
    cohort.to_csv(output_dir / "cohort.csv", index=False)
    logger.info("wrote %s", output_dir / "cohort.csv")


if __name__ == "__main__":
    main()
