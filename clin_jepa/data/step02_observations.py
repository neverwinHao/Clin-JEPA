"""Step 02: extract observation events from the mimic-code concept tables and the ICD codes, one parquet per stay."""

import gc
import logging

import numpy as np
import pandas as pd

from clin_jepa.data.helpers.io_utils import (
    add_events,
    extraction_args,
    load_and_filter_concepts_table,
    load_and_filter_raw_table,
    load_chunk_cohort,
    resolve_table_path,
    resolve_value_columns,
    write_stay_events,
)
from clin_jepa.data.helpers.schema import OUTPUT_COLUMNS, empty_events_df
from clin_jepa.data.helpers.temporal_utils import (
    assign_stay_ids_from_subject,
    build_icu_windows,
    build_intime_map,
    clip_to_icu_window,
    compute_hours_since_admission,
)
from clin_jepa.data.helpers.units import get_unit
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


def melt_to_long(
    df: pd.DataFrame,
    source_name: str,
    category: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    timestamp_col: str,
    intime_map: dict,
) -> pd.DataFrame:
    """Melt wide-format table to long-format event rows."""
    frames = []

    if numeric_cols:
        melted = df.melt(
            id_vars=["stay_id", timestamp_col],
            value_vars=numeric_cols,
            var_name="variable_name",
            value_name="numeric_value",
        )
        melted = melted.dropna(subset=["numeric_value"])
        melted["text_value"] = ""
        frames.append(melted)

    for col in categorical_cols:
        subset = df[["stay_id", timestamp_col, col]].copy()
        subset = subset.dropna(subset=[col])
        subset = subset[subset[col].astype(str).str.strip() != ""]
        subset = subset.rename(columns={col: "text_value"})
        subset["variable_name"] = col
        subset["numeric_value"] = np.nan
        subset["text_value"] = subset["text_value"].astype(str)
        frames.append(subset)

    if not frames:
        return empty_events_df()

    result = pd.concat(frames, ignore_index=True)

    result["timestamp"] = pd.to_datetime(result[timestamp_col])
    if timestamp_col != "timestamp":
        result = result.drop(columns=[timestamp_col])

    result["timestamp_hours"] = compute_hours_since_admission(
        result["timestamp"], result["stay_id"], intime_map
    )

    result["domain"] = "observation"
    result["source_table"] = source_name
    result["category"] = category
    result["end_timestamp"] = pd.NaT
    result["end_timestamp_hours"] = np.nan
    result["is_continuous"] = False

    result["unit"] = result["variable_name"].map(
        lambda vn: get_unit(source_name, vn)
    )

    return result[OUTPUT_COLUMNS]


def process_observation_source(
    source_name: str,
    source_config: dict,
    paths_config: dict,
    chunk_stay_ids_set: set,
    chunk_subject_ids_set: set,
    windows_df: pd.DataFrame,
    intime_map: dict,
) -> pd.DataFrame:
    """Process a single observation source table."""
    join_col = source_config["join_col"]
    timestamp_col = source_config["timestamp_col"]
    category = source_config["category"]
    source_dotpath = source_config["source"]

    if join_col == "stay_id":
        cohort_ids = chunk_stay_ids_set
    elif join_col == "subject_id":
        cohort_ids = chunk_subject_ids_set
    else:
        raise ValueError(f"Unknown join_col: {join_col} for source {source_name}")

    logger.info(
        "Processing %s (join=%s, category=%s)", source_name, join_col, category
    )

    df = load_and_filter_concepts_table(
        paths_config, source_dotpath, join_col, cohort_ids
    )

    if len(df) == 0:
        logger.warning("  No data found for %s", source_name)
        return empty_events_df()

    if join_col == "subject_id":
        df = assign_stay_ids_from_subject(df, windows_df, timestamp_col)
        if len(df) == 0:
            logger.warning("  No rows matched ICU windows for %s", source_name)
            return empty_events_df()
        logger.info("  After ICU window matching: %d rows", len(df))
    else:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = clip_to_icu_window(df, windows_df, timestamp_col)
        if len(df) == 0:
            logger.warning("  No rows within ICU window for %s", source_name)
            return empty_events_df()

    row_filter = source_config.get("row_filter")
    if row_filter:
        allowed = set(row_filter["values"])
        before = len(df)
        df = df[df[row_filter["column"]].isin(allowed)]
        logger.info(
            "  Row filter %s in %s: %d → %d rows",
            row_filter["column"], allowed, before, len(df),
        )
        if len(df) == 0:
            return empty_events_df()

    config_value_cols = source_config["value_cols"]
    numeric_cols, categorical_cols = resolve_value_columns(df, config_value_cols)
    logger.info(
        "  Value columns: %d numeric, %d categorical",
        len(numeric_cols), len(categorical_cols),
    )

    events = melt_to_long(
        df, source_name, category, numeric_cols, categorical_cols,
        timestamp_col, intime_map,
    )

    logger.info("  Produced %d event rows", len(events))

    del df
    gc.collect()

    return events


def process_diagnoses(
    source_config: dict,
    paths_config: dict,
    cohort: pd.DataFrame,
    intime_map: dict,
) -> pd.DataFrame:
    """Admission-level ICD codes with their titles, as observation rows at hour 0."""
    stay_of_admission = dict(zip(cohort["hadm_id"], cohort["stay_id"]))
    codes = load_and_filter_raw_table(
        paths_config, source_config["source"], "hadm_id", set(stay_of_admission),
        usecols=["hadm_id", "icd_code", "icd_version"],
    )
    if len(codes) == 0:
        return empty_events_df()

    titles = load_and_filter_raw_table(
        paths_config, source_config["titles"], "icd_code", set(codes["icd_code"]),
        usecols=["icd_code", "icd_version", "long_title"],
    )
    codes["icd_code"] = codes["icd_code"].astype(str)
    titles["icd_code"] = titles["icd_code"].astype(str)
    codes = codes.merge(titles, on=["icd_code", "icd_version"], how="left")

    stay_id = codes["hadm_id"].map(stay_of_admission).astype(int)
    events = pd.DataFrame({
        "stay_id": stay_id,
        "timestamp": stay_id.map(intime_map),
        "timestamp_hours": 0.0,
        "domain": "observation",
        "source_table": "diagnoses",
        "category": source_config["category"],
        "variable_name": codes["icd_code"],
        "numeric_value": np.nan,
        "text_value": codes["long_title"].fillna(""),
        "unit": "",
        "end_timestamp": pd.NaT,
        "end_timestamp_hours": np.nan,
        "is_continuous": False,
    })
    logger.info("  Produced %d diagnosis rows", len(events))
    return events[OUTPUT_COLUMNS]


def main() -> None:
    args = extraction_args(__doc__)
    setup_logging()
    paths_config = load_config(args.paths_config)
    features_config = load_config(args.features_config)

    stay_ids, cohort = load_chunk_cohort(paths_config, args.chunk_idx, args.total_chunks)
    stay_id_set = set(stay_ids.tolist())
    subject_id_set = set(cohort["subject_id"].tolist())
    intime_map = build_intime_map(cohort)
    windows_df = build_icu_windows(cohort)

    output_dir = ensure_dir(resolve_table_path(paths_config, "output.observations"))

    all_events: dict[int, list[pd.DataFrame]] = {}
    for source_name, source_config in features_config["observations"].items():
        events = process_observation_source(
            source_name, source_config, paths_config,
            stay_id_set, subject_id_set, windows_df, intime_map,
        )
        add_events(all_events, events)
        del events
        gc.collect()

    add_events(all_events, process_diagnoses(features_config["diagnoses"], paths_config, cohort, intime_map))
    write_stay_events(all_events, stay_ids, output_dir)


if __name__ == "__main__":
    main()
