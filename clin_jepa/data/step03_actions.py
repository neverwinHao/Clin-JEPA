"""Step 03: extract intervention events from concept tables and raw MIMIC-IV tables, one parquet per stay."""

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
    assign_stay_ids_from_hadm,
    build_hadm_windows,
    build_icu_windows,
    build_intime_map,
    build_outtime_map,
    clip_to_icu_window,
    compute_hours_since_admission,
)
from clin_jepa.data.helpers.units import get_unit
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


VALID_EVENT_STATUS = {"FinishedRunning", "Paused", "Stopped"}

VALID_EMAR_EVENTS = {"Administered", "Applied"}


def melt_to_long_action(
    df: pd.DataFrame,
    source_name: str,
    category: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    timestamp_col: str,
    intime_map: dict,
    outtime_map: dict,
    end_col: str | None,
    is_continuous: bool,
) -> pd.DataFrame:
    """Melt wide-format action table to long-format event rows."""
    frames = []

    has_end = end_col is not None and end_col in df.columns
    id_vars = ["stay_id", timestamp_col]
    if has_end:
        id_vars.append(end_col)

    if numeric_cols:
        melted = df.melt(
            id_vars=id_vars,
            value_vars=numeric_cols,
            var_name="variable_name",
            value_name="numeric_value",
        )
        melted = melted.dropna(subset=["numeric_value"])
        melted["text_value"] = ""
        frames.append(melted)

    for col in categorical_cols:
        keep_cols = ["stay_id", timestamp_col]
        if has_end:
            keep_cols.append(end_col)
        keep_cols.append(col)
        subset = df[keep_cols].copy()
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
        result["timestamp"], result["stay_id"], intime_map,
    )

    if has_end:
        result["end_timestamp"] = pd.to_datetime(result[end_col])
        outtimes = result["stay_id"].map(outtime_map)
        exceed_mask = result["end_timestamp"] > outtimes
        if exceed_mask.any():
            logger.info(
                "  Clamped %d end_timestamps to icu_outtime", exceed_mask.sum(),
            )
            result.loc[exceed_mask, "end_timestamp"] = outtimes[exceed_mask]
        result["end_timestamp_hours"] = compute_hours_since_admission(
            result["end_timestamp"], result["stay_id"], intime_map,
        )
        if end_col != "end_timestamp":
            result = result.drop(columns=[end_col])
    else:
        result["end_timestamp"] = pd.NaT
        result["end_timestamp_hours"] = np.nan

    result["domain"] = "action"
    result["source_table"] = source_name
    result["category"] = category
    result["is_continuous"] = is_continuous

    result["unit"] = result["variable_name"].map(
        lambda vn: get_unit(source_name, vn)
    )

    return result[OUTPUT_COLUMNS]


def process_concepts_action_source(
    source_name: str,
    source_config: dict,
    paths_config: dict,
    chunk_stay_ids_set: set,
    windows_df: pd.DataFrame,
    intime_map: dict,
    outtime_map: dict,
) -> pd.DataFrame:
    """Process a single concepts-table action source."""
    join_col = source_config["join_col"]
    timestamp_col = source_config["timestamp_col"]
    category = source_config["category"]
    source_dotpath = source_config["source"]
    end_col = source_config.get("end_col")
    is_continuous = source_config.get("is_continuous", False)

    logger.info(
        "Processing %s (join=%s, category=%s, end_col=%s, continuous=%s)",
        source_name, join_col, category, end_col, is_continuous,
    )

    df = load_and_filter_concepts_table(
        paths_config, source_dotpath, join_col, chunk_stay_ids_set,
    )

    if len(df) == 0:
        logger.warning("  No data found for %s", source_name)
        return empty_events_df()

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    if end_col and end_col in df.columns:
        df[end_col] = pd.to_datetime(df[end_col])
    df = clip_to_icu_window(df, windows_df, timestamp_col)
    if len(df) == 0:
        logger.warning("  No rows within ICU window for %s", source_name)
        return empty_events_df()

    config_value_cols = source_config.get("value_cols", "all")
    numeric_cols, categorical_cols = resolve_value_columns(df, config_value_cols)
    logger.info(
        "  Value columns: %d numeric, %d categorical",
        len(numeric_cols), len(categorical_cols),
    )

    events = melt_to_long_action(
        df, source_name, category, numeric_cols, categorical_cols,
        timestamp_col, intime_map, outtime_map, end_col, is_continuous,
    )

    logger.info("  Produced %d event rows", len(events))

    del df
    gc.collect()

    return events


def load_d_items(paths_config: dict) -> dict[int, str]:
    """Load d_items.csv.gz and return itemid -> label mapping."""
    df = pd.read_csv(resolve_table_path(paths_config, "icu.d_items"), usecols=["itemid", "label"])
    mapping = dict(zip(df["itemid"].astype(int), df["label"]))
    logger.info("Loaded d_items: %d item mappings", len(mapping))
    return mapping


def process_inputevents(
    paths_config: dict,
    chunk_stay_ids_set: set,
    d_items_map: dict[int, str],
    windows_df: pd.DataFrame,
    intime_map: dict,
    outtime_map: dict,
) -> pd.DataFrame:
    """Process inputevents (IV fluids, medications, blood products)."""
    logger.info("Processing inputevents (raw, stay_id, continuous)")

    usecols = [
        "stay_id", "starttime", "endtime", "itemid",
        "amount", "amountuom", "rate", "rateuom",
        "ordercategorydescription", "statusdescription",
    ]

    df = load_and_filter_raw_table(
        paths_config, "icu.inputevents", "stay_id",
        chunk_stay_ids_set, usecols=usecols,
    )

    if len(df) == 0:
        logger.warning("  No inputevents data found")
        return empty_events_df()

    n_before = len(df)
    df = df[df["statusdescription"].isin(VALID_EVENT_STATUS)].reset_index(drop=True)
    logger.info(
        "  Status filter: %d -> %d rows (dropped %d)",
        n_before, len(df), n_before - len(df),
    )

    if len(df) == 0:
        return empty_events_df()

    df["starttime"] = pd.to_datetime(df["starttime"])
    df["endtime"] = pd.to_datetime(df["endtime"])
    df = clip_to_icu_window(df, windows_df, "starttime")

    if len(df) == 0:
        return empty_events_df()

    df["variable_name"] = df["itemid"].map(d_items_map)
    unmapped = df["variable_name"].isna()
    if unmapped.any():
        logger.warning("  %d rows with unmapped itemid", unmapped.sum())
        df.loc[unmapped, "variable_name"] = (
            "unknown_item_" + df.loc[unmapped, "itemid"].astype(str)
        )

    df["timestamp"] = df["starttime"]
    df["timestamp_hours"] = compute_hours_since_admission(
        df["timestamp"], df["stay_id"], intime_map,
    )

    df["end_timestamp"] = df["endtime"]
    outtimes = df["stay_id"].map(outtime_map)
    exceed_mask = df["end_timestamp"] > outtimes
    if exceed_mask.any():
        logger.info("  Clamped %d end_timestamps to icu_outtime", exceed_mask.sum())
        df.loc[exceed_mask, "end_timestamp"] = outtimes[exceed_mask]
    df["end_timestamp_hours"] = compute_hours_since_admission(
        df["end_timestamp"], df["stay_id"], intime_map,
    )

    df["numeric_value"] = pd.to_numeric(df["amount"], errors="coerce")

    rate_str = df["rate"].astype(str).replace("nan", "")
    rateuom_str = df["rateuom"].fillna("")
    cat_str = df["ordercategorydescription"].fillna("")
    df["text_value"] = np.where(
        rate_str != "",
        "rate:" + rate_str + " " + rateuom_str + " | " + cat_str,
        cat_str,
    )
    df["text_value"] = df["text_value"].str.strip(" |")

    df["unit"] = df["amountuom"].fillna("")
    df["domain"] = "action"
    df["source_table"] = "inputevents"
    df["category"] = "fluid_med_input"
    df["is_continuous"] = True

    result = df[OUTPUT_COLUMNS].copy()
    logger.info("  Produced %d event rows", len(result))

    del df
    gc.collect()

    return result


def process_procedureevents(
    paths_config: dict,
    chunk_stay_ids_set: set,
    d_items_map: dict[int, str],
    windows_df: pd.DataFrame,
    intime_map: dict,
    outtime_map: dict,
) -> pd.DataFrame:
    """Process procedureevents (ICU procedures)."""
    logger.info("Processing procedureevents (raw, stay_id, continuous)")

    usecols = [
        "stay_id", "starttime", "endtime", "itemid",
        "value", "valueuom", "ordercategorydescription", "statusdescription",
    ]

    df = load_and_filter_raw_table(
        paths_config, "icu.procedureevents", "stay_id",
        chunk_stay_ids_set, usecols=usecols,
    )

    if len(df) == 0:
        logger.warning("  No procedureevents data found")
        return empty_events_df()

    n_before = len(df)
    df = df[df["statusdescription"].isin(VALID_EVENT_STATUS)].reset_index(drop=True)
    logger.info(
        "  Status filter: %d -> %d rows (dropped %d)",
        n_before, len(df), n_before - len(df),
    )

    if len(df) == 0:
        return empty_events_df()

    df["starttime"] = pd.to_datetime(df["starttime"])
    df["endtime"] = pd.to_datetime(df["endtime"])
    df = clip_to_icu_window(df, windows_df, "starttime")

    if len(df) == 0:
        return empty_events_df()

    df["variable_name"] = df["itemid"].map(d_items_map)
    unmapped = df["variable_name"].isna()
    if unmapped.any():
        logger.warning("  %d rows with unmapped itemid", unmapped.sum())
        df.loc[unmapped, "variable_name"] = (
            "unknown_item_" + df.loc[unmapped, "itemid"].astype(str)
        )

    df["timestamp"] = df["starttime"]
    df["timestamp_hours"] = compute_hours_since_admission(
        df["timestamp"], df["stay_id"], intime_map,
    )

    df["end_timestamp"] = df["endtime"]
    outtimes = df["stay_id"].map(outtime_map)
    exceed_mask = df["end_timestamp"] > outtimes
    if exceed_mask.any():
        logger.info("  Clamped %d end_timestamps to icu_outtime", exceed_mask.sum())
        df.loc[exceed_mask, "end_timestamp"] = outtimes[exceed_mask]
    df["end_timestamp_hours"] = compute_hours_since_admission(
        df["end_timestamp"], df["stay_id"], intime_map,
    )

    df["numeric_value"] = pd.to_numeric(df["value"], errors="coerce")
    df["text_value"] = df["ordercategorydescription"].fillna("")
    df["unit"] = df["valueuom"].fillna("")
    df["domain"] = "action"
    df["source_table"] = "procedureevents"
    df["category"] = "procedure"
    df["is_continuous"] = True

    result = df[OUTPUT_COLUMNS].copy()
    logger.info("  Produced %d event rows", len(result))

    del df
    gc.collect()

    return result


def process_prescriptions(
    paths_config: dict,
    chunk_hadm_ids_set: set,
    hadm_windows_df: pd.DataFrame,
    intime_map: dict,
    outtime_map: dict,
) -> pd.DataFrame:
    """Process prescriptions (medication orders, hospital level)."""
    logger.info("Processing prescriptions (raw, hadm_id, continuous)")

    usecols = [
        "hadm_id", "starttime", "stoptime",
        "drug", "prod_strength", "dose_val_rx", "dose_unit_rx", "route",
    ]

    df = load_and_filter_raw_table(
        paths_config, "hosp.prescriptions", "hadm_id",
        chunk_hadm_ids_set, usecols=usecols,
    )

    if len(df) == 0:
        logger.warning("  No prescriptions data found")
        return empty_events_df()

    df = assign_stay_ids_from_hadm(df, hadm_windows_df, "starttime")

    if len(df) == 0:
        logger.warning("  No prescriptions rows matched ICU windows")
        return empty_events_df()

    df["stoptime"] = pd.to_datetime(df["stoptime"])
    bad_order = df["starttime"] > df["stoptime"]
    if bad_order.any():
        logger.info(
            "  Dropped %d rows with starttime > stoptime", bad_order.sum(),
        )
        df = df[~bad_order].reset_index(drop=True)

    if len(df) == 0:
        return empty_events_df()

    df["variable_name"] = df["drug"].fillna("unknown_drug")
    df["timestamp"] = df["starttime"]
    df["timestamp_hours"] = compute_hours_since_admission(
        df["timestamp"], df["stay_id"], intime_map,
    )

    df["end_timestamp"] = df["stoptime"]
    outtimes = df["stay_id"].map(outtime_map)
    exceed_mask = df["end_timestamp"].notna() & (df["end_timestamp"] > outtimes)
    if exceed_mask.any():
        logger.info("  Clamped %d end_timestamps to icu_outtime", exceed_mask.sum())
        df.loc[exceed_mask, "end_timestamp"] = outtimes[exceed_mask]
    df["end_timestamp_hours"] = compute_hours_since_admission(
        df["end_timestamp"], df["stay_id"], intime_map,
    )

    df["numeric_value"] = pd.to_numeric(df["dose_val_rx"], errors="coerce")

    route_str = df["route"].fillna("")
    strength_str = df["prod_strength"].fillna("")
    df["text_value"] = np.where(
        route_str != "",
        "route:" + route_str + " | " + strength_str,
        strength_str,
    )
    df["text_value"] = df["text_value"].str.strip(" |")

    df["unit"] = df["dose_unit_rx"].fillna("")
    df["domain"] = "action"
    df["source_table"] = "prescriptions"
    df["category"] = "medication_order"
    df["is_continuous"] = True

    result = df[OUTPUT_COLUMNS].copy()
    logger.info("  Produced %d event rows", len(result))

    del df
    gc.collect()

    return result


def process_emar(
    paths_config: dict,
    chunk_hadm_ids_set: set,
    hadm_windows_df: pd.DataFrame,
    intime_map: dict,
) -> pd.DataFrame:
    """Process emar (medication administration records)."""
    logger.info("Processing emar (raw, hadm_id, point events)")

    usecols = ["hadm_id", "charttime", "medication", "event_txt"]

    df = load_and_filter_raw_table(
        paths_config, "hosp.emar", "hadm_id",
        chunk_hadm_ids_set, usecols=usecols,
    )

    if len(df) == 0:
        logger.warning("  No emar data found")
        return empty_events_df()

    n_before = len(df)
    df = df[df["event_txt"].isin(VALID_EMAR_EVENTS)].reset_index(drop=True)
    logger.info(
        "  Event filter (Administered/Applied): %d -> %d rows (dropped %d)",
        n_before, len(df), n_before - len(df),
    )

    if len(df) == 0:
        return empty_events_df()

    df = assign_stay_ids_from_hadm(df, hadm_windows_df, "charttime")

    if len(df) == 0:
        logger.warning("  No emar rows matched ICU windows")
        return empty_events_df()

    df["variable_name"] = df["medication"].fillna("unknown_medication")
    df["timestamp"] = df["charttime"]
    df["timestamp_hours"] = compute_hours_since_admission(
        df["timestamp"], df["stay_id"], intime_map,
    )

    df["numeric_value"] = np.nan
    df["text_value"] = df["event_txt"].fillna("")
    df["unit"] = ""
    df["end_timestamp"] = pd.NaT
    df["end_timestamp_hours"] = np.nan
    df["domain"] = "action"
    df["source_table"] = "emar"
    df["category"] = "medication_admin"
    df["is_continuous"] = False

    result = df[OUTPUT_COLUMNS].copy()
    logger.info("  Produced %d event rows", len(result))

    del df
    gc.collect()

    return result


def main() -> None:
    args = extraction_args(__doc__)
    setup_logging()
    paths_config = load_config(args.paths_config)
    features_config = load_config(args.features_config)

    stay_ids, cohort = load_chunk_cohort(paths_config, args.chunk_idx, args.total_chunks)
    stay_id_set = set(stay_ids.tolist())
    hadm_id_set = set(cohort["hadm_id"].tolist())
    intime_map = build_intime_map(cohort)
    outtime_map = build_outtime_map(cohort)
    windows_df = build_icu_windows(cohort)
    hadm_windows_df = build_hadm_windows(cohort)
    d_items_map = load_d_items(paths_config)

    output_dir = ensure_dir(resolve_table_path(paths_config, "output.actions"))

    all_events: dict[int, list[pd.DataFrame]] = {}
    for source_name, source_config in features_config["actions"].items():
        if not source_config.get("is_raw_table", False):
            events = process_concepts_action_source(
                source_name, source_config, paths_config,
                stay_id_set, windows_df, intime_map, outtime_map,
            )
        elif source_name == "inputevents":
            events = process_inputevents(
                paths_config, stay_id_set, d_items_map, windows_df, intime_map, outtime_map,
            )
        elif source_name == "procedureevents":
            events = process_procedureevents(
                paths_config, stay_id_set, d_items_map, windows_df, intime_map, outtime_map,
            )
        elif source_name == "prescriptions":
            events = process_prescriptions(
                paths_config, hadm_id_set, hadm_windows_df, intime_map, outtime_map,
            )
        elif source_name == "emar":
            events = process_emar(paths_config, hadm_id_set, hadm_windows_df, intime_map)
        else:
            raise ValueError(f"no handler for raw action source {source_name!r}")
        add_events(all_events, events)
        del events
        gc.collect()

    write_stay_events(all_events, stay_ids, output_dir)


if __name__ == "__main__":
    main()
