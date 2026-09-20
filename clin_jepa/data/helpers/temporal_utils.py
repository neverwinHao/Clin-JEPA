"""Hours since ICU admission, ICU-window clipping, and stay assignment for tables keyed by subject or admission."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_icu_windows(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """ICU admission and discharge times per stay, with the subject and stay identifiers."""
    windows = cohort_df[["subject_id", "stay_id", "icu_intime", "icu_outtime"]].copy()
    windows["icu_intime"] = pd.to_datetime(windows["icu_intime"])
    windows["icu_outtime"] = pd.to_datetime(windows["icu_outtime"])
    logger.info(
        "Built ICU windows: %d stays across %d subjects",
        len(windows), windows["subject_id"].nunique(),
    )
    return windows


def build_intime_map(cohort_df: pd.DataFrame) -> dict:
    """Build stay_id → icu_intime mapping."""
    df = cohort_df[["stay_id", "icu_intime"]].copy()
    df["icu_intime"] = pd.to_datetime(df["icu_intime"])
    return dict(zip(df["stay_id"], df["icu_intime"]))


def build_outtime_map(cohort_df: pd.DataFrame) -> dict:
    """Build stay_id → icu_outtime mapping."""
    df = cohort_df[["stay_id", "icu_outtime"]].copy()
    df["icu_outtime"] = pd.to_datetime(df["icu_outtime"])
    return dict(zip(df["stay_id"], df["icu_outtime"]))


def build_hadm_windows(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame for hadm_id → ICU window matching."""
    windows = cohort_df[["hadm_id", "stay_id", "icu_intime", "icu_outtime"]].copy()
    windows["icu_intime"] = pd.to_datetime(windows["icu_intime"])
    windows["icu_outtime"] = pd.to_datetime(windows["icu_outtime"])
    logger.info(
        "Built hadm windows: %d stays across %d admissions",
        len(windows), windows["hadm_id"].nunique(),
    )
    return windows


def compute_hours_since_admission(
    timestamps: pd.Series,
    stay_ids: pd.Series,
    intime_map: dict,
) -> pd.Series:
    """Convert timestamps to hours since ICU admission."""
    intimes = stay_ids.map(intime_map)
    return (timestamps - intimes).dt.total_seconds() / 3600.0


def clip_to_icu_window(
    df: pd.DataFrame,
    windows_df: pd.DataFrame,
    timestamp_col: str,
) -> pd.DataFrame:
    """Filter stay_id-joined table rows to ICU window timestamps."""
    if len(df) == 0:
        return df

    n_before = len(df)
    merged = df.merge(
        windows_df[["stay_id", "icu_intime", "icu_outtime"]],
        on="stay_id",
        how="inner",
    )
    mask = (
        (merged[timestamp_col] >= merged["icu_intime"])
        & (merged[timestamp_col] < merged["icu_outtime"])
    )
    result = merged.loc[mask].drop(columns=["icu_intime", "icu_outtime"])
    n_after = len(result)

    if n_after < n_before:
        logger.info(
            "  ICU window clip: %d → %d rows (dropped %d outside window)",
            n_before, n_after, n_before - n_after,
        )

    return result.reset_index(drop=True)


def assign_stay_ids_from_subject(
    df: pd.DataFrame,
    windows_df: pd.DataFrame,
    timestamp_col: str,
) -> pd.DataFrame:
    """Assign stay_id to rows from subject_id-joined tables."""
    if len(df) == 0:
        return df

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    merged = df.merge(windows_df, on="subject_id", how="inner")

    mask = (
        (merged[timestamp_col] >= merged["icu_intime"])
        & (merged[timestamp_col] < merged["icu_outtime"])
    )
    result = merged.loc[mask].drop(columns=["icu_intime", "icu_outtime"])

    return result.reset_index(drop=True)


def assign_stay_ids_from_hadm(
    df: pd.DataFrame,
    hadm_windows_df: pd.DataFrame,
    timestamp_col: str,
) -> pd.DataFrame:
    """Assign stay_id to rows from hadm_id-joined tables."""
    if len(df) == 0:
        return df

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    merged = df.merge(hadm_windows_df, on="hadm_id", how="inner")

    mask = (
        (merged[timestamp_col] >= merged["icu_intime"])
        & (merged[timestamp_col] < merged["icu_outtime"])
    )
    result = merged.loc[mask].drop(columns=["icu_intime", "icu_outtime"])
    n_dropped = len(df) - len(result)

    if n_dropped > 0:
        logger.info(
            "  hadm_id window matching: %d → %d rows (dropped %d outside ICU)",
            len(df), len(result), n_dropped,
        )

    return result.reset_index(drop=True)
