"""Table loading, cohort chunking and per-stay event writing for the data pipeline."""

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from clin_jepa.data.helpers.schema import validate_schema
from clin_jepa.utils import expand_path as resolve_path

logger = logging.getLogger(__name__)


def resolve_table_path(paths_config: dict, dotpath: str) -> Path:
    """Resolve a dot-notation path to an absolute file path."""
    node = paths_config
    for key in dotpath.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(
                f"Key '{key}' not found in config at path '{dotpath}'. "
                f"Available keys: {list(node.keys()) if isinstance(node, dict) else 'N/A'}"
            )
        node = node[key]
    if not isinstance(node, str):
        raise ValueError(
            f"Dotpath '{dotpath}' did not resolve to a file path string. "
            f"Got {type(node).__name__}: {node}"
        )
    return resolve_path(node)


def load_concepts_table(
    paths_config: dict,
    dotpath: str,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Load a concepts table (plain CSV) from MIMIC-IV."""
    filepath = resolve_table_path(paths_config, dotpath)
    if not filepath.exists():
        raise FileNotFoundError(f"Concepts table not found: {filepath}")

    logger.info("Loading concepts table: %s", filepath.name)
    df = pd.read_csv(filepath, usecols=usecols)
    logger.info("  Loaded %d rows x %d cols from %s", len(df), len(df.columns), filepath.name)
    return df


LARGE_TABLE_THRESHOLD = 300_000_000
DEFAULT_CHUNKSIZE = 1_000_000


def load_and_filter_concepts_table(
    paths_config: dict,
    dotpath: str,
    join_col: str,
    cohort_ids: set,
) -> pd.DataFrame:
    """Load a concepts table, filtering to cohort IDs during read."""
    filepath = resolve_table_path(paths_config, dotpath)
    if not filepath.exists():
        raise FileNotFoundError(f"Concepts table not found: {filepath}")

    file_size = filepath.stat().st_size

    if file_size > LARGE_TABLE_THRESHOLD:
        logger.info(
            "Loading %s in chunks (%.0f MB, filtering on %s)",
            filepath.name, file_size / 1e6, join_col,
        )
        frames = []
        n_total = 0
        for chunk in pd.read_csv(filepath, chunksize=DEFAULT_CHUNKSIZE, encoding_errors="replace"):
            n_total += len(chunk)
            filtered = chunk[chunk[join_col].isin(cohort_ids)]
            if len(filtered) > 0:
                frames.append(filtered)
            del chunk
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        logger.info(
            "  Loaded %d / %d rows from %s",
            len(result), n_total, filepath.name,
        )
        return result
    else:
        logger.info("Loading %s (%.0f MB)", filepath.name, file_size / 1e6)
        df = pd.read_csv(filepath, low_memory=False, encoding_errors="replace")
        result = df[df[join_col].isin(cohort_ids)]
        logger.info(
            "  Loaded %d / %d rows from %s",
            len(result), len(df), filepath.name,
        )
        del df
        gc.collect()
        return result.reset_index(drop=True)


RAW_TABLE_CHUNKSIZE = 500_000


def load_and_filter_raw_table(
    paths_config: dict,
    source_dotpath: str,
    join_col: str,
    cohort_ids: set,
    usecols: list[str],
) -> pd.DataFrame:
    """Load a raw MIMIC-IV table (.csv.gz), filtering to cohort during read."""
    filepath = resolve_table_path(paths_config, source_dotpath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw table not found: {filepath}")

    compression = "gzip" if filepath.suffix == ".gz" else None
    logger.info(
        "Loading raw table %s in chunks (compression=%s, join=%s)",
        filepath.name, compression, join_col,
    )

    frames = []
    n_total = 0
    for chunk in pd.read_csv(
        filepath, compression=compression, usecols=usecols,
        chunksize=RAW_TABLE_CHUNKSIZE, low_memory=False,
    ):
        n_total += len(chunk)
        filtered = chunk[chunk[join_col].isin(cohort_ids)]
        if len(filtered) > 0:
            frames.append(filtered)
        del chunk

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.info(
        "  Loaded %d / %d rows from %s",
        len(result), n_total, filepath.name,
    )
    del frames
    gc.collect()
    return result


def chunk_stay_ids(
    stay_ids: np.ndarray,
    chunk_idx: int,
    total_chunks: int,
) -> np.ndarray:
    """Split stay_ids into chunks for SLURM array parallelization."""
    if total_chunks < 1:
        raise ValueError(f"total_chunks must be >= 1, got {total_chunks}")
    if not 0 <= chunk_idx < total_chunks:
        raise ValueError(f"chunk_idx ({chunk_idx}) must be in [0, {total_chunks})")

    chunks = np.array_split(stay_ids, total_chunks)
    selected = chunks[chunk_idx]
    logger.info(
        "Chunk %d/%d: %d stay_ids (total: %d)",
        chunk_idx, total_chunks, len(selected), len(stay_ids),
    )
    return selected


EXCLUDE_COLS = {
    "subject_id", "hadm_id", "stay_id", "specimen_id", "specimen",
    "charttime", "starttime", "endtime", "stoptime", "hr",
    "icu_intime", "icu_outtime",
}


def resolve_value_columns(
    df: pd.DataFrame, config_value_cols: list[str] | str,
) -> tuple[list[str], list[str]]:
    """Resolve value column specification to (numeric_cols, categorical_cols)."""
    if config_value_cols == "all":
        candidates = [c for c in df.columns if c not in EXCLUDE_COLS]
    else:
        candidates = [c for c in config_value_cols if c in df.columns]

    numeric_cols = []
    categorical_cols = []
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    return numeric_cols, categorical_cols


def extraction_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--paths_config", default="configs/data/paths.yaml")
    parser.add_argument("--features_config", default="configs/data/features.yaml")
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--total_chunks", type=int, default=1)
    return parser.parse_args()


def load_chunk_cohort(
    paths_config: dict, chunk_idx: int, total_chunks: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return this chunk's sorted stay_ids and the cohort rows of those stays."""
    cohort = pd.read_csv(resolve_table_path(paths_config, "output.cohort") / "cohort.csv")
    cohort["icu_intime"] = pd.to_datetime(cohort["icu_intime"])
    cohort["icu_outtime"] = pd.to_datetime(cohort["icu_outtime"])
    stay_ids = chunk_stay_ids(np.sort(cohort["stay_id"].values), chunk_idx, total_chunks)
    return stay_ids, cohort[cohort["stay_id"].isin(set(stay_ids.tolist()))].copy()


def add_events(all_events: dict[int, list[pd.DataFrame]], events: pd.DataFrame) -> None:
    if len(events) > 0:
        for stay_id, group in events.groupby("stay_id"):
            all_events.setdefault(stay_id, []).append(group)


def write_stay_events(
    all_events: dict[int, list[pd.DataFrame]], stay_ids: np.ndarray, output_dir: Path,
) -> None:
    """Write one time-sorted parquet per stay that has events."""
    n_written = n_events = 0
    for stay_id in stay_ids:
        stay_id = int(stay_id)
        if stay_id not in all_events:
            continue
        stay_df = pd.concat(all_events[stay_id], ignore_index=True)
        stay_df = stay_df.sort_values("timestamp_hours").reset_index(drop=True)
        validate_schema(stay_df)
        stay_df.to_parquet(output_dir / f"{stay_id}.parquet", index=False, engine="pyarrow")
        n_written += 1
        n_events += len(stay_df)
    logger.info("wrote %d of %d stays, %d events, to %s", n_written, len(stay_ids), n_events, output_dir)
