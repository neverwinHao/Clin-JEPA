"""Step 04: merge each stay's observations and actions and bin them into hourly steps, one pickle per stay."""

import logging
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from clin_jepa.data.helpers.io_utils import chunk_stay_ids, extraction_args, resolve_table_path
from clin_jepa.data.helpers.schema import validate_schema
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

OBS_COLS = [
    "step_idx", "variable_name", "numeric_value",
    "text_value", "unit", "category", "source_table",
]
ACT_COLS = OBS_COLS + ["is_continuous"]


def build_stay_timeline(stay_id: int, obs_dir: Path, act_dir: Path) -> pd.DataFrame | None:
    """Concatenate a stay's observation and action events, de-duplicated and time-sorted."""
    frames = [
        pd.read_parquet(path)
        for path in (obs_dir / f"{stay_id}.parquet", act_dir / f"{stay_id}.parquet")
        if path.exists()
    ]
    if not frames:
        return None
    timeline = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    timeline = timeline.drop_duplicates(ignore_index=True)
    return timeline.sort_values(["timestamp_hours", "timestamp"], ignore_index=True)


def extract_metadata(cohort_row: pd.Series) -> dict:
    return {
        "age": float(cohort_row["admission_age"]),
        "gender": str(cohort_row["gender"]),
        "race": str(cohort_row["race"]),
        "icu_los_hours": float(cohort_row["icu_los_hours"]),
    }


def _bins(hours: np.ndarray, step_size: float, n_steps: int) -> np.ndarray:
    return np.clip(np.floor(np.round(hours, 6) / step_size).astype(int), 0, n_steps - 1)


def discretize_stay(
    timeline: pd.DataFrame,
    stay_id: int,
    metadata: dict,
    step_size: float,
    max_steps: int,
    min_obs: int,
) -> dict | None:
    """Bin a stay's events into hourly steps; continuous actions fill every hour they overlap."""
    obs_df = timeline[timeline["domain"] == "observation"]
    act_df = timeline[timeline["domain"] == "action"]
    if len(obs_df) < min_obs:
        return None

    n_steps = min(max_steps, max(1, math.ceil(metadata["icu_los_hours"] / step_size)))
    window_end = n_steps * step_size
    obs_df = obs_df[obs_df["timestamp_hours"] < window_end].copy()
    act_df = act_df[act_df["timestamp_hours"] < window_end].copy()

    if len(act_df) > 0:
        mask = act_df["end_timestamp_hours"].notna()
        act_df.loc[mask, "end_timestamp_hours"] = act_df.loc[
            mask, "end_timestamp_hours"
        ].clip(upper=window_end)

    if len(obs_df) > 0:
        obs_events = obs_df.assign(
            step_idx=_bins(obs_df["timestamp_hours"].values, step_size, n_steps)
        )[OBS_COLS].reset_index(drop=True)
    else:
        obs_events = pd.DataFrame(columns=OBS_COLS)

    point_acts = act_df[~act_df["is_continuous"]].copy()
    cont_acts = act_df[act_df["is_continuous"]].copy()
    act_frames = []

    if len(point_acts) > 0:
        point_acts = point_acts.assign(
            step_idx=_bins(point_acts["timestamp_hours"].values, step_size, n_steps)
        )
        act_frames.append(point_acts[ACT_COLS])

    if len(cont_acts) > 0:
        start_h = cont_acts["timestamp_hours"].values
        end_h = cont_acts["end_timestamp_hours"].values
        start_bins = np.clip(np.floor(start_h / step_size).astype(int), 0, n_steps - 1)
        end_bins = start_bins.copy()
        valid_end = ~np.isnan(end_h) & (end_h > start_h)
        end_bins[valid_end] = np.clip(
            np.ceil(end_h[valid_end] / step_size).astype(int) - 1,
            start_bins[valid_end], n_steps - 1,
        )
        spans = end_bins - start_bins + 1
        row_indices = np.repeat(np.arange(len(cont_acts)), spans)
        offsets = np.arange(int(spans.sum())) - np.repeat(
            np.concatenate([[0], np.cumsum(spans[:-1])]), spans,
        )
        keep_cols = ["variable_name", "numeric_value", "text_value",
                     "unit", "category", "source_table"]
        expanded = cont_acts.iloc[row_indices][keep_cols].reset_index(drop=True)
        expanded["step_idx"] = np.repeat(start_bins, spans) + offsets
        expanded["is_continuous"] = True
        act_frames.append(expanded)

    if act_frames:
        act_events = pd.concat(act_frames, ignore_index=True)[ACT_COLS]
    else:
        act_events = pd.DataFrame(columns=ACT_COLS)

    return {
        "stay_id": stay_id,
        "metadata": metadata,
        "n_steps": n_steps,
        "obs_events": obs_events,
        "act_events": act_events,
    }


def main() -> None:
    args = extraction_args(__doc__)
    setup_logging()
    paths_config = load_config(args.paths_config)
    temporal = load_config(args.features_config)["temporal"]
    step_size = float(temporal["step_size_hours"])
    max_steps = int(temporal["max_steps"])
    min_obs = int(temporal["min_obs_per_stay"])

    obs_dir = resolve_table_path(paths_config, "output.observations")
    act_dir = resolve_table_path(paths_config, "output.actions")
    output_dir = ensure_dir(resolve_table_path(paths_config, "output.discretized"))

    cohort = pd.read_csv(resolve_table_path(paths_config, "output.cohort") / "cohort.csv")
    cohort = cohort.set_index("stay_id")
    stay_ids = chunk_stay_ids(np.sort(cohort.index.values), args.chunk_idx, args.total_chunks)

    n_written = 0
    for stay_id in stay_ids:
        stay_id = int(stay_id)
        timeline = build_stay_timeline(stay_id, obs_dir, act_dir)
        if timeline is None:
            continue
        validate_schema(timeline)
        result = discretize_stay(
            timeline, stay_id, extract_metadata(cohort.loc[stay_id]),
            step_size, max_steps, min_obs,
        )
        if result is None:
            continue
        with open(output_dir / f"{stay_id}.pkl", "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        n_written += 1

    logger.info("wrote %d of %d stays to %s", n_written, len(stay_ids), output_dir)


if __name__ == "__main__":
    main()
