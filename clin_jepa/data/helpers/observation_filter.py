"""Drop the observation rows that the state text leaves out."""

import pandas as pd


def filter_obs_events(df: pd.DataFrame, state_text: dict) -> pd.DataFrame:
    """Rows outside the excluded categories and variables of ``state_text``."""
    excluded = df["category"].isin(state_text["exclude_categories"]) \
        | df["variable_name"].isin(state_text["exclude_variables"])
    return df.loc[~excluded]
