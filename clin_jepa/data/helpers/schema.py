"""Column schema shared by the per-stay observation and action event tables."""

import pandas as pd

OUTPUT_COLUMNS = [
    "stay_id",
    "timestamp",
    "timestamp_hours",
    "domain",
    "source_table",
    "category",
    "variable_name",
    "numeric_value",
    "text_value",
    "unit",
    "end_timestamp",
    "end_timestamp_hours",
    "is_continuous",
]

OUTPUT_DTYPES = {
    "stay_id": "int64",
    "timestamp": "datetime64[ns]",
    "timestamp_hours": "float64",
    "domain": "object",
    "source_table": "object",
    "category": "object",
    "variable_name": "object",
    "numeric_value": "float64",
    "text_value": "object",
    "unit": "object",
    "end_timestamp": "datetime64[ns]",
    "end_timestamp_hours": "float64",
    "is_continuous": "bool",
}


def validate_schema(df: pd.DataFrame) -> None:
    """Raise if the columns are not exactly ``OUTPUT_COLUMNS``."""
    missing = set(OUTPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    extra = set(df.columns) - set(OUTPUT_COLUMNS)
    if extra:
        raise ValueError(f"Unexpected columns: {extra}")


def empty_events_df() -> pd.DataFrame:
    """Return an empty DataFrame with the correct output schema."""
    return pd.DataFrame({col: pd.Series(dtype=OUTPUT_DTYPES[col]) for col in OUTPUT_COLUMNS})
