"""Drop the listed intervention rows and keep each distinct prescription row at its first hour."""

from __future__ import annotations

import pandas as pd

ACTION_SOURCE_TABLES_TO_DROP: set[str] = {
    "procedureevents",
    "invasive_line",
}


ACTION_VARIABLES_TO_DROP: set[tuple[str, str]] = {
    ("inputevents", "Solution"),
    ("inputevents", "Sterile Water"),
    ("inputevents", "Piggyback"),
    ("inputevents", "GT Flush"),
    ("prescriptions", "Bag"),
    ("prescriptions", "Sterile Water"),
    ("prescriptions", "SW"),
    ("prescriptions", "Iso-Osmotic Dextrose"),
    ("prescriptions", "Iso-Osmotic Sodium Chloride"),
    ("prescriptions", "Sodium Chloride 0.9%  Flush"),
    ("prescriptions", "Sodium Chloride 0.9%  Flush for CRRT"),
    ("prescriptions", "Heparin Flush (10 units/ml)"),
    ("prescriptions", "Heparin Flush (100 units/ml)"),
    ("prescriptions", "Heparin Flush (1000 units/mL)"),
    ("prescriptions", "Heparin Flush (5000 Units/mL)"),
    ("ventilator_setting", "ventilator_type"),
    ("emar", "unknown_medication"),
    ("emar", "Sodium Chloride 0.9%  Flush"),
    ("emar", "Heparin Flush (10 units/ml)"),
    ("prescriptions", "Syringe (Iso-Osmotic Dextrose)"),
    ("prescriptions", "Solution"),
    ("prescriptions", "Vial"),
    ("prescriptions", "Syringe"),
    ("prescriptions", "Bottle"),
    ("prescriptions", "Heparin Flush CVL  (100 units/ml)"),
    ("prescriptions", "Heparin Flush CRRT (5000 Units/mL)"),
    ("prescriptions", "Heparin Flush PICC (100 units/ml)"),
    ("prescriptions", "Heparin Flush Hickman (100 units/ml)"),
    ("prescriptions", "Heparin Flush Port (10 units/mL)"),
    ("prescriptions", "Heparin Flush"),
    ("prescriptions", "Heparin Flush "),
    ("prescriptions", "Heparin Flush (100 units/mL)"),
    ("prescriptions", "Heparin Flush (10 Units/mL)"),
    ("prescriptions", "Syringe (0.9% Sodium Chloride)"),
    ("prescriptions", "Syringe (SW)"),
    ("prescriptions", "Syringe (Sterile Water)"),
    ("prescriptions", "Syringe (Chemo)"),
    ("prescriptions", "Syringe (NS)"),
    ("prescriptions", "Syringe (Intraventricular)"),
    ("prescriptions", "Syringe (Intraperitoneal)"),
    ("prescriptions", "Syringe (Inhalation)"),
    ("prescriptions", "Syringe (Intrathecal)"),
    ("prescriptions", "Syringe (LR)"),
    ("prescriptions", "Syringe (PD)"),
    ("prescriptions", "Syringe (PCA)"),
    ("prescriptions", "Syringe (Intrapleural)"),
    ("prescriptions", "Syringe (0.9% Sodium Chloride"),
    ("prescriptions", "VIal (Sterile Water)"),
    ("prescriptions", "Sterile Water For Irrigation"),
    ("prescriptions", "Sterile Diluent for Flolan"),
    ("prescriptions", "Sterile Diluent for Remodulin"),
    ("prescriptions", "Sodium Chloride 0.45% Flush"),
}


_PRESCRIPTIONS_COLLAPSE_SUBSET = [
    "variable_name",
    "text_value",
    "numeric_value",
    "unit",
]


def filter_act_events(df: pd.DataFrame) -> pd.DataFrame:
    """Events without the listed rows, each distinct prescription row kept at its first hour."""
    if len(df) == 0:
        return df

    mask = ~df["source_table"].isin(ACTION_SOURCE_TABLES_TO_DROP)
    for source_table, variable_name in ACTION_VARIABLES_TO_DROP:
        mask &= ~(
            (df["source_table"] == source_table)
            & (df["variable_name"] == variable_name)
        )
    df = df.loc[mask].reset_index(drop=True)

    if len(df) == 0 or not (df["source_table"] == "prescriptions").any():
        return df

    presc_mask = df["source_table"] == "prescriptions"
    non_presc = df.loc[~presc_mask]
    presc = df.loc[presc_mask]

    presc_first = (
        presc.sort_values("step_idx", kind="stable")
             .drop_duplicates(
                 subset=_PRESCRIPTIONS_COLLAPSE_SUBSET,
                 keep="first",
             )
    )

    df = pd.concat([non_presc, presc_first], ignore_index=True)
    df = df.sort_values(
        ["step_idx", "source_table", "variable_name"],
        kind="stable",
    ).reset_index(drop=True)
    return df
