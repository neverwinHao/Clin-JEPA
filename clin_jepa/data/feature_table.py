"""Write the source-table and feature inventory (paper, Appendix A) from ``configs/data/features.yaml``."""

import argparse

from clin_jepa.utils import ensure_dir, expand_path, load_config

ALL_VALUE_COLS: dict[str, list[str]] = {
    "enzyme": ["alt", "ast", "alp", "ggt", "ld_ldh", "amylase", "ck_cpk", "ck_mb",
               "bilirubin_total", "bilirubin_direct", "bilirubin_indirect"],
    "inflammation": ["crp"],
    "cardiac_marker": ["troponin_t", "ck_mb", "ntprobnp"],
    "gcs": ["gcs", "gcs_motor", "gcs_verbal", "gcs_eyes", "gcs_unable"],
    "ventilator_setting": ["fio2", "peep", "plateau_pressure", "minute_volume",
                           "flow_rate", "tidal_volume_set", "tidal_volume_observed",
                           "tidal_volume_spontaneous", "respiratory_rate_set",
                           "respiratory_rate_total", "respiratory_rate_spontaneous",
                           "ventilator_mode", "ventilator_mode_hamilton",
                           "ventilator_type"],
    "crrt": ["crrt_mode", "system_active", "blood_flow", "dialysate_rate",
             "dialysate_fluid", "replacement_rate", "replacement_fluid",
             "prefilter_replacement_rate", "postfilter_replacement_rate",
             "ultrafiltrate_output", "hourly_patient_fluid_removal", "current_goal",
             "citrate", "heparin_dose", "heparin_concentration", "access_pressure",
             "filter_pressure", "effluent_pressure", "return_pressure",
             "clots", "clotted", "clots_increasing"],
    "invasive_line": ["line_type", "line_site"],
}

RAW_TABLE_VARS: dict[str, str] = {
    "inputevents": ("IV fluids, infusions, and blood products (crystalloids, "
                    "colloids, albumin, PRBC, FFP, platelets) plus "
                    "continuously-infused drugs (vasoactives, sedatives, "
                    "analgesics, insulin); 260+ distinct items"),
    "procedureevents": ("ICU procedures and device-duration events (arterial and "
                        "central lines, tubes, imaging, bronchoscopy, "
                        "extubation); 150+ distinct items"),
    "prescriptions": ("Inpatient medication orders across all routes (PO, IV, SC, "
                      "PR); 1500+ distinct named drugs and fluids"),
    "emar": ("Electronic medication administration records (drug administered, "
             "route, dose); 900+ distinct items"),
}


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def allowbreaks(s: str) -> str:
    return s.replace(".", r".\allowbreak ").replace(r"\_", r"\_\allowbreak ")


def variables_cell(domain: str, info: dict) -> str:
    value_cols = info.get("value_cols")
    if value_cols is None:
        return RAW_TABLE_VARS[domain]
    if value_cols == "all":
        value_cols = ALL_VALUE_COLS[domain]
    return ", ".join(latex_escape(v) for v in value_cols)


def xltabular(caption: str, label: str, sources: dict, font_size: str) -> list[str]:
    header = r"\textbf{Domain} & \textbf{Source table} & \textbf{Category} & \textbf{Variables} \\"
    lines = [
        "{" + font_size,
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{xltabular}{\linewidth}{@{}"
        r">{\raggedright\arraybackslash}p{0.16\linewidth} "
        r">{\raggedright\arraybackslash}p{0.25\linewidth} "
        r">{\raggedright\arraybackslash}p{0.14\linewidth} "
        r">{\raggedright\arraybackslash}X@{}}",
        rf"\caption{{{caption}}} \label{{{label}}} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead", "",
        r"\multicolumn{4}{c}{\textit{(\tablename~\thetable\ continued)}} \\",
        r"\toprule", header, r"\midrule", r"\endhead", "",
        r"\midrule",
        r"\multicolumn{4}{r}{\textit{Continued on next page\ldots}} \\",
        r"\endfoot", "",
        r"\bottomrule", r"\endlastfoot", "",
    ]
    for domain, info in sources.items():
        cells = [latex_escape(domain), latex_escape(info.get("source", "—")), latex_escape(info.get("category", "—"))]
        lines.append(
            f"{allowbreaks(cells[0])} & \\texttt{{{allowbreaks(cells[1])}}} & "
            f"{allowbreaks(cells[2])} & {allowbreaks(variables_cell(domain, info))} \\\\"
        )
    return lines + [r"\end{xltabular}", "}", ""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_config", default="configs/data/features.yaml")
    parser.add_argument("--output_dir", default="${CLIN_JEPA_OUTPUT}/tables")
    args = parser.parse_args()

    features = load_config(args.features_config)
    lines = xltabular(
        r"Observation features (patient state) extracted from MIMIC-IV. "
        r"Source tables prefixed \texttt{concepts.*} are official mimic-code "
        r"derived concepts; raw tables are from MIMIC-IV directly.",
        "tab:obs_features", features["observations"], r"\footnotesize",
    ) + xltabular(
        r"Action features (clinical interventions) extracted from MIMIC-IV. "
        r"Source tables prefixed \texttt{concepts.*} are official mimic-code "
        r"derived concepts; \texttt{inputevents}, \texttt{prescriptions}, etc.\ "
        r"are MIMIC-IV raw tables.",
        "tab:action_features", features["actions"], r"\scriptsize",
    )
    path = ensure_dir(expand_path(args.output_dir)) / "appendix_features_table.tex"
    path.write_text("\n".join(lines) + "\n")
    print(path)


if __name__ == "__main__":
    main()
