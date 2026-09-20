import numpy as np
import pandas as pd

from clin_jepa.data.helpers.observation_filter import filter_obs_events
from clin_jepa.data.step01_cohort import _flag
from clin_jepa.data.step02_observations import process_diagnoses
from clin_jepa.data.step06_trajectories import NO_OBSERVATIONS, hourly_sofa, hourly_texts, slice_windows

WINDOWS = {"t_max": 72, "min_len": 6, "stride": 12}


def test_window_slicing():
    assert slice_windows(5, WINDOWS) == []
    assert slice_windows(40, WINDOWS) == [(0, 40)]
    assert slice_windows(72, WINDOWS) == [(0, 72)]
    assert slice_windows(100, WINDOWS) == [(0, 72), (12, 72), (24, 72)]


def test_hour_without_events_gets_the_placeholder_text():
    events = pd.DataFrame(columns=["step_idx", "source_table", "variable_name", "numeric_value", "text_value", "unit"])
    assert hourly_texts(events, 2, NO_OBSERVATIONS) == [f"t=0h | {NO_OBSERVATIONS}", f"t=1h | {NO_OBSERVATIONS}"]


def test_hourly_sofa_keeps_the_first_value_of_an_hour():
    events = pd.DataFrame({
        "step_idx": [0, 0, 2, 2], "source_table": ["sofa", "sofa", "sofa", "vitalsign"],
        "variable_name": ["sofa_24hours", "sofa_24hours", "sofa_24hours", "heart_rate"],
        "numeric_value": [3.0, 5.0, 7.0, 90.0],
    })
    sofa = hourly_sofa(events, 4, {"variable_name": "sofa_24hours", "source_table": "sofa"})
    assert sofa.dtype == np.float32
    np.testing.assert_array_equal(sofa, np.array([3.0, np.nan, 7.0, np.nan], dtype=np.float32))


def test_diagnoses_become_rows_at_hour_zero(tmp_path):
    codes, titles = tmp_path / "diagnoses_icd.csv.gz", tmp_path / "d_icd_diagnoses.csv.gz"
    pd.DataFrame({"subject_id": [1, 1, 2], "hadm_id": [10, 10, 20], "seq_num": [1, 2, 1],
                  "icd_code": ["I10", "V1582", "I10"], "icd_version": [10, 9, 10]}).to_csv(codes, index=False)
    long_titles = ["Essential (primary) hypertension", "Personal history of tobacco use"]
    pd.DataFrame({"icd_code": ["I10", "V1582"], "icd_version": [10, 9],
                  "long_title": long_titles}).to_csv(titles, index=False)
    paths = {"hosp": {"diagnoses_icd": str(codes), "d_icd_diagnoses": str(titles)}}
    source = {"source": "hosp.diagnoses_icd", "titles": "hosp.d_icd_diagnoses", "category": "diagnosis"}
    admitted = pd.Timestamp("2150-01-01 08:00")
    cohort = pd.DataFrame({"stay_id": [100], "hadm_id": [10]})

    events = process_diagnoses(source, paths, cohort, {100: admitted})

    assert events["stay_id"].tolist() == [100, 100]
    assert events["variable_name"].tolist() == ["I10", "V1582"]
    assert events["text_value"].tolist() == long_titles
    assert (events["timestamp_hours"] == 0).all() and (events["timestamp"] == admitted).all()
    assert (events["category"] == "diagnosis").all()


def test_state_text_filter_drops_categories_and_variables():
    events = pd.DataFrame({"category": ["vitals", "diagnosis", "kidney"],
                           "variable_name": ["heart_rate", "I10", "aki_stage_uo"]})
    kept = filter_obs_events(events, {"exclude_categories": ["diagnosis"], "exclude_variables": ["aki_stage_uo"]})
    assert kept["variable_name"].tolist() == ["heart_rate"]


def test_concept_table_flags_in_either_export_format():
    column = pd.Series(["t", "f", "true", "False", True, False, None])
    assert _flag(column).tolist() == [True, False, True, False, True, False, False]
