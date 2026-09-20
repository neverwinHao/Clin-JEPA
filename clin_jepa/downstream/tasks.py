"""The 34 downstream tasks, their benchmarks and horizons, and the named task sets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    benchmark: str
    horizon: int | None
    title: str

    @property
    def hourly(self) -> bool:
        return self.horizon is not None


BENCHMARKS = {"E1": "Acute Deterioration", "E2": "Treatment Response", "E3": "Stay-Level Prognosis"}
CONTEXT = 24
ROLLOUT_HORIZON = 48

TASKS = [
    Task("circ_failure_8h", "E1", 8, "Circulatory failure, 8 h"),
    Task("resp_failure_24h", "E1", 24, "Respiratory failure, 24 h"),
    Task("kidney_failure_48h", "E1", 48, "Kidney failure, 48 h"),
    Task("liver_failure_48h", "E1", 48, "Liver failure, 48 h"),
    Task("hyperglycemia_8h", "E1", 8, "Hyperglycemia, 8 h"),
    Task("sepsis3_8h", "E1", 8, "Sepsis-3 onset, 8 h"),
    Task("decompensation_24h", "E1", 24, "Decompensation (ICU death), 24 h"),
    Task("composite_deterioration_24h", "E1", 24, "Composite deterioration, 24 h"),
    Task("vaso_onset_8h", "E2", 8, "Vasopressor onset, 8 h"),
    Task("vaso_onset_24h", "E2", 24, "Vasopressor onset, 24 h"),
    Task("vaso_escalation_24h", "E2", 24, "Vasopressor escalation, 24 h"),
    Task("vaso_liberation_24h", "E2", 24, "Vasopressor liberation, 24 h"),
    Task("imv_onset_24h", "E2", 24, "Invasive ventilation onset, 24 h"),
    Task("imv_liberation_48h", "E2", 48, "Ventilator liberation, 48 h"),
    Task("rrt_onset_48h", "E2", 48, "Renal replacement onset, 48 h"),
    Task("sofa_worsen_2_48h", "E2", 48, r"SOFA increase $\geq 2$, 48 h"),
    Task("sofa_improve_2_48h", "E2", 48, r"SOFA decrease $\geq 2$, 48 h"),
    Task("transfusion_24h", "E2", 24, "Transfusion, 24 h"),
    Task("prolonged_imv_96h", "E2", None, r"Prolonged ventilation $\geq$ 96 h"),
    Task("tracheostomy_ever", "E2", None, "Tracheostomy"),
    Task("extubation_failure_48h", "E2", None, "Extubation failure, 48 h"),
    Task("new_rrt_after_24h", "E2", None, "New renal replacement"),
    Task("aki_kdigo23_after_24h", "E2", None, "AKI, KDIGO stage 2--3"),
    Task("discharge_home", "E2", None, "Discharge to home"),
    Task("los_gt_14d", "E2", None, r"ICU stay $>$ 14 d"),
    Task("vasopressor_free_7d", "E2", None, "Vasopressor-free, days 2--8"),
    Task("hospital_mortality", "E3", None, "In-hospital mortality"),
    Task("icu_mortality", "E3", None, "ICU mortality"),
    Task("mortality_7d", "E3", None, "Mortality, 7 d"),
    Task("mortality_14d", "E3", None, "Mortality, 14 d"),
    Task("mortality_30d", "E3", None, "Mortality, 30 d"),
    Task("mortality_90d", "E3", None, "Mortality, 90 d"),
    Task("los_gt_7d", "E3", None, r"ICU stay $>$ 7 d"),
    Task("sepsis_ever", "E3", None, "Sepsis during the stay"),
]
BY_NAME = {task.name: task for task in TASKS}

TASK_SETS = {
    "acute_hourly": [t.name for t in TASKS if t.benchmark == "E1"],
    "treatment_hourly": [t.name for t in TASKS if t.benchmark == "E2" and t.hourly],
    "prognosis": ["hospital_mortality", "death_in_icu", "mortality_7d", "mortality_14d", "mortality_30d",
                  "mortality_90d", "los_gt_7d", "sepsis_onset_after_24h", "hospital_readmission_30d",
                  "hospital_readmission_60d", "icu_readmission_same_hosp"],
    "treatment_stay": ["sepsis_ever"] + [t.name for t in TASKS if t.benchmark == "E2" and not t.hourly],
    "icu_mortality": ["icu_mortality"],
}


def benchmark_tasks(benchmark: str) -> list[Task]:
    return [task for task in TASKS if task.benchmark == benchmark]


def feature_horizon(task: Task) -> int:
    """Rollout horizon whose pooled block a read-out of this task uses."""
    return task.horizon if task.hourly else ROLLOUT_HORIZON
