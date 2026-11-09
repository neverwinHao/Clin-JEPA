"""The optimized sepsis join must preserve mimic-code results, including time boundaries."""
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")


def test_suspicion_join_matches_upstream():
    code = os.environ.get("MIMIC_CODE_SOURCE")
    if not code:
        pytest.skip("Set MIMIC_CODE_SOURCE to a mimic-code checkout")
    original_path = Path(code) / "mimic-iv/concepts_duckdb/sepsis/suspicion_of_infection.sql"
    optimized_path = Path(__file__).resolve().parents[1] / "scripts/sql/suspicion_of_infection.sql"
    original = original_path.read_text().split(" AS\n", 1)[1]
    optimized = optimized_path.read_text().split(" AS\n", 1)[1]
    db = duckdb.connect(":memory:")
    db.execute("CREATE SCHEMA mimiciv_hosp; CREATE SCHEMA mimiciv_derived")
    db.execute("CREATE TABLE mimiciv_derived.antibiotic(subject_id INTEGER, hadm_id INTEGER, stay_id INTEGER, antibiotic VARCHAR, starttime TIMESTAMP, stoptime TIMESTAMP)")
    db.execute("CREATE TABLE mimiciv_hosp.microbiologyevents(micro_specimen_id INTEGER, subject_id INTEGER, hadm_id INTEGER, chartdate DATE, charttime TIMESTAMP, spec_type_desc VARCHAR, org_name VARCHAR, org_itemid INTEGER)")
    base = datetime(2020, 1, 10, 12, 0)
    antibiotics, microbes = [], []
    specimen = 0
    for subject in range(1, 13):
        for day in (0, 1, 5):
            when = base + timedelta(days=day, minutes=subject)
            antibiotics.append((subject, subject + 10, subject + 100, "test", when, when + timedelta(hours=12)))
        if subject == 1:
            antibiotics.append((subject, None, None, "empty", None, None))
        for delta in (-97, -73, -72, -71, -1, 0, 1, 23, 24, 25, 72, 73):
            specimen += 1
            when = base + timedelta(hours=delta, minutes=subject)
            microbes.append((specimen, subject, subject + 10, when.date(), when, "blood", "organism", 1))
            specimen += 1
            microbes.append((specimen, subject, subject + 10, when.date(), None, "urine", None, None))
            if delta == 0:
                microbes.append((specimen, subject, subject + 10, when.date(), None, "urine", "organism", 1))
    db.executemany("INSERT INTO mimiciv_derived.antibiotic VALUES (?, ?, ?, ?, ?, ?)", antibiotics)
    db.executemany("INSERT INTO mimiciv_hosp.microbiologyevents VALUES (?, ?, ?, ?, ?, ?, ?, ?)", microbes)
    db.execute("CREATE TABLE old_result AS " + original)
    db.execute("CREATE TABLE new_result AS " + optimized)
    difference = db.execute(
        "SELECT count(*) FROM ((SELECT * FROM old_result EXCEPT ALL SELECT * FROM new_result) "
        "UNION ALL (SELECT * FROM new_result EXCEPT ALL SELECT * FROM old_result))"
    ).fetchone()[0]
    assert difference == 0
    assert db.execute("SELECT COUNT(*) FROM old_result").fetchone()[0] == len(antibiotics)
    plan = db.execute("EXPLAIN " + optimized).fetchone()[1]
    if duckdb.__version__.startswith("1.1."):
        assert "BLOCKWISE_NL_JOIN" not in plan
