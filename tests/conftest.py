import pytest


@pytest.fixture(autouse=True)
def paths(monkeypatch, tmp_path):
    for name in ("MIMIC_RAW", "CLIN_JEPA_DATA", "CLIN_JEPA_OUTPUT", "CLMBR_MODEL", "MIMIC_CODE"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
