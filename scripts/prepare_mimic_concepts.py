"""Build the Clin-JEPA mimic-code concepts; keep all derived data in a work directory."""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[1]
CREATE = re.compile(r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?mimiciv_derived\.([a-z0-9_]+)", re.I)
DERIVED = re.compile(r"\bmimiciv_derived\.([a-z0-9_]+)", re.I)
RAW = re.compile(r"\bmimiciv_(hosp|icu)\.([a-z0-9_]+)", re.I)


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def plan(code_dir):
    definitions = {}
    for path in (code_dir / "mimic-iv/concepts_duckdb").rglob("*.sql"):
        text = path.read_text()
        match = CREATE.search(text)
        if match:
            name = match.group(1).lower()
            if name in definitions:
                raise ValueError(f"duplicate concept SQL: {name}")
            definitions[name] = (path, text)
    config = yaml.safe_load((ROOT / "configs/data/paths.yaml").read_text())
    targets = {Path(path).stem: group for group, values in config["concepts"].items() for path in values.values()}
    visited, visiting, order = set(), set(), []

    def visit(name):
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"cyclic concept dependency at {name}")
        if name not in definitions:
            raise ValueError(f"missing SQL for {name}")
        visiting.add(name)
        text = re.sub(r"--[^\n]*", "", definitions[name][1])
        for dep in sorted({s.lower() for s in DERIVED.findall(text)} - {name}):
            visit(dep)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for name in sorted(targets):
        visit(name)
    raw = set()
    for name in order:
        text = re.sub(r"--[^\n]*", "", definitions[name][1])
        raw.update((a.lower(), b.lower()) for a, b in RAW.findall(text))
    return targets, definitions, order, sorted(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-code", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--temp-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="24GB")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    targets, definitions, order, raw = plan(args.mimic_code)
    print(f"{len(targets)} target concepts, {len(order)} with dependencies, {len(raw)} raw tables", flush=True)
    if args.dry_run:
        print("raw tables:", ", ".join(f"{a}.{b}" for a, b in raw), flush=True)
        return
    args.db.parent.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.db))
    con.execute(f"SET threads = {args.threads}")
    con.execute("SET memory_limit = " + quote(args.memory_limit))
    con.execute("SET temp_directory = " + quote(args.temp_dir))
    con.execute("CREATE TABLE IF NOT EXISTS main.clin_jepa_progress (step VARCHAR PRIMARY KEY)")

    def done(step):
        return bool(con.execute("SELECT count(*) FROM main.clin_jepa_progress WHERE step = ?", [step]).fetchone()[0])

    def transaction(step, action):
        if done(step):
            print(f"{step}: already complete", flush=True)
            return
        print(f"{step}: starting", flush=True)
        start = time.monotonic()
        con.execute("BEGIN TRANSACTION")
        try:
            action()
            con.execute("INSERT INTO main.clin_jepa_progress VALUES (?)", [step])
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        print(f"{step}: complete ({time.monotonic() - start:.1f}s)", flush=True)

    def create_schema():
        count = con.execute("SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name IN ('mimiciv_hosp','mimiciv_icu','mimiciv_derived')").fetchone()[0]
        if count:
            raise RuntimeError("MIMIC schemas exist without a completed schema step; refusing to overwrite")
        sql = (args.mimic_code / "mimic-iv/buildmimic/postgres/create.sql").read_text()
        sql = re.sub(r"TIMESTAMP\([0-9]+\)", "TIMESTAMP", sql)
        sql = re.sub(r"spec_type_desc(.+)NOT NULL", r"spec_type_desc\1", sql)
        sql = re.sub(r"drug +(VARCHAR.+)NOT NULL", r"drug \1", sql)
        con.execute(sql)

    transaction("schema", create_schema)
    for group, name in raw:
        file = args.raw / group / (name + ".csv.gz")
        if not file.is_file():
            raise FileNotFoundError(file)
        table = f"mimiciv_{group}.{name}"
        def load(table=table, file=file):
            con.execute(f"COPY {table} FROM {quote(file)} (HEADER true, DELIM ',')")
        transaction("raw:" + group + "." + name, load)
    for name in order:
        path, text = definitions[name]
        if name == "suspicion_of_infection":
            # Equivalent join rewrite avoids a quadratic DuckDB 1.1 nested-loop plan.
            text = (ROOT / "scripts/sql/suspicion_of_infection.sql").read_text()
        transaction("concept:" + name, lambda text=text: con.execute(text))
    for name, group in targets.items():
        out = args.mimic_root / "concepts" / group / (name + ".csv")
        if out.exists() and out.stat().st_size:
            print(f"export:{group}/{name}: already present", flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_suffix(".csv.tmp")
        print(f"export:{group}/{name}: starting", flush=True)
        con.execute(f"COPY (SELECT * FROM mimiciv_derived.{name}) TO {quote(temp)} (FORMAT CSV, HEADER true)")
        temp.replace(out)
        print(f"export:{group}/{name}: complete", flush=True)
    con.close()


if __name__ == "__main__":
    main()
