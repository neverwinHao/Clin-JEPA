"""Figure and tables of the downstream evaluation."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from clin_jepa.downstream.tasks import BENCHMARKS, TASKS, benchmark_tasks
from clin_jepa.utils import ensure_dir, load_config

BASELINES = {"ridge": "Ridge", "lightgbm": "LightGBM", "lstm": "LSTM", "tcn": "TCN"}
FOUNDATION_MODEL = {"clmbr": "CLMBR-T-base"}
ENCODERS = {"sft_baseline": "SFT baseline w/o JEPA", "vjepa2ac": "V-JEPA 2-AC style", "clin_jepa": "Clin-JEPA (ours)"}
VARIANTS = {"hist": "history only", "full": "history + future"}
OURS, PAIRED = "clin_jepa", ("clin_jepa_hist_paired", "clmbr_paired")
METRICS = {"auroc": "AUROC", "auprc": "AUPRC"}
COLORS = {
    "ridge": "#9E9E9E", "lightgbm": "#795548", "lstm": "#607D8B", "tcn": "#4DB6AC", "clmbr": "#6A1B9A",
    "sft_baseline_hist": "#FFB74D", "sft_baseline_full": "#EF6C00", "vjepa2ac_hist": "#90CAF9",
    "vjepa2ac_full": "#1565C0", "clin_jepa_hist": "#81C784", "clin_jepa_full": "#2E7D32",
}
HATCHES = {"E1": None, "E2": "///", "E3": "xxx"}
ANCHORS = {"auroc": 0.80, "auprc": 0.29}
MIN_FINITE_DRAWS = 50


def summarise(vector: np.ndarray) -> tuple[float, float, float]:
    finite = vector[np.isfinite(vector)]
    if len(finite) < MIN_FINITE_DRAWS:
        return float(np.nanmean(vector)) if len(finite) else float("nan"), float("nan"), float("nan")
    return (float(np.mean(finite)), *map(float, np.percentile(finite, (2.5, 97.5))))


class Results:
    def __init__(self, directory: Path) -> None:
        self.cells = {task.name: {p.stem: dict(np.load(p)) for p in sorted((directory / task.name).glob("*.npz"))}
                      for task in TASKS}
        names = list(BASELINES) + list(FOUNDATION_MODEL) + [f"{e}_{v}" for e in ENCODERS for v in VARIANTS]
        self.methods = [m for m in names if any(m in cells for cells in self.cells.values())]
        self.n_draws = max((len(v["auroc"]) for cells in self.cells.values() for v in cells.values()), default=0)

    def task(self, task: str, method: str, metric: str) -> tuple[float, float, float] | None:
        cell = self.cells[task].get(method)
        return None if cell is None else summarise(cell[metric])

    def benchmark(self, benchmark: str, method: str, metric: str) -> tuple[float, float, float] | None:
        cells = [self.cells[task.name] for task in benchmark_tasks(benchmark)]
        vectors = [c[method][metric] for c in cells if method in c]
        return summarise(np.mean(vectors, axis=0)) if vectors else None


def method_name(method: str, tex: bool) -> tuple[str, str]:
    if method in BASELINES or method in FOUNDATION_MODEL:
        return {**BASELINES, **FOUNDATION_MODEL}[method], ""
    encoder, variant = method.rsplit("_", 1)
    name = r"\textsc{Clin-JEPA} (ours)" if tex and encoder == OURS else ENCODERS[encoder]
    return name, f"({VARIANTS[variant]})"


def row_name(method: str) -> str:
    name, variant = method_name(method, tex=True)
    if not variant:
        return name
    if method.startswith(OURS):
        return rf"\textsc{{Clin-JEPA}} (ours, {variant[1:-1]})"
    return f"{name} {variant}"


def cell(entry: tuple | None, bold: bool = False) -> str:
    if entry is None or not np.isfinite(entry[0]):
        return "--"
    value = f"{entry[0]:.3f}"
    if bold:
        value = rf"\textbf{{{value}}}"
    if not np.isfinite(entry[1]):
        return value
    return (rf"\begin{{tabular}}[t]{{@{{}}c@{{}}}}{value}\\[-1.5pt]"
            rf"{{\fontsize{{5.4}}{{6.2}}\selectfont[{entry[1]:.3f},\,{entry[2]:.3f}]}}\end{{tabular}}")


def best(entries: dict) -> str | None:
    scored = {k: v[0] for k, v in entries.items() if v is not None and np.isfinite(v[0])}
    return max(scored, key=scored.get) if scored else None


def table_summary(res: Results) -> str:
    paired = all(any(p in cells for cells in res.cells.values()) for p in PAIRED)
    caption = (r"\caption{\textbf{Downstream multi-task evaluation.} Mean AUROC and AUPRC over the tasks of each "
               rf"benchmark, with 95\% stay-clustered bootstrap intervals ({res.n_draws} draws); "
               r"best per column in bold.")
    if paired:
        caption += (r" The last two rows score \textsc{Clin-JEPA} and CLMBR-T-base on exactly the stays CLMBR-T-base "
                    r"covers, with shared resamples.")
    lines = [
        r"\begin{table}[H]\centering\small\renewcommand{\arraystretch}{1.28}", caption + "}",
        r"\label{tab:downstream_summary}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l cc cc cc}\toprule",
        "& " + " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{b}}} {name}}}" for b, name in BENCHMARKS.items())
        + r" \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"\textbf{Method} & AUROC & AUPRC & AUROC & AUPRC & AUROC & AUPRC \\",
        " & ".join([""] + [x for b in BENCHMARKS for x in (rf"\scriptsize({len(benchmark_tasks(b))} tasks)", "")])
        + r" \\\midrule",
    ]
    columns = [(b, metric) for b in BENCHMARKS for metric in METRICS]
    top = {c: best({m: res.benchmark(c[0], m, c[1]) for m in res.methods}) for c in columns}
    for method in res.methods:
        cells = [cell(res.benchmark(b, method, metric), top[(b, metric)] == method) for b, metric in columns]
        lines.append(" & ".join([row_name(method)] + cells) + r" \\")
        if method in ("tcn", "clmbr"):
            lines.append(r"\addlinespace[3pt]")
    if paired:
        lines.append(r"\midrule\multicolumn{7}{l}{\textit{Restricted to the stays CLMBR-T-base covers, "
                     r"shared resamples}}\\")
        for method, name in zip(PAIRED, (r"\quad \textsc{Clin-JEPA} (ours, history only)", r"\quad CLMBR-T-base")):
            cells = [cell(res.benchmark(b, method, metric)) for b, metric in columns]
            lines.append(" & ".join([name] + cells) + r" \\")
    lines.append(r"\bottomrule\end{tabular}}\end{table}")
    return "\n".join(lines) + "\n"


def count(value) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def table_per_task(res: Results, benchmark: str, metric: str) -> str:
    tasks, label = benchmark_tasks(benchmark), f"{benchmark} --- {BENCHMARKS[benchmark]}"
    header = [r"\textbf{Task}", r"\rotatebox{90}{\# test rows}", r"\rotatebox{90}{\# positives}"]
    for method in res.methods:
        name, variant = method_name(method, tex=True)
        header.append(r"\rotatebox{90}{\shortstack[l]{" + name + (r"\\" + variant if variant else "") + "}}")
    lines = [
        r"\begin{landscape}\begin{table}[p]\centering\scriptsize\renewcommand{\arraystretch}{1.12}"
        r"\setlength{\tabcolsep}{2.6pt}",
        rf"\caption{{\textbf{{{label}: per-task {METRICS[metric]}}} with 95\% stay-clustered bootstrap intervals "
        rf"({res.n_draws} draws); best per row in bold.}}",
        rf"\label{{tab:downstream_{benchmark}_{metric}}}",
        r"\begin{tabular}{l r r " + "c" * len(res.methods) + r"}\toprule",
        " & ".join(header) + r" \\\midrule",
    ]
    for task in tasks:
        entries = {m: res.task(task.name, m, metric) for m in res.methods}
        reference = res.cells[task.name].get(f"{OURS}_hist") or next(iter(res.cells[task.name].values()), None)
        counts = ["--", "--"] if reference is None else [count(reference["n"]), count(reference["n_positive"])]
        top = best(entries)
        lines.append(" & ".join([task.title, *counts] + [cell(entries[m], m == top) for m in res.methods]) + r" \\")
    entries = {m: res.benchmark(benchmark, m, metric) for m in res.methods}
    top = best(entries)
    lines += [r"\midrule",
              rf"\textbf{{Mean over {len(tasks)} tasks}} & & & "
              + " & ".join(cell(entries[m], m == top) for m in res.methods) + r" \\",
              r"\bottomrule\end{tabular}\end{table}\end{landscape}"]
    return "\n".join(lines) + "\n"


def figure(res: Results, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(19.5, 6.4))
    width, x = 0.26, np.arange(len(res.methods))
    for ax, (metric, title) in zip(axes, METRICS.items()):
        anchor = ANCHORS[metric]
        rows = {(b, m): res.benchmark(b, m, metric) for b in BENCHMARKS for m in res.methods}
        rows = {k: np.where(np.isfinite(v), v, v[0]) for k, v in rows.items() if v is not None and np.isfinite(v[0])}
        top = max(v[2] - anchor for v in rows.values())
        bottom = min(min(v[1] - anchor for v in rows.values()), 0)
        for i, benchmark in enumerate(BENCHMARKS):
            for j, method in enumerate(res.methods):
                if (benchmark, method) not in rows:
                    continue
                value, lo, hi = rows[(benchmark, method)]
                ours = method.startswith(OURS)
                position = x[j] + (i - 1) * width
                ax.bar(position, value - anchor, width, yerr=[[value - lo], [hi - value]], color=COLORS[method],
                       hatch=HATCHES[benchmark], edgecolor="#1B5E20" if ours else "white",
                       linewidth=1.7 if ours else 0.5, capsize=2,
                       error_kw={"lw": 0.8, "capthick": 0.8, "ecolor": "#333333"})
                above = value >= anchor
                ax.text(position, (hi - anchor + top * 0.025) if above else (lo - anchor - top * 0.025), f"{value:.3f}",
                        ha="center", va="bottom" if above else "top", fontsize=10.2, fontweight="bold", rotation=90,
                        clip_on=False)
        ax.axhline(0, color="#444444", lw=1.0, ls="-", alpha=0.85, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(["\n".join(filter(None, method_name(m, tex=False))) for m in res.methods], rotation=30,
                           ha="right", rotation_mode="anchor", fontsize=11.5, fontweight="bold", linespacing=1.15)
        ax.set_ylabel(f"$\\Delta$ {title}   (anchor = {anchor:.2f})", fontsize=14.5, fontweight="bold")
        ax.tick_params(axis="y", labelsize=12.5)
        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.grid(axis="y", alpha=0.25, ls="--")
        ax.set_axisbelow(True)
        span = top - bottom
        ax.set_ylim(bottom - span * 0.20, top + span * 0.215)
    handles = [Patch(facecolor="white", edgecolor="black", lw=1.2, hatch=HATCHES[b],
                     label=f"{b} — {label} ({len(benchmark_tasks(b))} tasks)") for b, label in BENCHMARKS.items()]
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, fontsize=13.5,
                        framealpha=1.0, edgecolor="#666666", columnspacing=2.4, handlelength=1.6)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def text_numbers(res: Results) -> dict:
    def value(entry):
        return None if entry is None or not np.isfinite(entry[0]) else entry[0]

    out = {"benchmarks": {}, "tasks": {}}
    for benchmark in BENCHMARKS:
        means = {f"{m}|{metric}": value(res.benchmark(benchmark, m, metric)) for m in res.methods for metric in METRICS}
        for metric in METRICS:
            scored = [means[f"{m}|{metric}"] for m in BASELINES if means.get(f"{m}|{metric}") is not None]
            means[f"best_baseline|{metric}"] = max(scored) if scored else None
        out["benchmarks"][benchmark] = means
    for task in TASKS:
        out["tasks"][task.name] = {f"{m}|{metric}": value(res.task(task.name, m, metric))
                                   for m in res.methods for metric in METRICS}

    def wins(tasks, ours: str, others: list[str], metric: str) -> list:
        margins = []
        for task in tasks:
            scores = out["tasks"][task.name]
            rivals = [scores[f"{m}|{metric}"] for m in others if scores.get(f"{m}|{metric}") is not None]
            if rivals and scores.get(f"{ours}|{metric}") is not None:
                margins.append((task.name, scores[f"{ours}|{metric}"] - max(rivals)))
        return margins

    baselines, other_encoders = list(BASELINES), [f"{e}_hist" for e in ENCODERS if e != OURS]
    for name, tasks, ours, others, metric in (
        ("full_vs_best_baseline_E2_auprc", benchmark_tasks("E2"), f"{OURS}_full", baselines, "auprc"),
        ("full_vs_lightgbm_E1_auroc", benchmark_tasks("E1"), f"{OURS}_full", ["lightgbm"], "auroc"),
        ("hist_vs_other_encoders_auroc", TASKS, f"{OURS}_hist", other_encoders, "auroc"),
        ("hist_vs_other_encoders_E3_auroc", benchmark_tasks("E3"), f"{OURS}_hist", other_encoders, "auroc"),
        ("full_vs_hist_auroc", TASKS, f"{OURS}_full", [f"{OURS}_hist"], "auroc"),
    ):
        margins = wins(tasks, ours, others, metric)
        out[name] = {"ahead": sum(m > 0 for _, m in margins), "of": len(margins),
                     "mean_margin": float(np.mean([m for _, m in margins])) if margins else None,
                     "margins": dict(sorted(margins, key=lambda item: -item[1]))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/downstream.yaml")
    args = parser.parse_args()

    root = Path(load_config(args.config)["output_dir"])
    res = Results(root / "bootstrap")
    tables, figures = ensure_dir(root / "tables"), ensure_dir(root / "figures")
    (tables / "Table4_downstream_summary.tex").write_text(table_summary(res))
    for benchmark in BENCHMARKS:
        for metric in METRICS:
            (tables / f"AppTable5_{benchmark}_{metric}.tex").write_text(table_per_task(res, benchmark, metric))
    numbers = text_numbers(res)
    coverage = root / "clmbr" / "timelines" / "stay" / "test" / "coverage.csv"
    if coverage.exists():
        table = pd.read_csv(coverage).set_index("table")
        numbers["clmbr_vocabulary_coverage"] = (table["in_vocabulary"] / table["events"]).to_dict()
    (tables / "text_numbers.json").write_text(json.dumps(numbers, indent=2))
    figure(res, figures / "Fig4_downstream.pdf")


if __name__ == "__main__":
    main()
