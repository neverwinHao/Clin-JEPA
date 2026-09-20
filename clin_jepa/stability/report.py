"""Figures and tables of the stability analysis, rendered from the three result sets of this package."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.lines import Line2D

from clin_jepa.utils import ensure_dir, load_config

H_COLUMNS = [1, 6, 12, 24, 36, 48]
LEGEND_ORDER = ["clin_jepa", "separate_predictor", "vjepa2ac", "no_warmup", "sft_baseline", "no_alignment"]
STYLE = {
    "clin_jepa": dict(color="#2E7D32", ls="-", lw=1.7, marker="o", ms=3.6, z=6),
    "vjepa2ac": dict(color="#1565C0", ls="-", lw=1.2, marker="s", ms=3.2, z=5),
    "sft_baseline": dict(color="#EF6C00", ls="-", lw=1.2, marker="^", ms=3.4, z=5),
    "separate_predictor": dict(color="#0097A7", ls="-.", lw=1.2, marker="v", ms=3.4, z=4),
    "no_warmup": dict(color="#8E24AA", ls="--", lw=1.2, marker="x", ms=3.6, z=4),
    "no_alignment": dict(color="#C62828", ls="--", lw=1.2, marker="d", ms=3.2, z=4),
}
METRICS = {
    "drift": dict(name="rollout drift", tex=r"$\ell_1(h)\,/\,\ell_1(1)$", tex48=r"$\ell_1(48)/\ell_1(1)$", arrow=""),
    "degradation": dict(name="predictor degradation", tex=r"MASE$(h)\,/\,$MASE$(1)$",
                        tex48=r"MASE$(48)$/MASE$(1)$", arrow=r"$\downarrow$"),
    "displacement": dict(name="state displacement", tex=r"$\ell_1^{\mathrm{cf}}(h)\,/\,\ell_1^{\mathrm{cf}}(1)$",
                         tex48=r"$\ell_1^{\mathrm{cf}}(48)/\ell_1^{\mathrm{cf}}(1)$", arrow=r"$\uparrow$"),
}
INK, INK2, GRID = "#222222", "#555555", "#e6e6e6"


class Results:
    """The outputs of ``drift``, ``encoder_side`` and ``predictor_side`` behind one model-keyed view."""

    def __init__(self, config: dict) -> None:
        root = Path(config["output_dir"])
        self.config = config
        self.models = config["models"]
        self.drift = json.loads((root / "drift.json").read_text())
        self.encoders = {p.stem: json.loads(p.read_text()) for p in (root / "encoder_side").glob("*.json")}
        self.ranks = {p.stem: json.loads(p.read_text()) for p in (root / "predictor_side").glob("*.json")}

    def group(self, name: str) -> list[str]:
        return [key for key, entry in self.models.items() if entry["group"] == name]

    def encoder(self, key: str) -> dict:
        return self.encoders[self.models[key]["encoder"]]

    def curve(self, key: str, name: str) -> np.ndarray:
        return np.array(self.drift["models"][key]["curves"][name])

    def rank_curve(self, key: str, name: str) -> np.ndarray:
        """Mean over the model's rollouts of a predictor-side quantity, per horizon."""
        per_rollout = []
        for rollout in self.models[key]["rollouts"]:
            rows = self.ranks[rollout]
            per_rollout.append([r["rankme_predicted"] / r["rankme_true"] if name == "rankme_ratio" else r[name]
                                for r in rows])
        return np.mean(per_rollout, axis=0)


def fmt_int(x: float) -> str:
    return f"{x:,.0f}"


def fmt_big(x: float) -> str:
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def times(x: float, big: bool = False) -> str:
    return rf"$\times${fmt_big(x) if big else f'{x:.2f}'}"


def plus_minus(value: float, lo: float, hi: float) -> str:
    half = (hi - lo) / 2
    if half == 0:
        return fmt_big(value)
    half_text = f"{half:.1f}" if half >= 10 else (f"{half:.2f}" if half >= 0.01 else f"{half:.3f}")
    return rf"{fmt_big(value)}\,$\pm$\,{half_text}"


def bold(text: str, on: bool) -> str:
    return rf"\textbf{{{text}}}" if on else text


def table_main(res: Results) -> str:
    paradigms, ablations = res.group("paradigm"), res.group("ablation")
    values = {}
    for key in paradigms + ablations:
        enc = res.encoder(key)
        values[key] = dict(
            static=enc["static"]["rankme"], dynamic=enc["dynamic"]["rankme"],
            top_static=enc["static"]["top10_share"], top_dynamic=enc["dynamic"]["top10_share"],
            displacement=res.curve(key, "displacement")[-1], degradation=res.curve(key, "degradation")[-1],
            rankme_predicted=res.rank_curve(key, "rankme_predicted")[-1], overlap=res.rank_curve(key, "overlap")[-1],
        )
    best = {
        "static": max(paradigms, key=lambda k: values[k]["static"]),
        "dynamic": max(paradigms, key=lambda k: values[k]["dynamic"]),
        "top": min(paradigms, key=lambda k: values[k]["top_static"] + values[k]["top_dynamic"]),
        "displacement": max(paradigms, key=lambda k: values[k]["displacement"]),
        "degradation": min(paradigms, key=lambda k: values[k]["degradation"]),
        "rankme_predicted": max(paradigms, key=lambda k: values[k]["rankme_predicted"]),
        "overlap": max(paradigms, key=lambda k: values[k]["overlap"]),
    }
    lines = [
        r"\begin{tabular}{@{}l rr c c c rr@{}}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Encoder} & \multicolumn{3}{c}{Predictor rollout} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-8}",
        r"Model & \makecell{RankMe\\static $\uparrow$} & \makecell{RankMe\\dynamic $\uparrow$} & "
        r"\makecell{top-10 share\\static / dyn.\ $\downarrow$} & "
        rf"\makecell{{state\\displacement {METRICS['displacement']['arrow']}}} & "
        rf"\makecell{{predictor\\degradation {METRICS['degradation']['arrow']}}} & "
        r"\makecell{RankMe$(\hat z)$\\$\uparrow$} & \makecell{top-64\\overlap $\uparrow$} \\",
        r"\midrule",
    ]
    reference_encoder = res.models[res.config["reference"]]["encoder"]
    for key in paradigms + ablations:
        v, label = values[key], res.models[key]["label"]
        top = f"{v['top_static']:.2f} / {v['top_dynamic']:.2f}"
        if key == ablations[0]:
            lines.append(r"\midrule")
        if key in paradigms:
            cells = [label, bold(fmt_int(v["static"]), best["static"] == key),
                     bold(fmt_int(v["dynamic"]), best["dynamic"] == key), bold(top, best["top"] == key),
                     bold(times(v["displacement"]), best["displacement"] == key),
                     bold(times(v["degradation"]), best["degradation"] == key),
                     bold(f"{v['rankme_predicted']:.1f}", best["rankme_predicted"] == key),
                     bold(f"{v['overlap']:.2f}", best["overlap"] == key)]
        else:
            if res.models[key]["encoder"] == reference_encoder:
                head = [rf"\multicolumn{{2}}{{@{{}}l}}{{{label}}}",
                        r"\multicolumn{3}{c}{\textit{same encoder as Clin-JEPA}}"]
            else:
                head = [label, fmt_int(v["static"]), fmt_int(v["dynamic"]), top, times(v["displacement"])]
            cells = head + [times(v["degradation"], big=True), f"{v['rankme_predicted']:.1f}", f"{v['overlap']:.2f}"]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def table_per_horizon(res: Results) -> str:
    statistics = [("mase", "MASE$(h)$"), ("degradation", METRICS["degradation"]["name"]),
                  ("displacement", METRICS["displacement"]["name"]), ("drift", METRICS["drift"]["name"])]
    lines = [r"\begin{tabular}{@{}ll" + "r" * len(H_COLUMNS) + r"@{}}", r"\toprule",
             r"Model & Statistic & " + " & ".join(rf"$h{{=}}{h}$" for h in H_COLUMNS) + r" \\", r"\midrule"]
    for g, group in enumerate([res.group("paradigm"), res.group("ablation")]):
        if g:
            lines.append(r"\midrule")
        for key in group:
            model = res.drift["models"][key]
            for s, (name, title) in enumerate(statistics):
                lo, hi = model["intervals"][name]
                cells = [plus_minus(model["curves"][name][h - 1], lo[h - 1], hi[h - 1]) for h in H_COLUMNS]
                lines.append(" & ".join([res.models[key]["label"] if s == 0 else "", title] + cells) + r" \\")
            if key != group[-1]:
                lines.append(r"\addlinespace[2pt]")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def table_trivial(res: Results) -> str:
    deg = METRICS["degradation"]
    lines = [
        r"\begin{tabular}{@{}l rrrrrr r rr@{}}", r"\toprule",
        r" & \multicolumn{6}{c}{$\ell_1$ error at $h{=}48$ (latent units of each encoder)} & & "
        rf"\multicolumn{{2}}{{c}}{{paired Wilcoxon vs.\ Clin-JEPA, {deg['name']}}} \\",
        r"\cmidrule(lr){2-7} \cmidrule(lr){9-10}",
        r"Model & model & \makecell{copy-\\forward} & \makecell{linear\\extrap.} & \makecell{mean state\\(global)} & "
        r"\makecell{mean state\\(per $h$)} & \makecell{mean state\\(per context)} & "
        rf"\makecell{{{deg['name']} {deg['arrow']}\\{deg['tex48']}}} & "
        r"\makecell{share of windows in which\\Clin-JEPA degrades less} & "
        r"\makecell{$p$ (one-sided,\\Clin-JEPA lower)} \\",
        r"\midrule",
    ]
    paradigms = res.group("paradigm")
    best = min(paradigms, key=lambda k: res.curve(k, "degradation")[-1])
    for g, group in enumerate([paradigms, res.group("ablation")]):
        if g:
            lines.append(r"\midrule")
        for key in group:
            model, reference = res.drift["models"][key], res.encoder(key)["mean_state_l1"]
            errors = [model["l1"][-1], model["l1_copy_forward"][-1], model["l1_linear_extrap"][-1],
                      reference["global"], reference["horizon"], reference["context"]]
            test = model.get("wilcoxon")
            if test is None:
                test_cells = ["--", "--"]
            else:
                p = test["p_value"]
                test_cells = [f"{test['share_reference_lower']:.3f}",
                              r"$<10^{-6}$" if p < 1e-6 else (f"{p:.2g}" if p < 0.999 else "1.0")]
            degradation = bold(times(res.curve(key, "degradation")[-1], big=True), key == best)
            lines.append(" & ".join([res.models[key]["label"]] + [fmt_int(e) for e in errors]
                                    + [degradation] + test_cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def table_checkpoints(res: Results) -> str:
    deg, dis, dri = METRICS["degradation"], METRICS["displacement"], METRICS["drift"]
    lines = [
        r"\begin{tabular}{@{}l r rr rr rrr@{}}", r"\toprule",
        r"Checkpoint & step & \makecell{RankMe\\static $\uparrow$} & \makecell{RankMe\\dynamic $\uparrow$} & "
        r"MASE$(1)$ $\downarrow$ & MASE$(48)$ $\downarrow$ & "
        rf"\makecell{{{deg['name']} {deg['arrow']}\\{deg['tex48']}}} & "
        rf"\makecell{{{dis['name']} {dis['arrow']}\\{dis['tex48']}}} & \makecell{{{dri['name']}\\{dri['tex48']}}} \\",
        r"\midrule",
    ]
    rows = []
    for entry in res.config["checkpoints"]:
        curves, enc = res.drift["checkpoints"][entry["rollout"]], res.encoders[entry["encoder"]]
        rows.append(dict(static=enc["static"]["rankme"], dynamic=enc["dynamic"]["rankme"],
                         mase1=curves["mase"][0], mase48=curves["mase"][-1], degradation=curves["degradation"][-1],
                         displacement=curves["displacement"][-1], drift=curves["drift"][-1]))
    best = {c: (max if c in ("static", "dynamic", "displacement") else min)(r[c] for r in rows)
            for c in ("static", "dynamic", "mase1", "mase48", "degradation", "displacement")}

    def cell(row: dict, column: str, text: str) -> str:
        return bold(text, abs(row[column] - best[column]) <= 1e-9 * max(1.0, abs(best[column])))

    for entry, r in zip(res.config["checkpoints"], rows):
        lines.append(" & ".join([
            entry["label"], f"{entry['step']:,}", cell(r, "static", fmt_int(r["static"])),
            cell(r, "dynamic", fmt_int(r["dynamic"])), cell(r, "mase1", f"{r['mase1']:.3f}"),
            cell(r, "mase48", fmt_big(r["mase48"])), cell(r, "degradation", times(r["degradation"], big=True)),
            cell(r, "displacement", times(r["displacement"])), times(r["drift"], big=True),
        ]) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def set_style() -> None:
    plt.rcParams.update({
        "font.family": ["Nimbus Sans", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 2.6, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2, "legend.labelspacing": 0.25,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "text.color": INK,
        "mathtext.fontset": "dejavusans",
    })


def log_axis(ax, ticks: list[float], lo: float, hi: float) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylim(lo, hi)


def draw_line(ax, key: str, y: np.ndarray) -> None:
    style, h = STYLE[key], np.arange(1, len(y) + 1)
    marked = [t - 1 for t in H_COLUMNS]
    ax.plot(h, y, color=style["color"], ls=style["ls"], lw=style["lw"], zorder=style["z"], solid_capstyle="round")
    ax.plot(h[marked], y[marked], ls="none", marker=style["marker"], ms=style["ms"], color=style["color"],
            mec="white" if style["marker"] != "x" else style["color"], mew=0.6, zorder=style["z"] + 1)


def add_legend(fig, res: Results) -> None:
    handles = [Line2D([], [], color=s["color"], ls=s["ls"], lw=s["lw"], marker=s["marker"], ms=s["ms"],
                      mec="white" if s["marker"] != "x" else s["color"], mew=0.6, label=res.models[key]["label"])
               for key, s in ((k, STYLE[k]) for k in LEGEND_ORDER)]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005),
               handlelength=2.8, columnspacing=0.9)


def end_labels(ax, items: list[tuple[float, str]], x: float = 48.0, min_gap_pt: float = 7.0) -> None:
    """Value labels to the right of the last horizon, pushed apart so that they do not overlap."""
    gap = min_gap_pt * ax.figure.dpi / 72
    heights = [ax.transData.transform((x, y))[1] for y, _ in items]
    placed: list[tuple[int, float]] = []
    for i in np.argsort(heights):
        y = heights[i] if not placed or heights[i] - placed[-1][1] >= gap else placed[-1][1] + gap
        placed.append((i, y))
    to_data = ax.transData.inverted()
    for i, y in placed:
        ax.annotate(items[i][1], xy=(x, items[i][0]), xytext=(x + 1.5, to_data.transform((0, y))[1]),
                    textcoords="data", fontsize=6.2, color=INK2, va="center", ha="left", zorder=9)


def figure_drift(res: Results, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.15))
    labelled = {
        "drift": ["clin_jepa", "no_warmup", "no_alignment"],
        "degradation": ["clin_jepa", "separate_predictor", "no_warmup", "no_alignment"],
        "displacement": ["clin_jepa", "vjepa2ac", "sft_baseline"],
    }
    for letter, ax, name in zip("abc", axes, labelled):
        metric = METRICS[name]
        ax.axhline(1.0, color="#9e9e9e", lw=0.7, ls=(0, (3, 2)), zorder=1)
        for key in LEGEND_ORDER:
            draw_line(ax, key, res.curve(key, name))
        if name == "displacement":
            ax.set_ylim(0.9, 2.3)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        else:
            log_axis(ax, [1, 2, 5, 10, 20, 50, 100], 0.85, 130)
        ax.set_xlim(0, 57)
        ax.set_xticks(H_COLUMNS)
        ax.set_xlabel("rollout horizon $h$ (hours)")
        arrow = f"  {metric['arrow']}" if metric["arrow"] else ""
        ax.set_title(f"({letter}) {metric['name']}{arrow}", loc="left", pad=4, color=INK, fontweight="bold")
        ax.set_ylabel(metric["tex"], labelpad=1.5)
        if name == "degradation":
            inset = ax.inset_axes([0.15, 0.56, 0.30, 0.37])
            for key in res.group("paradigm"):
                s = STYLE[key]
                curve = res.curve(key, name)
                inset.plot(np.arange(1, len(curve) + 1), curve, color=s["color"], ls=s["ls"],
                           lw=s["lw"] * 0.8, zorder=s["z"])
            inset.axhline(1.0, color="#9e9e9e", lw=0.5, ls=(0, (3, 2)), zorder=1)
            inset.set_xlim(0, 49)
            inset.set_ylim(0.95, 1.45)
            inset.set_xticks([1, 24, 48])
            inset.set_yticks([1.0, 1.2, 1.4])
            inset.tick_params(labelsize=5.4, length=1.5, pad=1.5, width=0.5)
            for spine in inset.spines.values():
                spine.set_linewidth(0.5)
            inset.grid(True, color=GRID, lw=0.4)
            inset.set_title("full paradigms, linear axis", fontsize=5.6, pad=2, color=INK2, loc="left")
    add_legend(fig, res)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.36, wspace=0.42)
    for ax, name in zip(axes, labelled):
        last = [res.curve(key, name)[-1] for key in labelled[name]]
        end_labels(ax, [(y, f"×{y:.2f}" if y < 10 else f"×{y:.1f}") for y in last])
    fig.savefig(path)
    plt.close(fig)


def figure_rank(res: Results, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0))
    panels = [("rankme_ratio", r"(a) RankMe$(\hat z_h)\,/\,$RankMe$(z_h)$"), ("overlap", "(b) top-64 subspace overlap")]
    for ax, (name, title) in zip(axes, panels):
        for key in LEGEND_ORDER:
            draw_line(ax, key, res.rank_curve(key, name))
        if name == "rankme_ratio":
            log_axis(ax, [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1], 0.0008, 1.5)
        else:
            ax.set_ylim(0, 1.02)
        ax.set_xlim(0, 50)
        ax.set_xticks(H_COLUMNS)
        ax.set_xlabel("rollout horizon $h$ (hours)")
        ax.set_title(title, loc="left", pad=4)
    add_legend(fig, res)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.40, wspace=0.3)
    fig.savefig(path)
    plt.close(fig)


def text_numbers(res: Results) -> dict:
    """Quantities the paper states in running text."""
    n_windows = res.drift["models"][res.config["reference"]]["n_windows"]
    return {
        "windows_test": res.encoder(res.config["reference"])["n_windows"],
        "windows_h1": n_windows[0],
        "windows_h48": n_windows[-1],
        "one_hour_relative_step": {
            key: res.drift["models"][key]["l1_copy_forward"][0] / res.encoder(key)["mean_l1_norm"]
            for key in res.group("paradigm")
        },
        "z_std": {name: enc["z_std"] for name, enc in res.encoders.items()},
        "mean_l1_norm": {name: enc["mean_l1_norm"] for name, enc in res.encoders.items()},
        "effective_rank": {name: [enc["static"]["effective_rank"], enc["dynamic"]["effective_rank"]]
                           for name, enc in res.encoders.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stability.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    res = Results(config)
    tables = ensure_dir(Path(config["output_dir"]) / "tables")
    figures = ensure_dir(Path(config["output_dir"]) / "figures")
    (tables / "Table2_encoder_predictor.tex").write_text(table_main(res))
    (tables / "AppTable1_per_horizon_ci.tex").write_text(table_per_horizon(res))
    (tables / "AppTable2_trivial_baselines_wilcoxon.tex").write_text(table_trivial(res))
    (tables / "AppTable3_curriculum_checkpoints.tex").write_text(table_checkpoints(res))
    (tables / "text_numbers.json").write_text(json.dumps(text_numbers(res), indent=2))
    set_style()
    figure_drift(res, figures / "Fig2a_drift_decomposition.pdf")
    figure_rank(res, figures / "AppFig1_rollout_rank_vs_h.pdf")


if __name__ == "__main__":
    main()
