"""Figures and tables of the latent-geometry diagnosis."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from clin_jepa.utils import ensure_dir, load_config

MAIN = "main|all|l2"
ROBUSTNESS = [
    (MAIN, "main definition, all stays, L2"), ("main|all|cos", "main, all, cosine"),
    ("main|all|pca", "main, all, PCA-50"), ("main|subsample|l2", "main, 50 + 50 subsample, L2"),
    ("delta2|all|l2", r"$\Delta$SOFA $\ge$ 2, all, L2"), ("delta4|all|l2", r"$\Delta$SOFA $\ge$ 4, all, L2"),
    ("stable_std_only|all|l2", "stable = std $\\le$ 1 only, all, L2"),
]
PANEL_ORDER = ["sft_baseline", "vjepa2ac", "clin_jepa"]
STYLE = {
    "clin_jepa": dict(color="#2E7D32", ls="-", lw=1.7, marker="o", ms=3.4),
    "vjepa2ac": dict(color="#1565C0", ls="-", lw=1.2, marker="s", ms=3.0),
    "sft_baseline": dict(color="#EF6C00", ls="-", lw=1.2, marker="^", ms=3.2),
}
COHORT_COLOR = {"deteriorating": "#d62728", "stable": "#1f77b4"}
MEAN_COLOR = "#FFC107"
INK, INK2, GRID = "#222222", "#555555", "#e6e6e6"
X_TICKS = [0, 12, 24, 36, 48, 60, 71]
TEXTWIDTH = 5.5
SMOOTHING = 0.1
JITTER_SEED = 7


def interval(pair: list) -> str:
    return f"[{pair[0]:.2f}, {pair[1]:.2f}]"


def mean_sd(values: np.ndarray, digits: int) -> str:
    mean = f"{values.mean():.{digits}f}"
    return f"{mean.replace('-0.', '0.') if float(mean) == 0 else mean} $\\pm$ {values.std(ddof=1):.{digits}f}"


def table_main(results: dict, labels: dict) -> str:
    lines = [
        r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
        r"Model & \makecell{net-displacement\\Cohen's $d$ $\uparrow$ [95\% CI]} & "
        r"\makecell{stable / deteriorating\\net displacement$^{\dagger}$} & "
        r"\makecell{cohort divergence$^{\dagger}$\\$\uparrow$ [95\% CI]} & "
        r"\makecell{nearest-centroid\\accuracy $\uparrow$ [95\% CI]} & "
        r"\makecell{direction\\accuracy $\uparrow$ [95\% CI]} & "
        r"\makecell{cohort-direction\\angle $\uparrow$ [95\% CI]} \\",
        r"\midrule",
    ]
    rows = {}
    for key in labels:
        r, d = results[key]["configurations"][MAIN], results[key]["direction"]
        net = r["net_displacement"]
        rows[key] = {
            "d": (net["cohens_d"], net["cohens_d_interval"]),
            "divergence": (r["divergence"][-1], [b[-1] for b in r["divergence_interval"]]),
            "centroid": (r["centroid_accuracy"][-1], [b[-1] for b in r["centroid_accuracy_interval"]]),
            "direction": (d["accuracy"][-1], [b[-1] for b in d["accuracy_interval"]]),
            "angle": (d["angle"], d["angle_interval"]),
            "displacement": f"{net['stable_mean']:.2f} / {net['deteriorating_mean']:.2f}",
        }
    best = {c: max(rows, key=lambda k: rows[k][c][0]) for c in ("d", "divergence", "centroid", "direction", "angle")}

    def cell(key: str, column: str) -> str:
        value, bounds = rows[key][column]
        if column == "angle":
            text, tail = f"{value:.0f}$^\\circ$", f"[{bounds[0]:.0f}, {bounds[1]:.0f}]"
        else:
            text, tail = f"{value:.2f}", interval(bounds)
        return (rf"\textbf{{{text}}}" if best[column] == key else text) + " " + tail

    for key, label in labels.items():
        lines.append(" & ".join([label, cell(key, "d"), rows[key]["displacement"], cell(key, "divergence"),
                                 cell(key, "centroid"), cell(key, "direction"), cell(key, "angle")]) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def table_robustness(results: dict, labels: dict) -> str:
    lines = [
        r"\begin{tabular}{@{}l r " + "rrr " * len(labels) + r"@{}}", r"\toprule",
        r" & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{label}}}" for label in labels.values()) + r" \\",
        " ".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(labels))),
        r"Configuration & $n$ (det./stable) & "
        + " & ".join(r"$d$ & div.\ $h{=}71$ & acc.\ $h{=}71$" for _ in labels) + r" \\", r"\midrule",
    ]
    first = next(iter(labels))
    for name, title in ROBUSTNESS:
        reference = results[first]["configurations"][name]
        cells = [title, f"{reference['n_deteriorating']} / {reference['n_stable']}"]
        for key in labels:
            r = results[key]["configurations"][name]
            cells += [f"{r['net_displacement']['cohens_d']:.2f}", f"{r['divergence'][-1]:.3f}",
                      f"{r['centroid_accuracy'][-1]:.2f}"]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    reference = results[first]["configurations"][MAIN]
    n_null = len(results[first]["null"])
    cells = [f"random relabeling of the main pool ({n_null} splits, mean $\\pm$ sd)",
             f"{reference['n_deteriorating']} / {reference['n_stable']}"]
    for key in labels:
        null = results[key]["null"]
        d, div, acc = (np.array([row[c] for row in null])
                       for c in ("cohens_d", "divergence_last", "centroid_accuracy_last"))
        cells += [mean_sd(d, 2), mean_sd(div, 3), mean_sd(acc, 2)]
    lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def set_style() -> None:
    plt.rcParams.update({
        "font.family": ["Nimbus Sans", "DejaVu Sans"], "font.size": 7.5, "axes.titlesize": 7.5,
        "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.dpi": 300, "text.color": INK, "mathtext.fontset": "dejavusans",
    })


def two_line(label: str) -> str:
    return label.replace(" (ours)", "\n(ours)").replace(" w/o ", "\nw/o ").replace(" style", "\nstyle")


def figure_net_displacement(results: dict, labels: dict, path: Path) -> None:
    rng = np.random.default_rng(JITTER_SEED)
    fig, ax = plt.subplots(figsize=(0.36 * TEXTWIDTH, 2.05))
    per_stay = {key: results[key]["configurations"][MAIN]["net_displacement"]["per_stay"] for key in PANEL_ORDER}
    cap = max(max(v) for stays in per_stay.values() for v in stays.values())
    for g, key in enumerate(PANEL_ORDER):
        data = [np.asarray(per_stay[key][cohort]) for cohort in COHORT_COLOR]
        positions = [g - 0.21, g + 0.21]
        parts = ax.violinplot(data, positions=positions, widths=0.38, showextrema=False)
        for body, cohort in zip(parts["bodies"], COHORT_COLOR):
            body.set_facecolor(COHORT_COLOR[cohort])
            body.set_edgecolor(COHORT_COLOR[cohort])
            body.set_alpha(0.40)
            body.set_linewidth(0.8)
        for position, values, cohort in zip(positions, data, COHORT_COLOR):
            density = gaussian_kde(values)(values)
            spread = 0.44 * 0.38 * density / density.max()
            ax.scatter(position + rng.uniform(-1, 1, len(values)) * spread, values, s=1.2,
                       c=COHORT_COLOR[cohort], alpha=0.28, lw=0, zorder=3, rasterized=True)
        ax.boxplot(data, positions=positions, widths=0.16 * 0.38 / 0.8, showfliers=False, patch_artist=True, zorder=6,
                   boxprops=dict(facecolor="#3a3a3a", edgecolor="#2a2a2a", lw=0.5),
                   medianprops=dict(color="white", lw=1.0), whiskerprops=dict(color="#2a2a2a", lw=0.6),
                   capprops=dict(color="#2a2a2a", lw=0.6))
        for position, values in zip(positions, data):
            ax.plot(position, values.mean(), marker="D", ms=4.2, color=MEAN_COLOR, mec="#222222", mew=0.6,
                    zorder=8, ls="none")
        y, dy = cap * 1.02, cap * 0.03
        ax.plot([positions[0], positions[0], positions[1], positions[1]], [y, y + dy, y + dy, y],
                color="#333333", lw=0.7, zorder=7, clip_on=False)
        d = results[key]["configurations"][MAIN]["net_displacement"]["cohens_d"]
        ax.text(g, y + 1.6 * dy, f"$d$ = {d:.2f}", ha="center", va="bottom", fontsize=6.0, color=INK,
                fontweight="bold")
    ax.set_xticks(range(len(PANEL_ORDER)))
    ax.set_xticklabels([two_line(labels[k]) for k in PANEL_ORDER], fontsize=5.3, fontweight="bold", linespacing=1.0)
    ax.set_xlim(-0.55, len(PANEL_ORDER) - 0.45)
    ax.set_ylim(0, cap * 1.2)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("72-h net displacement\n(÷ typical inter-patient distance)", fontsize=6.2)
    ax.tick_params(axis="y", labelsize=6.0)
    ax.set_title("cohort separation, Cohen's $d$ ↑", fontsize=6.4, fontweight="bold", pad=3)
    reference = results[PANEL_ORDER[-1]]["configurations"][MAIN]
    handles = [
        Line2D([], [], marker="o", color=COHORT_COLOR["deteriorating"], lw=0, ms=5, alpha=0.8,
               label=f"deteriorating (n = {reference['n_deteriorating']:,})"),
        Line2D([], [], marker="o", color=COHORT_COLOR["stable"], lw=0, ms=5, alpha=0.8,
               label=f"stable (n = {reference['n_stable']:,})"),
        Line2D([], [], marker="D", color=MEAN_COLOR, mec="#222222", mew=0.6, lw=0, ms=5, label="mean"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, fontsize=5.8,
               frameon=False, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0)
    fig.subplots_adjust(left=0.235, right=0.99, top=0.9, bottom=0.3)
    fig.savefig(path)
    plt.close(fig)


def draw_curve(ax, key: str, hours: np.ndarray, values: list, bounds: list | None, label: str | None = None) -> None:
    style, values = STYLE[key], np.asarray(values, dtype=float)
    if bounds is not None:
        ax.fill_between(hours, np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float),
                        color=style["color"], alpha=0.16, lw=0)
    ax.plot(hours, values, color=style["color"], ls=style["ls"], lw=style["lw"], label=label, zorder=4)
    ticks = [t for t in X_TICKS if hours[0] <= t <= hours[-1]]
    ax.plot(ticks, values[[t - hours[0] for t in ticks]], ls="none", marker=style["marker"], ms=style["ms"],
            color=style["color"], mec="white", mew=0.5, zorder=5)


def hour_axis(ax, size: float | None = None) -> None:
    ax.set_xticks(X_TICKS)
    ax.set_xlim(-1.5, 73)
    if size is None:
        ax.set_xlabel("hour of ICU stay")
    else:
        ax.set_xticklabels([str(t) for t in X_TICKS], fontsize=size)
        ax.set_xlabel("hour of ICU stay", fontsize=6.0, labelpad=1.5)


def figure_curves(results: dict, labels: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(0.61 * TEXTWIDTH, 2.05))
    for key in PANEL_ORDER:
        r, d = results[key]["configurations"][MAIN], results[key]["direction"]
        hours = np.arange(len(r["divergence"]))
        draw_curve(axes[0], key, hours, r["divergence"], r["divergence_interval"], label=labels[key])
        draw_curve(axes[1], key, hours, r["centroid_accuracy"], r["centroid_accuracy_interval"])
        draw_curve(axes[2], key, hours[1:], d["accuracy"], d["accuracy_interval"])
    for ax in axes[1:]:
        ax.axhline(0.5, color="#9e9e9e", lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.text(71, 0.5, "chance", ha="right", va="bottom", fontsize=5.4, color="#8a8a8a")
        ax.set_ylim(0.45, 0.95)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    titles = ["centroid divergence ↑", "nearest-centroid acc. ↑", "direction acc. ↑"]
    y_labels = ["÷ typical distance", "balanced accuracy (LOO)", "balanced accuracy (LOO)"]
    for ax, title, y_label in zip(axes, titles, y_labels):
        ax.set_title(title, fontsize=6.4, fontweight="bold", pad=3)
        ax.set_ylabel(y_label, fontsize=5.9, labelpad=1.5)
        hour_axis(ax, size=5.6)
        ax.tick_params(axis="y", labelsize=5.6)
    axes[0].set_ylim(0, None)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
               fontsize=5.9, frameon=False, handlelength=1.8, columnspacing=1.2, borderaxespad=0.0)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.9, bottom=0.27, wspace=0.45)
    fig.savefig(path)
    plt.close(fig)


def figure_knn(results: dict, labels: dict, path: Path) -> None:
    neighbours = list(results[PANEL_ORDER[0]]["configurations"][MAIN]["knn_purity"])
    fig, axes = plt.subplots(1, len(neighbours), figsize=(TEXTWIDTH, 1.9))
    chance = results[PANEL_ORDER[-1]]["configurations"][MAIN]["knn_chance"]
    for ax, k in zip(axes, neighbours):
        for key in PANEL_ORDER:
            r = results[key]["configurations"][MAIN]
            draw_curve(ax, key, np.arange(len(r["knn_purity"][k])), r["knn_purity"][k], r["knn_purity_interval"][k],
                       label=labels[key] if k == neighbours[0] else None)
        ax.axhline(chance, color="#9e9e9e", lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.text(71, chance, f"chance {chance:.3f}", ha="right", va="bottom", fontsize=5.6, color="#8a8a8a")
        ax.set_title(f"kNN cohort purity, k = {k}", fontsize=7, fontweight="bold", loc="left")
        ax.set_ylabel("share of same-cohort neighbours")
        hour_axis(ax)
        ax.set_ylim(0.45, 0.9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
               frameon=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.3, wspace=0.32)
    fig.savefig(path)
    plt.close(fig)


def figure_from_start(results: dict, labels: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(PANEL_ORDER), figsize=(TEXTWIDTH, 1.9), sharey=True)
    for ax, key in zip(axes, PANEL_ORDER):
        for cohort, color in COHORT_COLOR.items():
            curve = results[key]["configurations"][MAIN]["from_start"][cohort]
            ax.plot(np.arange(len(curve)), curve, color=color, lw=1.4, label=cohort if key == PANEL_ORDER[0] else None)
        ax.set_title(labels[key], fontsize=7, fontweight="bold", loc="left")
        hour_axis(ax)
    axes[0].set_ylabel("distance from admission\n(÷ typical distance)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0),
               frameon=False)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.3, wspace=0.12)
    fig.savefig(path)
    plt.close(fig)


def smooth(trajectory: np.ndarray) -> np.ndarray:
    out = np.zeros_like(trajectory)
    out[0] = trajectory[0]
    for t in range(1, len(trajectory)):
        out[t] = SMOOTHING * trajectory[t] + (1 - SMOOTHING) * out[t - 1]
    return out


def square_limits(ax, points: np.ndarray, padding: float, percentiles: tuple[float, float]) -> None:
    lo, hi = np.percentile(points, percentiles, axis=0)
    half = max(hi - lo) * (1 + 2 * padding) / 2
    centre = (lo + hi) / 2
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal")


UMAP_LEGEND = {"deteriorating": "SOFA rising (Δ≥3)", "stable": "SOFA stable (std≤1)"}
UMAP_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.edgecolor": "#333333", "axes.linewidth": 0.9, "axes.grid": True,
    "grid.color": "#e0e0e0", "grid.linewidth": 0.4, "grid.alpha": 0.6, "pdf.fonttype": 42,
}


def umap_axes(labels: dict):
    fig, axes = plt.subplots(1, len(PANEL_ORDER), figsize=(18, 6))
    for ax, key in zip(axes, PANEL_ORDER):
        ax.set_title(labels[key], fontsize=24, fontweight="bold", pad=10)
        ax.set_xlabel("UMAP-1", fontsize=22, fontweight="bold")
        ax.set_ylabel("UMAP-2" if key == PANEL_ORDER[0] else "", fontsize=22, fontweight="bold")
        ax.tick_params(axis="both", labelsize=19)
    return fig, axes


def figure_umap_trajectories(projections: dict, labels: dict, path: Path) -> None:
    fig, axes = umap_axes(labels)
    for ax, key in zip(axes, PANEL_ORDER):
        projection = projections[key]
        smoothed = {cohort: np.stack([smooth(tr) for tr in projection[cohort]]) for cohort in COHORT_COLOR}
        square_limits(ax, np.concatenate([s.reshape(-1, 2) for s in smoothed.values()]), 0.02, (2, 98))
        for cohort, color in COHORT_COLOR.items():
            for trajectory in smoothed[cohort]:
                ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.28, linewidth=0.9, zorder=3,
                        solid_capstyle="round")
            mean = smoothed[cohort].mean(axis=0)
            points = mean.reshape(-1, 1, 2)
            band = LineCollection(np.concatenate([points[:-1], points[1:]], axis=1), cmap="viridis",
                                  norm=plt.Normalize(0, len(mean) - 1), linewidth=5.0, zorder=8, capstyle="round")
            band.set_array(np.arange(len(mean) - 1, dtype=float))
            ax.add_collection(band)
            ax.scatter(*mean[0], s=240, marker="o", facecolor="white", edgecolor=color, linewidth=2.5, zorder=10)
            ax.scatter(*mean[-1], s=380, marker="*", facecolor=color, edgecolor="black", linewidth=1.4, zorder=10,
                       label=f"{UMAP_LEGEND[cohort]} (n={len(smoothed[cohort])})")
        if key == PANEL_ORDER[0]:
            ax.legend(loc="best", fontsize=18, framealpha=0.95, edgecolor="#666666", fancybox=True)
    n_hours = projections[PANEL_ORDER[0]]["stable"].shape[1]
    bar = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, n_hours - 1)),
                       cax=fig.add_axes([0.925, 0.20, 0.012, 0.60]))
    bar.set_label(f"Hour in {n_hours}-h admission window", fontsize=20, fontweight="bold")
    bar.ax.tick_params(labelsize=18)
    bar.outline.set_edgecolor("#333333")
    fig.subplots_adjust(top=0.92, right=0.91, wspace=0.07, left=0.05)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_umap_density(projections: dict, labels: dict, path: Path) -> None:
    fig, axes = umap_axes(labels)
    for ax, key in zip(axes, PANEL_ORDER):
        projection = projections[key]
        square_limits(ax, np.concatenate([projection[c].reshape(-1, 2) for c in COHORT_COLOR]), 0.06, (0, 100))
        grid_x, grid_y = np.meshgrid(np.linspace(*ax.get_xlim(), 120), np.linspace(*ax.get_ylim(), 120))
        for cohort, color in COHORT_COLOR.items():
            first, last = projection[cohort][:, 0], projection[cohort][:, -1]
            for points, line_style, fill, width in ((first, ":", 0.0, 1.6), (last, "-", 0.22, 1.8)):
                density = gaussian_kde(points.T, bw_method=0.35)(np.vstack([grid_x.ravel(), grid_y.ravel()]))
                density = density.reshape(grid_x.shape)
                contours = ax.contour(grid_x, grid_y, density, levels=4, colors=color, alpha=0.88,
                                      linestyles=line_style, linewidths=width, zorder=4)
                if fill:
                    levels = np.asarray(contours.levels)
                    ax.contourf(grid_x, grid_y, density, levels=levels[levels > 1e-3 * density.max()],
                                colors=[color], alpha=fill, extend="max", zorder=2)
            ax.annotate("", xy=last.mean(axis=0), xytext=first.mean(axis=0), zorder=8,
                        arrowprops=dict(arrowstyle="-|>,head_width=0.7,head_length=0.9", color=color, lw=3.0,
                                        alpha=0.95))
            ax.scatter(*first.mean(axis=0), s=240, marker="o", facecolor="white", edgecolor=color, linewidth=2.5,
                       zorder=10)
            ax.scatter(*last.mean(axis=0), s=380, marker="*", facecolor=color, edgecolor="white", linewidth=2.0,
                       zorder=10, label=f"{UMAP_LEGEND[cohort]} (n={len(last)})")
        if key == PANEL_ORDER[0]:
            ax.legend(loc="upper right", fontsize=16, framealpha=0.95, edgecolor="#666666", fancybox=True)
    fig.subplots_adjust(top=0.92, wspace=0.10)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def text_numbers(results: dict, cohorts: dict) -> dict:
    """Quantities the paper states in running text."""
    labels = np.array(cohorts["labels"]["main"])
    statements = np.array([r["statements"] for r in cohorts["eligible"]])[labels >= 0]
    out = {
        "eligible": len(labels), "deteriorating": int((labels == 1).sum()), "stable": int((labels == 0).sum()),
        "state_text_statements_h0_h1_h71": statements.mean(0).tolist(),
    }
    for key, result in results.items():
        r, d = result["configurations"][MAIN], result["direction"]
        out[key] = {
            "p_value": r["net_displacement"]["p_value"], "pooled_sd": r["net_displacement"]["pooled_sd"],
            "divergence_h0_h71": [r["divergence"][0], r["divergence"][-1]],
            "centroid_accuracy_h0_h24_h71": [r["centroid_accuracy"][h] for h in (0, 24, 71)],
            "direction_accuracy_h1_h24_h71": [d["accuracy"][h - 1] for h in (1, 24, 71)],
            "direction_cosine": d["cosine"],
            "knn10_purity_h71": r["knn_purity"]["10"][-1], "knn_chance": r["knn_chance"],
            "from_start_h1": [r["from_start"]["deteriorating"][1], r["from_start"]["stable"][1]],
            "pca_explained_variance": result["pca_explained_variance"],
            "null_direction_cosine_mean": float(np.mean([row["direction_cosine_last"] for row in result["null"]])),
            "null_direction_accuracy_mean": float(np.mean([row["direction_accuracy_last"] for row in result["null"]])),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/geometry.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["output_dir"])
    labels = {key: entry["label"] for key, entry in config["encoders"].items()}
    results = {key: json.loads((root / "metrics" / f"{key}.json").read_text()) for key in labels}
    cohorts = json.loads((root / "cohorts.json").read_text())

    tables, figures = ensure_dir(root / "tables"), ensure_dir(root / "figures")
    (tables / "Table3_geometry.tex").write_text(table_main(results, labels))
    (tables / "AppTable4_geometry_robust.tex").write_text(table_robustness(results, labels))
    (tables / "text_numbers.json").write_text(json.dumps(text_numbers(results, cohorts), indent=2))
    projections = {key: np.load(root / "umap" / f"{key}.npz") for key in labels}
    with plt.rc_context(UMAP_STYLE):
        figure_umap_trajectories(projections, labels, figures / "Fig3a_trajectories.pdf")
        figure_umap_density(projections, labels, figures / "Fig3b_density.pdf")
    set_style()
    figure_net_displacement(results, labels, figures / "Fig3c_net_displacement.pdf")
    figure_curves(results, labels, figures / "Fig3d_geometry_curves.pdf")
    figure_knn(results, labels, figures / "AppFig2_knn_purity.pdf")
    figure_from_start(results, labels, figures / "AppFig3_from_start.pdf")


if __name__ == "__main__":
    main()
