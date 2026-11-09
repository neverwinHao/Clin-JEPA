"""Render 34 sampled clinical task results as compact, presentation-ready tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches

FONT = "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=FONT).get_name(),
    "axes.unicode_minus": False,
    "svg.fonttype": "path",
    "savefig.facecolor": "white",
})

TITLES = {
    "circ_failure_8h": "循环衰竭（8h）",
    "resp_failure_24h": "呼吸衰竭（24h）",
    "kidney_failure_48h": "肾衰竭（48h）",
    "liver_failure_48h": "肝衰竭（48h）",
    "hyperglycemia_8h": "高血糖（8h）",
    "sepsis3_8h": "脓毒症发生（8h）",
    "decompensation_24h": "ICU 死亡（24h）",
    "composite_deterioration_24h": "复合恶化（24h）",
    "vaso_onset_8h": "升压药启用（8h）",
    "vaso_onset_24h": "升压药启用（24h）",
    "vaso_escalation_24h": "升压药升级（24h）",
    "vaso_liberation_24h": "升压药停用（24h）",
    "imv_onset_24h": "有创通气开始（24h）",
    "imv_liberation_48h": "有创通气撤离（48h）",
    "rrt_onset_48h": "肾脏替代治疗开始（48h）",
    "sofa_worsen_2_48h": "SOFA 恶化至少 2 分（48h）",
    "sofa_improve_2_48h": "SOFA 改善至少 2 分（48h）",
    "transfusion_24h": "输血（24h）",
    "prolonged_imv_96h": "通气时长至少 96h",
    "tracheostomy_ever": "气管切开",
    "extubation_failure_48h": "拔管失败（48h）",
    "new_rrt_after_24h": "24h 后新发肾脏替代治疗",
    "aki_kdigo23_after_24h": "24h 后 AKI 2-3 期",
    "discharge_home": "出院回家",
    "los_gt_14d": "ICU 停留超过 14 天",
    "vasopressor_free_7d": "第 2-8 天未用升压药",
    "hospital_mortality": "住院死亡",
    "icu_mortality": "ICU 死亡",
    "mortality_7d": "7 天死亡",
    "mortality_14d": "14 天死亡",
    "mortality_30d": "30 天死亡",
    "mortality_90d": "90 天死亡",
    "los_gt_7d": "ICU 停留超过 7 天",
    "sepsis_ever": "住院期间脓毒症",
}

METHODS = ("history_32", "history_64", "history_32_plus_future_32")
SPECS = [
    ("08_tasks_e1", "E1  急性恶化", "E1", 0, 8),
    ("09_tasks_e2_a", "E2  治疗及状态变化（1/2）", "E2", 0, 9),
    ("10_tasks_e2_b", "E2  治疗及状态变化（2/2）", "E2", 9, 18),
    ("11_tasks_e3", "E3  住院结局", "E3", 0, 8),
]


def render(rows, title, out, stem):
    n = len(rows)
    fig, ax = plt.subplots(figsize=(13.1, 4.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(.015, .96, title, fontsize=15, fontweight="bold", va="top")
    ax.text(.985, .96, "测试集结果", fontsize=11, ha="right", va="top", color="#333333")

    xbounds = [0.015, .290, .400, .535, .670, .815, .985]
    centers = [(xbounds[i] + xbounds[i+1]) / 2 for i in range(len(xbounds)-1)]
    header_y = .835
    headings = ["下游任务", "患者数\n（阳性）", "历史 32\nAUROC / AUPRC", "历史 64\nAUROC / AUPRC",
                "历史 32 + 预测 32\nAUROC / AUPRC", "ΔAUROC\n（95% CI）"]
    ax.axhline(.895, xmin=.015, xmax=.985, color="#222222", lw=1.1)
    ax.axhline(.770, xmin=.015, xmax=.985, color="#222222", lw=.9)
    for i, heading in enumerate(headings):
        x = xbounds[0]+.008 if i == 0 else centers[i]
        ax.text(x, header_y, heading, ha="left" if i == 0 else "center",
                va="center", fontsize=11.2, fontweight="bold", linespacing=1.15)

    bottom = .205
    step = (.770-bottom) / n
    for i, row in enumerate(rows):
        top = .770-i*step
        y = top-step/2
        if i % 2:
            ax.add_patch(patches.Rectangle((.015, top-step), .970, step,
                         facecolor="#f5f5f5", edgecolor="none", zorder=0))
        ax.add_patch(patches.Rectangle((xbounds[4], top-step), xbounds[5]-xbounds[4], step,
                     facecolor="#e8f0f7" if i%2==0 else "#e2ebf3", edgecolor="none", zorder=0))
        flag = "*" if min(row["positive_patients"], row["negative_patients"]) < 10 else ""
        counts = f"{row['patients']}（{row['positive_patients']}）{flag}"
        metric = [f"{row['scores'][k]['auroc']:.3f} / {row['scores'][k]['auprc']:.3f}" for k in METHODS]
        lo, hi = row["delta_auroc_ci95"]
        delta = f"{row['delta_auroc']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        values = [TITLES[row["task"]], counts, *metric, delta]
        for j, value in enumerate(values):
            x = xbounds[0]+.008 if j == 0 else centers[j]
            ax.text(x,y,value,ha="left" if j == 0 else "center",va="center",fontsize=10.6)
    ax.axhline(bottom, xmin=.015, xmax=.985, color="#222222", lw=1.1)
    ax.text(.015, .145, "* 测试集中任一类别患者少于 10 人；部分患者贡献多个窗口。",
            fontsize=9.7, va="center", color="#444444")
    ax.text(.015, .085, "ΔAUROC =（历史 32 + 预测 32）− 历史 32；区间为患者级 bootstrap 95% CI（500 次）。",
            fontsize=9.7, va="center", color="#444444")
    ax.text(.015, .025, "预测状态基于无干预假设；住院级任务使用 48h 模拟未来。",
            fontsize=9.7, va="center", color="#444444")
    fig.subplots_adjust(left=.01,right=.99,top=.98,bottom=.02)
    for ext in ("png","svg"):
        path=out / (stem + "." + ext)
        fig.savefig(path,dpi=240,bbox_inches="tight",pad_inches=.05)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    rows = data["downstream"]["tasks"]
    if len(rows) != 34 or set(TITLES) != {row["task"] for row in rows}:
        raise ValueError("The task names or number do not match the sampled 34-task evaluation")
    args.output.mkdir(parents=True,exist_ok=True)
    for stem,title,group,start,end in SPECS:
        group_rows=[row for row in rows if row["group"]==group]
        selected=group_rows[start:end]
        if len(selected) != end-start:
            raise ValueError(f"Incorrect number of results for {stem}")
        render(selected,title,args.output,stem)
        print(f"Created {stem}: {len(selected)} tasks",flush=True)


if __name__ == "__main__":
    main()
