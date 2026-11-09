"""Render sampled clinical tasks as paper-style, method-by-task comparison tables."""
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

SHORT_TITLES = {
    "circ_failure_8h": "循环衰竭\n8h",
    "resp_failure_24h": "呼吸衰竭\n24h",
    "kidney_failure_48h": "肾衰竭\n48h",
    "liver_failure_48h": "肝衰竭\n48h",
    "hyperglycemia_8h": "高血糖\n8h",
    "sepsis3_8h": "脓毒症发生\n8h",
    "decompensation_24h": "ICU 死亡\n24h",
    "composite_deterioration_24h": "复合恶化\n24h",
    "vaso_onset_8h": "升压药启用\n8h",
    "vaso_onset_24h": "升压药启用\n24h",
    "vaso_escalation_24h": "升压药升级\n24h",
    "vaso_liberation_24h": "升压药停用\n24h",
    "imv_onset_24h": "有创通气开始\n24h",
    "imv_liberation_48h": "有创通气撤离\n48h",
    "rrt_onset_48h": "肾脏替代开始\n48h",
    "sofa_worsen_2_48h": "SOFA 恶化\n至少 2 分 / 48h",
    "sofa_improve_2_48h": "SOFA 改善\n至少 2 分 / 48h",
    "transfusion_24h": "输血\n24h",
    "prolonged_imv_96h": "通气时长\n至少 96h",
    "tracheostomy_ever": "气管切开\n",
    "extubation_failure_48h": "拔管失败\n48h",
    "new_rrt_after_24h": "新发肾脏替代\n24h 后",
    "aki_kdigo23_after_24h": "AKI 2-3 期\n24h 后",
    "discharge_home": "出院回家\n",
    "los_gt_14d": "ICU 停留\n超过 14 天",
    "vasopressor_free_7d": "未用升压药\n第 2-8 天",
    "hospital_mortality": "住院死亡\n",
    "icu_mortality": "ICU 死亡\n",
    "mortality_7d": "7 天死亡\n",
    "mortality_14d": "14 天死亡\n",
    "mortality_30d": "30 天死亡\n",
    "mortality_90d": "90 天死亡\n",
    "los_gt_7d": "ICU 停留\n超过 7 天",
    "sepsis_ever": "住院期间\n脓毒症",
}

METHODS = ("history_32", "history_64", "history_32_plus_future_32")
METHOD_LABELS = ("历史 32", "历史 64", "历史 32 + 预测 32")
SPECS = [
    ("08_tasks_e1", "E1  急性恶化", "E1", 0, 8),
    ("09_tasks_e2_a", "E2  治疗及状态变化（1/2）", "E2", 0, 9),
    ("10_tasks_e2_b", "E2  治疗及状态变化（2/2）", "E2", 9, 18),
    ("11_tasks_e3", "E3  住院结局", "E3", 0, 8),
]


def render(rows, title, out, stem):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    left, right, task_start = .016, .984, .237
    task_step = (right - task_start) / len(rows)
    centers = [task_start + (i + .5) * task_step for i in range(len(rows))]
    ax.text(left, .965, title, fontsize=15, fontweight="bold", va="top")
    ax.text(right, .965, "测试集结果", fontsize=11, ha="right", va="top", color="#333333")
    ax.plot([left,right],[.898,.898],color="#222222",lw=1.1)

    ax.text(left+.005, .815, "方法", fontsize=11.4, va="center", fontweight="bold")
    ax.text(.19, .815, "指标", fontsize=11.4, va="center", ha="center", fontweight="bold")
    for center, row in zip(centers, rows):
        ax.text(center, .835, SHORT_TITLES[row["task"]], ha="center", va="center",
                fontsize=10.4, linespacing=1.18)

    ax.text(left+.005, .725, "测试患者（阳性）", fontsize=10.2, va="center")
    for center, row in zip(centers, rows):
        flag = "*" if min(row["positive_patients"], row["negative_patients"]) < 10 else ""
        ax.text(center, .725, f"{row['patients']} ({row['positive_patients']}){flag}",
                ha="center", va="center", fontsize=9.9)
    ax.plot([left,right],[.675,.675],color="#222222",lw=.9)

    top, step = .675, .069
    # Shade the proposed comparison row, without implying it is the best.
    ax.add_patch(patches.Rectangle((left, top-6*step), right-left, 2*step,
                 facecolor="#e8f0f7", edgecolor="none", zorder=0))
    for j,(method,label) in enumerate(zip(METHODS,METHOD_LABELS)):
        method_center = top - (2*j+1)*step
        ax.text(left+.005, method_center, label, fontsize=11.3, va="center",
                fontweight="bold" if j==2 else "normal")
        for k,metric in enumerate(("auroc","auprc")):
            y = top - (2*j+k+.5)*step
            ax.text(.19, y, "AUROC" if metric=="auroc" else "AUPRC",
                    ha="center", va="center", fontsize=10.5)
            for x,row in zip(centers,rows):
                value = row["scores"][method][metric]
                # Bold ties at the displayed precision. These are point estimates only.
                displayed_max = max(round(row["scores"][m][metric],3) for m in METHODS)
                best = round(value,3) == displayed_max
                ax.text(x,y,f"{value:.3f}",ha="center",va="center",fontsize=11.1,
                        fontfamily="DejaVu Serif", fontweight="bold" if best else "normal")
        if j<2:
            yy=top-2*(j+1)*step
            ax.plot([left,right],[yy,yy],color="#aaaaaa",lw=.45)
    ax.plot([left,right],[top-6*step,top-6*step],color="#222222",lw=1.1)
    ax.text(left, .175, "粗体为三种方案的最高点估计，不代表统计显著；浅蓝色行表示加入预测状态。",
            fontsize=10, va="center", color="#333333")
    ax.text(left, .112, "* 测试集中任一类别患者少于 10 人；部分患者贡献多个窗口。",
            fontsize=10, va="center", color="#333333")
    ax.text(left, .049, "预测状态按无干预假设生成，住院级任务使用 48h 预测特征；置信区间见完整结果表。",
            fontsize=10, va="center", color="#333333")
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
    if len(rows) != 34 or set(TITLES) != {row["task"] for row in rows} or set(SHORT_TITLES) != set(TITLES):
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
