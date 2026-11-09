"""Export aggregate-only incremental-value results and paper-style tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.create_task_tables import TITLES, SPECS, render
from scripts.evaluate_predictor_increment import METHODS

LABELS = ("历史 64", "历史 64 + 预测 32", "历史 64 + 保持 32")
GROUPS = {"E1": "急性恶化", "E2": "治疗及状态变化", "E3": "住院结局"}


def aggregate(data):
    tasks = []
    for row in data["tasks"]:
        if row["status"] != "ok":
            raise ValueError(f"Task not evaluated: {row['task']}")
        counts = row["counts"]["test"]
        tasks.append({
            "task": row["task"], "group": row["benchmark"], "title": row["title"],
            "patients": int(counts["patients"]), "positive_patients": int(counts["positive_patients"]),
            "negative_patients": int(counts["negative_patients"]),
            "scores": {m: {"auroc": row["models"][m]["test_auroc"],
                           "auprc": row["models"][m]["test_auprc"],
                           "auroc_ci95": row["models"][m]["test_auroc_ci95"],
                           "auprc_ci95": row["models"][m]["test_auprc_ci95"]}
                       for m in METHODS},
            "comparisons": row["comparisons"],
            "warnings": row["warnings"],
        })
    if len(tasks) != 34 or set(TITLES) != {r["task"] for r in tasks}:
        raise ValueError("Expected 34 named clinical task summaries")
    groups = []
    for group in ("E1","E2","E3"):
        subset = [r for r in tasks if r["group"] == group]
        scores = {m: {metric: float(np.mean([r["scores"][m][metric] for r in subset]))
                      for metric in ("auroc","auprc")} for m in METHODS}
        groups.append({"group": group, "tasks": len(subset), "scores": scores})
    return {"protocol": data["protocol"], "groups": groups, "tasks": tasks}


def pair_text(row, name):
    comp = row["comparisons"][name]["auroc"]
    lo, hi = comp["ci95"]
    return f"{comp['difference']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--doc",type=Path,required=True)
    parser.add_argument("--assets",type=Path,required=True)
    args = parser.parse_args()
    summary = aggregate(json.loads(args.results.read_text()))
    args.assets.mkdir(parents=True,exist_ok=True)
    args.doc.parent.mkdir(parents=True,exist_ok=True)
    (args.assets/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    figs = []
    for stem,title,group,start,end in SPECS:
        selected=[r for r in summary["tasks"] if r["group"]==group][start:end]
        figure_stem=stem.replace("_tasks_", "_increment_")
        render(selected,title,args.assets,figure_stem,
               methods=METHODS,method_labels=LABELS,highlight_index=1)
        figs.append((title,figure_stem))

    rows=summary["tasks"]
    better_history=sum(r["comparisons"]["prediction_vs_history"]["auroc"]["difference"]>0 for r in rows)
    better_copy=sum(r["comparisons"]["prediction_vs_copy"]["auroc"]["difference"]>0 for r in rows)
    cross_history=sum(r["comparisons"]["prediction_vs_history"]["auroc"]["ci95"][0] <= 0 <=
                      r["comparisons"]["prediction_vs_history"]["auroc"]["ci95"][1] for r in rows)
    mortality=next(r for r in rows if r["task"]=="hospital_mortality")
    m_auc=[mortality["scores"][m]["auroc"] for m in METHODS]
    m_pair=mortality["comparisons"]["prediction_vs_history"]["auroc"]
    m_lo,m_hi=m_pair["ci95"]
    text = [
        "# 冻结基座实验：预测状态有没有帮助",
        "",
        "在同一批患者和同一 64 维历史特征上，比较加入预测状态与加入简单外推状态。"
        "这是小样本流程验证，不是原论文模型的完整复现，也不是临床部署结果。",
        "",
        "## 对照方式",
        "",
        "1. **历史 64**：历史状态、历史治疗和人口学特征降到 64 维，训练逻辑回归。",
        "2. **历史 64 + 预测 32**：保留相同历史特征，加上预测器生成的未来状态的 32 维特征。",
        "3. **历史 64 + 保持 32**：保留相同历史特征，假设最后一个已观测状态在未来保持不变，"
        "从这条简单轨迹生成 32 维特征。",
        "",
        "预测和保持轨迹都按均值、标准差、末状态汇总。"
        "预测未来特征沿用原实验按预测时长分别在训练集拟合的 PCA；保持不变特征另用训练集拟合 PCA。"
        "这两组都是 96 维输入，虽然特征变换不同。编码器与预测器均冻结，不重新训练。",
        "",
        "预测器使用已观测历史，之后的治疗设为“无干预”；未用实际未来治疗或观察。"
        "这是人工假设，不是治疗建议或已验证的因果反事实。"
        "住院级任务使用 48 小时预测，先前单独的死亡率实验使用 24 小时预测。",
        "",
        "三个对照的逻辑回归均用同一患者划分，正则参数独立按验证集 AUROC 选择。"
        "每项使用相同测试患者和窗口；按患者配对重采样 2,000 次计算差值区间。"
        "若某次重采样只有一种标签，则该次指标无效并跳过。区间以当前训练好的模型和单个随机种子为条件，不包括重新训练带来的波动。",
        "",
        "## 各任务组平均",
        "",
        "每项任务等权平均；任务之间阳性比例和样本量不同，不把组均值当作显著性检验。",
        "",
        "| 任务组 | 历史 64 | 历史 64 + 预测 32 | 历史 64 + 保持 32 |",
        "|---|---:|---:|---:|",
    ]
    for g in summary["groups"]:
        cells=[f"{g['scores'][m]['auroc']:.3f} / {g['scores'][m]['auprc']:.3f}" for m in METHODS]
        text.append(f"| {g['group']} {GROUPS[g['group']]}（{g['tasks']} 项） | " + " | ".join(cells) + " |")
    text += [
        "",
        "表内数字为 **AUROC / AUPRC**；AUPRC 使用 average precision。",
        f"相对历史 64，加入预测状态后有 {better_history}/34 项 AUROC 点估计更高；"
        f"有 {cross_history}/34 项的配对差值 95% 区间包含 0。"
        f"相对保持不变，有 {better_copy}/34 项 AUROC 点估计更高。"
        "这些任务存在多重比较，点估计和单项未校正区间不等于临床效果证据。",
        "",
        f"例如，住院死亡的 AUROC：历史 64 为 {m_auc[0]:.3f}，加入预测为 {m_auc[1]:.3f}，加入保持不变为 {m_auc[2]:.3f}。"
        f"预测相对历史的差值为 {m_pair['difference']:+.3f}，95% CI [{m_lo:+.3f}, {m_hi:+.3f}]；"
        f"测试集为 {mortality['patients']} 名患者、{mortality['positive_patients']} 个阳性，不能断言有益或有害。",
        "",
        "## 按任务展开",
        "",
        "图片按组会中的论文结果表排版：方法在行，任务在列；浅蓝表示加入预测状态，"
        "粗体为三组中的最高点估计，不代表统计显著。点击 PNG 或 SVG 链接可单独使用。",
        "",
    ]
    for title,stem in figs:
        image=f"assets/predictor-increment/{stem}"
        text += [f"### {title}", "", f"![{title}]({image}.png)", "",
                 f"[PNG]({image}.png) · [SVG]({image}.svg)", ""]
    text += [
        "## 完整数值表",
        "",
        "每格为测试集 AUROC / AUPRC；括号内为阳性患者数。"
        "* 表示某一类别患者少于 10 人，结果尤其不稳定。"
        "两个差值均为配对患者级 bootstrap 95% 区间。",
        "",
        "| 任务 | 测试患者（阳性） | 历史 64 | 历史 64 + 预测 32 | 历史 64 + 保持 32 |"
        " ΔAUROC：预测−历史 [95% CI] | ΔAUROC：预测−保持 [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("E1","E2","E3"):
        text.append(f"| **{group} {GROUPS[group]}** | | | | | | |")
        for r in rows:
            if r["group"] != group:
                continue
            flag="*" if min(r["positive_patients"],r["negative_patients"])<10 else ""
            count=f"{r['patients']} ({r['positive_patients']}){flag}"
            metrics=[f"{r['scores'][m]['auroc']:.3f} / {r['scores'][m]['auprc']:.3f}" for m in METHODS]
            text.append("| " + " | ".join([TITLES[r["task"]], count, *metrics,
                         pair_text(r,"prediction_vs_history"), pair_text(r,"prediction_vs_copy")]) + " |")
    text += [
        "",
        "图表使用的 [汇总 JSON](assets/predictor-increment/summary.json) 不包含患者级记录。"
        "患者级预测和模型留在服务器的 "
        "`/media/cuichang/liuhao/clin-jepa/sample_3h/clinical_data/all_tasks/predictor_increment_20261109/`。",
        "",
        "数据处理与实验来源见 [前一版 34 项结果](13-all-downstream-results.md)。",
        "",
        "## 运行",
        "",
        "```bash",
        "cd /home/cuichang/liuhao/Clin-JEPA",
        "set -a; source .env; set +a",
        "EXP=/media/cuichang/liuhao/clin-jepa/sample_3h",
        "OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \"$CLIN_JEPA_PYTHON\" -m scripts.evaluate_predictor_increment \\",
        "  --experiment \"$EXP\" --output \"$EXP/clinical_data/all_tasks/predictor_increment_new\" --bootstrap 2000",
        "\"$CLIN_JEPA_PYTHON\" -m scripts.write_predictor_increment_report \\",
        "  --results \"$EXP/clinical_data/all_tasks/predictor_increment_new/results.json\" \\",
        "  --doc docs/17-predictor-increment-results.md --assets docs/assets/predictor-increment",
        "```",
        "",
    ]
    args.doc.write_text("\n".join(text))
    print(f"Created 34-task report: {args.doc}",flush=True)


if __name__ == "__main__":
    main()
