"""Create an aggregate-only Chinese Markdown report from the clinical task results."""
import argparse
import json
from pathlib import Path

NAMES = {"circ_failure_8h":"\u5faa\u73af\u8870\u7aed\uff088h\uff09","resp_failure_24h":"\u547c\u5438\u8870\u7aed\uff0824h\uff09","kidney_failure_48h":"\u80be\u8870\u7aed\uff0848h\uff09","liver_failure_48h":"\u809d\u8870\u7aed\uff0848h\uff09","hyperglycemia_8h":"\u9ad8\u8840\u7cd6\uff088h\uff09","sepsis3_8h":"\u8113\u6bd2\u75c7\u53d1\u751f\uff088h\uff09","decompensation_24h":"ICU \u6b7b\u4ea1\uff0824h\uff09","composite_deterioration_24h":"\u590d\u5408\u6076\u5316\uff0824h\uff09","vaso_onset_8h":"\u5347\u538b\u836f\u542f\u7528\uff088h\uff09","vaso_onset_24h":"\u5347\u538b\u836f\u542f\u7528\uff0824h\uff09","vaso_escalation_24h":"\u5347\u538b\u836f\u5347\u7ea7\uff0824h\uff09","vaso_liberation_24h":"\u5347\u538b\u836f\u505c\u7528\uff0824h\uff09","imv_onset_24h":"\u6709\u521b\u901a\u6c14\u5f00\u59cb\uff0824h\uff09","imv_liberation_48h":"\u6709\u521b\u901a\u6c14\u64a4\u79bb\uff0848h\uff09","rrt_onset_48h":"\u80be\u810f\u66ff\u4ee3\u6cbb\u7597\u5f00\u59cb\uff0848h\uff09","sofa_worsen_2_48h":"SOFA \u6076\u5316\u81f3\u5c11 2 \u5206\uff0848h\uff09","sofa_improve_2_48h":"SOFA \u6539\u5584\u81f3\u5c11 2 \u5206\uff0848h\uff09","transfusion_24h":"\u8f93\u8840\uff0824h\uff09","prolonged_imv_96h":"\u901a\u6c14\u65f6\u957f\u81f3\u5c11 96h","tracheostomy_ever":"\u6c14\u7ba1\u5207\u5f00","extubation_failure_48h":"\u62d4\u7ba1\u5931\u8d25\uff0848h\uff09","new_rrt_after_24h":"24h \u540e\u65b0\u53d1\u80be\u810f\u66ff\u4ee3\u6cbb\u7597","aki_kdigo23_after_24h":"24h \u540e AKI 2-3 \u671f","discharge_home":"\u51fa\u9662\u56de\u5bb6","los_gt_14d":"ICU \u505c\u7559\u8d85\u8fc7 14 \u5929","vasopressor_free_7d":"\u7b2c 2-8 \u5929\u672a\u7528\u5347\u538b\u836f","hospital_mortality":"\u4f4f\u9662\u6b7b\u4ea1","icu_mortality":"ICU \u6b7b\u4ea1","mortality_7d":"7 \u5929\u6b7b\u4ea1","mortality_14d":"14 \u5929\u6b7b\u4ea1","mortality_30d":"30 \u5929\u6b7b\u4ea1","mortality_90d":"90 \u5929\u6b7b\u4ea1","los_gt_7d":"ICU \u505c\u7559\u8d85\u8fc7 7 \u5929","sepsis_ever":"\u4f4f\u9662\u671f\u95f4\u8113\u6bd2\u75c7"}
TITLES = {"E1":"E1 \u6025\u6027\u6076\u5316\uff088 \u9879\uff09","E2":"E2 \u6cbb\u7597\u53ca\u72b6\u6001\u53d8\u5316\uff0818 \u9879\uff09","E3":"E3 \u4f4f\u9662\u7ed3\u5c40\uff088 \u9879\uff09"}
METHODS = ("history_32", "history_64", "history_32_plus_future_32")


def metric(task, method):
    m = task["models"][method]
    return f"{m['test_auroc']:.3f} / {m['test_auprc']:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    tasks = data["tasks"]
    if len(tasks) != 34 or set(NAMES) != {t["task"] for t in tasks}:
        raise ValueError("Expected the 34 released downstream tasks")
    lines = [
        "# \u51bb\u7ed3 Qwen \u5c0f\u6837\u672c\uff1a34 \u9879\u4e0b\u6e38\u4efb\u52a1\u7ed3\u679c",
        "",
        "\u8fd9\u662f\u6d41\u7a0b\u9a8c\u8bc1\u548c\u63a2\u7d22\u6027\u5bf9\u7167\uff0c\u4e0d\u662f\u8bba\u6587\u7684\u5168\u91cf\u590d\u73b0\u6216\u53ef\u7528\u7684\u4e34\u5e8a\u9884\u6d4b\u6a21\u578b\u3002",
        "\u4efb\u52a1\u548c\u6807\u7b7e\u6765\u81ea\u4f5c\u8005\u4ee3\u7801\uff1b\u4f7f\u7528\u51bb\u7ed3\u7684 Qwen \u57fa\u5ea7\u548c\u5df2\u8bad\u7ec3\u9884\u6d4b\u5668\uff0c\u60a3\u8005\u6309\u539f\u5212\u5206\u62bd\u6837\uff0c\u8bad\u7ec3/\u9a8c\u8bc1/\u6d4b\u8bd5\u60a3\u8005\u4e92\u4e0d\u91cd\u53e0\u3002",
        "\u524d\u4e09\u4e2a\u5bf9\u7167\u5206\u522b\u4e3a\uff1a\u5386\u53f2\u4fe1\u606f 32 \u7ef4\u3001\u5386\u53f2\u4fe1\u606f 64 \u7ef4\u3001\u5386\u53f2\u4fe1\u606f 32 \u7ef4\u52a0\u9884\u6d4b\u672a\u6765\u72b6\u6001 32 \u7ef4\uff1b\u5206\u7c7b\u5668\u4e3a\u903b\u8f91\u56de\u5f52\uff0cPCA \u548c\u6807\u51c6\u5316\u53ea\u5728\u8bad\u7ec3\u96c6\u62df\u5408\uff0c\u8d85\u53c2\u6570\u7531\u9a8c\u8bc1\u96c6\u9009\u62e9\u3002",
        "\u9884\u6d4b\u672a\u6765\u65f6\uff0c24 \u5c0f\u65f6\u4e4b\u540e\u5047\u8bbe\u65e0\u5e72\u9884\uff0c\u672a\u4f7f\u7528\u771f\u5b9e\u672a\u6765\u68c0\u67e5\u6216\u6cbb\u7597\uff1b\u8fd9\u662f\u53ef\u80fd\u4e0d\u7b26\u5408\u4e34\u5e8a\u5b9e\u9645\u7684\u4eba\u5de5\u5047\u8bbe\uff0c\u4e0d\u662f\u6cbb\u7597\u5efa\u8bae\u6216\u5df2\u9a8c\u8bc1\u7684\u56e0\u679c\u53cd\u4e8b\u5b9e\u3002",
        "\u6bcf\u9879\u6307\u6807\u4e3a\u6d4b\u8bd5\u96c6 AUROC / AUPRC\uff1b\u540e\u8005\u662f average precision\u3002\u7f6e\u4fe1\u533a\u95f4\u6309\u60a3\u8005\u91cd\u91c7\u6837 500 \u6b21\u3002",
        "\u4f4f\u9662\u7ea7\u4efb\u52a1\u5728\u672c\u8868\u4f7f\u7528 48 \u5c0f\u65f6\u9884\u6d4b\u7279\u5f81\uff1b\u5148\u524d\u5355\u72ec\u6b7b\u4ea1\u7387\u5b9e\u9a8c\u4f7f\u7528 24 \u5c0f\u65f6\uff0c\u56e0\u6b64\u52a0\u9884\u6d4b\u72b6\u6001\u7684\u5206\u6570\u53ef\u7565\u6709\u4e0d\u540c\u3002",
        "",
        "## \u6309\u4efb\u52a1\u7ec4\u53d6\u7b80\u5355\u5e73\u5747",
        "",
        "\u5e73\u5747\u5bf9\u6bcf\u4e2a\u4efb\u52a1\u7b49\u6743\uff0c\u4e0d\u8003\u8651\u4efb\u52a1\u9633\u6027\u7387\u5dee\u5f02\uff0c\u4ec5\u7528\u4e8e\u63a2\u7d22\u3002",
        "",
        "| \u4efb\u52a1\u7ec4 | \u5386\u53f2 32 | \u5386\u53f2 64 | \u5386\u53f2 32 + \u9884\u6d4b 32 |",
        "|---|---:|---:|---:|",
    ]
    for bench in ("E1", "E2", "E3"):
        subset = [t for t in tasks if t["benchmark"] == bench and t["status"] == "ok"]
        values = []
        for method in METHODS:
            auc = sum(t["models"][method]["test_auroc"] for t in subset) / len(subset)
            ap = sum(t["models"][method]["test_auprc"] for t in subset) / len(subset)
            values.append(f"{auc:.3f} / {ap:.3f}")
        lines.append(f"| {TITLES[bench]} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## \u6bcf\u9879\u7ed3\u679c",
        "",
        "\u5217\u4e2d\u7684\u6570\u5b57\u4e3a AUROC / AUPRC\u3002\u5e26 * \u7684\u4efb\u52a1\u5728\u6d4b\u8bd5\u96c6\u9633\u6027\u6216\u9634\u6027\u60a3\u8005\u5c11\u4e8e 10 \u4eba\uff0c\u7ed3\u679c\u5c24\u5176\u4e0d\u7a33\u5b9a\uff1b\u591a\u4e2a\u91cd\u53e0\u7a97\u53e3\u53ef\u80fd\u6765\u81ea\u540c\u4e00\u60a3\u8005\u3002",
        "\u6700\u540e\u4e00\u5217\u662f\u52a0\u5165\u9884\u6d4b\u72b6\u6001\u76f8\u5bf9\u5386\u53f2 32 \u7684 AUROC \u5dee\u503c\u53ca\u60a3\u8005\u7ea7 95% \u7f6e\u4fe1\u533a\u95f4\u3002",
        "",
        "| \u4efb\u52a1 | \u6d4b\u8bd5\u60a3\u8005\uff08\u9633\u6027\uff09 | \u5386\u53f2 32 | \u5386\u53f2 64 | \u5386\u53f2 32 + \u9884\u6d4b 32 | \u0394AUROC [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bench in ("E1", "E2", "E3"):
        lines.append(f"| **{TITLES[bench]}** | | | | | |")
        for task in (t for t in tasks if t["benchmark"] == bench):
            if task["status"] != "ok":
                lines.append(f"| {NAMES[task['task']]} | - | - | - | - | {task['status']} |")
                continue
            c = task["counts"]["test"]
            star = "*" if min(c["positive_patients"], c["negative_patients"]) < 10 else ""
            count = f"{c['patients']} ({c['positive_patients']}){star}"
            values = [metric(task, method) for method in METHODS]
            delta = task["future_minus"]["history_32"]
            ci = delta["auroc_difference_ci95"]
            diff = f"{delta['auroc_difference']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "-"
            lines.append(f"| {NAMES[task['task']]} | {count} | " + " | ".join(values) + f" | {diff} |")
    lines += [
        "",
        "\u547c\u5438\u8870\u7aed\u3001\u809d\u8870\u7aed\u3001\u8113\u6bd2\u75c7\u548c\u6c14\u7ba1\u5207\u5f00\u7684\u9a8c\u8bc1\u96c6\u4e5f\u6709\u5c11\u4e8e 5 \u540d\u67d0\u4e00\u7c7b\u60a3\u8005\uff0c\u8d85\u53c2\u6570\u9009\u62e9\u4e0d\u7a33\u5b9a\uff1b\u6c14\u7ba1\u5207\u5f00\u5728\u6d4b\u8bd5\u96c6\u4ec5\u6709 1 \u540d\u9633\u6027\u60a3\u8005\u3002",
        "\u8fd9\u662f\u5bf9 34 \u4e2a\u4efb\u52a1\u7684\u63a2\u7d22\u6027\u591a\u91cd\u6bd4\u8f83\uff1b\u5355\u9879\u533a\u95f4\u4e0d\u80fd\u4f5c\u4e3a\u672a\u7ecf\u6821\u6b63\u7684\u663e\u8457\u6027\u7ed3\u8bba\u3002",
        "\u4e0e\u539f\u6587\u6bd4\u8f83\u65f6\u8fd8\u9700\u6ce8\u610f\uff1a\u539f\u6587\u4f7f\u7528\u4e0d\u540c\u7684\u7f16\u7801\u5668\u3001MLP \u548c\u771f\u5b9e\u8bb0\u5f55\u7684\u540e\u7eed\u6cbb\u7597\uff0c\u672c\u5b9e\u9a8c\u4f7f\u7528\u51bb\u7ed3\u57fa\u5ea7\u3001PCA \u52a0\u903b\u8f91\u56de\u5f52\u53ca\u56fa\u5b9a\u5047\u8bbe\u3002",
        "",
        "\u5b8c\u6574\u6570\u503c\u3001AUPRC \u533a\u95f4\u53ca\u60a3\u8005\u7ea7\u6a21\u578b\u8f93\u51fa\u4fdd\u5b58\u5728\u670d\u52a1\u5668\u7684 `sample_3h/clinical_data/all_tasks/evaluation/`\uff1b\u75c5\u4eba\u7ea7\u6587\u4ef6\u4e0d\u8fdb\u5165 Git\u3002",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(tasks)} tasks to {args.output}")


if __name__ == "__main__":
    main()
