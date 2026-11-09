"""Generate presentation-ready figures from the aggregate-only experiment data."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, patches

BLUE, GREEN, ORANGE, GREY = "#346FA6", "#3C8B75", "#D58B4B", "#83909C"
INK, TITLE, GRID = "#263746", "#182C3A", "#E6EBEF"
font_manager.fontManager.addfont("/usr/share/fonts/wqy-microhei/wqy-microhei.ttc")
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname="/usr/share/fonts/wqy-microhei/wqy-microhei.ttc").get_name(),
    "axes.unicode_minus": False, "svg.fonttype": "path", "font.size": 15,
    "axes.titlesize": 18, "axes.labelsize": 15, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "legend.fontsize": 12, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": "#C6CFD6", "savefig.facecolor": "white",
})


def canvas(title, subtitle):
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(.055, .947, title, fontsize=28, fontweight="bold", color=TITLE, va="top")
    fig.text(.055, .887, subtitle, fontsize=15, color="#5F7280", va="top")
    fig.add_artist(plt.Line2D([.055, .945], [.855, .855], transform=fig.transFigure, color=GRID))
    return fig


def finish(fig, out, stem, note):
    fig.text(.055, .044, note, fontsize=11, color="#617482", va="bottom")
    fig.text(.945, .044, "Clin-JEPA · 冻结基座 · 小样本实验", fontsize=11,
             color="#617482", ha="right", va="bottom")
    for ext in ("png", "svg"):
        path = out / (stem + "." + ext)
        fig.savefig(path, dpi=160)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def tidy(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def pipeline(data, out):
    fig = canvas("冻结基座实验：从 MIMIC 数据到下游任务",
                 "一次小样本流程验证｜患者级划分，未经 SFT 的 Qwen3-8B 冻结，仅训练预测器")
    ax = fig.add_axes([.055, .19, .89, .61]); ax.set_xlim(-.01, 1.02); ax.set_ylim(0, 1); ax.axis("off")
    cards = [
        (.005, BLUE, "01  数据与轨迹", "MIMIC-IV · 每小时轨迹\n训练 1,000 / 验证 200\n测试 200 名患者\n训练窗口 4,190 个"),
        (.265, BLUE, "02  冻结编码器", "Qwen3-8B 基座\n不使用 SFT / LoRA\n编码状态、治疗与\n患者基本信息"),
        (.525, GREEN, "03  状态预测器", "Clin-JEPA 预测器 92.5M\n20 轮 · 仅训练预测器\n训练约 287 秒\n状态 + 治疗 → 未来状态"),
        (.785, ORANGE, "04  测试与评估", "1–48 小时潜在状态预测\n简单基线与表示诊断\n34 项临床下游任务\n逻辑回归探针"),
    ]
    for x, color, title, body in cards:
        ax.add_patch(patches.FancyBboxPatch((x, .32), .21, .61,
                     boxstyle="round,pad=0.012,rounding_size=0.018",
                     facecolor="white", edgecolor="#D2DDE3", linewidth=1.5))
        ax.add_patch(patches.FancyBboxPatch((x+.016, .84), .055, .045,
                     boxstyle="round,pad=0.002,rounding_size=0.01", facecolor=color, edgecolor="none"))
        ax.text(x+.016, .81, title, fontsize=16, fontweight="bold", color=TITLE, va="top")
        ax.text(x+.016, .74, body, fontsize=13.2, linespacing=1.5, color=INK, va="top")
    for x1, x2 in ((.224,.258),(.484,.518),(.744,.778)):
        ax.annotate("", xy=(x2,.62), xytext=(x1,.62),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color="#9FB0BC"))
    ax.text(.01, .13, "滚动评估：使用真实记录的后续治疗（回顾性）", fontsize=14, color=BLUE, fontweight="bold")
    ax.text(.51, .13, "下游评估：24 小时后假设无干预（模拟）", fontsize=14, color=GREEN, fontweight="bold")
    ax.text(.01, .04, "两种评估的动作条件不同，均不是临床部署或因果结论。", fontsize=12.5, color="#617482")
    finish(fig, out, "01_pipeline", "数据：训练 / 验证 / 测试患者互不重叠；预测器训练不含编码缓存生成时间。")


def training(data, out):
    t = data["training"]; ep = t["epochs"]; x = np.array([v["epoch"] for v in ep])
    fig = canvas("预测器训练过程：20 轮后验证误差下降",
                 "训练冻结 Qwen 的状态预测器｜训练日志原始记录，L1 是 4,096 维潜在状态的误差之和")
    gs = fig.add_gridspec(1, 3, left=.065, right=.95, bottom=.22, top=.79, wspace=.34, width_ratios=[1.25, 1, .9])
    a, b, c = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax in (a,b,c): tidy(ax)
    a.plot(x, [v["train_tf"] for v in ep], color=BLUE, lw=2.8, label="训练 · 单步")
    a.plot(x, [v["validation_tf"] for v in ep], color=GREEN, lw=2.8, label="验证 · 单步")
    a.scatter([20], [ep[-1]["validation_tf"]], s=45, c=GREEN, zorder=4)
    a.annotate("最佳验证 L1\n1,653", (20, ep[-1]["validation_tf"]), xytext=(11.5, 2700),
               fontsize=13, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN))
    a.set_title("A  单步 teacher forcing", loc="left", fontweight="bold")
    a.set_xlabel("训练轮次"); a.set_ylabel("L1 误差"); a.legend(frameon=False, loc="upper right")
    a.set_xticks([1,5,10,15,20]); a.set_ylim(bottom=0)
    b.plot(x, [v["train_rollout_2step_sum"] for v in ep], color=ORANGE, lw=2.8)
    b.scatter([20], [ep[-1]["train_rollout_2step_sum"]], s=45, color=ORANGE, zorder=3)
    b.set_title("B  训练两步 rollout", loc="left", fontweight="bold")
    b.set_xlabel("训练轮次"); b.set_ylabel("两步 L1 累计"); b.set_xticks([1,5,10,15,20]); b.set_ylim(bottom=0)
    lrs=t["logged_learning_rates"]
    c.plot([v["step"] for v in lrs], [v["learning_rate"]*1e4 for v in lrs], "o--", color=GREY, lw=2, markersize=6)
    c.set_title("C  学习率日志点", loc="left", fontweight="bold")
    c.set_xlabel("优化器 step"); c.set_ylabel("学习率（× 10^(-4)）"); c.set_ylim(bottom=0)
    fig.text(.065, .135, "20 epochs  ·  batch 128 × 梯度累积 2  ·  320 steps  ·  约 287 秒",
             fontsize=15, color=TITLE, fontweight="bold")
    finish(fig, out, "02_training", "两步累计误差和单步验证误差尺度不同；编码与数据准备时间未计入训练耗时。")


def normalized(data, out):
    rows=data["forecast"]["horizons"]; x=np.array([r["horizon"] for r in rows])
    fig=canvas("滚动预测：时间越远，潜在状态误差如何变化？",
               "参考论文 Fig. 2｜每个指标以 1 小时数值归一化；阴影为患者级 bootstrap 95% 区间")
    gs=fig.add_gridspec(1,3,left=.065,right=.95,bottom=.22,top=.79,wspace=.31)
    names=[("rollout_drift","A  预测误差变化","预测器 L1 / 1h L1",GREEN),
           ("predictor_degradation","B  相对表现变化","MASE / 1h MASE",BLUE),
           ("state_displacement","C  状态位移变化","保持不变 L1 / 1h L1",ORANGE)]
    for i,(key,title,ylabel,color) in enumerate(names):
        ax=fig.add_subplot(gs[0,i]); tidy(ax)
        y=np.array([r[key] for r in rows]); ci=np.array([r[key+"_ci"] for r in rows])
        ax.fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.14,linewidth=0)
        ax.plot(x,y,color=color,lw=2.7); ax.axhline(1,color=GREY,lw=1,ls="--")
        ax.scatter([48],[y[-1]],color=color,s=45,zorder=3)
        ax.text(.97,.93,f"48h  {y[-1]:.2f}",transform=ax.transAxes,ha="right",va="top",
                color=color,fontsize=15,fontweight="bold")
        ax.set_title(title,loc="left",fontweight="bold"); ax.set_xlabel("预测步数（小时）")
        ax.set_ylabel(ylabel); ax.set_xlim(1,48); ax.set_xticks([1,12,24,36,48])
    fig.text(.065,.14,"MASE = 预测器 L1 ÷ 保持最后状态不变的 L1；数值低于 1 表示优于该基线。", fontsize=14)
    finish(fig,out,"03_rollout_stability","使用记录中的后续治疗；1h 为 162 人，48h 为 64 人；不同时间点可用患者不同。")


def baselines(data, out):
    rows=data["forecast"]["horizons"]; x=np.array([r["horizon"] for r in rows])
    fig=canvas("预测效果：与简单基线相比",
               "回顾性 1–48 小时预测｜L1 越低越好，但潜在状态 L1 并不是临床指标")
    gs=fig.add_gridspec(1,2,left=.07,right=.95,bottom=.22,top=.79,wspace=.3,width_ratios=[1.1,1])
    a,b=[fig.add_subplot(gs[0,i]) for i in range(2)]
    for ax in (a,b): tidy(ax)
    for key,label,color,style in [
        ("predictor","预测器",GREEN,"-"), ("context_mean","过去 24h 均值",ORANGE,"-"),
        ("train_mean","训练集全局均值",BLUE,"--"), ("copy_forward","保持最后状态",GREY,"--")]:
        a.plot(x,[r[key] for r in rows],label=label,color=color,ls=style,lw=2.5)
    a.set_title("A  潜在状态预测误差",loc="left",fontweight="bold")
    a.set_xlabel("预测步数（小时）"); a.set_ylabel("L1 误差（维度求和）")
    a.set_xlim(1,48); a.set_xticks([1,12,24,36,48]); a.legend(frameon=False,loc="upper left",ncol=2)
    y=np.array([r["context_improvement"]*100 for r in rows])
    ci=np.array([r["context_improvement_ci"] for r in rows])*100
    b.fill_between(x,ci[:,0],ci[:,1],color=GREEN,alpha=.15,linewidth=0)
    b.plot(x,y,color=GREEN,lw=2.7); b.axhline(0,color=GREY,ls="--",lw=1.2)
    b.set_title("B  相对过去 24h 均值的改善",loc="left",fontweight="bold")
    b.set_xlabel("预测步数（小时）"); b.set_ylabel("L1 相对降低（%）")
    b.set_xlim(1,48); b.set_xticks([1,12,24,36,48])
    b.text(.95,.90,f"48h  {y[-1]:+.1f}%",transform=b.transAxes,ha="right",fontsize=15,color=GREEN,fontweight="bold")
    fig.text(.07,.14,"预测器优于“保持不变”基线；相对历史均值的差距较小，部分时间点的患者级区间跨 0。",fontsize=14)
    finish(fig,out,"04_prediction_baselines","阴影：患者级 bootstrap 95% 区间；不同预测时长的可用患者与窗口数不同。")


def representation(data,out):
    r=data["rank"]; hs=[1,8,24,48]
    fig=canvas("预测状态的多样性随滚动步数下降",
               "表示诊断｜RankMe 衡量表示的有效维度；更低提示状态变化集中于较少方向")
    gs=fig.add_gridspec(1,2,left=.08,right=.94,bottom=.23,top=.78,wspace=.28)
    a,b=[fig.add_subplot(gs[0,i]) for i in range(2)]
    for ax in (a,b): tidy(ax)
    for key,label,color in [("true","真实编码状态",BLUE),("predicted","预测状态",GREEN)]:
        a.plot(hs,[r[str(h)][key]["rankme"] for h in hs],"-o",label=label,color=color,lw=2.8,markersize=7)
        b.plot(hs,[r[str(h)][key]["top10_share"]*100 for h in hs],"-o",label=label,color=color,lw=2.8,markersize=7)
    a.set_title("A  RankMe：有效维度",loc="left",fontweight="bold"); a.set_ylabel("有效维度（越低越集中）")
    b.set_title("B  前 10 个主方向的方差占比",loc="left",fontweight="bold"); b.set_ylabel("解释的方差（%）")
    for ax in (a,b):
        ax.set_xlabel("预测步数（小时）"); ax.set_xticks(hs); ax.legend(frameon=False)
    predicted_rank=r["48"]["predicted"]["rankme"]; true_rank=r["48"]["true"]["rankme"]
    fig.text(.08,.14,f"48h：预测 RankMe {predicted_rank:.1f}，真实 {true_rank:.1f}；提示预测趋同，需要进一步验证。",fontsize=14)
    finish(fig,out,"05_representation","不同步数对应不同可用测试窗口；此诊断不能单独证明临床任务有效或无效。")


def mortality(data,out):
    d=data["mortality"]; result=d["results"]
    keys=["history_32","history_64","history_32_plus_future_32"]
    labels=["历史 32 维","历史 64 维","历史 32 维\n+ 预测 32 维"]
    colors=[BLUE, "#A0B7CC", GREEN]
    fig=canvas("下游示例：住院死亡预测",
               "前 24 小时历史 + 模拟未来状态｜测试集 210 次入住、165 人、23 次死亡")
    gs=fig.add_gridspec(1,2,left=.09,right=.94,bottom=.25,top=.78,wspace=.3)
    for i,(metric,title) in enumerate([("auroc","A  AUROC"),("auprc","B  AUPRC")]):
        ax=fig.add_subplot(gs[0,i]); tidy(ax)
        vals=np.array([result[k]["test_"+metric] for k in keys])
        ci=np.array([result[k]["test_"+metric+"_ci95"] for k in keys])
        xx=np.arange(3)
        ax.bar(xx,vals,color=colors,width=.55,zorder=2)
        ax.errorbar(xx,vals,yerr=np.stack([vals-ci[:,0],ci[:,1]-vals]),fmt="none",ecolor=INK,
                    capsize=5,lw=1.4,zorder=3)
        for j,v in enumerate(vals):
            ax.text(j,ci[j,1]+.025,f"{v:.3f}",ha="center",fontsize=13,color=INK)
        baseline=.5 if metric=="auroc" else d["test_positive_prevalence"]
        ax.axhline(baseline,ls="--",lw=1.2,color=GREY)
        ax.text(2.55,baseline+.015,"随机参考 0.50" if metric=="auroc" else "阳性比例 0.110",
                color=INK,ha="right",fontsize=11, bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2))
        ax.set_ylim(0,1 if metric=="auroc" else .70)
        ax.set_xticks(xx,labels); ax.set_title(title,loc="left",fontweight="bold")
        ax.set_ylabel("测试集分数")
    pair=d["paired_comparisons"]["history_32"]["auroc"]; delta=pair["difference"]; lo,hi=pair["ci95"]
    fig.text(.09,.145,f"历史 32 + 预测 32 相比历史 32：AUROC 差值 {delta:+.3f}，95% CI [{lo:+.3f}, {hi:+.3f}]。",fontsize=14)
    finish(fig,out,"06_mortality","误差线为患者级 bootstrap 95% 区间；未来治疗按无干预模拟，不是实际未来治疗。")


def downstream(data,out):
    groups=data["downstream"]["groups"]
    keys=["history_32","history_64","history_32_plus_future_32"]
    labels=["历史 32 维","历史 64 维","历史 32 + 预测 32 维"]
    colors=[BLUE,"#A0B7CC",GREEN]
    fig=canvas("34 项下游任务：加入预测状态后的探索性对照",
               "参考论文 Fig. 4｜每个任务等权平均；小样本探索，不与论文全量结果直接比较")
    gs=fig.add_gridspec(1,2,left=.07,right=.96,bottom=.25,top=.76,wspace=.26)
    xx=np.arange(3); width=.22
    group_labels=["E1 急性恶化\n8 项","E2 治疗及状态变化\n18 项","E3 住院结局\n8 项"]
    for i,(metric,title) in enumerate([("auroc","A  任务平均 AUROC"),("auprc","B  任务平均 AUPRC")]):
        ax=fig.add_subplot(gs[0,i]); tidy(ax)
        for j,key in enumerate(keys):
            vals=[g["scores"][key][metric] for g in groups]
            bars=ax.bar(xx+(j-1)*width,vals,width=width,color=colors[j],label=labels[j],zorder=2)
            for bar,val in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2,val+.012,f"{val:.3f}",
                        ha="center",va="bottom",fontsize=10,color=INK,rotation=90)
        ax.set_ylim(0,.94 if metric=="auroc" else .55)
        ax.set_xticks(xx,group_labels); ax.set_title(title,loc="left",fontweight="bold"); ax.set_ylabel("任务等权平均")
    fig.legend(labels,loc="upper center",bbox_to_anchor=(.50,.845),ncol=3,frameon=False,fontsize=13)
    fig.text(.07,.14,"三组平均 AUROC 均未因加入预测状态而提高；部分任务阳性样本极少，差异不宜过度解读。",fontsize=14)
    finish(fig,out,"07_downstream","下游假设无干预；住院级任务用 48h 预测，和单独死亡率图的 24h 方案不同。")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.data.read_text()); args.output.mkdir(parents=True,exist_ok=True)
    for fn in (pipeline,training,normalized,baselines,representation,mortality,downstream):
        fn(data,args.output)
        print("Created:", fn.__name__, flush=True)


if __name__=="__main__":
    main()
