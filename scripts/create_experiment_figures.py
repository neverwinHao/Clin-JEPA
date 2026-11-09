"""Generate compact paper-style charts for the sampled frozen-base experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator

GREEN = "#2D8537"
BLUE = "#2876B9"
ORANGE = "#E68728"
GREY = "#777777"
PURPLE = "#8B58A2"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": .8,
    "axes.edgecolor": "#444444",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def save(fig, out, stem):
    for ext in ("png", "svg"):
        path = out / (stem + "." + ext)
        fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=.05)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def tidy(ax):
    ax.set_axisbelow(True)
    ax.grid(color="#e4e4e4", linewidth=.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=.7)


def panel(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", pad=5)


def pipeline(data, out):
    counts = data["sampling"]["splits"]
    params = data["training"]["parameters"] / 1e6
    fig, ax = plt.subplots(figsize=(11.4, 2.45))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.1); ax.axis("off")
    boxes = [
        (0.05, 2.35, "#eaf1f7", "MIMIC-IV trajectories",
         "Hourly clinical states\nPatient-level splits\nTrain: %s patients\n%s windows" % (
             f"{counts['train']['patients']:,}", f"{counts['train']['windows']:,}")),
        (3.15, 2.30, "#eaf1f7", "Frozen encoder",
         "Qwen3-8B base\nNo SFT / LoRA\nState and actions\nStatic features"),
        (6.20, 2.35, "#edf5ec", "State predictor",
         "Clin-JEPA predictor: %.1fM\nTrain predictor only\n20 epochs" % params),
        (9.30, 2.60, "#fff3e8", "Evaluation",
         "1–48 h state rollout\n34 clinical tasks\nLinear probes"),
    ]
    for x, w, fill, title, body in boxes:
        ax.add_patch(Rectangle((x, .60), w, 2.05, facecolor=fill, edgecolor="#8999a0", linewidth=.8))
        ax.text(x+w/2, 2.30, title, fontsize=11, fontweight="bold", ha="center", va="center")
        ax.plot([x+.12,x+w-.12],[2.06,2.06],color="#aab7bb",lw=.6)
        ax.text(x+w/2,1.78,body,fontsize=9.6,ha="center",va="top",linespacing=1.5)
    for start, end in [(2.43,3.10),(5.48,6.15),(8.58,9.25)]:
        ax.add_patch(FancyArrowPatch((start,1.65),(end,1.65),arrowstyle="-|>",mutation_scale=12,
                                     linewidth=.9,color="#56646b"))
    fig.subplots_adjust(left=.01,right=.99,top=.95,bottom=.08)
    save(fig,out,"01_pipeline")


def training(data, out):
    t=data["training"]; ep=t["epochs"]; x=np.array([v["epoch"] for v in ep])
    fig, axes=plt.subplots(1,3,figsize=(10.7,3.25))
    for ax in axes: tidy(ax)
    a,b,c=axes
    a.plot(x,[v["train_tf"] for v in ep],color=BLUE,lw=1.8,label="Train")
    a.plot(x,[v["validation_tf"] for v in ep],color=GREEN,lw=1.8,label="Validation")
    panel(a,"(a) Next-state L1")
    a.set_xlabel("Epoch"); a.set_ylabel("L1 error")
    a.set_xticks([1,5,10,15,20]); a.set_ylim(bottom=0); a.legend(frameon=False,loc="upper right")
    b.plot(x,[v["train_rollout_2step_sum"] for v in ep],color=ORANGE,lw=1.8)
    panel(b,"(b) Two-step rollout loss")
    b.set_xlabel("Epoch"); b.set_ylabel("Sum of two L1 errors")
    b.set_xticks([1,5,10,15,20]); b.set_ylim(bottom=0)
    lr=t["logged_learning_rates"]
    c.plot([v["step"] for v in lr],[v["learning_rate"]*1e4 for v in lr],
           color=PURPLE,lw=1.3,marker="o",ms=3)
    panel(c,"(c) Learning rate")
    c.set_xlabel("Optimizer step"); c.set_ylabel("Learning rate (×10⁻⁴)")
    c.set_ylim(bottom=0); c.xaxis.set_major_locator(MaxNLocator(5,integer=True))
    fig.tight_layout(w_pad=1.6)
    save(fig,out,"02_training")


def normalized(data, out):
    r=data["forecast"]["horizons"]; x=np.array([v["horizon"] for v in r])
    fig,axes=plt.subplots(1,3,figsize=(10.6,3.1))
    specs=[("rollout_drift","(a) Rollout drift",r"$L_1(h) / L_1(1)$",GREEN),
           ("predictor_degradation","(b) Predictor degradation","MASE(h) / MASE(1)",BLUE),
           ("state_displacement","(c) State displacement",r"$L_1^{CF}(h) / L_1^{CF}(1)$",ORANGE)]
    for ax,(key,title,ylabel,color) in zip(axes,specs):
        tidy(ax); panel(ax,title)
        y=np.array([v[key] for v in r]); ci=np.array([v[key+"_ci"] for v in r])
        ax.fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.12,lw=0)
        ax.plot(x,y,color=color,lw=1.55)
        ax.axhline(1,color=GREY,ls="--",lw=.8)
        ax.set_xlim(1,48); ax.set_xticks([1,12,24,36,48])
        ax.set_xlabel("Rollout horizon (hours)"); ax.set_ylabel(ylabel)
    fig.tight_layout(w_pad=1.6)
    save(fig,out,"03_rollout_stability")


def baselines(data, out):
    r=data["forecast"]["horizons"]; x=np.array([v["horizon"] for v in r])
    fig,(a,b)=plt.subplots(1,2,figsize=(9.0,3.3))
    for ax in (a,b): tidy(ax)
    specs=[("predictor","Predictor",GREEN,"-"),("context_mean","History mean",ORANGE,"-"),
           ("train_mean","Training-set mean",PURPLE,"--"),("copy_forward","Copy-forward",BLUE,"--")]
    for key,label,color,style in specs:
        a.plot(x,[v[key] for v in r],label=label,color=color,ls=style,lw=1.5)
    panel(a,"(a) Prediction error")
    a.set_ylabel("L1 error"); a.set_xlabel("Rollout horizon (hours)")
    a.set_xlim(1,48); a.set_xticks([1,12,24,36,48])
    a.legend(frameon=True,framealpha=.9,edgecolor="#dddddd",fontsize=8,ncol=2,loc="upper left")
    y=np.array([v["context_improvement"]*100 for v in r])
    ci=np.array([v["context_improvement_ci"] for v in r])*100
    b.fill_between(x,ci[:,0],ci[:,1],color=GREEN,alpha=.14,lw=0)
    b.plot(x,y,color=GREEN,lw=1.5)
    b.axhline(0,color=GREY,ls="--",lw=.8)
    panel(b,"(b) Relative to history mean")
    b.set_ylabel("Reduction in L1 (%)"); b.set_xlabel("Rollout horizon (hours)")
    b.set_xlim(1,48); b.set_xticks([1,12,24,36,48])
    fig.tight_layout(w_pad=2.2)
    save(fig,out,"04_prediction_baselines")


def representation(data,out):
    r=data["rank"]; hs=[1,8,24,48]
    fig,(a,b)=plt.subplots(1,2,figsize=(8.6,3.25))
    for ax in (a,b): tidy(ax)
    for key,label,color in [("true","Encoded target",BLUE),("predicted","Prediction",GREEN)]:
        a.plot(hs,[r[str(h)][key]["rankme"] for h in hs],"-o",label=label,color=color,lw=1.6,ms=3.5)
        b.plot(hs,[r[str(h)][key]["top10_share"]*100 for h in hs],"-o",
               label=label,color=color,lw=1.6,ms=3.5)
    panel(a,"(a) RankMe"); panel(b,"(b) Top-10 variance share")
    a.set_ylabel("Effective rank"); b.set_ylabel("Explained variance (%)")
    a.set_ylim(bottom=0); b.set_ylim(85,101)
    for ax in (a,b):
        ax.set_xlabel("Rollout horizon (hours)"); ax.set_xticks(hs)
    a.legend(frameon=False,loc="upper right")
    fig.tight_layout(w_pad=2.3)
    save(fig,out,"05_representation")


def mortality(data,out):
    d=data["mortality"]; res=d["results"]
    keys=["history_32","history_64","history_32_plus_future_32"]
    labels=["History\n32","History\n64","History 32\n+ predicted 32"]
    colors=[BLUE,"#98b5ce",GREEN]
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.5))
    for ax,(metric,title) in zip(axes,[("auroc","(a) AUROC"),("auprc","(b) AUPRC")]):
        tidy(ax); panel(ax,title)
        vals=np.array([res[k]["test_"+metric] for k in keys])
        ci=np.array([res[k]["test_"+metric+"_ci95"] for k in keys])
        xx=np.arange(3)
        ax.bar(xx,vals,color=colors,width=.57,edgecolor="white",linewidth=.4,zorder=2)
        ax.errorbar(xx,vals,yerr=np.stack([vals-ci[:,0],ci[:,1]-vals]),fmt="none",
                    ecolor="#333333",capsize=3,lw=.9,zorder=3)
        for j,v in enumerate(vals):
            ax.text(j,ci[j,1]+.025,f"{v:.3f}",ha="center",va="bottom",fontsize=9)
        baseline=.5 if metric=="auroc" else d["test_positive_prevalence"]
        ax.axhline(baseline,ls="--",lw=.8,color=GREY)
        ax.set_ylim(0,1.03 if metric=="auroc" else .70)
        ax.set_xticks(xx,labels); ax.set_ylabel("Test score")
    fig.tight_layout(w_pad=2.0)
    save(fig,out,"06_mortality")


def downstream(data,out):
    groups=data["downstream"]["groups"]
    keys=["history_32","history_64","history_32_plus_future_32"]
    labels=["History 32","History 64","History 32 + predicted 32"]
    colors=[BLUE,"#98b5ce",GREEN]
    xx=np.arange(3); width=.23
    fig,axes=plt.subplots(1,2,figsize=(9.1,3.55))
    for ax,(metric,title) in zip(axes,[("auroc","(a) AUROC"),("auprc","(b) AUPRC")]):
        tidy(ax); panel(ax,title)
        for j,key in enumerate(keys):
            vals=[g["scores"][key][metric] for g in groups]
            bars=ax.bar(xx+(j-1)*width,vals,width=width,color=colors[j],label=labels[j],zorder=2)
            for bar,val in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2,val+.012,f"{val:.3f}",ha="center",
                        va="bottom",fontsize=7.6,rotation=90)
        ax.set_ylim(0,1 if metric=="auroc" else .54)
        ax.set_xticks(xx,["E1 (8 tasks)","E2 (18 tasks)","E3 (8 tasks)"])
        ax.set_ylabel("Mean test score")
    fig.legend(labels,loc="lower center",bbox_to_anchor=(.5,-.01),ncol=3,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.08,1,1),w_pad=2.1)
    save(fig,out,"07_downstream")


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
