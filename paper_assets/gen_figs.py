#!/usr/bin/env python3
"""Generate publication figures for the AQLoRA paper. Run from paper/tmlr/.
Outputs vector PDFs: fig_pareto, fig_sensitivity, fig_levers, fig_memory,
and fig_placement (when E3 lands). Colorblind-safe, print-friendly."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = "../.."
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
# Okabe-Ito colorblind-safe palette
C = {"aqs": "#D55E00", "aqs2": "#E69F00", "qlora": "#0072B2", "rslora": "#56B4E9",
     "loftq": "#009E73", "dora": "#CC79A7", "fp16": "#555555", "grid": "#DDDDDD"}


TASKS = ["boolq", "piqa", "social_i_qa", "hellaswag",
         "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]


def avg8(cs):
    # simple mean of the 8 task accuracies (DoRA / LLM-Adapters convention),
    # not the example-weighted pooled "average" field
    return sum(cs[t] for t in TASKS) / 8 * 100


def rj(name):
    p = f"{ROOT}/results-cs-phi15-{name}/report.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d if "commonsense" in d else None


# gather main-table points: (label, avg%, min, peak, base, color, is_ours, marker)
# accuracy is the mean over every seed on disk, matching Table 1; minutes and
# memory come from the seed-42 run, matching the table's stated policy (the
# repeat seeds trained concurrently, so their wall-clock is contended)
SEEDS = ["", "-s43", "-s44", "-s45", "-s46", "-s47"]


def pt(name, label, color, ours=False, mk="o"):
    d = rj(name)
    if not d:
        return None
    accs = [avg8(x["commonsense"])
            for x in (rj(name + s) for s in SEEDS) if x]
    return dict(label=label, avg=sum(accs) / len(accs),
                mins=d.get("train_minutes"), peak=d.get("peak_train_gib"),
                base=d.get("base_model_gib"), color=color, ours=ours, mk=mk)


PTS = [p for p in [
    pt("aqslora-q-nors", "AQLoRA-q", C["aqs"], True, "*"),
    pt("aqslora-s-nors", "AQLoRA-s", C["aqs2"], True, "D"),
    pt("qlora", "QLoRA", C["qlora"], mk="o"),
    pt("rslora", "rsLoRA", C["rslora"], mk="o"),
    pt("loftq", "LoftQ", C["loftq"], mk="o"),
    pt("dora", "QDoRA", C["dora"], mk="s"),
    pt("lora16", "LoRA (fp16)", C["fp16"], mk="P"),
    pt("dora16", "DoRA (fp16)", C["fp16"], mk="X"),
] if p]


# FIG 1: Pareto (accuracy vs training time). Legend, not inline labels,
#      so the crowded cluster stays readable. Faster = left (natural).
def fig_pareto():
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.set_axisbelow(True)
    is_fp16 = lambda p: "fp16" in p["label"]
    # frontier over the quantized class only (fp16 is a different memory class)
    Q = [p for p in PTS if p["mins"] and not is_fp16(p)]
    dom = [(p["mins"], p["avg"]) for p in Q
           if not any(q["mins"] <= p["mins"] and q["avg"] >= p["avg"] and q is not p for q in Q)]
    dom.sort()
    ax.plot([d[0] for d in dom], [d[1] for d in dom], "--", color=C["aqs"],
            lw=1.3, zorder=1)
    handles = [Line2D([0], [0], ls="--", color=C["aqs"], lw=1.3,
                      label="quantized frontier")]
    # plot points, larger + bordered for ours; fixed-size legend handles
    for p in PTS:
        if p["mins"] is None:
            continue
        ax.scatter(p["mins"], p["avg"], s=170 if p["ours"] else 85, marker=p["mk"],
                   facecolor=p["color"], edgecolor="black",
                   linewidth=1.1 if p["ours"] else 0.6,
                   alpha=0.5 if is_fp16(p) else 0.95,
                   zorder=4 if p["ours"] else 3)
        lab = (p["label"] + " (ours)") if p["ours"] else p["label"]
        handles.append(Line2D([0], [0], marker=p["mk"], ls="", markersize=7,
                              markerfacecolor=p["color"], markeredgecolor="black",
                              markeredgewidth=0.6, label=lab))
    ax.set_xlabel("training time (min, single V100); lower is faster")
    ax.set_ylabel("Commonsense 8-task accuracy (%)")
    ax.grid(True, color=C["grid"], lw=0.5)
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # "better" direction arrow in the empty upper-right corner
    x1 = max(p["mins"] for p in PTS if p["mins"])
    y1 = max(p["avg"] for p in PTS if p["mins"])
    ax.annotate("better", xy=(x1 * 0.78, y1 + 0.04), xytext=(x1 * 0.95, y1 - 0.22),
                fontsize=8, color="#333333", ha="center",
                arrowprops=dict(arrowstyle="->", lw=1.1, color="#333333"))
    ax.legend(handles=handles, loc="lower right", fontsize=7.2, ncol=2,
              handletextpad=0.3, columnspacing=0.8, borderpad=0.4)
    fig.savefig("fig_pareto.pdf")
    plt.close(fig)
    print("wrote fig_pareto.pdf")


# FIG 2: per-layer NF4 sensitivity (motivating: damage concentrated)
def fig_sensitivity():
    prof = json.load(open("profile_phi15.json"))
    eps = np.array(sorted([l["eps"] for l in prof], reverse=True)) * 1e5
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.set_axisbelow(True)
    x = np.arange(1, len(eps) + 1)
    k = int(round(0.20 * len(eps)))
    ax.bar(x[:k], eps[:k], color=C["aqs"], width=1.0, label=f"top-20% $\\to$ fp16 (K={k})")
    ax.bar(x[k:], eps[k:], color=C["qlora"], width=1.0, label="quantized to NF4")
    ax.set_xlabel("linear layer (sorted by NF4 error $\\varepsilon_\\ell$)")
    ax.set_ylabel("NF4 recon. error $\\varepsilon_\\ell$ ($\\times10^{-5}$)")
    frac = eps[:k].sum() / eps.sum() * 100
    ax.annotate(f"top 20% of layers carry\n{frac:.0f}% of total NF4 error",
                (k, eps[k]), xytext=(k + 28, eps[3]),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.7, color="black"))
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, axis="y", color=C["grid"], lw=0.5)
    fig.savefig("fig_sensitivity.pdf")
    plt.close(fig)
    print(f"wrote fig_sensitivity.pdf (top-20% carry {frac:.0f}% of error)")


# FIG 3: speed-lever waterfall (tokens/s across the ladder)
def fig_levers():
    rows = {json.load(open(j))["tag"]: json.load(open(j))
            for j in glob.glob(f"{ROOT}/ablation-phi-1_5/*/timing.json")}
    ladder = [("q0-eager", "QLoRA, eager", C["qlora"]),
              ("q1-sdpa", "+SDPA", C["qlora"]),
              ("q2-dropout0", "+dropout 0", C["qlora"]),
              ("q3-grouplen", "+length-group", C["qlora"]),
              ("q4-fusedopt", "+fused AdamW", C["qlora"]),
              ("a1-speedprof", "AQLoRA sparsity", C["aqs"]),
              ("a5-depth07-notrunc", "AQLoRA depth .7", C["aqs"]),
              ("a2-depth07", "+truncation", C["aqs"]),
              ("a4-depth05", "AQLoRA depth .5", C["aqs"]),
              ("l1-lora16", "fp16 LoRA", C["fp16"])]
    labels, vals, cols = [], [], []
    for k, lab, c in ladder:
        if k in rows:
            labels.append(lab)
            vals.append(rows[k]["real_tokens_per_sec"])
            cols.append(c)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.set_axisbelow(True)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=cols, edgecolor="black", linewidth=0.5, height=0.72)
    for yi, v in zip(y, vals):
        ax.text(v + 15, yi, f"{v:,.0f}", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("real (non-pad) tokens / s  (phi-1.5, 220 steps, V100)")
    # color key lives in the caption (blue=shared levers, orange=AQLoRA, grey=fp16)
    # to avoid overlapping the value labels in this monotone waterfall.
    ax.grid(True, axis="x", color=C["grid"], lw=0.5)
    ax.set_xlim(0, max(vals) * 1.12)
    fig.savefig("fig_levers.pdf")
    plt.close(fig)
    print("wrote fig_levers.pdf")


# FIG 4: memory (base + trainable/activation overhead)
def fig_memory():
    order = ["lora16", "dora16", "qlora", "rslora", "loftq",
             "aqslora-q-nors", "aqslora-s-nors"]
    labs = {"lora16": "LoRA fp16", "dora16": "DoRA fp16", "qlora": "QLoRA",
            "rslora": "rsLoRA", "loftq": "LoftQ", "aqslora-q-nors": "AQLoRA-q",
            "aqslora-s-nors": "AQLoRA-s"}
    base, over, labels, ours = [], [], [], []
    for n in order:
        d = rj(n)
        if not d:
            continue
        b = d.get("base_model_gib") or 0
        pk = d.get("peak_train_gib") or 0
        base.append(b)
        over.append(max(0, pk - b))
        labels.append(labs[n])
        ours.append(n.startswith("aqslora"))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.set_axisbelow(True)
    x = np.arange(len(labels))
    ax.bar(x, base, color="#B0B0B0", edgecolor="black", linewidth=0.5, label="base model (weights)")
    ax.bar(x, over, bottom=base, color=C["qlora"], edgecolor="black", linewidth=0.5,
           label="training overhead (act.+optim.)")
    for xi, (b, o, isours) in enumerate(zip(base, over, ours)):
        ax.text(xi, b + o + 0.1, f"{b+o:.1f}", ha="center", fontsize=7,
                fontweight="bold" if isours else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("peak GPU memory (GiB)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, axis="y", color=C["grid"], lw=0.5)
    fig.savefig("fig_memory.pdf")
    plt.close(fig)
    print("wrote fig_memory.pdf")


# FIG 5: placement-criterion ablation (E3; the audit-mandatory control)
def fig_placement():
    def a(name):
        d = rj(name)
        return d["commonsense"]["average"] * 100 if d else None
    grid = {  # budget -> {criterion: accuracy}
        "0.5": {"sparsity": a("e3-phi15-sparsity05") or a("../results-e3-phi15-sparsity05"),
                "weightnorm": a("../results-e3-phi15-weightnorm05"),
                "random": a("../results-e3-phi15-random05")},
        "0.7": {"sparsity": a("../results-e3-phi15-sparsity07"),
                "weightnorm": a("../results-e3-phi15-weightnorm07"),
                "random": a("../results-e3-phi15-random07")}}
    # helper reused with e3 prefix
    def e3(name):
        p = f"{ROOT}/results-{name}/report.json"
        if not os.path.exists(p):
            return None
        d = json.load(open(p))
        return avg8(d["commonsense"]) if "commonsense" in d else None
    budgets = ["0.5", "0.7"]
    crits = [("sparsity", "near-zero frac (ours)", C["aqs"]),
             ("weightnorm", "weight-norm", C["qlora"]),
             ("random", "random", "#999999")]
    data = {b: {c[0]: e3(f"e3-phi15-{c[0]}{b.replace('0.','0')}") for c in crits}
            for b in budgets}
    if all(data[b][c[0]] is None for b in budgets for c in crits):
        print("fig_placement: E3 data not ready")
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.set_axisbelow(True)
    w = 0.25
    x = np.arange(len(budgets))
    for i, (ckey, clab, col) in enumerate(crits):
        vals = [data[b][ckey] or 0 for b in budgets]
        ax.bar(x + (i - 1) * w, vals, w, color=col, edgecolor="black",
               linewidth=0.5, label=clab)
        for xi, v in zip(x + (i - 1) * w, vals):
            if v:
                ax.text(xi, v + 0.06, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\alpha={b}$" for b in budgets])
    ax.set_ylabel("Commonsense 8-task accuracy (%)")
    ax.set_xlabel("adapter budget (fraction of layers)")
    allv = [data[b][c[0]] for b in budgets for c in crits if data[b][c[0]]]
    ax.set_ylim(min(allv) - 1.0, max(allv) + 1.2)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, axis="y", color=C["grid"], lw=0.5)
    fig.savefig("fig_placement.pdf")
    plt.close(fig)
    print("wrote fig_placement.pdf")


if __name__ == "__main__":
    fig_pareto()
    fig_sensitivity()
    fig_levers()
    fig_memory()
    fig_placement()
