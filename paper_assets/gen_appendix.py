#!/usr/bin/env python3
"""Regenerate appendix tables from raw run artifacts. Run from paper/tmlr/.
   ../../ is the bundle root. Writes table_lever_full.tex, table_edge.tex,
   table_e3.tex, table_precision.tex as data lands."""
import json, glob, os
ROOT = "../.."

TASKS = ["boolq", "piqa", "social_i_qa", "hellaswag",
         "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]


def avg8(cs):
    # simple mean of the 8 task accuracies (DoRA / LLM-Adapters convention)
    return sum(cs[t] for t in TASKS) / 8 * 100


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


# C: full lever ladder
order = ['q0-eager', 'q1-sdpa', 'q2-dropout0', 'q3-grouplen', 'q4-fusedopt', 'q5-nogc',
         'a0-classic', 'a1-speedprof', 'a5-depth07-notrunc', 'a2-depth07', 'a4-depth05',
         'a3-depth07-nogc', 'l1-lora16', 'r1-rslora', 'e1-edge-aqs', 'e2-edge-qlora']
pretty = {'q0-eager': 'QLoRA, eager attention', 'q1-sdpa': r'\quad + SDPA',
          'q2-dropout0': r'\quad + dropout 0', 'q3-grouplen': r'\quad + length-grouped batching',
          'q4-fusedopt': r'\quad + fused AdamW (= QLoRA best)', 'q5-nogc': r'\quad + no grad-ckpt',
          'a0-classic': 'AQLoRA all-layer 1.0 (paged 8-bit opt)', 'a1-speedprof': 'AQLoRA all-layer 1.0 + levers',
          'a5-depth07-notrunc': 'AQLoRA depth .7, no truncation', 'a2-depth07': 'AQLoRA depth .7 + truncation',
          'a4-depth05': 'AQLoRA depth .5 + truncation', 'a3-depth07-nogc': 'AQLoRA depth .7 + trunc, no GC',
          'l1-lora16': 'fp16 LoRA + levers', 'r1-rslora': 'rsLoRA + levers',
          'e1-edge-aqs': 'AQLoRA edge (bs1/seq256/r4)', 'e2-edge-qlora': 'QLoRA edge (bs1/seq256/r4)'}
rows = {json.load(open(j))['tag']: json.load(open(j))
        for j in glob.glob(f'{ROOT}/ablation-phi-1_5/*/timing.json')}
with open('table_lever_full.tex', 'w') as f:
    f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
    f.write("Configuration & steps/s & real tok/s & pad waste & peak GiB \\\\\n\\midrule\n")
    for k in order:
        if k not in rows:
            continue
        d = rows[k]
        if k == 'e1-edge-aqs':
            f.write("\\midrule\n")
        f.write(f"{pretty[k]} & {d['steps_per_sec']:.3f} & {d['real_tokens_per_sec']:,.0f} & "
                f"{d['padding_waste_frac']*100:.1f}\\% & {d['peak_gib']:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("wrote table_lever_full.tex")

# 6: edge feasibility (measured V100 + bandwidth-bound extrapolation)
# At batch 1 / short sequence, LoRA training is memory-bandwidth-bound, not
# compute-bound: the V100 edge run reached only ~1.3 effective TFLOPS despite
# the card's ~112 peak, because it streams weights once per token. So the
# honest scaling variable to a phone is DRAM bandwidth (V100 ~900 GB/s vs
# phone LPDDR5X ~80 GB/s), NOT TFLOPS. We cross-check the projection against
# QVAC's measured Qwen3-1.7B LoRA training on an Adreno 830 (~1.7 h/epoch).
V100_BW = 900.0  # GB/s (HBM2)
aqs = rows.get('e1-edge-aqs')
ql = rows.get('e2-edge-qlora')
if aqs and ql:
    toks = aqs['real_tokens_per_sec']
    # examples/s = optimizer-steps/s * grad-accum (batch 1); an "epoch" here is a
    # fixed 1000 examples, timed directly from the measured throughput.
    def epoch_min(d):
        ex_per_s = d['steps_per_sec'] * d.get('grad_accum', 16)
        return 1000.0 / ex_per_s / 60.0
    aqs_ep, ql_ep = epoch_min(aqs), epoch_min(ql)
    with open('table_edge.tex', 'w') as f:
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Device & DRAM BW & proj.\\ tok/s & peak GiB & 1k-example epoch \\\\\n\\midrule\n")
        f.write(f"V100 (\\textbf{{measured}}), AQLoRA-edge & 900 GB/s & "
                f"{toks:,.0f} & {aqs['peak_gib']:.2f} & "
                f"{aqs_ep:.1f} min \\\\\n")
        f.write(f"V100 (\\textbf{{measured}}), QLoRA-edge & 900 GB/s & "
                f"{ql['real_tokens_per_sec']:,.0f} & {ql['peak_gib']:.2f} & "
                f"{ql_ep:.1f} min \\\\\n")
        f.write("\\midrule\n\\multicolumn{5}{l}{\\emph{bandwidth-bound projection for AQLoRA-edge "
                "(upper bound; on-device kernel maturity will lower it):}} \\\\\n")
        for name, bw in [("Jetson AGX Orin (LPDDR5)", 205.0),
                         ("iPhone 16 Pro (LPDDR5X)", 85.0),
                         ("OnePlus 15 / SD 8 Elite (LPDDR5X)", 80.0)]:
            scaled = toks * bw / V100_BW
            ep = aqs_ep * V100_BW / bw
            f.write(f"{name} & {bw:.0f} GB/s & $\\sim${scaled:,.0f} & "
                    f"{aqs['peak_gib']:.2f} & $\\sim${ep/60:.1f} h \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("wrote table_edge.tex")

# 7.1: E3 placement grid
e3 = {}
for name in ['sparsity07', 'weightnorm07', 'random07', 'sparsity05', 'weightnorm05', 'random05']:
    d = load(f'{ROOT}/results-e3-phi15-{name}/report.json')
    if d and 'commonsense' in d:
        e3[name] = d
# also reuse main-table depth points + quality point
extra = {'depth07': load(f'{ROOT}/results-cs-phi15-aqslora-speed/report.json'),
         'depth085': load(f'{ROOT}/results-cs-phi15-aqslora-d085/report.json'),
         'sparsity10': load(f'{ROOT}/results-cs-phi15-aqslora-rs/report.json')}
if e3:
    with open('table_e3.tex', 'w') as f:
        f.write("\\begin{tabular}{llcc}\n\\toprule\n")
        f.write("Placement & budget $\\alpha$ & 8-task Avg & train min \\\\\n\\midrule\n")
        label = {'sparsity': 'near-zero frac (ours)', 'weightnorm': 'weight-norm',
                 'random': 'random', 'depth': 'depth (top blocks)'}
        for a in ['05', '07']:
            for crit in ['sparsity', 'weightnorm', 'random']:
                k = f'{crit}{a}'
                if k in e3:
                    cs = avg8(e3[k]['commonsense'])
                    mn = e3[k].get('train_minutes', '--')
                    f.write(f"{label[crit]} & 0.{a[1]} & {cs:.1f} & {mn:.0f} \\\\\n")
            f.write("\\midrule\n")
        # depth rows for the trade-off
        for k, lab, a in [('depth07', 'depth (top blocks)', '0.7'),
                          ('depth085', 'depth (top blocks)', '0.85'),
                          ('sparsity10', 'near-zero frac (ours)', '1.0')]:
            if extra.get(k):
                cs = avg8(extra[k]['commonsense'])
                mn = extra[k].get('train_minutes', '--')
                f.write(f"{lab} & {a} & {cs:.1f} & {mn:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote table_e3.tex ({len(e3)}/6 grid cells present)")
else:
    print("E3 grid not ready yet")
