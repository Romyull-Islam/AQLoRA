"""Does NF4-error ranking beat budget-matched random selection?

Reads every ranking_null artifact and reports where top-K lands in the null
distribution of random selections at the same protected memory. Reproduces the
appendix table and the sign test quoted in the paper.

Two controls are present. The count-matched one holds K fixed; it cannot match
the parameter total where layer sizes vary, so its Llama-3.2-1B rows are not
memory-matched and are reported only for the record. The parameter-matched one
holds the protected parameters fixed and lets K float; that is the control the
paper draws on.

usage: python summarize_ranking_null.py [results_dir]
"""
import glob
import json
import os
import re
import statistics as st
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "results"
NM = {"phi15": "phi-1.5", "llama-3.2-1b": "Llama-3.2-1B", "llama-3.2-3b": "Llama-3.2-3B",
      "phi-3-mini": "Phi-3-mini", "qwen2.5-7b": "Qwen2.5-7B"}
ORDER = ["phi-1.5", "Llama-3.2-1B", "Llama-3.2-3B", "Phi-3-mini", "Qwen2.5-7B"]
LAYERS = {"phi-1.5": 144, "Llama-3.2-1B": 112, "Llama-3.2-3B": 196,
          "Phi-3-mini": 128, "Qwen2.5-7B": 196}


def load():
    rows = []
    for p in sorted(glob.glob(f"{D}/ranking_null*.json")):
        b = os.path.basename(p)
        m = re.match(r"ranking_null_(.+?)(-pm)?_(?:k(\d+)|r(\d+))\.json$", b)
        if m:
            tag, pm, k, r = m.group(1), bool(m.group(2)), m.group(3), m.group(4)
        elif b == "ranking_null.json":
            tag, pm, k, r = "phi15", False, "29", None      # the first run, phi-1.5
        else:
            continue
        d = json.load(open(p))
        if "top" not in d:
            continue
        rnd = [v for key, v in d.items() if key.startswith("random")]
        if not rnd:
            continue
        ppl = sorted(v["ppl"] for v in rnd)
        model = NM[tag]
        rows.append(dict(
            model=model, pm=pm, k=d["top"]["k"],
            rho=int(r) / 100 if r else int(k) / LAYERS[model],
            top=d["top"]["ppl"], bot=d["bottom"]["ppl"],
            lo=min(ppl), med=st.median(ppl), hi=max(ppl),
            beat=sum(1 for x in ppl if x < d["top"]["ppl"]),
            # protected memory of top-K against the mean of the draws: the
            # control is only meaningful where this is 1.000
            cov=d["top"]["base_gib"] / st.mean(v["base_gib"] for v in rnd)))
    return rows


rows = load()
if not rows:
    sys.exit(f"no ranking_null artifacts under {D}/")

for pm, label in ((True, "parameter-matched (the control the paper uses)"),
                  (False, "count-matched (original, kept for the record)")):
    sel = [r for r in rows if r["pm"] == pm]
    if not sel:
        continue
    print(f"\n=== {label} ===")
    print(f"  {'model':13} {'rho':>4} {'K':>3} {'top-K':>8} {'bottom-K':>8} "
          f"{'random min/med/max':>25} {'rnd<top':>8} {'memory':>7}")
    for model in ORDER:
        for r in sorted((x for x in sel if x["model"] == model), key=lambda x: x["rho"]):
            print(f"  {r['model']:13} {r['rho']:4.2f} {r['k']:3d} {r['top']:8.3f} "
                  f"{r['bot']:8.3f} {r['lo']:8.3f}/{r['med']:8.3f}/{r['hi']:8.3f} "
                  f"{r['beat']:5d}/10 {r['cov']:6.3f}x")
    ahead = sum(1 for r in sel if r["top"] < r["med"])
    print(f"  top-K below the random median in {ahead}/{len(sel)} settings; "
          f"{sum(r['beat'] for r in sel)}/{10*len(sel)} draws beat top-K "
          f"(chance predicts {5*len(sel)})")
    try:
        from scipy import stats as sps
        print(f"  sign test: p = {sps.binomtest(ahead, len(sel), 0.5).pvalue:.2f}")
    except ImportError:
        pass
    print(f"  median gap (random median - top): "
          f"{st.median([r['med'] - r['top'] for r in sel]):+.4f} ppl")
    unmatched = [r for r in sel if abs(r["cov"] - 1) > 0.001]
    if unmatched:
        print(f"  NOT memory-matched, so not comparable: "
              f"{', '.join(f'{r['model']} rho={r['rho']:.2f}' for r in unmatched)}")
