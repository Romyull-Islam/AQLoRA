#!/usr/bin/env python3
"""Robust timing estimate from repeated runs on a shared node.

Contention can only ever SLOW a run, never speed it up, so the samples are
one-sided: the true uncontended speed sits at the fast end, and slow runs are
interference. A plain mean is therefore biased downward by however much
interference happened to occur, and a plain max keeps a single lucky run.

Estimator: find the TIGHTEST CLUSTER of k consecutive samples (smallest spread
among the fastest values) and average it. This discards interference-slowed
outliers while still averaging several runs, so it is stable without being
a single-sample maximum.
"""
import json, glob, statistics as st, sys

DIR = sys.argv[1] if len(sys.argv) > 1 else "timing-session2-phi15"
METHODS = ["qlora", "aq-q", "aq-s", "rslora", "loftq", "lora16"]


def fast_anchored(vals, frac=0.5):
    """Average the fastest ceil(frac*n) samples.

    Contention only ever slows a run, so the sample distribution has a hard
    ceiling at the true uncontended speed and a long slow tail. The estimator
    must therefore be anchored at the FAST end: averaging the fastest half keeps
    several runs (so it is not a single lucky maximum) while excluding the
    interference-slowed tail. A smallest-spread window is NOT valid here, since
    slow runs can cluster tightly too.
    """
    v = sorted(vals, reverse=True)
    k = max(2, int(round(frac * len(v))))
    k = min(k, len(v))
    w = v[:k]
    return st.mean(w), (max(w) - min(w)), k


print(f"=== robust timing from {DIR} ===")
print(f"{'method':8} {'n':>2} {'all samples (steps/s, fastest first)':<44} "
      f"{'fast-k':>8} {'spread':>7} {'mean':>7} {'max':>7}")
res = {}
for m in METHODS:
    vals = []
    for j in sorted(glob.glob(f"{DIR}/{m}-r*/timing.json")):
        try:
            vals.append(json.load(open(j))["steps_per_sec"])
        except Exception:
            pass
    if not vals:
        continue
    est, spread, k = fast_anchored(vals)
    res[m] = est
    shown = " ".join(f"{x:.4f}" for x in sorted(vals, reverse=True))
    print(f"{m:8} {len(vals):2d} {shown:<44} {est:8.4f} {spread:7.4f} "
          f"{st.mean(vals):7.4f} {max(vals):7.4f}")

if "qlora" in res:
    q = res["qlora"]
    print(f"\nrelative to QLoRA (cluster estimate):")
    for m, v in res.items():
        if m != "qlora":
            print(f"  {m:8} {(v/q-1)*100:+6.1f}%")
    print("\nnote: rsLoRA and LoftQ have the same per-step compute as QLoRA "
          "(scaling constant / init only), so agreement among those three is a "
          "sanity check on the measurement, not a finding.")
