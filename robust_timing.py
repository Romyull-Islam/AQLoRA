#!/usr/bin/env python3
"""Timing estimate from repeated runs on a shared node.

Contention only slows a run, never speeds it up, so the samples have a ceiling at
the true speed and a slow tail. The estimate is the mean of the fastest half:
several runs, so not one lucky maximum, but without the interference-slowed tail.
"""
import json, glob, os, statistics as st, sys, time

DIR = sys.argv[1] if len(sys.argv) > 1 else "timing-session2-phi15"
METHODS = ["qlora", "qlora-b", "aq-q", "aq-s", "rslora", "loftq", "lora16",
           "aq-f10", "aq-f05", "aq-r20", "aq-r40"]
SESSION_GAP_H = 6.0   # hours of silence that separate one sweep from the next


def newest_session(paths):
    """Keep only the most recent sweep's files.

    Sweeps write into the same directory and overwrite by tag, so a sweep that
    dies partway leaves the tail of an older one behind. Those files look
    identical to fresh ones. Mixing a discarded sweep into a clean one is the
    failure this whole protocol exists to prevent, so cut at the largest gap in
    write times and report what was dropped rather than averaging across it.
    """
    stamped = sorted(((os.path.getmtime(p), p) for p in paths))
    cut = 0
    for i in range(1, len(stamped)):
        if stamped[i][0] - stamped[i - 1][0] > SESSION_GAP_H * 3600:
            cut = i
    return [p for _, p in stamped[cut:]], [p for _, p in stamped[:cut]]


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
keep, dropped = newest_session(glob.glob(f"{DIR}/*-r*/timing.json"))
if dropped:
    old = time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(dropped[-1])))
    new = time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(keep[0])))
    print(f"provenance: {len(dropped)} file(s) from an earlier sweep (to {old}) "
          f"excluded; this sweep starts {new}")
    for p in sorted(dropped):
        print(f"  dropped {os.path.basename(os.path.dirname(p))}")
keep = set(keep)
print(f"{'method':8} {'n':>2} {'all samples (steps/s, fastest first)':<44} "
      f"{'fast-k':>8} {'spread':>7} {'mean':>7} {'max':>7}")
res = {}
for m in METHODS:
    vals = []
    for j in sorted(glob.glob(f"{DIR}/{m}-r*/timing.json")):
        if j not in keep:
            continue
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
