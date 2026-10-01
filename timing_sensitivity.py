"""Does the speed result depend on the estimator or the exclusion rule?

Recomputes the pooled speed numbers under four estimators, with and without
the sessions that fail the duplicate-arm floor test. Reproduces Appendix M.

usage: python timing_sensitivity.py [results_dir]
"""
import glob
import json
import os
import statistics as st
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
GAP_H = 6.0          # hours of silence that separate one sweep from the next
FLOOR = 3.0          # duplicate-arm disagreement (%) above which a session is out

# The thirteen sessions that pass the floor test, and the five that do not.
CLEAN = ["timing-session2-phi15", "timing-sess-phi15-s2", "timing-llama-fixedwindow",
         "timing-sess-llama-3.2-1b-s1", "timing-llama-3.2-3b-retry",
         "timing-floor-phi-3-mini", "timing-phi-3-mini", "timing-sess-phi-3-mini-s2",
         "timing-clean-qwen2.5-7b-validated",
         # between-session replicates for the two single-session models
         "timing-sess-qwen2.5-7b-s2", "timing-sess-qwen2.5-7b-s3",
         "timing-sess-llama-3.2-3b-s2", "timing-sess-llama-3.2-3b-s3"]
EXCLUDED = ["timing-floor-phi15", "timing-sess-phi15-s1", "timing-sess-llama-3.2-1b-s2",
            "timing-llama-3.2-3b", "timing-sess-phi-3-mini-s1-starved"]


def newest_session(paths):
    """Sweeps overwrite by tag, so cut at the largest gap in write times."""
    stamped = sorted((os.path.getmtime(p), p) for p in paths)
    cut = 0
    for i in range(1, len(stamped)):
        if stamped[i][0] - stamped[i - 1][0] > GAP_H * 3600:
            cut = i
    return [p for _, p in stamped[cut:]]


def samples(session):
    by = {}
    for p in newest_session(glob.glob(f"{ROOT}/{session}/*-r*/timing.json")):
        try:
            v = json.load(open(p))["steps_per_sec"]
        except Exception:
            continue
        m = os.path.basename(os.path.dirname(p)).rsplit("-r", 1)[0]
        by.setdefault(m, []).append(v)
    return by


def fastest_half(v):
    """Contention only slows a run, so anchor the estimate at the fast end."""
    v = sorted(v, reverse=True)
    return st.mean(v[:min(len(v), max(2, round(0.5 * len(v))))])


ESTIMATORS = {"fastest half": fastest_half, "plain mean": st.mean,
              "median": st.median, "single fastest": max}


def floor_of(by):
    """Disagreement between the two identically-configured QLoRA arms."""
    if "qlora-b" not in by:
        return None
    return abs(fastest_half(by["qlora-b"]) / fastest_half(by["qlora"]) - 1) * 100


def gains(sessions, est):
    q, s = [], []
    for name in sessions:
        by = samples(name)
        if "qlora" not in by:
            continue
        ref = est(by["qlora"])
        if "aq-q" in by:
            q.append((est(by["aq-q"]) / ref - 1) * 100)
        if "aq-s" in by:
            s.append((est(by["aq-s"]) / ref - 1) * 100)
    return q, s


def fmt(v):
    return f"{st.mean(v):+5.1f} +/- {st.stdev(v):3.1f}" if len(v) > 1 else f"{v[0]:+5.1f}"


print("per-session duplicate-arm floor (blank = proxy floor only)\n")
for name in CLEAN + EXCLUDED:
    by = samples(name)
    if not by:
        print(f"  {name:36} MISSING")
        continue
    f = floor_of(by)
    tag = "excluded" if name in EXCLUDED else "kept"
    print(f"  {name:36} floor={'' if f is None else f'{f:5.2f}%':>6}  {tag}")

print(f"\npooled speed vs QLoRA (exclusion threshold {FLOOR}%)\n")
print(f"  {'estimator':15} {'sessions':16} {'n':>2} {'AQLoRA-q':>14} {'AQLoRA-s':>14}")
for label, est in ESTIMATORS.items():
    for sess, tag in ((CLEAN, "clean"), (CLEAN + EXCLUDED, "all")):
        q, s = gains(sess, est)
        print(f"  {label:15} {tag:16} {len(q):2d} {fmt(q):>14} {fmt(s):>14}")
