"""Build control layer sets to test whether the NF4 error ranking does any work.

usage: python make_ranking_controls.py [model_id] [K]
"""
import json
import random
import sys

from aqslora_core import profile_layers


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "microsoft/phi-1_5"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 29

    stats, _ = profile_layers(model_id)
    by_err = sorted(stats, key=lambda s: s["quant_mse"], reverse=True)

    top = by_err[:K]
    bottom = by_err[-K:]
    target = sum(s["numel"] for s in top)

    # sample until the parameter count is close to top-K, so the comparison is a
    # budget-matched one and not an accidental test of protecting bigger layers
    rng = random.Random(0)
    best, best_gap = None, None
    for _ in range(20000):
        cand = rng.sample(stats, K)
        gap = abs(sum(s["numel"] for s in cand) - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = cand, gap
        if best_gap == 0:
            break

    for name, sel in (("top", top), ("bottom", bottom), ("random", best)):
        params = sum(s["numel"] for s in sel)
        mse = sum(s["quant_mse"] * s["numel"] for s in sel) / params
        json.dump([s["name"] for s in sel], open(f"protect_{name}_{K}.json", "w"), indent=1)
        print(f"  {name:7} K={len(sel)}  params={params/1e6:8.1f}M  "
              f"({100*params/target:5.1f}% of top-K)  mean NF4 mse={mse:.3e}")
    print(f"\nwrote protect_{{top,bottom,random}}_{K}.json")


if __name__ == "__main__":
    main()
