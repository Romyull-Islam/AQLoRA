"""What the allocation rule actually sees in each architecture.

usage: python arch_profile.py [model_id ...]
"""
import sys
import statistics as st

from aqslora_core import profile_layers

MODELS = sys.argv[1:] or [
    "microsoft/phi-1_5",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "microsoft/Phi-3-mini-4k-instruct",
]

print(f"{'model':<34}{'layers':>7}{'leaf types':>12}{'size CV':>9}"
      f"{'K@.20':>7}{'prot %params':>14}{'premium GiB':>13}")
print("-" * 96)
for mid in MODELS:
    try:
        stats, leaves = profile_layers(mid)
    except Exception as e:
        print(f"{mid:<34}  FAILED: {type(e).__name__}")
        continue
    sizes = [s["numel"] for s in stats]
    total = sum(sizes)
    cv = st.stdev(sizes) / st.mean(sizes)          # how uneven the layers are
    by_err = sorted(stats, key=lambda s: s["quant_mse"], reverse=True)
    k = int(round(len(stats) * 0.20))
    prot = sum(s["numel"] for s in by_err[:k])
    # fp16 storage costs 16 bits where NF4 costs about 4.13
    premium = prot * (16 - 4.127) / 8 / 2**30
    print(f"{mid:<34}{len(stats):>7}{len(leaves):>12}{cv:>9.2f}"
          f"{k:>7}{100*prot/total:>13.1f}%{premium:>12.3f}")
print()
print("size CV is the coefficient of variation of layer parameter counts:")
print("  low means a top-K cut costs roughly K/L of the weights,")
print("  high means the cut's cost depends heavily on which layers it lands on.")
