"""Is the NF4-error ranking better than picking layers at random?

usage: python ranking_null_distribution.py [--n 10] [--model ...] [--k 29]
"""
import argparse
import gc
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="microsoft/phi-1_5")
ap.add_argument("--k", type=int, default=29)
ap.add_argument("--rho", type=float, default=None,
                help="budget as a fraction of linear layers; overrides --k")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--out", default="ranking_null.json")
ap.add_argument("--match", choices=["count", "params"], default="count",
                help="hold the layer count fixed, or the parameter total")
args = ap.parse_args()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer
import aqslora_core as core
from aqslora_core import log

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

stats, _ = core.profile_layers(args.model)
if args.rho is not None:
    args.k = max(1, round(args.rho * len(stats)))
    log(f"rho={args.rho} over {len(stats)} linear layers -> K={args.k}")
by_err = sorted(stats, key=lambda s: s["quant_mse"], reverse=True)
top = by_err[:args.k]
target = sum(s["numel"] for s in top)


def matched_random(seed):
    """K layers sampled to sit as close as possible to top-K's parameter count.

    Holding the count fixed cannot always match the parameters: where layer
    sizes vary widely, top-K selects systematically small layers and a random
    draw of the same count carries more of them. Then the control is unmatched
    on memory and the comparison is meaningless, so --match params instead
    holds the parameter total and lets the count float.
    """
    rng = random.Random(seed)
    best, gap = None, None
    for _ in range(5000):
        if args.match == "count":
            c = rng.sample(stats, args.k)
        else:
            pool = stats[:]
            rng.shuffle(pool)
            c, tot = [], 0
            for s in pool:
                if tot + s["numel"] <= target * 1.005:
                    c.append(s)
                    tot += s["numel"]
                if tot >= target * 0.995:
                    break
        g = abs(sum(s["numel"] for s in c) - target)
        if gap is None or g < gap:
            best, gap = c, g
        if g <= target * 0.001:
            break
    return best


def ppl_of(names, label):
    skip = [s for s in names]
    model, _, gib = core.load_base_model(
        args.model, "aqslora", skip, for_training=False)
    p = core.wikitext2_ppl(model, tok)
    log(f"{label}: ppl={p:.4f}  base={gib:.3f} GiB")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return p, gib


cases = [("top", [s["name"] for s in top]),
         ("bottom", [s["name"] for s in by_err[-args.k:]])]
for i in range(args.n):
    cases.append((f"random{i}", [s["name"] for s in matched_random(i)]))

out = {}
for label, names in cases:
    p, gib = ppl_of(names, label)
    out[label] = {"ppl": p, "base_gib": round(gib, 3), "k": len(names)}
    json.dump(out, open(args.out, "w"), indent=1)

rnd = sorted(v["ppl"] for k, v in out.items() if k.startswith("random"))
t = out["top"]["ppl"]
better = sum(1 for x in rnd if x < t)
print(f"\n  top-K   ppl = {t:.4f}")
print(f"  bottom  ppl = {out['bottom']['ppl']:.4f}")
print(f"  random  ppl = {min(rnd):.4f} .. {max(rnd):.4f}  (n={len(rnd)}, "
      f"median {rnd[len(rnd)//2]:.4f})")
print(f"  random draws beating top-K: {better}/{len(rnd)}")
print(f"\n  wrote {args.out}")
