#!/usr/bin/env python3
"""Precision-budget dial: WikiText-2 PPL of the quantized base (no adapters).

Regenerates the numbers behind the paper's precision-budget table with artifacts
kept in this bundle. For each quantizer (NF4+DQ, fp4 without double quant) and
each protection budget rho, build the plan with the same profiler the method
uses, load the base, and measure WikiText-2 perplexity. No training, no adapters,
so this isolates the quantization itself.

Run: CUDA_VISIBLE_DEVICES=<gpu> python precision_dial_ppl.py --model microsoft/phi-1_5
"""
import argparse, gc, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="microsoft/phi-1_5")
ap.add_argument("--out", default="results-precision-dial")
ap.add_argument("--rhos", default="0,0.10,0.20")
args = ap.parse_args()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer
import aqslora_core as core
from aqslora_core import log

os.makedirs(args.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

stats, leaves = core.profile_layers(args.model)
results = []

# fp16 anchor (no quantization at all)
log("==== fp16 anchor ====")
model, _, fp_gib = core.load_base_model(args.model, "lora16", [], for_training=False)
ppl = core.wikitext2_ppl(model, tok)
log(f"[fp16] footprint={fp_gib:.3f} GiB  ppl={ppl:.4f}")
results.append({"quantizer": "fp16", "rho": None, "ppl": round(ppl, 4),
                "base_gib": round(fp_gib, 3)})
del model; gc.collect(); torch.cuda.empty_cache()

for qt, dq in [("nf4", True), ("fp4", False)]:
    for rho in [float(x) for x in args.rhos.split(",")]:
        skip = core.make_quant_plan(stats, rho) if rho > 0 else []
        mode = "aqslora" if rho > 0 else "qlora"
        log(f"==== {qt} (dq={dq}) rho={rho} -> {len(skip)} layers fp16 ====")
        t0 = time.time()
        model, _, gib = core.load_base_model(
            args.model, mode, skip, quant_type=qt, double_quant=dq,
            for_training=False)
        ppl = core.wikitext2_ppl(model, tok)
        log(f"[{qt} rho={rho}] footprint={gib:.3f} GiB  ppl={ppl:.4f}  "
            f"({time.time()-t0:.0f}s)")
        results.append({"quantizer": f"{qt}{'+DQ' if dq else ' (no DQ)'}",
                        "rho": rho, "n_fp16_layers": len(skip),
                        "ppl": round(ppl, 4), "base_gib": round(gib, 3)})
        del model; gc.collect(); torch.cuda.empty_cache()
        json.dump({"model": args.model, "results": results},
                  open(os.path.join(args.out, "precision_dial.json"), "w"), indent=2)

log("\n=========== PRECISION DIAL SUMMARY ===========")
log(f"{'quantizer':16} {'rho':>5} {'fp16 layers':>12} {'GiB':>7} {'wt2 ppl':>9}")
for r in results:
    log(f"{r['quantizer']:16} {str(r['rho']):>5} {str(r.get('n_fp16_layers','-')):>12} "
        f"{r['base_gib']:7.3f} {r['ppl']:9.4f}")
print("PRECISION-DIAL-DONE")
