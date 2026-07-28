#!/usr/bin/env python3
"""
Phase-0a: iso-memory PPL + footprint ladder.

Question this answers (training-free, minutes):
  At a matched ~4-bit MEMORY budget, does ADAPTIVE mixed-precision quantization
  match (or beat) UNIFORM NF4 (QLoRA's base) on WikiText-2 PPL?

For each quant config we report (a) real GPU memory footprint and (b) WikiText-2
PPL of the quantized base (no adapters, no training). bnb-NF4+double-quant is the
QLoRA reference point; the HQQ configs let us push some layers below 4 bit and
spend the saved bits protecting the most quantization-sensitive layer types.

The adaptive plans are built from the SAME profiler AQSLoRA uses: per-linear-type
mean NF4 reconstruction error -> protect the top-sensitive TYPES at higher bits.
(HQQ's transformers integration maps bits by linear-tag/type, not per-layer-index.)

Run:
  CUDA_VISIBLE_DEVICES=0 python phase0_isomem_ppl.py --model microsoft/phi-1_5
"""
import argparse
import collections
import gc
import json
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/phi-1_5")
    p.add_argument("--out", default="results-phase0-isomem")
    p.add_argument("--skip-fp16", action="store_true", help="skip the fp16 anchor PPL")
    return p.parse_args()


ARGS = parse_args()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from hqq.core.quantize import BaseQuantizeConfig
from hqq.models.hf.base import AutoHQQHFModel

import aqslora_core as core
from aqslora_core import log


def linear_tag(name):
    """model.layers.31.self_attn.k_proj -> self_attn.k_proj  (HQQ's tag rule)."""
    return ".".join(n for n in name.split(".") if n not in ("model", "layers") and not n.isnumeric())


def profile_types(model_id):
    """Per-linear-TYPE mean NF4 error + numel share, ranked most->least sensitive."""
    stats, _ = core.profile_layers(model_id)
    by_type = collections.defaultdict(lambda: {"mse": [], "numel": 0, "n": 0})
    for s in stats:
        t = linear_tag(s["name"])
        by_type[t]["mse"].append(s["quant_mse"])
        by_type[t]["numel"] += s["numel"]
        by_type[t]["n"] += 1
    rows = []
    for t, d in by_type.items():
        rows.append({"type": t, "mean_mse": float(sum(d["mse"]) / len(d["mse"])),
                     "numel": d["numel"], "n": d["n"]})
    rows.sort(key=lambda r: -r["mean_mse"])
    total = sum(r["numel"] for r in rows)
    for r in rows:
        r["numel_frac"] = r["numel"] / total
    log("per-TYPE sensitivity (most->least fragile):")
    for r in rows:
        log(f"  {r['type']:22s} mean_nf4_mse={r['mean_mse']:.3e}  "
            f"numel_frac={r['numel_frac']:.3f}  (n={r['n']})")
    return rows


def nominal_avg_bits(type_rows, bits_map):
    """Weight-averaged nominal bits (ignores scale/zero overhead) for labelling.
    bits_map: {type_tag: nbits}; a type absent from the map is fp16 (16 bits)."""
    tot = sum(r["numel"] for r in type_rows)
    acc = sum(r["numel"] * bits_map.get(r["type"], 16) for r in type_rows)
    return acc / tot


def build_configs(type_rows):
    """Assemble the ladder. Each HQQ config is a full {type_tag: nbits} map so no
    type is accidentally left fp16. Adaptive plans protect the top-sensitive TYPES."""
    tags = [r["type"] for r in type_rows]            # all linear types
    ranked = [r["type"] for r in type_rows]          # already sorted most->least fragile
    top1, top2 = ranked[0], ranked[1]                # e.g. q_proj, k_proj

    def uniform(b):
        return {t: b for t in tags}

    def base_protect(base_b, protect):
        m = {t: base_b for t in tags}
        m.update(protect)
        return m

    cfgs = []
    cfgs.append(("bnb-nf4-dq (QLoRA ref)", "bnb", None))
    cfgs.append(("hqq-uniform-4bit", "hqq", uniform(4)))
    cfgs.append(("hqq-uniform-3bit", "hqq", uniform(3)))
    cfgs.append(("hqq-uniform-2bit", "hqq", uniform(2)))
    # adaptive iso-memory candidates: push base low, protect top-sensitive types high
    cfgs.append((f"hqq-adapt base3 [{top1},{top2}]@4", "hqq",
                 base_protect(3, {top1: 4, top2: 4})))
    cfgs.append((f"hqq-adapt base2 [{top1},{top2}]@8", "hqq",
                 base_protect(2, {top1: 8, top2: 8})))
    cfgs.append((f"hqq-adapt base2 [{top1},{top2}]@4", "hqq",
                 base_protect(2, {top1: 4, top2: 4})))
    # 3-tier AQLoRA-style: fp16 top fragile (omitted from map -> stays fp16),
    # 4-bit middle, 2-bit for the two least-fragile types.
    lo1, lo2 = ranked[-1], ranked[-2]
    m3 = {t: 4 for t in tags if t != top1}
    m3[lo1] = 2; m3[lo2] = 2
    cfgs.append((f"hqq-3tier fp16[{top1}] base4 2bit[{lo1},{lo2}]", "hqq", m3))
    return cfgs


def load_bnb_nf4(model_id):
    qc = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda:0", torch_dtype=torch.float16, quantization_config=qc)


def load_hqq(model_id, bits_map):
    """Load fp16, then HQQ-quantize each linear TYPE to its assigned bits (native API)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16)
    quant_config = {t: BaseQuantizeConfig(nbits=b, group_size=64)
                    for t, b in bits_map.items()}
    AutoHQQHFModel.quantize_model(
        model, quant_config=quant_config, compute_dtype=torch.float16, device="cuda:0")
    return model


def measure(model_id, name, kind, bits_map, type_rows, tokenizer):
    log(f"==== loading [{name}] ====")
    t0 = time.time()
    if kind == "bnb":
        model = load_bnb_nf4(model_id)
        avg_bits = 4.0
    else:
        model = load_hqq(model_id, bits_map)
        avg_bits = nominal_avg_bits(type_rows, bits_map)
    load_s = time.time() - t0
    footprint = model.get_memory_footprint() / 2**30
    t1 = time.time()
    ppl = core.wikitext2_ppl(model, tokenizer)
    ppl_s = time.time() - t1
    log(f"[{name}]  footprint={footprint:.3f} GiB  nominal_avg_bits={avg_bits:.2f}  "
        f"ppl={ppl:.4f}  (load {load_s:.1f}s, ppl {ppl_s:.1f}s)")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"name": name, "kind": kind, "footprint_gib": round(footprint, 3),
            "nominal_avg_bits": round(avg_bits, 3), "wikitext2_ppl": round(ppl, 4),
            "load_s": round(load_s, 1), "ppl_s": round(ppl_s, 1)}


def main():
    os.makedirs(ARGS.out, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ARGS.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    type_rows = profile_types(ARGS.model)
    results = []

    if not ARGS.skip_fp16:
        log("==== loading [fp16 anchor] ====")
        model = AutoModelForCausalLM.from_pretrained(
            ARGS.model, device_map="cuda:0", torch_dtype=torch.float16)
        fp = model.get_memory_footprint() / 2**30
        ppl = core.wikitext2_ppl(model, tokenizer)
        log(f"[fp16 anchor]  footprint={fp:.3f} GiB  ppl={ppl:.4f}")
        results.append({"name": "fp16 (anchor)", "kind": "fp16",
                        "footprint_gib": round(fp, 3), "nominal_avg_bits": 16.0,
                        "wikitext2_ppl": round(ppl, 4)})
        del model
        gc.collect()
        torch.cuda.empty_cache()

    for name, kind, bits_map in build_configs(type_rows):
        try:
            results.append(measure(ARGS.model, name, kind, bits_map,
                                   type_rows, tokenizer))
        except Exception as e:
            log(f"[{name}] FAILED: {type(e).__name__}: {e}")
            results.append({"name": name, "kind": kind, "error": f"{type(e).__name__}: {e}"})
        with open(os.path.join(ARGS.out, "phase0_ppl.json"), "w") as f:
            json.dump({"model": ARGS.model, "results": results}, f, indent=2)

    # Summary table
    log("\n================ PHASE-0a SUMMARY ================")
    log(f"{'config':40s} {'GiB':>7s} {'avgbit':>7s} {'wt2_ppl':>9s}")
    for r in results:
        if "error" in r:
            log(f"{r['name']:40s}  ERROR: {r['error']}")
        else:
            log(f"{r['name']:40s} {r['footprint_gib']:7.3f} "
                f"{r['nominal_avg_bits']:7.2f} {r['wikitext2_ppl']:9.4f}")
    log(f"\nwrote {os.path.join(ARGS.out, 'phase0_ppl.json')}")


if __name__ == "__main__":
    main()
