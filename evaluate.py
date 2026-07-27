#!/usr/bin/env python3
"""
AQSLoRA — EVALUATION (no training).

Loads a base model + a trained adapter (produced by finetune.py) and scores it.
Because it reads <adapter-dir>/plan.json, it rebuilds the *exact* quantized base
the adapter was trained on (same fp16-protected layers for AQSLoRA).

Metrics (choose any; commonsense is the DoRA/LLM-Adapters standard):
  --commonsense   accuracy on the 8 held-out test sets  -> report["commonsense"]
  --ppl           WikiText-2 perplexity                  -> report["wikitext2_ppl_final"]
  --gen           leak-free webglm ROUGE/BLEU/EM         -> report["generation"]

Writes/merges <adapter-dir>/report.json (combining train_report.json + these metrics).

Examples
--------
  python evaluate.py --adapter-dir results-aqslora-commonsense-r8 --commonsense
  python evaluate.py --adapter-dir results-aqslora-commonsense-r8 --commonsense --ppl \
                     --cs-max-eval 300          # quick subset
"""
import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(description="AQSLoRA evaluation (no training)")
    p.add_argument("--adapter-dir", required=True,
                   help="dir from finetune.py (contains final/ and plan.json)")
    p.add_argument("--gpus", default="6,7")
    p.add_argument("--commonsense", action="store_true",
                   help="score the 8 commonsense test sets (DoRA protocol)")
    p.add_argument("--gsm8k", action="store_true",
                   help="GSM8K test accuracy (ABBA/LoRA-One protocol)")
    p.add_argument("--gsm8k-max-eval", type=int, default=-1)
    p.add_argument("--ppl", action="store_true", help="WikiText-2 perplexity")
    p.add_argument("--gen", action="store_true", help="webglm ROUGE/BLEU/EM")
    p.add_argument("--cs-eval-dir", default="./cs_eval")
    p.add_argument("--cs-max-eval", type=int, default=-1, help="cap per-task eval (-1=full)")
    p.add_argument("--eval-samples", type=int, default=200, help="webglm gen samples")
    p.add_argument("--base-only", action="store_true",
                   help="evaluate the quantized base WITHOUT the adapter "
                        "(isolates pure quantization damage)")
    return p.parse_args()


ARGS = parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ARGS.gpus)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402

import aqslora_core as core  # noqa: E402
from aqslora_core import log  # noqa: E402


def main():
    if not (ARGS.commonsense or ARGS.ppl or ARGS.gen or ARGS.gsm8k):
        raise SystemExit("choose a metric: --commonsense / --ppl / --gen / --gsm8k")

    plan = core.load_json(os.path.join(ARGS.adapter_dir, "plan.json"))
    log(f"plan: mode={plan['mode']} model={plan['model']} "
        f"sensitive={plan.get('sensitive_frac')} skip={len(plan.get('skip_modules') or [])} layers")

    # ---- rebuild the identical quantized base
    model, tokenizer, base_gib = core.load_base_model(
        plan["model"], plan["mode"], plan.get("skip_modules") or [],
        quant_type=plan.get("quant_type", "nf4"),
        double_quant=bool(plan.get("double_quant", True)),
        for_training=False,
        quant_backend=plan.get("quant_backend") or "bnb",
        attn_impl=plan.get("attn", "sdpa"),
        fp32_protected=bool(plan.get("fp32_protected")))

    # ---- attach the trained adapter (unless we want the raw base)
    if not ARGS.base_only:
        adapter_path = os.path.join(ARGS.adapter_dir, "final")
        model = PeftModel.from_pretrained(model, adapter_path)
        log(f"loaded adapter from {adapter_path}")
    model.eval()

    # ---- start from the training-side report if present, else fresh
    report_path = os.path.join(ARGS.adapter_dir, "report.json")
    train_report_path = os.path.join(ARGS.adapter_dir, "train_report.json")
    report = core.load_json(train_report_path) if os.path.exists(train_report_path) else dict(plan)

    if ARGS.ppl:
        ppl = core.wikitext2_ppl(model, tokenizer)
        key = "wikitext2_ppl_base" if ARGS.base_only else "wikitext2_ppl_final"
        report[key] = round(ppl, 4)
        log(f"wikitext-2 ppl ({'base' if ARGS.base_only else 'final'}): {ppl:.4f}")

    if ARGS.commonsense:
        report["commonsense"] = core.commonsense_accuracy(
            model, tokenizer, ARGS.cs_eval_dir, max_eval=ARGS.cs_max_eval)

    if ARGS.gsm8k:
        report["gsm8k"] = core.gsm8k_accuracy(
            model, tokenizer, max_eval=ARGS.gsm8k_max_eval)

    if ARGS.gen:
        _, _, raw_test = core.prepare_webglm(tokenizer, 600, smoke=False)
        scores, samples = core.generation_metrics(
            model, tokenizer, raw_test, ARGS.eval_samples)
        report["generation"] = {k: round(float(v), 4) for k, v in scores.items()}

    core.save_json(report, report_path)
    log(f"REPORT written to {report_path}")
    import json
    log(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
