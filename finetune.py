#!/usr/bin/env python3
"""Training entry point. Every method and lever is a command-line flag.

Writes the adapter, plan.json (so evaluate.py can rebuild the identical base) and
train_report.json. Evaluation is a separate step: run evaluate.py afterwards.
"""
import argparse
import os
import time


def parse_args():
    p = argparse.ArgumentParser(description="AQSLoRA fine-tuning (training only)")
    p.add_argument("--model", default="microsoft/phi-1_5")
    p.add_argument("--mode",
                   choices=["aqslora", "qlora", "lora16", "dora", "dora16",
                            "loftq", "pissa", "rslora", "adalora", "loraplus"],
                   default="aqslora")
    p.add_argument("--task", choices=["webglm", "commonsense", "math"], default="commonsense")
    p.add_argument("--gpus", default="6,7", help="CUDA_VISIBLE_DEVICES")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-seq-len", type=int, default=600)
    p.add_argument("--rslora-scaling", action="store_true")
    p.add_argument("--quant-type", choices=["nf4", "fp4"], default="nf4")
    p.add_argument("--no-double-quant", action="store_true")
    p.add_argument("--sensitive-frac", type=float, default=0.10)
    # how a protected layer is stored: fp16 (exact, +11.9 bits/param) or a
    # second NF4 stage over its residual (99% of the error gone, +4.1 bits)
    p.add_argument("--protect-tier", choices=["fp16", "residual"], default="fp16")
    p.add_argument("--adapter-frac", type=float, default=0.70)
    p.add_argument("--cs-max-train", type=int, default=-1)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    # Speed levers (each one is an ablation dial)
    p.add_argument("--placement",
                   choices=["sparsity", "depth", "weightnorm", "random", "depthmatched"],
                   default="sparsity",
                   help="adapter placement: 'sparsity' (least-sparse anywhere), "
                        "'depth' (top blocks -> backprop truncation), "
                        "'weightnorm'/'random' (reviewer control baselines)")
    p.add_argument("--truncate-backprop", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="stop backward at the lowest adapted block "
                        "(default: on for aqslora+depth, off otherwise)")
    p.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--optim", default="paged_adamw_8bit",
                   choices=["paged_adamw_8bit", "adamw_8bit",
                            "adamw_torch_fused", "adamw_torch"])
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="disable gradient checkpointing (faster, more activation mem)")
    p.add_argument("--group-by-length", action="store_true",
                   help="length-grouped batching (less padding waste)")
    p.add_argument("--quant-backend", choices=["bnb", "hqq"], default="bnb")
    p.add_argument("--hqq-bits", type=int, default=4,
                   help="HQQ base bit-width (smoke-test lever for sub-4-bit)")
    p.add_argument("--backward-drop", type=float, default=0.0,
                   help="sensitivity-gated backward-layer drop: base prob for the "
                        "most-robust block (0 = off). Requires --no-grad-ckpt.")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--loraplus-ratio", type=float, default=4.0,
                   help="LoRA+ B-matrix lr ratio (paper: 2-4 optimal for "
                        "LLaMA-class causal LMs; 16 was the RoBERTa regime)")
    p.add_argument("--fp32-protected", action="store_true",
                   help="compute the fp16-protected layers in fp32 (fixes AMP "
                        "overflow at 14B scale on pre-bf16 GPUs; storage unchanged)")
    p.add_argument("--skip-modules-file", default=None,
                   help="JSON list of module names to keep fp16, overriding the "
                        "profiler's plan (e.g. an Unsloth dynamic-4bit skip list)")
    p.add_argument("--no-loftq-callback", action="store_true",
                   help="disable the PEFT-recommended LoftQ logits callback "
                        "(falls back to the weaker one-shot init)")
    p.add_argument("--speed-profile", action="store_true",
                   help="preset: --lora-dropout 0 --group-by-length "
                        "--optim adamw_torch_fused")
    p.add_argument("--ppl-base", action="store_true",
                   help="also measure WikiText-2 PPL of the quantized base (pre-train)")
    p.add_argument("--smoke-test", action="store_true",
                   help="tiny run: 30 steps")
    return p.parse_args()


ARGS = parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ARGS.gpus)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

import aqslora_core as core
from aqslora_core import log

if ARGS.smoke_test:
    ARGS.max_steps = 30

if ARGS.speed_profile:
    ARGS.lora_dropout = 0.0
    ARGS.group_by_length = True
    ARGS.optim = "adamw_torch_fused"
    log("speed profile: lora_dropout=0, group-by-length, fused AdamW")

# backprop truncation defaults on only where it is both sound and useful
if ARGS.truncate_backprop is None:
    ARGS.truncate_backprop = (ARGS.mode == "aqslora" and ARGS.placement == "depth")

FP16_MODES = ("lora16", "dora16", "pissa")
OUTPUT_DIR = ARGS.output_dir or f"./results-{ARGS.mode}-{ARGS.task}-r{ARGS.rank}"


def main():
    t0 = time.time()
    torch.cuda.empty_cache()

    # Profile + adaptive plan (deterministic from weights)
    skip_modules, adapter_layers, leaves = core.build_plan(
        ARGS.model, ARGS.mode, ARGS.sensitive_frac, ARGS.adapter_frac,
        placement=ARGS.placement, seed=ARGS.seed)
    if ARGS.skip_modules_file:
        import json as _json
        skip_modules = [s for s in _json.load(open(ARGS.skip_modules_file))
                        if s != "lm_head"]  # lm_head re-added by loader
        log(f"OVERRIDE: fp16 skip list from {ARGS.skip_modules_file} "
            f"({len(skip_modules)} entries)")

    # Load quantized base
    model, tokenizer, base_gib = core.load_base_model(
        ARGS.model, ARGS.mode, skip_modules,
        quant_type=ARGS.quant_type, double_quant=not ARGS.no_double_quant,
        for_training=True, quant_backend=ARGS.quant_backend,
        attn_impl=ARGS.attn, grad_ckpt=not ARGS.no_grad_ckpt,
        fp32_protected=ARGS.fp32_protected, hqq_bits=ARGS.hqq_bits,
        protect_tier=ARGS.protect_tier)

    ppl_base = None
    if ARGS.ppl_base:
        ppl_base = core.wikitext2_ppl(model, tokenizer)
        log(f"wikitext-2 ppl (quantized base, no adapters): {ppl_base:.4f}")

    # Data
    if ARGS.task == "commonsense":
        train_ds, eval_ds = core.prepare_commonsense_train(
            tokenizer, ARGS.max_seq_len, max_train=ARGS.cs_max_train, smoke=ARGS.smoke_test)
    elif ARGS.task == "math":
        train_ds, eval_ds = core.prepare_math_train(
            tokenizer, ARGS.max_seq_len, smoke=ARGS.smoke_test)
    else:
        train_ds, eval_ds, _ = core.prepare_webglm(
            tokenizer, ARGS.max_seq_len, smoke=ARGS.smoke_test)

    # Adapters
    if ARGS.max_steps > 0:
        total_steps = ARGS.max_steps
    else:
        total_steps = int(np.ceil(
            len(train_ds) / (ARGS.batch_size * ARGS.grad_accum) * ARGS.epochs))
    # LoftQ fidelity: build a probe batch + fp16 reference logits so the
    # PEFT-recommended greedy callback can accept/reject each replacement.
    loftq_probe = loftq_ref = None
    if ARGS.mode == "loftq" and not ARGS.no_loftq_callback:
        probe_feats = [train_ds[i] for i in range(min(4, len(train_ds)))]
        loftq_probe = {k: v.to(model.device) for k, v in
                       core.CompletionCollator(tokenizer)(probe_feats).items()
                       if k != "labels"}
        log("computing fp16 reference logits for the LoftQ callback ...")
        ref_model = __import__("transformers").AutoModelForCausalLM.from_pretrained(
            ARGS.model, torch_dtype=torch.float16, device_map="auto",
            attn_implementation=ARGS.attn)
        with torch.no_grad():
            loftq_ref = ref_model(**loftq_probe).logits.float()
        del ref_model
        torch.cuda.empty_cache()

    model, n_trainable = core.attach_lora(
        model, ARGS.mode, ARGS.rank, adapter_layers, leaves,
        rslora_scaling=ARGS.rslora_scaling, total_steps=total_steps,
        lora_dropout=ARGS.lora_dropout,
        truncate_backprop=ARGS.truncate_backprop,
        loftq_probe=loftq_probe, loftq_ref_logits=loftq_ref)

    if ARGS.backward_drop > 0:
        if not ARGS.no_grad_ckpt:
            raise SystemExit("--backward-drop requires --no-grad-ckpt "
                             "(gradient checkpointing re-samples the drop RNG).")
        bd_stats, _ = core.profile_layers(ARGS.model)
        model = core.apply_backward_drop(model, bd_stats, ARGS.backward_drop)

    # Training arguments
    targs = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=ARGS.batch_size,
        per_device_eval_batch_size=ARGS.batch_size,
        gradient_accumulation_steps=ARGS.grad_accum,
        num_train_epochs=ARGS.epochs,
        max_steps=ARGS.max_steps,
        learning_rate=ARGS.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        optim=ARGS.optim,
        fp16=True,
        max_grad_norm=1.0,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200 if not ARGS.smoke_test else 15,
        save_strategy="no",
        gradient_checkpointing=not ARGS.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        torch_compile=ARGS.compile,
        report_to="none",
        seed=ARGS.seed,
    )

    optimizers = (None, None)
    if ARGS.mode == "loraplus":
        import bitsandbytes as bnb_lib
        b_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and "lora_B" in n]
        a_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and "lora_B" not in n]
        opt = bnb_lib.optim.PagedAdamW8bit(
            [{"params": a_params, "lr": ARGS.lr},
             {"params": b_params, "lr": ARGS.lr * ARGS.loraplus_ratio}], lr=ARGS.lr)
        optimizers = (opt, None)
        log(f"LoRA+ optimizer: {len(b_params)} B-matrices at lr ratio "
            f"{ARGS.loraplus_ratio:g} -> {ARGS.lr*ARGS.loraplus_ratio:.0e}")

    callbacks = []
    if ARGS.mode == "adalora":
        from transformers import TrainerCallback

        class AdaLoraStep(TrainerCallback):
            def __init__(self, peft_model):
                self.peft_model = peft_model

            def on_pre_optimizer_step(self, args, state, control, **kwargs):
                self.peft_model.base_model.update_and_allocate(state.global_step)

        callbacks.append(AdaLoraStep(model))

    trainer_cls = Trainer
    if ARGS.group_by_length:
        lengths = [len(x) for x in train_ds["input_ids"]]
        sampler = core.LengthGroupedSampler(lengths, ARGS.batch_size, seed=ARGS.seed)
        log(f"group-by-length: mega-batch sampler over {len(lengths)} examples")

        class GroupedTrainer(Trainer):
            # transformers 5.x removed group_by_length; signature of
            # _get_train_sampler differs across versions, so accept anything.
            def _get_train_sampler(self, *a, **k):
                return sampler

        trainer_cls = GroupedTrainer

    trainer = trainer_cls(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=core.CompletionCollator(tokenizer),
        callbacks=callbacks, optimizers=optimizers)

    torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    trainer.train()
    train_minutes = (time.time() - t_train) / 60
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    final_eval = trainer.evaluate()
    log(f"training done in {train_minutes:.1f} min | peak {peak_gib:.2f} GiB | "
        f"eval_loss {final_eval['eval_loss']:.4f}")

    # Save adapter + tokenizer
    adapter_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Save the plan so evaluate.py rebuilds the IDENTICAL base
    core.save_json({
        "model": ARGS.model,
        "mode": ARGS.mode,
        "task": ARGS.task,
        "rank": ARGS.rank,
        "seed": ARGS.seed,
        "quant_type": ARGS.quant_type if ARGS.mode not in FP16_MODES else "none",
        "double_quant": (not ARGS.no_double_quant) if ARGS.mode not in FP16_MODES else None,
        "sensitive_frac": ARGS.sensitive_frac if ARGS.mode == "aqslora" else None,
        "adapter_frac": ARGS.adapter_frac if ARGS.mode == "aqslora" else None,
        "placement": ARGS.placement if ARGS.mode == "aqslora" else None,
        "quant_backend": ARGS.quant_backend if ARGS.mode not in FP16_MODES else None,
        "attn": ARGS.attn,
        "fp32_protected": ARGS.fp32_protected,
        "skip_modules": skip_modules,
    }, os.path.join(OUTPUT_DIR, "plan.json"))

    # Training-side report (evaluate.py appends the quality metrics)
    core.save_json({
        "model": ARGS.model,
        "mode": ARGS.mode,
        "task": ARGS.task,
        "rank": ARGS.rank,
        "seed": ARGS.seed,
        "quant_type": ARGS.quant_type if ARGS.mode not in FP16_MODES else "none",
        "sensitive_frac": ARGS.sensitive_frac if ARGS.mode == "aqslora" else None,
        "adapter_frac": ARGS.adapter_frac if ARGS.mode == "aqslora" else None,
        "placement": ARGS.placement if ARGS.mode == "aqslora" else None,
        "truncate_backprop": ARGS.truncate_backprop,
        "quant_backend": ARGS.quant_backend if ARGS.mode not in FP16_MODES else None,
        "lora_dropout": ARGS.lora_dropout,
        "optim": ARGS.optim,
        "grad_ckpt": not ARGS.no_grad_ckpt,
        "group_by_length": ARGS.group_by_length,
        "compile": ARGS.compile,
        "trainable_params": n_trainable,
        "base_model_gib": round(base_gib, 3),
        "peak_train_gib": round(peak_gib, 3),
        "train_minutes": round(train_minutes, 2),
        "eval_loss": round(final_eval["eval_loss"], 4),
        "wikitext2_ppl_base": round(ppl_base, 4) if ppl_base else None,
        "backward_drop": ARGS.backward_drop,
    }, os.path.join(OUTPUT_DIR, "train_report.json"))

    log(f"adapter + plan saved to {OUTPUT_DIR}")
    log(f"next: python evaluate.py --adapter-dir {OUTPUT_DIR}")
    log(f"total wall time {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
