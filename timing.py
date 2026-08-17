#!/usr/bin/env python3
"""Throughput measurement: steady-state steps/s and real (non-pad) tokens/s
under cuda.synchronize, with a discarded warmup.

Use robust_timing.py to combine repeated runs into one estimate.
"""
import argparse
import json
import os
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/phi-1_5")
    p.add_argument("--mode", default="qlora",
                   choices=["aqslora", "qlora", "lora16", "dora", "loftq", "rslora"])
    p.add_argument("--gpus", default="0")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=220)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-seq-len", type=int, default=600)
    p.add_argument("--rslora-scaling", action="store_true")
    p.add_argument("--sensitive-frac", type=float, default=0.20)
    # a protected layer skips dequantization under the fp16 tier and does two
    # under the residual tier, so the tier is a speed variable, not just memory
    p.add_argument("--protect-tier", choices=["fp16", "residual"], default="fp16")
    p.add_argument("--adapter-frac", type=float, default=1.0)
    p.add_argument("--warmup-skip", type=int, default=20, help="steps to discard as warmup")
    p.add_argument("--output-dir", default="./timing-out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="", help="free-form label copied into timing.json")
    # Speed levers (mirror finetune.py)
    p.add_argument("--placement",
                   choices=["sparsity", "depth", "weightnorm", "random"],
                   default="sparsity")
    p.add_argument("--truncate-backprop", action=argparse.BooleanOptionalAction,
                   default=None)
    p.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--optim", default="paged_adamw_8bit",
                   choices=["paged_adamw_8bit", "adamw_8bit",
                            "adamw_torch_fused", "adamw_torch"])
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--group-by-length", action="store_true")
    p.add_argument("--quant-backend", choices=["bnb", "hqq"], default="bnb")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--speed-profile", action="store_true",
                   help="preset: --lora-dropout 0 --group-by-length --optim adamw_torch_fused")
    p.add_argument("--edge", action="store_true",
                   help="phone-GPU profile: bs=1, grad-accum 16, seq 256, rank 4 "
                        "+ TFLOPS extrapolation")
    return p.parse_args()


A = parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", A.gpus)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import Trainer, TrainingArguments, TrainerCallback  # noqa: E402

import aqslora_core as core  # noqa: E402
from aqslora_core import log  # noqa: E402

if A.speed_profile:
    A.lora_dropout = 0.0
    A.group_by_length = True
    A.optim = "adamw_torch_fused"
if A.edge:
    A.batch_size, A.grad_accum, A.max_seq_len, A.rank = 1, 16, 256, 4
    log("EDGE profile: bs=1 accum=16 seq=256 r=4 (phone-realistic shape)")
if A.truncate_backprop is None:
    A.truncate_backprop = (A.mode == "aqslora" and A.placement == "depth")


class CountingCollator(core.CompletionCollator):
    """CompletionCollator that also counts real vs padded tokens."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.real_tokens = 0
        self.padded_tokens = 0

    def __call__(self, features):
        batch = super().__call__(features)
        self.real_tokens += int(batch["attention_mask"].sum())
        self.padded_tokens += int(batch["attention_mask"].numel())
        return batch


def main():
    torch.cuda.empty_cache()
    skip, adapter_layers, leaves = core.build_plan(
        A.model, A.mode, A.sensitive_frac, A.adapter_frac, placement=A.placement,
        seed=A.seed)

    model, tok, base_gib = core.load_base_model(
        A.model, A.mode, skip, for_training=True,
        quant_backend=A.quant_backend, attn_impl=A.attn,
        grad_ckpt=not A.no_grad_ckpt, protect_tier=A.protect_tier)
    n_base = sum(p.numel() for p in model.parameters())

    model, tr = core.attach_lora(
        model, A.mode, A.rank, adapter_layers, leaves,
        rslora_scaling=A.rslora_scaling, total_steps=A.max_steps,
        lora_dropout=A.lora_dropout, truncate_backprop=A.truncate_backprop)

    ds = load_dataset("zwhe99/commonsense_170k", split="train").select(range(20000))

    def tk(batch):
        ii, ll = [], []
        for ins, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
            p = tok(core.alpaca_prompt(ins, inp), add_special_tokens=False)["input_ids"]
            a = tok(out + tok.eos_token, add_special_tokens=False)["input_ids"]
            ii.append((p + a)[:A.max_seq_len])
            ll.append(([-100] * len(p) + a)[:A.max_seq_len])
        return {"input_ids": ii, "labels": ll}

    ds = ds.map(tk, batched=True, remove_columns=ds.column_names, desc="tokenize")
    collator = CountingCollator(tok)

    # timer that starts AFTER warmup steps, so load/compile jitter is excluded
    class WarmTimer(TrainerCallback):
        def __init__(s):
            s.t0, s.n0, s.tok0, s.pad0 = None, 0, 0, 0

        def on_step_end(s, args, state, control, **k):
            if state.global_step == A.warmup_skip:
                torch.cuda.synchronize()
                s.t0, s.n0 = time.time(), state.global_step
                s.tok0, s.pad0 = collator.real_tokens, collator.padded_tokens
                torch.cuda.reset_peak_memory_stats()
    wt = WarmTimer()

    targs = TrainingArguments(
        output_dir=A.output_dir, per_device_train_batch_size=A.batch_size,
        gradient_accumulation_steps=A.grad_accum, max_steps=A.max_steps,
        learning_rate=A.lr, lr_scheduler_type="cosine", warmup_steps=10,
        optim=A.optim, fp16=True, max_grad_norm=1.0, logging_steps=50,
        save_strategy="no", report_to="none",
        gradient_checkpointing=not A.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        torch_compile=A.compile, seed=A.seed)

    trainer_cls = Trainer
    if A.group_by_length:
        lengths = [len(x) for x in ds["input_ids"]]
        sampler = core.LengthGroupedSampler(lengths, A.batch_size, seed=A.seed)

        class GroupedTrainer(Trainer):
            def _get_train_sampler(self, *a, **k):
                return sampler
        trainer_cls = GroupedTrainer

    trainer = trainer_cls(model=model, args=targs, train_dataset=ds,
                          data_collator=collator, callbacks=[wt])

    torch.cuda.reset_peak_memory_stats()
    t_all = time.time()
    trainer.train()
    torch.cuda.synchronize()
    total_min = (time.time() - t_all) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30
    steps = A.max_steps - wt.n0
    steady_s = (time.time() - wt.t0) if wt.t0 else total_min * 60
    sps = steps / steady_s if steady_s > 0 else 0
    real_tok = collator.real_tokens - wt.tok0
    padded_tok = collator.padded_tokens - wt.pad0
    tok_per_s = real_tok / steady_s if steady_s > 0 else 0
    pad_frac = 1 - real_tok / max(1, padded_tok)

    # crude fine-tuning FLOPs/token: fwd 2N + bwd 4N (+2N recompute if grad ckpt);
    # backprop truncation scales the bwd(+recompute) part by the adapted-depth frac.
    bwd_frac = (A.adapter_frac if (A.mode == "aqslora" and A.truncate_backprop)
                else 1.0)
    flops_per_tok = 2 * n_base + (4 + (0 if A.no_grad_ckpt else 2)) * n_base * bwd_frac
    achieved_tflops = flops_per_tok * tok_per_s / 1e12

    rep = {"mode": A.mode, "model": A.model, "rslora": A.rslora_scaling, "tag": A.tag,
           "levers": {"placement": A.placement, "truncate_backprop": A.truncate_backprop,
                      "attn": A.attn, "lora_dropout": A.lora_dropout, "optim": A.optim,
                      "grad_ckpt": not A.no_grad_ckpt, "group_by_length": A.group_by_length,
                      "quant_backend": A.quant_backend, "compile": A.compile,
                      "edge": A.edge},
           "batch_size": A.batch_size, "grad_accum": A.grad_accum,
           "max_seq_len": A.max_seq_len, "rank": A.rank, "max_steps": A.max_steps,
           "seed": A.seed, "warmup_skip": A.warmup_skip,
           "torch": __import__("torch").__version__,
           "total_minutes_all_steps": round(total_min, 3),
           "steady_seconds": round(steady_s, 2), "steady_steps": steps,
           "steps_per_sec": round(sps, 4),
           "real_tokens_per_sec": round(tok_per_s, 1),
           "padding_waste_frac": round(pad_frac, 4),
           "peak_gib": round(peak, 3), "base_gib": round(base_gib, 3),
           "trainable_params": tr,
           "est_flops_per_token": flops_per_tok,
           "achieved_tflops": round(achieved_tflops, 3)}
    os.makedirs(A.output_dir, exist_ok=True)
    json.dump(rep, open(os.path.join(A.output_dir, "timing.json"), "w"), indent=2)
    log(f"RESULT {json.dumps(rep)}")
    print(f"\n### {A.mode}{'+rs' if A.rslora_scaling else ''} [{A.tag}]: "
          f"{sps:.3f} steps/s | {tok_per_s:,.0f} tok/s (pad waste {pad_frac:.1%}) | "
          f"peak {peak:.2f} GiB | est. achieved {achieved_tflops:.2f} TFLOPS")

    if A.edge:
        print("\n--- phone-GPU extrapolation (same efficiency, FLOPs-scaled) ---")
        micro_tok = real_tok / max(1, steps * A.grad_accum)  # tokens per micro-batch
        # planned device ladder: Jetson AGX Orin (CUDA, direct port) ->
        # OnePlus 15 (Adreno) / iPhone 16 Pro (Metal) via an export runtime
        for name, tflops in [("OnePlus-15-class Adreno (3 TF fp16 eff.)", 3.0),
                             ("iPhone-16-Pro-class Metal (4 TF fp16 eff.)", 4.0),
                             ("Jetson AGX Orin GPU (10 TF fp16 eff.)", 10.0)]:
            scale = achieved_tflops / tflops
            print(f"  {name}: ~{tok_per_s / scale:,.0f} tok/s "
                  f"-> micro-step {micro_tok / (tok_per_s / scale):.2f}s, "
                  f"1k-example epoch ~{1000 * micro_tok * scale / tok_per_s / 60:.1f} min")
        print("  (efficiency-parity extrapolation; kernel maturity on mobile will "
              "lower this — treat as an upper bound and validate on-device)")


if __name__ == "__main__":
    main()
