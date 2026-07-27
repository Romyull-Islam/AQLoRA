#!/usr/bin/env python3
"""
AQSLoRA core — shared internals for `finetune.py` and `evaluate.py`.

This module holds everything both scripts need (so training and evaluation are
cleanly separated but never drift apart): the adaptive-precision profiler, the
quantized model loader, LoRA attachment, the Alpaca/webglm data pipelines, and
the metrics (WikiText-2 PPL, leak-free generation, commonsense accuracy).

AQSLoRA method in one paragraph
-------------------------------
1. Profile each linear layer on a CPU fp16 copy: block-wise NF4 reconstruction
   error (sensitivity) + near-zero weight fraction (sparsity). One pass, no search.
2. Keep the top-K most quantization-fragile layers in fp16; store the rest in
   bitsandbytes NF4 (+ double-quant). lm_head is always fp16.
3. Attach LoRA adapters only to the least-sparse (most informative) layers.
4. Optionally compose rank-stabilized scaling (rsLoRA, alpha/sqrt(r)).

Nothing here reads argv/env; callers pass every knob explicitly, so the exact
same functions build the model at train time and at eval time.
"""

import gc
import json
import os
import re

import numpy as np
import torch
from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

RESPONSE_MARKER = "Answer:\n"


def log(msg):
    print(f"[aqslora] {msg}", flush=True)


# ==========================================================================
# 1. Profiling: per-layer sensitivity (NF4 error) and sparsity, on CPU fp16.
# ==========================================================================
NF4_LEVELS = torch.tensor(
    [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
     -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
     0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
     0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
)


def nf4_quantization_error(w: torch.Tensor, block: int = 64) -> float:
    """MSE introduced by block-wise absmax NF4 quantization (mirrors bitsandbytes)."""
    flat = w.flatten().float()
    n = (flat.numel() // block) * block
    if n == 0:
        return 0.0
    if n > 2_000_000:  # subsample huge layers; this is an estimate, not exact
        n = 2_000_000
    blocks = flat[:n].view(-1, block)
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    normed = blocks / absmax
    idx = torch.bucketize(normed, (NF4_LEVELS[1:] + NF4_LEVELS[:-1]) / 2)
    deq = NF4_LEVELS[idx] * absmax
    return torch.mean((blocks - deq) ** 2).item()


def profile_layers(model_id: str):
    """Load weights on CPU once; return per-linear-layer stats + leaf module names."""
    log(f"profiling {model_id} on CPU (fp16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu")
    stats = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            w = module.weight.data
            stats.append({
                "name": name,
                "leaf": name.split(".")[-1],
                "numel": w.numel(),
                "sparsity": (w.abs() < 0.01).float().mean().item(),
                "mean_abs": w.abs().mean().item(),
                "quant_mse": nf4_quantization_error(w),
            })
    leaves = sorted({s["leaf"] for s in stats})
    log(f"found {len(stats)} linear layers; module types: {leaves}")
    del model
    gc.collect()
    return stats, leaves


def make_quant_plan(stats, sensitive_frac):
    """Top-K layers by NF4 error stay fp16; the rest go NF4. Returns skip list."""
    k = int(round(len(stats) * sensitive_frac))
    by_err = sorted(stats, key=lambda s: s["quant_mse"], reverse=True)
    skip = [s["name"] for s in by_err[:k]]
    log(f"adaptive precision: {k}/{len(stats)} most sensitive layers kept fp16")
    for s in by_err[:k]:
        log(f"  fp16  {s['name']}  (nf4_mse={s['quant_mse']:.2e})")
    return skip


def _block_idx(name):
    """Transformer block index of a layer name, e.g. 'model.layers.12.mlp.fc1' -> 12."""
    m = re.search(r"\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def make_adapter_plan(stats, adapter_frac, placement="sparsity", seed=42):
    """Choose which layers get LoRA adapters.

    placement:
      'sparsity'   — least-sparse (fewest near-zero weights) layers anywhere.
      'depth'      — the TOP `adapter_frac` fraction of layers by block depth
                     (ties broken by density). Because no adapter exists below
                     the cut, autograd stops the backward pass at the lowest
                     adapted block — lower blocks skip recompute + backprop.
      'weightnorm' — largest mean |w| layers (Act-LoRA-style baseline; reviewer
                     control for the sparsity criterion).
      'random'     — uniform random layer subset (control baseline).
    """
    k = max(1, int(len(stats) * adapter_frac))
    if placement == "weightnorm":
        order = sorted(stats, key=lambda s: -s.get("mean_abs", 0.0))
        chosen = [s["name"] for s in order[:k]]
        log(f"adapters (weightnorm baseline): LoRA on {k}/{len(stats)} layers")
    elif placement == "random":
        import random as _rnd
        rng = _rnd.Random(seed)
        chosen = [s["name"] for s in rng.sample(stats, k)]
        log(f"adapters (random baseline, seed {seed}): LoRA on {k}/{len(stats)} layers")
    elif placement == "depth":
        max_blk = max((b for b in (_block_idx(s["name"]) for s in stats)
                       if b is not None), default=0)
        order = sorted(stats, key=lambda s: (
            -(max_blk + 1 if _block_idx(s["name"]) is None else _block_idx(s["name"])),
            s["sparsity"]))
        chosen = [s["name"] for s in order[:k]]
        cut = min((b for b in (_block_idx(n) for n in chosen) if b is not None),
                  default=0)
        log(f"adaptive adapters (depth): LoRA on {k}/{len(stats)} layers, "
            f"lowest adapted block = {cut} -> backward pass truncates below it")
    else:
        by_density = sorted(stats, key=lambda s: s["sparsity"])
        chosen = [s["name"] for s in by_density[:k]]
        dropped = by_density[k:]
        log(f"adaptive adapters (sparsity): LoRA on {k}/{len(stats)} layers "
            f"({len(dropped)} sparse layers get no adapter)")
        if dropped:
            worst = ", ".join(f"{s['name']}({s['sparsity']:.2f})" for s in dropped[:5])
            log(f"  e.g. skipped (sparsity): {worst} ...")
    return chosen


def build_plan(model_id, mode, sensitive_frac, adapter_frac, placement="sparsity",
               seed=42):
    """Convenience: profile once and return the (skip_modules, adapter_layers, leaves)."""
    stats, leaves = profile_layers(model_id)
    skip_modules = make_quant_plan(stats, sensitive_frac)
    adapter_layers = make_adapter_plan(stats, adapter_frac, placement=placement,
                                       seed=seed)
    return skip_modules, adapter_layers, leaves


# --------------------------------------------------------------------------
# Sensitivity-gated backward-layer dropping ("zero-search DropBP").
# The per-block drop probability comes from the SAME profiling pass that sets
# precision and placement: robust (low NF4-error) blocks are dropped in the
# backward pass more often, fragile ones are kept. A dropped block still runs
# its forward (model output intact) but under no_grad, so its activations are
# not stored and its adapters get no gradient that step; the residual passes the
# gradient through as identity. Cuts backward compute + activation memory.
# Requires gradient checkpointing OFF (otherwise the recompute re-samples the
# drop RNG). Mechanism follows DropBP (Woo et al., 2024); our part is the
# calibration-free per-block schedule from the profiling map.
# --------------------------------------------------------------------------
def _find_decoder_layers(model):
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith("layers"):
            if best is None or len(mod) > len(best):
                best = mod
    return best


def block_drop_probs(stats, base_p):
    """robust (low NF4 error) block -> high drop prob; fragile -> ~0."""
    from collections import defaultdict
    err = defaultdict(list)
    for s in stats:
        b = _block_idx(s["name"])
        if b is not None:
            err[b].append(s["quant_mse"])
    mean = {b: sum(v) / len(v) for b, v in err.items()}
    lo, hi = min(mean.values()), max(mean.values())
    return {b: (base_p * (hi - mean[b]) / (hi - lo) if hi > lo else base_p)
            for b in mean}


def apply_backward_drop(model, stats, base_p):
    layers = _find_decoder_layers(model)
    if layers is None:
        log("backward-drop: no decoder ModuleList found; skipping")
        return model
    probs = block_drop_probs(stats, base_p)
    for i, layer in enumerate(layers):
        p_i = probs.get(i, 0.0)
        orig = layer.forward

        def make(orig, p_i):
            def fwd(hidden_states, *a, **k):
                if p_i > 0 and torch.is_grad_enabled() \
                        and getattr(hidden_states, "requires_grad", False) \
                        and torch.rand((), device=hidden_states.device) < p_i:
                    with torch.no_grad():
                        out = orig(hidden_states, *a, **k)
                    h = out[0] if isinstance(out, tuple) else out
                    h = hidden_states + (h - hidden_states).detach()
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                return orig(hidden_states, *a, **k)
            return fwd
        layer.forward = make(orig, p_i)
    log(f"backward-drop ON: {len(layers)} blocks, base_p={base_p}, "
        f"mean p={sum(probs.values())/len(probs):.2f}, "
        f"max p={max(probs.values()):.2f} (robust)")
    return model


# ==========================================================================
# 2. Model + adapters (explicit params — no globals)
# ==========================================================================
def _wrap_fp32_compute(model, skip_modules):
    """Compute protected (fp16-stored) layers in fp32.

    The protected layers are by construction the widest-range layers; at 14B
    scale their fp16 backward can overflow and silently kill AMP training
    (GradScaler collapse -> zero grads). Upcasting just these matmuls to fp32
    restores range at unchanged storage. Needed only on pre-bf16 GPUs (V100).
    """
    import torch.nn.functional as F
    import types
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and any(s in name for s in skip_modules):
            def fp32_forward(self, x):
                with torch.autocast("cuda", enabled=False):
                    b = self.bias.float() if self.bias is not None else None
                    return F.linear(x.float(), self.weight.float(), b).to(x.dtype)
            mod.forward = types.MethodType(fp32_forward, mod)
            n += 1
    log(f"fp32-compute wrapper on {n} protected layers (storage stays fp16)")
    return model


def load_base_model(model_id, mode, skip_modules, quant_type="nf4",
                    double_quant=True, for_training=True,
                    quant_backend="bnb", attn_impl="sdpa", grad_ckpt=True,
                    fp32_protected=False, hqq_bits=4):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if mode in ("lora16", "dora16", "pissa"):
        quant_cfg = None
    elif quant_backend == "hqq":
        from transformers import HqqConfig
        # HQQ 4-bit (torch backend; compile-friendly dequant). Adaptive fp16
        # protection uses the same skip mechanism as bitsandbytes.
        quant_cfg = HqqConfig(
            nbits=hqq_bits, group_size=64,
            skip_modules=(skip_modules + ["lm_head"]) if mode == "aqslora"
            else ["lm_head"])
    else:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_compute_dtype=torch.float16,
            # AQSLoRA: sensitive layers stay fp16. NOTE passing skip_modules
            # REPLACES bnb's default ["lm_head"], so re-add it explicitly.
            llm_int8_skip_modules=(skip_modules + ["lm_head"]) if mode == "aqslora" else None,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16,
        attn_implementation=attn_impl,
        quantization_config=quant_cfg)
    if for_training:
        model.config.use_cache = False
        if grad_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
    if fp32_protected and mode == "aqslora" and skip_modules:
        model = _wrap_fp32_compute(model, skip_modules)
    footprint = model.get_memory_footprint() / 2**30
    log(f"base model footprint ({mode}, {quant_backend if quant_cfg else 'fp16'}): "
        f"{footprint:.2f} GiB")
    return model, tokenizer, footprint


def attach_lora(model, mode, rank, adapter_layers, all_leaves,
                rslora_scaling=False, total_steps=1000,
                lora_dropout=0.05, truncate_backprop=False,
                loftq_probe=None, loftq_ref_logits=None, loftq_passes=2):
    target = adapter_layers if mode == "aqslora" else all_leaves
    if mode == "adalora":
        from peft import AdaLoraConfig
        cfg = AdaLoraConfig(
            init_r=2 * rank, target_r=rank, lora_alpha=2 * rank,
            lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM",
            target_modules=target, total_step=total_steps,
            tinit=max(1, int(0.1 * total_steps)),
            tfinal=max(1, int(0.3 * total_steps)), deltaT=10)
    else:
        cfg = LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=target,
            use_dora=(mode in ("dora", "dora16")),
            use_rslora=(mode == "rslora") or rslora_scaling,
            init_lora_weights="pissa" if mode == "pissa" else True)
    model = get_peft_model(model, cfg)
    if mode == "loftq":
        from peft.utils.loftq_utils import replace_lora_weights_loftq
        if loftq_probe is not None and loftq_ref_logits is not None:
            # PEFT-docs-recommended greedy callback: keep a replacement only if
            # it moves the quantized model's logits closer to the fp16
            # reference on a probe batch; repeated passes give incremental
            # gains (the plain one-shot call is a weaker approximation of the
            # paper's iterative init and can make layers WORSE).
            state = {"mse": None}

            @torch.no_grad()
            def _probe_mse():
                out = model(**loftq_probe).logits.float()
                return torch.mean((out - loftq_ref_logits) ** 2).item()

            def _cb(_model, _name):
                mse = _probe_mse()
                if state["mse"] is None or mse < state["mse"]:
                    state["mse"] = mse
                    return True
                return False

            state["mse"] = _probe_mse()
            log(f"LoftQ callback init: probe MSE vs fp16 = {state['mse']:.5f}")
            for it in range(loftq_passes):
                replace_lora_weights_loftq(model, callback=_cb)
                log(f"LoftQ pass {it + 1}/{loftq_passes}: probe MSE = {state['mse']:.5f}")
        else:
            log("applying LoftQ adapter re-initialization (one-shot, no callback) ...")
            replace_lora_weights_loftq(model)
    if truncate_backprop:
        # PEFT force-enables input-require-grads on quantized bases (a relic of
        # reentrant checkpointing). With use_reentrant=False it is unnecessary,
        # and removing it lets autograd STOP at the lowest adapted block —
        # blocks below the cut skip recompute + backward entirely. Only sound
        # when no adapter exists below the cut (placement='depth').
        try:
            model.get_base_model().disable_input_require_grads()
            log("backprop truncation ON: input-require-grads hook removed; "
                "backward stops at the lowest adapted block")
        except (AttributeError, ValueError) as e:
            log(f"backprop truncation: no input-require-grads hook to remove ({e})")
    trainable, total = model.get_nb_trainable_parameters()
    log(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
    return model, trainable


# ==========================================================================
# 3. Data pipelines
# ==========================================================================
def build_prompt(question):
    return f"Instruct: Answer the following question.\nQuestion: {question}\n{RESPONSE_MARKER}"


def prepare_webglm(tokenizer, max_len, smoke=False):
    ds = load_dataset("THUDM/webglm-qa")
    train, test = ds["train"], ds["test"]
    if smoke:
        train = train.select(range(2000))

    def tok(batch):
        input_ids, labels = [], []
        for q, a in zip(batch["question"], batch["answer"]):
            pids = tokenizer(build_prompt(q), add_special_tokens=False)["input_ids"]
            aids = tokenizer(a + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            ids = (pids + aids)[:max_len]
            lab = ([-100] * len(pids) + aids)[:max_len]
            input_ids.append(ids)
            labels.append(lab)
        return {"input_ids": input_ids, "labels": labels}

    cols = train.column_names
    train = train.map(tok, batched=True, remove_columns=cols, desc="tokenize train")
    test_tok = test.map(tok, batched=True, remove_columns=cols, desc="tokenize test")
    return train, test_tok, test


# ---- Commonsense reasoning (DoRA / LLM-Adapters protocol) -----------------
ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n")
ALPACA_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")

CS_TASKS = {
    "boolq": ["true", "false"],
    "piqa": ["solution1", "solution2"],
    "social_i_qa": ["answer1", "answer2", "answer3"],
    "hellaswag": ["ending1", "ending2", "ending3", "ending4"],
    "winogrande": ["option1", "option2"],
    "ARC-Easy": ["answer1", "answer2", "answer3", "answer4", "answer5"],
    "ARC-Challenge": ["answer1", "answer2", "answer3", "answer4", "answer5"],
    "openbookqa": ["answer1", "answer2", "answer3", "answer4", "answer5"],
}


def alpaca_prompt(instruction, inp=""):
    inp = (inp or "").strip()
    return (ALPACA_INPUT if inp else ALPACA_NO_INPUT).format(instruction=instruction, input=inp)


def prepare_commonsense_train(tokenizer, max_len, max_train=-1, smoke=False):
    """Tokenize Commonsense-170K, completion-only loss, Alpaca template."""
    ds = load_dataset("zwhe99/commonsense_170k", split="train")
    if smoke:
        ds = ds.select(range(2000))
    elif max_train > 0:
        ds = ds.select(range(min(max_train, len(ds))))
    n_eval = min(500, len(ds) // 20)
    eval_slice = ds.select(range(n_eval))
    train_slice = ds.select(range(n_eval, len(ds)))

    def tok(batch):
        input_ids, labels = [], []
        for ins, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
            pids = tokenizer(alpaca_prompt(ins, inp), add_special_tokens=False)["input_ids"]
            aids = tokenizer(out + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            ids = (pids + aids)[:max_len]
            lab = ([-100] * len(pids) + aids)[:max_len]
            input_ids.append(ids)
            labels.append(lab)
        return {"input_ids": input_ids, "labels": labels}

    cols = ds.column_names
    train_tok = train_slice.map(tok, batched=True, remove_columns=cols, desc="tok cs-train")
    eval_tok = eval_slice.map(tok, batched=True, remove_columns=cols, desc="tok cs-eval")
    log(f"commonsense-170k: {len(train_tok)} train / {len(eval_tok)} held-out")
    return train_tok, eval_tok


# ---- Math reasoning (MetaMathQA-20K -> GSM8K, ABBA/LoRA-One protocol) -----
def prepare_math_train(tokenizer, max_len, subset=20000, smoke=False):
    """Tokenize a MetaMathQA subset, completion-only loss, Alpaca template."""
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    ds = ds.shuffle(seed=42).select(range(min(2000 if smoke else subset, len(ds))))
    n_eval = min(500, len(ds) // 20)
    eval_slice = ds.select(range(n_eval))
    train_slice = ds.select(range(n_eval, len(ds)))

    def tok(batch):
        input_ids, labels = [], []
        for q, r in zip(batch["query"], batch["response"]):
            pids = tokenizer(alpaca_prompt(q), add_special_tokens=False)["input_ids"]
            aids = tokenizer(r + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            ids = (pids + aids)[:max_len]
            lab = ([-100] * len(pids) + aids)[:max_len]
            input_ids.append(ids)
            labels.append(lab)
        return {"input_ids": input_ids, "labels": labels}

    cols = ds.column_names
    train_tok = train_slice.map(tok, batched=True, remove_columns=cols, desc="tok math-train")
    eval_tok = eval_slice.map(tok, batched=True, remove_columns=cols, desc="tok math-eval")
    log(f"metamathqa: {len(train_tok)} train / {len(eval_tok)} held-out")
    return train_tok, eval_tok


_NUM_RE = r"-?\$?[\d,]*\.?\d+"


def _extract_answer(text):
    """Final numeric answer: prefer 'The answer is: X' (MetaMath), else last number."""
    import re
    m = re.findall(r"[Tt]he answer is:?\s*(" + _NUM_RE + ")", text)
    cand = m[-1] if m else None
    if cand is None:
        nums = re.findall(_NUM_RE, text)
        cand = nums[-1] if nums else None
    if cand is None:
        return None
    cand = cand.replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(cand)
    except ValueError:
        return None


@torch.no_grad()
def gsm8k_accuracy(model, tokenizer, max_eval=-1, batch_size=8, max_new_tokens=400):
    """Greedy generation on the GSM8K test set; exact-match on the final number."""
    import re
    data = load_dataset("openai/gsm8k", "main", split="test")
    if max_eval > 0:
        data = data.select(range(min(max_eval, len(data))))
    model.eval()
    tokenizer.padding_side = "left"
    correct = 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        prompts = [alpaca_prompt(q) for q in batch["question"]]
        toks = tokenizer(prompts, return_tensors="pt", padding=True,
                         truncation=True, max_length=768).to(model.device)
        out = model.generate(**toks, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.batch_decode(out[:, toks["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        for text, gold_raw in zip(gen, batch["answer"]):
            gold = _extract_answer("The answer is " +
                                   gold_raw.split("####")[-1].strip())
            pred = _extract_answer(text)
            correct += int(pred is not None and gold is not None
                           and abs(pred - gold) < 1e-4)
        log(f"  gsm8k {min(i + batch_size, len(data))}/{len(data)} "
            f"(acc so far {correct / min(i + batch_size, len(data)):.4f})")
    tokenizer.padding_side = "right"
    acc = correct / len(data)
    log(f"GSM8K accuracy = {acc:.4f}  ({correct}/{len(data)})")
    return round(acc, 4)


class LengthGroupedSampler(torch.utils.data.Sampler):
    """Shuffled mega-batches, sorted by length inside each mega-batch.

    With dynamic padding this removes most padding waste (Commonsense-170K
    lengths span ~20..600 tokens): batches become near-uniform in length while
    the data order stays stochastic across epochs. transformers 5.x dropped
    the built-in `group_by_length`, so we ship our own.
    """

    def __init__(self, lengths, batch_size, mega_factor=50, seed=42):
        self.lengths = lengths
        self.batch_size = batch_size
        self.mega = batch_size * mega_factor
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(len(self.lengths), generator=g).tolist()
        out = []
        for i in range(0, len(perm), self.mega):
            chunk = sorted(perm[i:i + self.mega], key=lambda j: -self.lengths[j])
            out.extend(chunk)
        return iter(out)


class CompletionCollator:
    """Dynamic padding per batch; pads labels with -100."""

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch_ids, batch_labels, batch_mask = [], [], []
        for f in features:
            ids, lab = list(f["input_ids"]), list(f["labels"])
            pad = max_len - len(ids)
            batch_mask.append([1] * len(ids) + [0] * pad)
            batch_ids.append(ids + [self.tok.pad_token_id] * pad)
            batch_labels.append(lab + [-100] * pad)
        return {
            "input_ids": torch.tensor(batch_ids),
            "attention_mask": torch.tensor(batch_mask),
            "labels": torch.tensor(batch_labels),
        }


# ==========================================================================
# 4. Metrics
# ==========================================================================
@torch.no_grad()
def wikitext2_ppl(model, tokenizer, max_len=2048, stride=512):
    """Sliding-window perplexity on wikitext-2-raw test (standard protocol)."""
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    input_ids = enc.input_ids
    model.eval()
    total_nll, n_tokens, prev_end = 0.0, 0, 0
    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + max_len, input_ids.size(1))
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(model.device)
        targets = ids.clone()
        targets[:, :-trg_len] = -100
        out = model(ids, labels=targets)
        num = (targets[:, 1:] != -100).sum().item()
        total_nll += out.loss.float().item() * num
        n_tokens += num
        prev_end = end
        if end == input_ids.size(1):
            break
    return float(np.exp(total_nll / n_tokens))


@torch.no_grad()
def generation_metrics(model, tokenizer, raw_test, n_samples, batch_size=16):
    """Leak-free webglm generation metrics: prompt contains the QUESTION ONLY."""
    import evaluate as hf_evaluate
    rouge = hf_evaluate.load("rouge")
    smoothie = SmoothingFunction().method4

    model.eval()
    tokenizer.padding_side = "left"
    preds, refs, bleus = [], [], []
    data = raw_test.select(range(min(n_samples, len(raw_test))))
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        prompts = [build_prompt(q) for q in batch["question"]]
        toks = tokenizer(prompts, return_tensors="pt", padding=True,
                         truncation=True, max_length=512).to(model.device)
        out = model.generate(**toks, max_new_tokens=150, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        gen_only = out[:, toks["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        for text, ref in zip(texts, batch["answer"]):
            preds.append(text.strip())
            refs.append(ref.strip())
            bleus.append(sentence_bleu([ref.split()], text.split(),
                                       smoothing_function=smoothie))
        log(f"  gen eval {min(i + batch_size, len(data))}/{len(data)}")
    tokenizer.padding_side = "right"
    scores = rouge.compute(predictions=preds, references=refs,
                           use_aggregator=True, use_stemmer=True)
    scores["bleu"] = float(np.mean(bleus))
    scores["exact_match"] = float(np.mean([p == r for p, r in zip(preds, refs)]))
    return scores, list(zip(preds[:3], refs[:3]))


def _parse_cs_pred(text, labels):
    """Return the first label (by position) that appears in the generation."""
    t = text.lower().replace(" ", "")
    best, best_pos = None, len(t) + 1
    for lab in labels:
        pos = t.find(lab)
        if 0 <= pos < best_pos:
            best, best_pos = lab, pos
    return best


@torch.no_grad()
def commonsense_accuracy(model, tokenizer, eval_dir, max_eval=-1, batch_size=16):
    """Greedy-generate on each of the 8 test sets; parse and score accuracy."""
    model.eval()
    tokenizer.padding_side = "left"
    per_task, all_correct, all_total = {}, 0, 0
    for task, labels in CS_TASKS.items():
        path = os.path.join(eval_dir, f"{task}.json")
        if not os.path.exists(path):
            log(f"  [skip] {task}: {path} not found")
            continue
        data = json.load(open(path))
        if max_eval > 0:
            data = data[:max_eval]
        correct = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            prompts = [alpaca_prompt(r["instruction"], r.get("input", "")) for r in batch]
            toks = tokenizer(prompts, return_tensors="pt", padding=True,
                             truncation=True, max_length=1024).to(model.device)
            out = model.generate(**toks, max_new_tokens=12, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            gen = tokenizer.batch_decode(out[:, toks["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
            for text, r in zip(gen, batch):
                pred = _parse_cs_pred(text, labels)
                gold = str(r["answer"]).lower().replace(" ", "")
                correct += int(pred == gold)
            log(f"  {task}: {min(i + batch_size, len(data))}/{len(data)}")
        acc = correct / len(data)
        per_task[task] = round(acc, 4)
        all_correct += correct
        all_total += len(data)
        log(f"  [{task}] accuracy = {acc:.4f}  ({correct}/{len(data)})")
    tokenizer.padding_side = "right"
    per_task["average"] = round(all_correct / all_total, 4) if all_total else 0.0
    log(f"COMMONSENSE 8-task average accuracy = {per_task['average']:.4f}")
    return per_task


# ==========================================================================
# 5. Small IO helpers shared by the two entry points
# ==========================================================================
def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
