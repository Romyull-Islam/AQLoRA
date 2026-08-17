"""Offline probe: is fp16 the right way to spend the protection budget?

usage: python residual_protect_probe.py [model_id] [sensitive_frac]
"""
import sys
import json
import torch
from transformers import AutoModelForCausalLM

from aqslora_core import NF4_LEVELS, nf4_quantization_error

BOUNDARIES = (NF4_LEVELS[1:] + NF4_LEVELS[:-1]) / 2


def nf4_round(blocks):
    """Quantize/dequantize one [n, block] tensor, block-wise absmax NF4."""
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    idx = torch.bucketize(blocks / absmax, BOUNDARIES)
    return NF4_LEVELS[idx] * absmax


def reshape(w, block):
    flat = w.flatten().float()
    n = (flat.numel() // block) * block
    if n > 4_000_000:            # subsample huge layers; same cap for every tier
        n = (4_000_000 // block) * block
    return flat[:n].view(-1, block)


def tiers(w):
    """MSE of each storage tier on one weight matrix."""
    b64 = reshape(w, 64)
    q1 = nf4_round(b64)
    # second stage sees only what the first could not represent
    q2 = nf4_round(b64 - q1)
    b32 = reshape(w, 32)
    return {
        "nf4_b64": torch.mean((b64 - q1) ** 2).item(),
        "nf4_b32": torch.mean((b32 - nf4_round(b32)) ** 2).item(),
        "residual": torch.mean((b64 - q1 - q2) ** 2).item(),
        "fp16": 0.0,
    }


# bits per parameter: payload + absmax scales (double-quantized, ~0.127 b/param
# at block 64, twice that at block 32, twice again for a second NF4 stage)
BITS = {"nf4_b64": 4.127, "nf4_b32": 4.254, "residual": 8.254, "fp16": 16.0}


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "microsoft/phi-1_5"
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20

    print(f"loading {model_id} on CPU ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu")

    layers = [(n, m.weight.data) for n, m in model.named_modules()
              if isinstance(m, torch.nn.Linear) and "lm_head" not in n]
    ranked = sorted(layers, key=lambda t: nf4_quantization_error(t[1]), reverse=True)
    k = int(round(len(ranked) * frac))
    protected = ranked[:k]
    print(f"{len(layers)} linear layers; probing the {k} protected at rho={frac}\n")

    acc, params = {t: 0.0 for t in BITS}, 0
    for name, w in protected:
        t = tiers(w)
        n = w.numel()
        params += n
        for key in acc:
            acc[key] += t[key] * n          # parameter-weighted, not layer-mean

    base = acc["nf4_b64"] / params
    print(f"{'tier':<12} {'bits/param':>10} {'MSE':>12} {'vs NF4':>10} {'premium':>9}")
    print("-" * 58)
    rows = {}
    for key in ("nf4_b64", "nf4_b32", "residual", "fp16"):
        mse = acc[key] / params
        red = "exact" if key == "fp16" else f"{base / mse:.1f}x" if mse > 0 else "-"
        prem = BITS[key] - BITS["nf4_b64"]
        print(f"{key:<12} {BITS[key]:>10.3f} {mse:>12.3e} {red:>10} {prem:>+8.2f}b")
        rows[key] = {"bits": BITS[key], "mse": mse, "premium_bits": prem}

    # what the premium costs on the whole model, protected layers only
    tot = sum(w.numel() for _, w in layers)
    print(f"\nprotected weights: {params / 1e6:.0f}M of {tot / 1e6:.0f}M "
          f"({100 * params / tot:.1f}%)")
    for key in ("nf4_b32", "residual", "fp16"):
        gib = params * rows[key]["premium_bits"] / 8 / 2**30
        print(f"  {key:<9} adds {gib:.3f} GiB over all-NF4")
        rows[key]["premium_gib"] = gib

    json.dump({"model": model_id, "rho": frac, "protected_params": params,
               "total_params": tot, "tiers": rows},
              open("residual_probe.json", "w"), indent=2)
    print("\nwrote residual_probe.json")


if __name__ == "__main__":
    main()
