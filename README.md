# AQLoRA: A Zero-Search Recipe for Fast Quantized LoRA Fine-Tuning

Code and run artifacts for the paper. One CPU pass over the weights sets
per-layer precision and adapter placement, with no search and no calibration
data. Everything runs on the stock HuggingFace / bitsandbytes / PEFT stack: no
custom kernels, no FP8 hardware.

## Layout

```
aqslora_core.py        profiling pass, precision and placement plans, model
                       loading, data pipelines, 8-task scoring, WikiText-2 PPL
finetune.py            training entry point; every method and lever is a flag
evaluate.py            evaluation; rebuilds the exact quantized base from plan.json
timing.py              throughput measurement (steady state, cuda.synchronize)
robust_timing.py       fast-anchored estimator over repeated timing runs
precision_dial_ppl.py  base perplexity against the protection budget
phase0_isomem_ppl.py   sub-4-bit and multi-tier study (the negative result)
scripts/               campaign launchers
paper_assets/          builds the figures and appendix tables from the artifacts
results/               report.json / timing.json for every run in the paper
```

`results/` holds the raw output of each run, so every number in the paper can be
recomputed without a GPU.

## Setup

```
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
```

On V100 (sm_70) use `torch==2.7.1+cu126`. Newer CUDA wheels ship no sm_70
kernels and bitsandbytes fails to load.

## Reproducing a result

```
python finetune.py --model microsoft/phi-1_5 --mode aqslora \
    --sensitive-frac 0.20 --adapter-frac 1.0 --speed-profile \
    --task commonsense --epochs 1 --batch-size 6 --gpus 0 \
    --output-dir results-aqslora-q
python evaluate.py --adapter-dir results-aqslora-q --gpus 0 --commonsense
```

Settings used in the paper:

| row | flags |
|---|---|
| AQLoRA-q (quality) | `--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0` |
| AQLoRA-s (speed) | `--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.85 --placement depth` |
| QLoRA | `--mode qlora` |
| rsLoRA | `--mode rslora` |
| LoftQ | `--mode loftq` |
| QDoRA | `--mode dora` |
| fp16 references | `--mode lora16`, `--mode dora16` |

All headline rows add `--speed-profile` (dropout 0, length-grouped batching,
fused AdamW), applied identically to every method.

## Reading the numbers

**Accuracy in the paper is the mean of the eight task scores.** `report.json`
also has an `average` field, which is example-weighted over the pooled test sets
and is a different number; the paper does not use it. To recompute a row:

```python
import json
T = ["boolq","piqa","social_i_qa","hellaswag",
     "winogrande","ARC-Easy","ARC-Challenge","openbookqa"]
c = json.load(open("results/<run>/report.json"))["commonsense"]
print(sum(c[t] for t in T) / 8 * 100)
```

## Timing

Timing runs on a shared node, so the samples are one-sided: interference can
only slow a run, never speed it up. Each configuration is measured several times,
interleaved across methods so drift affects all of them equally, and the estimate
is the mean of the fastest half. `robust_timing.py` prints the individual samples
alongside the estimate so it can be checked:

```
./scripts/retime_all.sh
python robust_timing.py timing-session2-phi15
```

The noise floor is measured, not assumed: QLoRA, rsLoRA and LoftQ have identical
per-step compute, so their spread bounds the measurement error.

Wall-clock minutes in the accuracy tables come from shared-node runs and are
indicative only. They are not timing measurements.

## Negative results

Two things in the repo did not work and are reported as such:

- `phase0_isomem_ppl.py` — pushing robust layers below 4 bits. Uniform 2-bit
  raises WikiText-2 perplexity from 18.97 to 112 at 1.4B and from 6.30 to 351 at
  7B, and every multi-tier plan containing a 2-bit tier inherits that.
- The weight-density rule for choosing which layers to adapt does no better than
  a random subset at the same budget (`--placement sparsity|weightnorm|random`).
