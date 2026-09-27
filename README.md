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
arch_profile.py        what a layer budget costs per architecture
make_ranking_controls.py  builds top-K / random / bottom-K layer sets
ranking_null_distribution.py  the ranking against budget-matched random draws
timing_sensitivity.py  speed result under four estimators, with and without
                       the floor-failing sessions
unsloth_overlap.py     our selection against Unsloth's hand-curated list
summarize_ranking_null.py  the error ranking against matched-random selection,
                       under both the count-matched and parameter-matched controls
residual_protect_probe.py  two-stage NF4 as a cheaper protection tier (offline)
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
interleaved across methods, and the estimate is the mean of the fastest half.
`robust_timing.py` prints the individual samples alongside the estimate:

```
python robust_timing.py results/timing-sess-phi15-s2
```

Three findings from this protocol shape how the numbers should be read, and the
paper's headline speed figures come from the last of them:

- Fix the measurement duration, not the step count. A fixed step count gives
  fast models systematically shorter measurements and wider noise floors; four
  sweeps were discarded before this was understood
  (`scripts/run_timing_llama_fixed.sh`).
- Measure the floor from a duplicated arm. rsLoRA and LoftQ are only nearly
  compute-identical to QLoRA; running QLoRA twice under different labels gives a
  floor with no assumption in it, and it is wider than the proxy
  (`scripts/run_timing_floors.sh`).
- A within-session floor understates the real uncertainty. Repeating whole
  sessions moves the numbers far more than any within-session error bar
  predicts, so the paper reports speed as mean and standard deviation over nine
  clean sessions (`scripts/run_timing_sessions.sh`; sweeps that fail the floor
  test are in `results/` too, as the audit trail).

Wall-clock minutes in the accuracy tables come from shared-node runs and are
indicative only. They are not timing measurements.

Seeds do not make these runs reproducible: three runs at one seed spread as
widely as six runs at different seeds, because the training does not enable
deterministic kernels. The error bars measure run-to-run variation.

## Negative results

Four things in the repo did not work and are reported as such:

- The NF4-error ranking itself. Tested on five models at three budgets against
  random selections matched on protected parameters, top-K beats the median draw
  in 11 of 15 settings, which a sign test does not separate from chance
  (p=0.12), and the median gap is 0.01 perplexity. Downstream accuracy cannot
  separate top-K, matched random, and bottom-K on either model tested
  (`ranking_null_distribution.py --match params`, `summarize_ranking_null.py`,
  `results/ranking_null_*.json`, `results/results-cs-phi-3-mini-ctrl-*`). The
  count of protected layers carries the effect; the identity does not measurably.
  Matching the draws on layer count instead of parameters is not a valid control:
  where layer sizes vary, the most damaged layers are smaller than average, so a
  same-count draw carries more memory. On Llama-3.2-1B that reached 11%.
- The weight-density rule for choosing which layers to adapt does no better than
  a random subset at the same budget (`--placement sparsity|weightnorm|random`).
- A two-stage NF4 protection tier (`residual_protect_probe.py`,
  `--protect-tier residual`) removes 110x of the protected layers' quantization
  error at a third of fp16's memory premium, and is still dominated: lowering
  the budget reaches the same accuracy at less memory and no speed cost.
- `phase0_isomem_ppl.py` — pushing robust layers below 4 bits. Uniform 2-bit
  raises WikiText-2 perplexity from 18.97 to 112 at 1.4B and from 6.30 to 351 at
  7B, and every multi-tier plan containing a 2-bit tier inherits that.

## License

MIT, see [LICENSE](LICENSE). The base models and the training data keep their
own licenses, which this grant does not extend.
