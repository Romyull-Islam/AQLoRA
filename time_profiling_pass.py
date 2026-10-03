"""Wall-time of the profiling pass on CPU, per model.

Each model is profiled twice; the second run reads the checkpoint from the page
cache ("warm"), the first from disk ("cold"). Writes results/profile_timing.json.

usage: python time_profiling_pass.py
"""
import json
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import aqslora_core as core

MODELS = ["microsoft/phi-1_5", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B",
          "microsoft/Phi-3-mini-4k-instruct", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B"]
out = {}
for m in MODELS:
    times = []
    for _ in range(2):
        t = time.time()
        stats, _ = core.profile_layers(m)
        times.append(time.time() - t)
    out[m] = {"cold_s": round(times[0], 1), "warm_s": round(times[1], 1),
              "linear_layers": len(stats),
              "params_M": round(sum(s["numel"] for s in stats) / 1e6)}
    print(f"{m:36} cold {times[0]:6.1f}s  warm {times[1]:6.1f}s  "
          f"({len(stats)} layers, {out[m]['params_M']}M params)", flush=True)
    json.dump({"threads": torch.get_num_threads(), "models": out},
              open("results/profile_timing.json", "w"), indent=1)
