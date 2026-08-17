#!/bin/bash
# Protection-budget sweep, downstream. rho=0.20 was fixed at the start of the
# project and never tuned against 8-task accuracy, only against base perplexity.
# The premium AQLoRA pays over QLoRA is entirely a function of rho, so if a
# smaller budget holds accuracy, the memory gap shrinks proportionally:
#   rho=0.20 -> 29 fp16 layers, +0.22 GiB
#   rho=0.10 -> 14 fp16 layers, +0.10 GiB
#   rho=0.05 ->  7 fp16 layers, +0.05 GiB
# Accuracy runs, so contention does not matter.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }
# hold off until the timing sweep is finished; it needs the node to itself
while ! grep -q 'GAP 1 done' gaps.log 2>/dev/null; do sleep 180; done
echo "[$(STAMP)] timing finished; starting rho sweep"
free_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
             awk -F', ' '$2 < 2000 {print $1; exit}'; }
wait_gpu() { local g; while :; do g=$(free_gpu); [ -n "$g" ] && { echo "$g"; return; }; sleep 240; done; }
for rho in 0.05 0.10 0.15; do
  G=$(wait_gpu)
  tag=$(echo $rho | tr -d '.')
  out=./results-rho-phi15-$tag
  [ -f "$out/report.json" ] && continue
  ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model microsoft/phi-1_5 \
      --mode aqslora --sensitive-frac $rho --adapter-frac 1.0 --speed-profile \
      --task commonsense --epochs 1 --batch-size 6 --rank 8 --gpus "$G" \
      --output-dir "$out" \
    && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$out" --gpus "$G" --commonsense \
    && echo "RHO-$rho-DONE" ) > "./rho-$tag.log" 2>&1 &
  echo "[$(STAMP)]   rho=$rho -> GPU$G"
  sleep 90
done
wait
echo "[$(STAMP)] RHO-SWEEP-DONE"
