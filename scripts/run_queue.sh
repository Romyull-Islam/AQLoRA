#!/bin/bash
# Priority job queue. Waits for a GENUINELY idle GPU before starting anything, so
# it neither contends with other users nor with its own earlier jobs.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%Y-%m-%d %H:%M:%S'; }

# A card is IDLE only if essentially nothing is resident and it is not computing.
idle_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
             --format=csv,noheader,nounits |
    awk -F', ' '$2 < 120 && $3 < 5 {print $1; exit}'
}
# Cards with room but possibly shared: fine for accuracy work.
roomy_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', ' '$2 < 6000 {print $1; exit}'
}
wait_for() {  # wait_for <fn> ; echoes the gpu index once available
  local f=$1 g=""
  while :; do g=$($f); [ -n "$g" ] && { echo "$g"; return; }; sleep 300; done
}

# timing jobs
# 7 interleaved repeats per method, matching the protocol already used for
# phi-1.5, so the new numbers are comparable within their own session.
timing_sweep() {   # timing_sweep <model> <tag> <gpu> <methods...>
  local model=$1 tag=$2 gpu=$3; shift 3
  local out=./timing-clean-$tag
  mkdir -p "$out"
  echo "[$(STAMP)] TIMING $tag on GPU$gpu: $*"
  for i in 1 2 3 4 5 6 7; do
    for spec in "$@"; do
      local n=${spec%%:*} f=${spec#*:}
      env CUDA_VISIBLE_DEVICES=$gpu $PY timing.py --model "$model" --gpus "$gpu" \
          --max-steps 80 --warmup-skip 20 --speed-profile $f \
          --output-dir "$out/$n-r$i" --tag "$n-r$i" > "$out/$n-r$i.log" 2>&1
      echo "[$(STAMP)]   $tag $n rep$i"
    done
  done
  $PY robust_timing.py "$out" > "$out/summary.txt" 2>&1
  echo "[$(STAMP)] TIMING-$tag-DONE"
  cat "$out/summary.txt"
}

AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"
AQS="--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.85 --placement depth"

echo "[$(STAMP)] queue started; waiting for an idle GPU"

G=$(wait_for idle_gpu)
timing_sweep meta-llama/Llama-3.2-1B llama-3.2-1b "$G" \
  "qlora:--mode qlora" "aq-q:$AQQ" "aq-s:$AQS" "lora16:--mode lora16" \
  "rslora:--mode rslora" "loftq:--mode loftq"

# accuracy jobs
echo "[$(STAMP)] timing done; starting E3 seed-43 grid"
for spec in sparsity:5 random:5 weightnorm:5 sparsity:7 random:7 weightnorm:7; do
  m=${spec%%:*}; a=${spec##*:}
  G=$(wait_for roomy_gpu)
  out=./results-e3s43-phi15-$m$a
  ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model microsoft/phi-1_5 \
      --mode aqslora --sensitive-frac 0.20 --adapter-frac 0.$a --placement $m \
      --seed 43 --speed-profile --task commonsense --epochs 1 --batch-size 6 \
      --gpus "$G" --output-dir "$out" \
    && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$out" \
      --gpus "$G" --commonsense \
    && echo "E3S43-$m$a-DONE" ) > "./e3s43-$m$a.log" 2>&1 &
  echo "[$(STAMP)] E3 seed43 $m 0.$a -> GPU$G"
  sleep 120
done
wait
echo "[$(STAMP)] E3-SEED43-ALL-DONE"

# Lowest priority, optional: 14B timing. Roughly 8-18 h and the least useful of
# the sweeps (14B has no fp16 baseline to compare against), so it runs last and
# never blocks anything above it.
G=$(wait_for idle_gpu)
timing_sweep Qwen/Qwen2.5-14B qwen2.5-14b "$G" \
  "qlora:--mode qlora" "aq-q:$AQQ" "aq-s:$AQS"

echo "[$(STAMP)] QUEUE-ALL-DONE"
