#!/bin/bash
# Commonsense-170K head-to-head: fine-tune (finetune.py) then 8-task accuracy
# eval (evaluate.py) for each method, one method per GPU, in parallel.
#
# usage: ./run-commonsense.sh MODEL EPOCHS BATCH "GPU:method GPU:method ..."
#   e.g. ./run-commonsense.sh microsoft/phi-1_5 1 6 "3:aqslora-rs 4:qlora 5:rslora 7:dora"
set -u
PY=${PYTHON:-python}
MODEL=$1; EPOCHS=$2; BATCH=$3; shift 3
TAG=$(echo "$MODEL" | sed 's#.*/##' | tr '[:upper:]' '[:lower:]')

# All methods get the universal speed levers (--speed-profile) so AQSLoRA's
# edge comes only from its own mechanisms — see ABLATION_PLAN.md §2.
declare -A FLAGS=(
  [aqslora]="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0 --speed-profile"
  [aqslora-rs]="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0 --rslora-scaling --speed-profile"
  [aqslora-rs70]="--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.70 --rslora-scaling --speed-profile"
  [aqslora-speed]="--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.70 --placement depth --rslora-scaling --speed-profile"
  [qlora]="--mode qlora --speed-profile"
  [rslora]="--mode rslora --speed-profile"
  [dora]="--mode dora --speed-profile"
  [loftq]="--mode loftq --speed-profile"
  [loraplus]="--mode loraplus --speed-profile"
  [lora16]="--mode lora16 --speed-profile"
  # protocol-parity variants (old defaults, comparable to published DoRA numbers)
  [aqslora-rs-paper]="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0 --rslora-scaling"
  [qlora-paper]="--mode qlora"
)

for pair in "$@"; do
  gpu=${pair%%:*}; method=${pair##*:}
  out=./results-cs-$TAG-$method
  log=./cs-$TAG-$method.log
  ( "$PY" finetune.py --model "$MODEL" ${FLAGS[$method]} --task commonsense \
        --epochs "$EPOCHS" --batch-size "$BATCH" --gpus "$gpu" --output-dir "$out" \
    && "$PY" evaluate.py --adapter-dir "$out" --gpus "$gpu" --commonsense ) > "$log" 2>&1 &
  echo "launched $method on GPU$gpu  ->  $out   (log: $log, pid $!)"
  sleep 5   # stagger CPU-side profiling / dataset cache
done
echo "all jobs for $TAG launched; monitor with: tail -f cs-$TAG-*.log"
