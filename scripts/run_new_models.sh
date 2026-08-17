#!/bin/bash
# Two more model families, for variation and for correctness of the scale story.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }
AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"

# fp16 references hold the whole model in 16-bit and need far more room than the
# 4-bit arms, so gate on free memory rather than on a card being idle
free_gpu() { local need=$1
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
    awk -F', ' -v n="$need" '{f=$3-$2; if(f>n && f>bf){bf=f;bi=$1}} END{print bi}'; }
wait_gpu() { local g; while :; do g=$(free_gpu "$1"); [ -n "$g" ] && { echo "$g"; return; }; sleep 300; done; }

run_model () {   # run_model <hf-id> <tag>
  local MODEL=$1 TAG=$2
  echo "[$(STAMP)] ===== $TAG ====="
  for seed in 42 43 44; do
    for spec in "aqslora-q:$AQQ:14000" "qlora:--mode qlora:14000" "lora16:--mode lora16:24000"; do
      local m=${spec%%:*}; local rest=${spec#*:}
      local f=${rest%:*}; local need=${rest##*:}
      local out=./results-cs-$TAG-$m-s$seed
      [ -f "$out/report.json" ] && { echo "[$(STAMP)]   $m s$seed exists, skip"; continue; }
      local G; G=$(wait_gpu "$need")
      ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model "$MODEL" $f \
          --task commonsense --epochs 1 --batch-size 6 --rank 8 --seed $seed \
          --speed-profile --ppl-base --gpus "$G" --output-dir "$out" \
        && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$out" --gpus "$G" --commonsense \
        && echo "NEWMODEL-$TAG-$m-$seed-DONE" ) > "./newmodel-$TAG-$m-s$seed.log" 2>&1 &
      echo "[$(STAMP)]   $TAG $m seed$seed -> GPU$G (needed ${need} MiB free)"
      sleep 420   # 3B loads slower than 1.4B; keep the stagger above load time
    done
  done
  wait
  echo "[$(STAMP)] $TAG accuracy done"
}

run_model meta-llama/Llama-3.2-3B llama-3.2-3b
run_model microsoft/Phi-3-mini-4k-instruct phi-3-mini
echo "[$(STAMP)] NEW-MODELS-DONE"
