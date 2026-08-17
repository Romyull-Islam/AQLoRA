#!/bin/bash
# Remaining baselines for the two new models, so their rows match the existing
# main tables rather than being a reduced comparison.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }
AQS="--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.85 --placement depth"

echo "[$(STAMP)] waiting for the first-pass runs to finish"
while ! grep -q "NEW-MODELS-DONE" newmodels.log 2>/dev/null; do sleep 300; done
echo "[$(STAMP)] first pass done"

free_gpu() { local need=$1
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
    awk -F', ' -v n="$need" '{f=$3-$2; if(f>n && f>bf){bf=f;bi=$1}} END{print bi}'; }
wait_gpu() { local g; while :; do g=$(free_gpu "$1"); [ -n "$g" ] && { echo "$g"; return; }; sleep 300; done; }

launch () {  # launch <model> <tag> <name> <flags>
  local MODEL=$1 TAG=$2 NAME=$3 FLAGS=$4
  for seed in 42 43 44; do
    local out=./results-cs-$TAG-$NAME-s$seed
    [ -f "$out/report.json" ] && continue
    local G; G=$(wait_gpu 14000)
    ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model "$MODEL" $FLAGS \
        --task commonsense --epochs 1 --batch-size 6 --rank 8 --seed $seed \
        --speed-profile --gpus "$G" --output-dir "$out" \
      && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$out" --gpus "$G" --commonsense \
      && echo "BASE-$TAG-$NAME-$seed-DONE" ) > "./base-$TAG-$NAME-s$seed.log" 2>&1 &
    echo "[$(STAMP)]   $TAG $NAME seed$seed -> GPU$G"
    sleep 420
  done
}

for pair in "meta-llama/Llama-3.2-3B:llama-3.2-3b" "microsoft/Phi-3-mini-4k-instruct:phi-3-mini"; do
  MODEL=${pair%%:*}; TAG=${pair##*:}
  echo "[$(STAMP)] ===== $TAG baselines ====="
  launch "$MODEL" "$TAG" rslora "--mode rslora"
  launch "$MODEL" "$TAG" loftq  "--mode loftq"
  launch "$MODEL" "$TAG" aqslora-s "$AQS"
  wait
done

# memory-matched budget for the architecture where rho=0.20 is expensive
echo "[$(STAMP)] ===== phi-3 memory-matched budget ====="
launch microsoft/Phi-3-mini-4k-instruct phi-3-mini aqslora-q-r005 \
  "--mode aqslora --sensitive-frac 0.05 --adapter-frac 1.0"
wait
echo "[$(STAMP)] NEW-MODEL-BASELINES-DONE"
