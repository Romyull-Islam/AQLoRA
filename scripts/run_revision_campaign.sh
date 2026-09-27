#!/bin/bash
# Everything the revision needs: the ranking null on two more models and under a
# parameter-matched control, plus the downstream control the null only implies.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
MAXJOBS=4   # other users need the node too; GPU0 is theirs in practice
STAMP() { date '+%m-%d %H:%M:%S'; }

throttle () {
  while [ "$(ps -eo args | grep -cE '[v]env/bin/python (finetune|evaluate|ranking_null_distribution)\.py')" -ge "$MAXJOBS" ]; do
    sleep 120
  done
}
pick_gpu () {   # GPU0 excluded, other users hold it most of the time
  local need=$1 g=""
  while :; do
    g=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
        awk -F', ' -v n="$need" '$1 != 0 {f=$3-$2; if (f>n && f>bf) {bf=f; bi=$1}} END {print bi}')
    [ -n "$g" ] && { echo "$g"; return; }
    sleep 180
  done
}

# 1. downstream control: does the ranking help accuracy where it helps perplexity?
#    Phi-3-mini at rho=.20, where count-matched and parameter-matched coincide.
echo "[$(STAMP)] ===== downstream ranking control, Phi-3-mini =====" 
for s in 42 43 44; do
  for arm in top random bottom; do
    OUT=./results-cs-phi-3-mini-ctrl-$arm-s$s
    [ -f "$OUT/report.json" ] && { echo "[$(STAMP)]   skip $arm s$s (done)"; continue; }
    if ps -eo args | grep -qE -- "--(output|adapter)-dir $OUT( |$)"; then
      echo "[$(STAMP)]   skip $arm s$s (in flight)"; continue; fi
    throttle; G=$(pick_gpu 16000)
    ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py \
        --model microsoft/Phi-3-mini-4k-instruct --mode aqslora \
        --skip-modules-file protect_${arm}_phi-3-mini_26.json --adapter-frac 1.0 \
        --task commonsense --epochs 1 --batch-size 6 --rank 8 --seed $s \
        --speed-profile --gpus "$G" --output-dir "$OUT" \
      && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$OUT" --gpus "$G" --commonsense \
      && echo "CTRL-DONE $arm s$s" ) > "./q-ctrl-$arm-s$s.log" 2>&1 &
    echo "[$(STAMP)]   ctrl $arm s$s -> GPU$G"
    sleep 240
  done
done

# 2. the null on two more models, and under the parameter-matched control
echo "[$(STAMP)] ===== ranking null, remaining models ====="
run_null () {   # run_null <model> <tag> <rho> <match>
  local OUT="ranking_null_$2_r${3#0.}.json"
  [ -f "$OUT" ] && { echo "[$(STAMP)]   skip $OUT (done)"; return; }
  if ps -eo args | grep -qE -- "--out $OUT( |$)"; then
    echo "[$(STAMP)]   skip $OUT (in flight)"; return; fi
  throttle; local G; G=$(pick_gpu 14000)
  ( env CUDA_VISIBLE_DEVICES=$G $PY ranking_null_distribution.py \
      --model "$1" --rho "$3" --n 10 --match "$4" --out "$OUT" ) > "rn-$2-r${3#0.}.log" 2>&1 &
  echo "[$(STAMP)]   null $2 rho=$3 ($4) -> GPU$G"
  sleep 120
}
for r in 0.10 0.20 0.30; do
  run_null "microsoft/phi-1_5"                  phi15-pm        "$r" params
done
for r in 0.10 0.20 0.30; do
  run_null "meta-llama/Llama-3.2-3B"            llama-3.2-3b-pm "$r" params
  run_null "microsoft/Phi-3-mini-4k-instruct"   phi-3-mini-pm   "$r" params
done
for r in 0.10 0.20 0.30; do
  run_null "Qwen/Qwen2.5-7B"                    qwen2.5-7b-pm   "$r" params
done

wait
echo "[$(STAMP)] CAMPAIGN-DONE"
