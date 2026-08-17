#!/bin/bash
# Close the three remaining gaps, in the order their constraints demand.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }
AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"

free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', ' '$2 < 2000 {print $1; exit}'
}
wait_gpu() { local g; while :; do g=$(free_gpu); [ -n "$g" ] && { echo "$g"; return; }; sleep 240; done; }

# ---------------------------------------------------------------- gap 1
echo "[$(STAMP)] GAP 1: Llama clean timing (alone on the node)"
./run_timing_clean.sh meta-llama/Llama-3.2-1B llama-3.2-1b 7
echo "[$(STAMP)] GAP 1 done"

# ---------------------------------------------------------------- gap 2
# Table 1 reports single-seed numbers for the methods the narrative compares.
# Two more seeds each turns those cells into mean +/- std.
echo "[$(STAMP)] GAP 2: extra seeds for the contested Table-1 cells"
for seed in 43 44; do
  for spec in "aqslora-q-nors:$AQQ" "qlora:--mode qlora" "loftq:--mode loftq"; do
    m=${spec%%:*}; f=${spec#*:}
    G=$(wait_gpu)
    out=./results-cs-phi15-$m-s$seed
    [ -f "$out/report.json" ] && { echo "[$(STAMP)]   $m s$seed exists, skip"; continue; }
    ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model microsoft/phi-1_5 $f \
        --task commonsense --epochs 1 --batch-size 6 --rank 8 --seed $seed \
        --speed-profile --gpus "$G" --output-dir "$out" \
      && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$out" --gpus "$G" --commonsense \
      && echo "SEED-$m-$seed-DONE" ) > "./seed-$m-s$seed.log" 2>&1 &
    echo "[$(STAMP)]   $m seed$seed -> GPU$G"
    sleep 90
  done
done
wait
echo "[$(STAMP)] GAP 2 done"

# ---------------------------------------------------------------- gap 3
# The rank sweep covers phi-1.5 only. r=8 already exists from the main table, so
# r=16 and r=32 are enough to show whether the ordering, and rsLoRA's collapse,
# reproduce on a second architecture.
echo "[$(STAMP)] GAP 3: rank sweep on Llama-3.2-1B"
for r in 16 32; do
  for spec in "aqslora:$AQQ" "qlora:--mode qlora" "loftq:--mode loftq" "rslora:--mode rslora"; do
    m=${spec%%:*}; f=${spec#*:}
    G=$(wait_gpu)
    out=./results-rank-llama-$m-r$r
    [ -f "$out/report.json" ] && { echo "[$(STAMP)]   $m r$r exists, skip"; continue; }
    ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model meta-llama/Llama-3.2-1B $f \
        --task commonsense --epochs 1 --batch-size 6 --rank $r \
        --speed-profile --gpus "$G" --output-dir "$out" \
      && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$out" --gpus "$G" --commonsense \
      && echo "RANK-llama-$m-r$r-DONE" ) > "./rank-llama-$m-r$r.log" 2>&1 &
    echo "[$(STAMP)]   llama $m r=$r -> GPU$G"
    sleep 90
  done
done
wait
echo "[$(STAMP)] ALL-GAPS-DONE"
