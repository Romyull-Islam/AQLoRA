#!/bin/bash
# Residual protection tier, downstream.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
STAMP() { date '+%m-%d %H:%M:%S'; }
while ! grep -q 'GAP 1 done' gaps.log 2>/dev/null; do sleep 180; done
echo "[$(STAMP)] timing finished; smoke-testing the residual tier"

# One short run first: if the second stage is wired up wrong, base perplexity
# blows up and there is no point spending hours on the full runs.
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', ' '$2 < 2000 {print $1; exit}')
env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model microsoft/phi-1_5 \
    --mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0 --protect-tier residual \
    --ppl-base --task commonsense --epochs 1 --batch-size 6 --rank 8 \
    --cs-max-train 64 --gpus "$G" --output-dir ./results-resid-smoke > resid-smoke.log 2>&1
ppl=$(grep -o 'quantized base, no adapters): [0-9.]*' resid-smoke.log | tail -1 | tr -d 'a-z(),: ')
echo "[$(STAMP)] smoke base ppl = ${ppl:-FAILED}"
# fp16 tier at rho=0.20 gives 18.935 and plain NF4 gives 18.972; anything far
# outside that band means the residual stage is not being applied correctly
ok=$($PY -c "p=float('${ppl:-999}');print(1 if 18.0<p<20.0 else 0)" 2>/dev/null || echo 0)
[ "$ok" = "1" ] || { echo "[$(STAMP)] SMOKE FAILED (ppl=$ppl) - stopping, not burning GPU hours"; exit 1; }
echo "[$(STAMP)] smoke passed; launching full runs"

free_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
             awk -F', ' '$2 < 2000 {print $1; exit}'; }
wait_gpu() { local g; while :; do g=$(free_gpu); [ -n "$g" ] && { echo "$g"; return; }; sleep 240; done; }

for rho in 0.20 0.40; do
  G=$(wait_gpu)
  tag=$(echo $rho | tr -d '.')
  out=./results-resid-phi15-$tag
  [ -f "$out/report.json" ] && continue
  ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model microsoft/phi-1_5 \
      --mode aqslora --sensitive-frac $rho --adapter-frac 1.0 --protect-tier residual \
      --ppl-base --speed-profile --task commonsense --epochs 1 --batch-size 6 \
      --rank 8 --gpus "$G" --output-dir "$out" \
    && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$out" --gpus "$G" --commonsense \
    && echo "RESID-$rho-DONE" ) > "./resid-$tag.log" 2>&1 &
  echo "[$(STAMP)]   residual rho=$rho -> GPU$G"
  sleep 90
done
wait
echo "[$(STAMP)] RESIDUAL-DONE"
