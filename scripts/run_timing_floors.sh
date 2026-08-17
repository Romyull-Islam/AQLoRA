#!/bin/bash
# Re-measure every timed model's noise floor with a duplicated arm.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
REPS=${1:-7}
MINREPS=5          # below this the sweep is not worth analysing
WAIT_CAP=40        # x30s = 20 min of waiting for a card to clear before giving up on the rep
STAMP() { date '+%m-%d %H:%M:%S'; }

AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"
AQS="--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.85 --placement depth"
METHODS=("qlora:--mode qlora" "qlora-b:--mode qlora" "rslora:--mode rslora"
         "aq-q:$AQQ" "aq-s:$AQS")

find_clean() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
               awk -F', ' '$2 < 120 && $3 < 5 {print $1; exit}'; }

sweep () {   # sweep <model> <tag>
  local MODEL=$1 TAG=$2 OUT=./timing-floor-$2 GPU
  [ -f "$OUT/DONE" ] && { echo "[$(STAMP)] $TAG already done"; return; }
  mkdir -p "$OUT"
  echo "[$(STAMP)] waiting for a clean card for $TAG"
  while :; do GPU=$(find_clean); [ -n "$GPU" ] && break; sleep 300; done
  echo "[$(STAMP)] $TAG on GPU$GPU"

  local skipped=0
  for i in $(seq 1 "$REPS"); do
    for spec in "${METHODS[@]}"; do
      local n=${spec%%:*} f=${spec#*:}
      [ -f "$OUT/$n-r$i/timing.json" ] && continue
      # wait for the card rather than dropping the run: a skipped rep is data lost
      local waited=0 before
      while :; do
        before=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
        [ "$before" -le 120 ] && break
        waited=$((waited+1))
        [ "$waited" -ge "$WAIT_CAP" ] && break
        [ "$waited" -eq 1 ] && echo "[$(STAMP)]   $n rep$i: card busy (${before} MiB), waiting"
        sleep 30
      done
      if [ "$before" -gt 120 ]; then
        echo "[$(STAMP)]   $n rep$i: still busy after $((WAIT_CAP/2)) min, skipping"
        skipped=$((skipped+1)); continue
      fi
      env CUDA_VISIBLE_DEVICES=$GPU $PY timing.py --model "$MODEL" --gpus "$GPU" \
          --max-steps 240 --warmup-skip 40 --speed-profile $f \
          --output-dir "$OUT/$n-r$i" --tag "$n-r$i" > "$OUT/$n-r$i.log" 2>&1
      echo "[$(STAMP)]   $TAG $n rep$i (card ${before} MiB)"
    done
  done

  # only claim the sweep if every arm actually has enough repeats
  local worst=999 m c
  for spec in "${METHODS[@]}"; do
    m=${spec%%:*}
    c=$(ls -d "$OUT/$m"-r*/timing.json 2>/dev/null | wc -l)
    [ "$c" -lt "$worst" ] && worst=$c
  done
  echo "[$(STAMP)] $TAG: thinnest arm has $worst/$REPS repeats, $skipped runs skipped"
  if [ "$worst" -ge "$MINREPS" ]; then
    touch "$OUT/DONE"; echo "[$(STAMP)] FLOOR-$TAG-DONE"
  else
    echo "[$(STAMP)] FLOOR-$TAG-STARVED: not marking done, rerun when the node is quieter"
  fi
}

echo "[$(STAMP)] waiting for our own training work to drain"
while ps -eo args | grep -qE "[v]env/bin/python (finetune|evaluate)\.py"; do sleep 300; done

sweep microsoft/phi-1_5 phi15
sweep meta-llama/Llama-3.2-1B llama-3.2-1b
sweep microsoft/Phi-3-mini-4k-instruct phi-3-mini
echo "[$(STAMP)] ALL-FLOORS-DONE"
