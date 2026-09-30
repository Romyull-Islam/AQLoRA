#!/bin/bash
# More between-session replicates for the two models that had one clean session.
# Same protocol as run_timing_sessions.sh: four interleaved arms including a
# duplicated QLoRA arm, five repeats, 200 measured steps after 40 warmup.
# usage: run_timing_sessions_r3.sh <model> <tag> <session numbers...>
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
MODEL=$1; TAG=$2; shift 2
REPS=5; MINREPS=4
STAMP() { date '+%m-%d %H:%M:%S'; }
AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"
AQS="--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.85 --placement depth"
METHODS=("qlora:--mode qlora" "qlora-b:--mode qlora" "aq-q:$AQQ" "aq-s:$AQS")

find_clean() {   # GPU0 excluded, other users hold it most of the time
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F', ' '$1 != 0 && $2 < 120 && $3 < 5 {print $1; exit}'
}

for N in "$@"; do
  OUT=./timing-sess-$TAG-s$N
  [ -f "$OUT/DONE" ] && { echo "[$(STAMP)] $TAG s$N already done"; continue; }
  mkdir -p "$OUT"
  if [ -f "$OUT/gpu" ]; then GPU=$(cat "$OUT/gpu"); else
    GPU=""; while [ -z "$GPU" ]; do GPU=$(find_clean); [ -z "$GPU" ] && sleep 300; done
    echo "$GPU" > "$OUT/gpu"; fi
  echo "[$(STAMP)] $TAG session $N on GPU$GPU"
  for i in $(seq 1 "$REPS"); do
    for spec in "${METHODS[@]}"; do
      n=${spec%%:*}; f=${spec#*:}
      [ -f "$OUT/$n-r$i/timing.json" ] && continue
      waited=0
      while :; do   # the card must be ours alone before each run
        before=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
        [ "$before" -le 120 ] && break
        waited=$((waited+1)); [ "$waited" -ge 40 ] && break
        sleep 30
      done
      [ "$before" -gt 120 ] && { echo "[$(STAMP)]   $n rep$i skipped (card busy)"; continue; }
      env CUDA_VISIBLE_DEVICES=$GPU $PY timing.py --model "$MODEL" --gpus "$GPU" \
          --max-steps 240 --warmup-skip 40 --speed-profile $f \
          --output-dir "$OUT/$n-r$i" --tag "$n-r$i" > "$OUT/$n-r$i.log" 2>&1
      echo "[$(STAMP)]   $TAG s$N $n rep$i"
    done
  done
  worst=999
  for spec in "${METHODS[@]}"; do
    m=${spec%%:*}; c=$(ls -d "$OUT/$m"-r*/timing.json 2>/dev/null | wc -l)
    [ "$c" -lt "$worst" ] && worst=$c
  done
  if [ "$worst" -ge "$MINREPS" ]; then touch "$OUT/DONE"; echo "[$(STAMP)] SESSION-$TAG-$N-DONE ($worst reps)"
  else echo "[$(STAMP)] SESSION-$TAG-$N-STARVED ($worst reps)"; fi
done
echo "[$(STAMP)] $TAG ALL-DONE"
