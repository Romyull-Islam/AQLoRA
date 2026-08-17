#!/bin/bash
# Between-session variance of the speed measurements.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
SESSIONS=${1:-2}
REPS=${2:-5}
MINREPS=4
STAMP() { date '+%m-%d %H:%M:%S'; }

AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"
AQS="--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.85 --placement depth"
METHODS=("qlora:--mode qlora" "qlora-b:--mode qlora" "aq-q:$AQQ" "aq-s:$AQS")

find_clean() {   # optionally avoid the card used last session
  local avoid=${1:-}
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F', ' -v a="$avoid" '$2 < 120 && $3 < 5 && $1 != a {print $1; exit}'
}

session () {   # session <model> <tag> <n> <avoid_gpu>
  local MODEL=$1 TAG=$2 N=$3 AVOID=${4:-} OUT=./timing-sess-$2-s$3 GPU
  [ -f "$OUT/DONE" ] && { echo "[$(STAMP)] $TAG session $N already done"; return; }
  mkdir -p "$OUT"
  echo "[$(STAMP)] $TAG session $N: waiting for a clean card (avoiding GPU${AVOID:-none})"
  local tries=0
  while :; do
    GPU=$(find_clean "$AVOID"); [ -n "$GPU" ] && break
    tries=$((tries+1))
    [ "$tries" -gt 20 ] && { GPU=$(find_clean); [ -n "$GPU" ] && break; }
    sleep 300
  done
  echo "[$(STAMP)] $TAG session $N on GPU$GPU"
  echo "$GPU" > "$OUT/gpu"

  for i in $(seq 1 "$REPS"); do
    for spec in "${METHODS[@]}"; do
      local n=${spec%%:*} f=${spec#*:}
      [ -f "$OUT/$n-r$i/timing.json" ] && continue
      local waited=0 before
      while :; do
        before=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
        [ "$before" -le 120 ] && break
        waited=$((waited+1)); [ "$waited" -ge 40 ] && break
        sleep 30
      done
      [ "$before" -gt 120 ] && { echo "[$(STAMP)]   $n rep$i skipped"; continue; }
      env CUDA_VISIBLE_DEVICES=$GPU $PY timing.py --model "$MODEL" --gpus "$GPU" \
          --max-steps 240 --warmup-skip 40 --speed-profile $f \
          --output-dir "$OUT/$n-r$i" --tag "$n-r$i" > "$OUT/$n-r$i.log" 2>&1
      echo "[$(STAMP)]   $TAG s$N $n rep$i"
    done
  done

  local worst=999 m c
  for spec in "${METHODS[@]}"; do
    m=${spec%%:*}; c=$(ls -d "$OUT/$m"-r*/timing.json 2>/dev/null | wc -l)
    [ "$c" -lt "$worst" ] && worst=$c
  done
  if [ "$worst" -ge "$MINREPS" ]; then touch "$OUT/DONE"; echo "[$(STAMP)] SESSION-$TAG-$N-DONE ($worst reps)"
  else echo "[$(STAMP)] SESSION-$TAG-$N-STARVED ($worst reps)"; fi
}

echo "[$(STAMP)] waiting for other timing work to finish"
while ps -eo args | grep -qE "[t]iming\.py"; do sleep 300; done

for s in $(seq 1 "$SESSIONS"); do
  for pair in "microsoft/phi-1_5:phi15" "meta-llama/Llama-3.2-1B:llama-3.2-1b" \
              "microsoft/Phi-3-mini-4k-instruct:phi-3-mini"; do
    M=${pair%%:*}; T=${pair##*:}
    prev=""
    [ -f "./timing-sess-$T-s$((s-1))/gpu" ] && prev=$(cat "./timing-sess-$T-s$((s-1))/gpu")
    session "$M" "$T" "$s" "$prev"
  done
done
echo "[$(STAMP)] ALL-SESSIONS-DONE"
