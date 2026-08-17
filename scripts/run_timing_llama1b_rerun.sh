#!/bin/bash
# A third Llama-3.2-1B sweep, to settle one disagreement.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
REPS=${1:-7}
MINREPS=6
STAMP() { date '+%m-%d %H:%M:%S'; }

AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"
AQS="--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.85 --placement depth"
METHODS=("qlora:--mode qlora" "qlora-b:--mode qlora" "rslora:--mode rslora"
         "aq-q:$AQQ" "aq-s:$AQS")

# a quiet node, not just a quiet card: the 3B sweep failed on a card that read
# 20 MiB for all 42 runs while the interference came from elsewhere in the box
node_quiet() {
  local busy
  busy=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
         awk -F', ' '$1 > 2000 || $2 > 10 {c++} END{print c+0}')
  [ "$busy" -le 1 ]
}
find_clean() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
               awk -F', ' '$2 < 120 && $3 < 5 {print $1; exit}'; }

echo "[$(STAMP)] waiting for the floor sweeps to finish"
while ps -eo pid,args --no-headers | awk '$2=="/bin/bash" && $3~/run_timing_floors/' | grep -q .; do sleep 300; done

OUT=./timing-llama1b-rerun; rm -rf "$OUT"; mkdir -p "$OUT"
echo "[$(STAMP)] waiting for a quiet node (at most one other GPU in use)"
while :; do node_quiet && break; sleep 600; done
while :; do GPU=$(find_clean); [ -n "$GPU" ] && break; sleep 300; done
echo "[$(STAMP)] node quiet; using GPU$GPU"

skipped=0
for i in $(seq 1 "$REPS"); do
  for spec in "${METHODS[@]}"; do
    n=${spec%%:*}; f=${spec#*:}
    [ -f "$OUT/$n-r$i/timing.json" ] && continue
    waited=0
    while :; do
      before=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
      [ "$before" -le 120 ] && break
      waited=$((waited+1)); [ "$waited" -ge 40 ] && break
      [ "$waited" -eq 1 ] && echo "[$(STAMP)]   $n rep$i: card busy (${before} MiB), waiting"
      sleep 30
    done
    if [ "$before" -gt 120 ]; then
      echo "[$(STAMP)]   $n rep$i skipped, card still busy"; skipped=$((skipped+1)); continue
    fi
    env CUDA_VISIBLE_DEVICES=$GPU $PY timing.py --model meta-llama/Llama-3.2-1B --gpus "$GPU" \
        --max-steps 240 --warmup-skip 40 --speed-profile $f \
        --output-dir "$OUT/$n-r$i" --tag "$n-r$i" > "$OUT/$n-r$i.log" 2>&1
    echo "[$(STAMP)]   $n rep$i (card ${before} MiB)"
  done
done

worst=999
for spec in "${METHODS[@]}"; do
  m=${spec%%:*}; c=$(ls -d "$OUT/$m"-r*/timing.json 2>/dev/null | wc -l)
  [ "$c" -lt "$worst" ] && worst=$c
done
echo "[$(STAMP)] thinnest arm $worst/$REPS, $skipped skipped"
[ "$worst" -ge "$MINREPS" ] && touch "$OUT/DONE"
echo "[$(STAMP)] LLAMA1B-RERUN-DONE"
