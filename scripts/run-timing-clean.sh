#!/bin/bash
# Publication-grade timing (short: minutes per config, ~1-2 h total).
# WHY THIS IS DEFENSIBLE ON A SHARED NODE:
set -u
PY=${PYTHON:-python}
MODEL=$1; TGPU=$2; HOLD=$3; N=${4:-3}; STEPS=${5:-80}
TAG=$(echo "$MODEL" | sed 's#.*/##' | tr '[:upper:]' '[:lower:]')
OUT=./timing-clean-$TAG; mkdir -p "$OUT"

# the 4 speed-story methods (all share the universal levers via --speed-profile)
declare -a METHODS=(
  "qlora:--mode qlora"
  "aq-q:--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0 --rslora-scaling"
  "aq-s:--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.85 --placement depth --rslora-scaling"
  "lora16:--mode lora16"
)

setsid nohup "$PY" gpu_hold.py "$HOLD" > "$OUT/hold.log" 2>&1 < /dev/null &
HOLD_PID=$!
echo "held GPUs $HOLD; timing on GPU $TGPU; $N repeats x ${STEPS} steps each"
sleep 5

summary="$OUT/summary.txt"; : > "$summary"
for spec in "${METHODS[@]}"; do
  name=${spec%%:*}; flags=${spec#*:}
  best=0
  for i in $(seq 1 "$N"); do
    "$PY" timing.py --model "$MODEL" --gpus "$TGPU" --max-steps "$STEPS" --warmup-skip 20 \
        --speed-profile $flags --output-dir "$OUT/$name-r$i" --tag "$name-r$i" \
        > "$OUT/$name-r$i.log" 2>&1
    sps=$("$PY" -c "import json;print(json.load(open('$OUT/$name-r$i/timing.json'))['steps_per_sec'])" 2>/dev/null || echo 0)
    tps=$("$PY" -c "import json;print(json.load(open('$OUT/$name-r$i/timing.json'))['real_tokens_per_sec'])" 2>/dev/null || echo 0)
    awk "BEGIN{exit !($sps>$best)}" && best=$sps
    echo "  $name run $i/$N: $sps steps/s, $tps tok/s"
  done
  line="$name: best $best steps/s (max over $N runs)"
  echo "### $line"; echo "$line" >> "$summary"
done

kill "$HOLD_PID" 2>/dev/null
echo "released holds. best-of-$N per method in $summary"
