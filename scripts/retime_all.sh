#!/bin/bash
# Re-time ALL methods in ONE session so every number is mutually comparable.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=./timing-session2-phi15; mkdir -p "$OUT"
setsid nohup "$PY" gpu_hold.py "4,5,6" > "$OUT/hold.log" 2>&1 < /dev/null & HOLD=$!
sleep 5
declare -a M=(
 "qlora:--mode qlora"
 "aq-q:--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0 --rslora-scaling"
 "aq-s:--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.85 --placement depth --rslora-scaling"
 "rslora:--mode rslora"
 "loftq:--mode loftq"
 "lora16:--mode lora16"
)
# interleave repeats (round-robin) so slow drift hits every method equally
for i in 1 2 3; do
  for spec in "${M[@]}"; do
    n=${spec%%:*}; f=${spec#*:}
    "$PY" timing.py --model microsoft/phi-1_5 --gpus 3 --max-steps 80 --warmup-skip 20 \
        --speed-profile $f --output-dir "$OUT/$n-r$i" --tag "$n-r$i" > "$OUT/$n-r$i.log" 2>&1
    echo "  $n rep$i done"
  done
done
kill "$HOLD" 2>/dev/null
"$PY" - << 'PY'
import json,glob,statistics as st
print("\n=== SESSION-2 (all methods, interleaved, one session) ===")
res={}
for m in ['qlora','aq-q','aq-s','rslora','loftq','lora16']:
    v=[json.load(open(j))['steps_per_sec'] for j in sorted(glob.glob(f'timing-session2-phi15/{m}-r*/timing.json'))]
    if v: res[m]=(st.mean(v), st.pstdev(v)); print(f"{m:8} mean={st.mean(v):.4f} +/- {st.pstdev(v):.4f}  (n={len(v)})")
if 'qlora' in res:
    q=res['qlora'][0]
    print("\nvs QLoRA:")
    for m,(mu,sd) in res.items():
        if m!='qlora': print(f"  {m:8} {(mu/q-1)*100:+6.1f}%")
PY
echo "RETIME-DONE"
