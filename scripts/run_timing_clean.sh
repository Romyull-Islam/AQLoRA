#!/bin/bash
# Timing sweep with contention guards.
#
# usage: ./run_timing_clean.sh <model> <tag> [reps]
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
MODEL=${1:?model}; TAG=${2:?tag}; REPS=${3:-7}
OUT=./timing-clean-$TAG; mkdir -p "$OUT"
STAMP() { date '+%H:%M:%S'; }

# Our own process is the only thing allowed on the card.
others_on() {  # others_on <gpu> -> prints foreign MiB
  local g=$1 mine=0 tot
  tot=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
  echo "$tot"
}
find_clean() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
             --format=csv,noheader,nounits |
    awk -F', ' '$2 < 120 && $3 < 5 {print $1; exit}'
}

echo "[$(STAMP)] waiting for a clean card for $TAG"
while :; do GPU=$(find_clean); [ -n "$GPU" ] && break; sleep 300; done
echo "[$(STAMP)] using GPU$GPU"

AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"
AQS="--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.85 --placement depth"
METHODS=("qlora:--mode qlora" "aq-q:$AQQ" "aq-s:$AQS" "lora16:--mode lora16"
         "rslora:--mode rslora" "loftq:--mode loftq")

for i in $(seq 1 "$REPS"); do
  for spec in "${METHODS[@]}"; do
    n=${spec%%:*}; f=${spec#*:}
    # baseline occupancy before the run: anything resident is someone else's
    before=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$before" -gt 120 ]; then
      echo "[$(STAMP)] SKIP $n rep$i: card not clean (${before} MiB resident)"
      continue
    fi
    env CUDA_VISIBLE_DEVICES=$GPU $PY timing.py --model "$MODEL" --gpus "$GPU" \
        --max-steps 80 --warmup-skip 20 --speed-profile $f \
        --output-dir "$OUT/$n-r$i" --tag "$n-r$i" > "$OUT/$n-r$i.log" 2>&1
    # foreign memory that appeared while we ran = someone joined mid-run
    peak_ours=$($PY -c "import json;print(int(json.load(open('$OUT/$n-r$i/timing.json'))['peak_gib']*1024))" 2>/dev/null || echo 0)
    now=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
    echo "{\"gpu\":$GPU,\"mib_before\":$before,\"mib_after_run\":$now,\"our_peak_mib\":$peak_ours}" \
        > "$OUT/$n-r$i/occupancy.json" 2>/dev/null
    echo "[$(STAMP)]   $n rep$i (card ${before} MiB before)"
  done
done

echo "[$(STAMP)] sweep finished; drift check:"
OUT_DIR="$OUT" $PY - << 'PY'
import json,glob,collections,statistics as st,os
out=os.environ.get('OUT_DIR')
R=collections.defaultdict(list)
for j in glob.glob(f"{out}/*/timing.json"):
    d=json.load(open(j)); R[d['tag'].rsplit('-r',1)[0]].append(d['steps_per_sec'])
def fa(v):
    v=sorted(v,reverse=True); return st.mean(v[:max(2,round(0.5*len(v)))])
est={m:fa(v) for m,v in R.items() if v}
q=est.get('qlora')
# The real validity test is the noise floor, not per-rep drift. rsLoRA and LoftQ
# have the same per-step compute as QLoRA, so whatever they differ by IS the
# measurement error. Per-rep drift was tried first and rejected a good sweep:
# the accepted phi-1.5 session drifts 6.6% yet has a 1.8% floor.
ident=[m for m in ('rslora','loftq') if m in est]
if q and ident:
    floor=max(abs(est[m]/q-1)*100 for m in ident)
    print(f"  noise floor (from {', '.join(ident)}): {floor:.1f}%")
    for m in ('aq-q','aq-s'):
        if m in est:
            r=(est[m]/q-1)*100
            print(f"  {m}: {r:+.1f}%  = {abs(r)/floor:.1f}x floor")
    print("  VERDICT:", "usable" if floor < 3.0 else "FLOOR TOO WIDE - discard, the run was contended")
else:
    print("  VERDICT: no compute-identical baseline timed, floor unmeasurable")
PY
echo "[$(STAMP)] TIMING-$TAG-DONE"
