#!/bin/bash
# One serial queue for all remaining work, capped at MAXJOBS concurrent jobs.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
MAXJOBS=5
STAMP() { date '+%m-%d %H:%M:%S'; }

AQQ="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0"
AQS="--mode aqslora --sensitive-frac 0.20 --adapter-frac 0.85 --placement depth"

# Wait until fewer than MAXJOBS of our training/eval processes are alive.
throttle () {
  while [ "$(ps -eo args | grep -cE '[v]env/bin/python (finetune|evaluate)\.py')" -ge "$MAXJOBS" ]; do
    sleep 120
  done
}
# Pick the card with the most free memory; fp16 arms need far more than 4-bit ones.
pick_gpu () {
  local need=$1 g=""
  while :; do
    g=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
        awk -F', ' -v n="$need" '{f=$3-$2; if(f>n && f>bf){bf=f;bi=$1}} END{print bi}')
    [ -n "$g" ] && { echo "$g"; return; }
    sleep 180
  done
}

job () {   # job <model> <outdir> <flags> <seed> <need_mib>
  local MODEL=$1 OUT=$2 FLAGS=$3 SEED=$4 NEED=$5
  [ -f "$OUT/report.json" ] && { echo "[$(STAMP)]   skip $(basename "$OUT") (done)"; return; }
  # also skip anything a live process is already writing, so restarting this
  # queue after a concurrency change does not duplicate in-flight runs.
  # training uses --output-dir and evaluation --adapter-dir, so match both or a
  # recovery eval looks idle and the run gets retrained from scratch.
  if ps -eo args | grep -qE -- "--(output|adapter)-dir $OUT( |$)"; then
    echo "[$(STAMP)]   skip $(basename "$OUT") (in flight)"; return
  fi
  throttle
  local G; G=$(pick_gpu "$NEED")
  ( env CUDA_VISIBLE_DEVICES=$G $PY finetune.py --model "$MODEL" $FLAGS \
      --task commonsense --epochs 1 --batch-size 6 --rank 8 --seed $SEED \
      --speed-profile --gpus "$G" --output-dir "$OUT" \
    && env CUDA_VISIBLE_DEVICES=$G $PY evaluate.py --adapter-dir "$OUT" --gpus "$G" --commonsense \
    && echo "QDONE $(basename "$OUT")" ) > "./q-$(basename "$OUT").log" 2>&1 &
  echo "[$(STAMP)]   $(basename "$OUT") -> GPU$G"
  sleep 240   # keep the stagger longer than model load so the next pick sees this card
}

echo "[$(STAMP)] ===== 1. Llama-3.2-1B seeds ====="
for s in 43 44 45; do
  job meta-llama/Llama-3.2-1B ./results-cs-llama-3.2-1b-aqslora-q-nors-s$s "$AQQ" $s 12000
  job meta-llama/Llama-3.2-1B ./results-cs-llama-3.2-1b-qlora-s$s "--mode qlora" $s 12000
done

echo "[$(STAMP)] ===== 2. same-seed determinism repeats ====="
for r in r2 r3; do
  job microsoft/phi-1_5 ./results-cs-phi15-aqslora-q-nors-s42$r "$AQQ" 42 10000
done

echo "[$(STAMP)] ===== 3. Llama-3.2-3B ====="
for s in 42 43 44; do
  job meta-llama/Llama-3.2-3B ./results-cs-llama-3.2-3b-aqslora-q-s$s "$AQQ" $s 16000
  job meta-llama/Llama-3.2-3B ./results-cs-llama-3.2-3b-qlora-s$s "--mode qlora" $s 16000
  job meta-llama/Llama-3.2-3B ./results-cs-llama-3.2-3b-lora16-s$s "--mode lora16" $s 26000
done

echo "[$(STAMP)] ===== 4. Phi-3-mini ====="
for s in 42 43 44; do
  job microsoft/Phi-3-mini-4k-instruct ./results-cs-phi-3-mini-aqslora-q-s$s "$AQQ" $s 16000
  job microsoft/Phi-3-mini-4k-instruct ./results-cs-phi-3-mini-qlora-s$s "--mode qlora" $s 16000
  job microsoft/Phi-3-mini-4k-instruct ./results-cs-phi-3-mini-lora16-s$s "--mode lora16" $s 26000
done

echo "[$(STAMP)] ===== 5. baselines for both new models ====="
for s in 42 43 44; do
  for pair in "meta-llama/Llama-3.2-3B:llama-3.2-3b" "microsoft/Phi-3-mini-4k-instruct:phi-3-mini"; do
    M=${pair%%:*}; T=${pair##*:}
    job "$M" ./results-cs-$T-rslora-s$s "--mode rslora" $s 16000
    job "$M" ./results-cs-$T-loftq-s$s  "--mode loftq"  $s 16000
    job "$M" ./results-cs-$T-aqslora-s-s$s "$AQS" $s 16000
  done
done

echo "[$(STAMP)] ===== 6. Phi-3 memory-matched budget ====="
for s in 42 43 44; do
  job microsoft/Phi-3-mini-4k-instruct ./results-cs-phi-3-mini-aqslora-q-r005-s$s \
      "--mode aqslora --sensitive-frac 0.05 --adapter-frac 1.0" $s 16000
done

wait
echo "[$(STAMP)] ALL ACCURACY DONE; timing sweeps need an empty node"
while ps -eo args | grep -qE "[v]env/bin/python (finetune|evaluate)\.py"; do sleep 300; done

echo "[$(STAMP)] ===== 7. timing sweeps (serial, one model at a time) ====="
./run_timing_llama_fixed.sh 7 >> llama-fixed-timing.log 2>&1
./run_timing_new_models.sh 7 >> newmodels-timing.log 2>&1
echo "[$(STAMP)] QUEUE-ALL-DONE"
