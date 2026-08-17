#!/bin/bash
set -u
PY=./venv/bin/python
cd "$(dirname "$0")/.."
launch() { # method rank gpu
  m=$1; r=$2; g=$3
  case $m in
    aqslora) F="--mode aqslora --sensitive-frac 0.20 --adapter-frac 1.0 --speed-profile" ;;
    qlora)   F="--mode qlora --speed-profile" ;;
  esac
  out=./results-rank-phi15-$m-r$r
  ( $PY finetune.py --model microsoft/phi-1_5 $F --task commonsense --rank $r \
      --epochs 1 --batch-size 6 --gpus $g --output-dir $out \
    && $PY evaluate.py --adapter-dir $out --gpus $g --commonsense \
    && echo "RANK-$m-r$r-DONE" ) > ./rank-$m-r$r.log 2>&1 &
  echo "  $m r=$r -> GPU$g (pid $!)"
  sleep 5
}
launch aqslora 16 3
launch qlora   16 4
launch aqslora 32 5
launch qlora   32 6
