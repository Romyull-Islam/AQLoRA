#!/bin/bash
source venv/bin/activate 2>/dev/null
cd "$(dirname "$0")/.."
python3 finetune.py --mode qlora --model microsoft/phi-1_5 --task commonsense \
  --no-grad-ckpt --group-by-length --lora-dropout 0 --optim adamw_torch_fused \
  --gpus 4 --output-dir results-bd-baseline && \
python3 evaluate.py --adapter-dir results-bd-baseline --gpus 4 --commonsense
echo "BD-BASELINE-DONE"
