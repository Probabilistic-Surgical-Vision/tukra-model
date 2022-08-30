#!/bin/bash

python main.py config.yml da-vinci --epochs 2 \
    --training-size 16 --validation-size 16 \
    --finetune-from trained/da-vinci/model_20220827011058/epoch_001.pt \
    --save-model-to trained/da-vinci --save-model-every 1 \
    --save-results-to results/da-vinci --evaluate-every 1 \
    --home ../
