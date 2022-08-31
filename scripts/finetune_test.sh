#!/bin/bash

. venv/bin/activate

python main.py config.yml scared --epochs 200 \
    --validation-size 1000 \
    --finetune-from RELATIVE_PATH_TO_MODEL \
    --save-model-to trained/scared --save-model-every 1 \
    --save-results-to results/da-vinci --evaluate-every 1 \
    --home ../
