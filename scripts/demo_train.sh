#!/bin/bash

. venv/bin/activate

python main.py config.yml da-vinci -b 8 -e 120 -lr 0.0001 -w 8 \
    --save-model-to trained/da-vinci --save-model-every 10 \
    --save-results-to results/da-vinci --evaluate-every 10 \
    --home ../