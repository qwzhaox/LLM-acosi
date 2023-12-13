#!/bin/bash

python3 scripts/get_score_csv.py --absa_task="acos" --dataset="laptop"
python3 scripts/get_score_csv.py --absa_task="acos" --dataset="rest"
python3 scripts/get_score_csv.py --absa_task="acosi" --dataset="shoes"