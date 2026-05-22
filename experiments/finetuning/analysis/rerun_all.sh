#!/bin/bash
source /home/psquare_a6000/Desktop/insideLLMs/venv/bin/activate
cd /home/psquare_a6000/Desktop/insideLLMs/FineTuning/experiments
python3 domain_specificity.py --epoch 1
python3 domain_specificity.py --epoch 2
python3 domain_specificity.py --epoch 3
python3 generate_all_plots.py
