#!/bin/bash
amplitude=$1
layers=$2
maxiter=$3
cd ~/vans
. ~/qenv_bilkis/bin/activate
#python3 simulate_bash.py --ratesiid $rates
python3 optimizer.py --amplitude $amplitude --layers $layers --maxiter $maxiter
deactivate
