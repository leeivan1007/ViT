#!/bin/bash

dataset="$1"
NUM="$2"
slight_disturbance="$3"

for (( i=1; i<=${NUM}; i++ )); do
    python train_vit.py --dataset ${dataset} ${slight_disturbance}
done
