#!/bin/bash

dataset="$1"
NUM="$2"

for (( i=1; i<=${NUM}; i++ )); do
    python train_vit.py --dataset ${dataset} --random ${i}
done
