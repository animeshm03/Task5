#!/bin/bash

python3 train.py --lr 0.003 --momentum 0.2 --num_hidden 3 --sizes 100,200,100 \
    --activation tanh --loss ce --opt momentum --batch_size 64 --noanneal \
    --save_dir pa1/ --expt_dir pa1/exp1/ --train cifar-10/train/ \
    --train_labels cifar-10/trainLabels.csv --test cifar-10/test_check/ --epochs 5