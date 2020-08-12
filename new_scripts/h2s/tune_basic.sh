#!/usr/bin/env bash

export TRAIN_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/train_h2s.json
export DEV_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_h2s.json
export TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_h2s.json

# original dp 0.2
for dp in 0.0 0.5 0.7
do
    export DP=$dp
    allennlp train new_scripts/h2s/tune_basic.jsonnet -s outputs/h2s_dp$DP --include-package new_allen_packages
done
