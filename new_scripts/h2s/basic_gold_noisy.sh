#!/usr/bin/env bash

export TRAIN_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/train_h2s_gold.noisy.json
export DEV_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_h2s_gold.noisy.json
export TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_h2s_gold.noisy.json

allennlp train new_scripts/h2s/basic.jsonnet -s outputs/h2s_debug_gold_noisy --include-package new_allen_packages
