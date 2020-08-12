#!/usr/bin/env bash

export TRAIN_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/train_h2s.json
export DEV_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_h2s.json
export TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_h2s.json

allennlp train new_scripts/h2s/basic.jsonnet -s outputs/h2s_debug --include-package new_allen_packages
