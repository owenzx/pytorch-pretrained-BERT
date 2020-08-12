#!/usr/bin/env bash

export TRAIN_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/train_out_short_h2s_gold.json
export DEV_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_out_short_h2s_gold.json
export TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_out_short_h2s_gold.json

allennlp train new_scripts/h2s/out_bert.jsonnet -s outputs/h2s_debug_out_bert_2 --include-package new_allen_packages
