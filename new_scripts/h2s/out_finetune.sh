#!/usr/bin/env bash

export TRAIN_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/train_out_short_h2s_gold.json
export DEV_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_out_short_h2s_gold.json
export TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_out_short_h2s_gold.json
export LOAD_MODEL_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/h2s_debug_gold/model.tar.gz

allennlp train new_scripts/h2s/out_finetune.jsonnet -s outputs/h2s_out_finetune --include-package new_allen_packages
