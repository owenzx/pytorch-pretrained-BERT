#!/usr/bin/env bash

export TRAIN_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/full_out_short_h2s.json
export DEV_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/test_out_short_h2s.json
export TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_out_short_h2s.json
export LOAD_MODEL_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/h2s_bert/model.tar.gz

allennlp train new_scripts/h2s/out_finetune_bert.jsonnet -s outputs/h2s_out_finetune_bert_debug_full --include-package new_allen_packages
