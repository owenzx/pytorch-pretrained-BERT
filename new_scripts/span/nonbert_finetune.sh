#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/train.out.parse.english.v4_gold_conll.short
export COREF_DEV_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/dev.out.parse.english.v4_gold_conll.short
export COREF_TEST_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/test.out.parse.english.v4_gold_conll.short
export LOAD_MODEL_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/baseline_span_new/model.tar.gz

allennlp train new_scripts/span/nonbert_finetune.jsonnet -s outputs/span_nonbert_finetune_35

