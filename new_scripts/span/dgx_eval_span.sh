#!/usr/bin/env bash

export COREF_TEST_DATA_PATH=/fortest/xzh/datasets/coref/allen/test.english.v4_gold_conll
#export COREF_TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/test.min_span


allennlp evaluate --include-package new_allen_packages --cuda-device 0 --output-file outputs/tune_coref_bert_baseline_0312_maxsen_60/eval.txt  outputs/tune_coref_bert_baseline_0312_maxsen_60 $COREF_TEST_DATA_PATH

