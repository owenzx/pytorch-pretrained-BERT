#!/usr/bin/env bash

#export COREF_TEST_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/test.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/test.min_span


allennlp evaluate --include-package new_allen_packages --cuda-device 0 --output-file outputs/head_debug_subword_light/eval.txt  outputs/head_debug_subword_light $COREF_TEST_DATA_PATH

