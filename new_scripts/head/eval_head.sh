#!/usr/bin/env bash

#export COREF_TEST_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/test.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/test.min_span


allennlp evaluate --include-package new_allen_packages --cuda-device 0 --output-file outputs/new_head_joint_debug/eval.txt  outputs/new_head_joint_debug $COREF_TEST_DATA_PATH

