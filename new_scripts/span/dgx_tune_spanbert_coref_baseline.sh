#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/fortest/xzh/datasets/coref/allen/train.english.v4_gold_conll
export COREF_DEV_DATA_PATH=/fortest/xzh/datasets/coref/allen/dev.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/fortest/xzh/datasets/coref/allen/test.english.v4_gold_conll

for max_sen in 100 80 60 40 20 11
do
    export COREF_MAX_SEN=$max_sen
    allennlp train new_scripts/span/tune_coref_spanbert_lstm.jsonnet -s outputs/tune_coref_spanbert_baseline_0312_maxsen_$COREF_MAX_SEN
done

