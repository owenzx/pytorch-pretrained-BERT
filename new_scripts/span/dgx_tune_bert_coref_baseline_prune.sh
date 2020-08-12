#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/fortest/xzh/datasets/coref/allen/train.english.v4_gold_conll
export COREF_DEV_DATA_PATH=/fortest/xzh/datasets/coref/allen/dev.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/fortest/xzh/datasets/coref/allen/test.english.v4_gold_conll

#for max_span in 5 10 20
#do
#    export MAX_SPAN_WIDTH=$max_span
allennlp train new_scripts/span/tune_coref_bert_lstm_prune.jsonnet -s outputs/tune_coref_bert_baseline_0312_maxsen60_verysmall
#done
