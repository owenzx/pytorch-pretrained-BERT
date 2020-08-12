#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/fortest/xzh/datasets/coref/allen/train.english.v4_gold_conll
export COREF_DEV_DATA_PATH=/fortest/xzh/datasets/coref/allen/dev.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/fortest/xzh/datasets/coref/allen/test.english.v4_gold_conll

allennlp train new_scripts/span/coref_bert_lstm_small.jsonnet -s outputs/coref_bert_small_0507 --include-package new_allen_packages
allennlp train new_scripts/span/coref_bert_lstm_allfeatures.jsonnet -s outputs/coref_bert_allfeatures_0507 --include-package new_allen_packages
allennlp train new_scripts/span/coref_bert_lstm_everything.jsonnet -s outputs/coref_bert_everything_0507 --include-package new_allen_packages
