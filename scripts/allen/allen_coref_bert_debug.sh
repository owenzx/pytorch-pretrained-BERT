#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/fortest/xzh/datasets/coref/allen/debug.english.v4_gold_conll
export COREF_DEV_DATA_PATH=/fortest/xzh/datasets/coref/allen/debug.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/fortest/xzh/datasets/coref/allen/debug.english.v4_gold_conll

allennlp train scripts/allen/coref_bert_uncased_feature_debug.jsonnet -s outputs/allen_debug --include-package allen_packages --force

