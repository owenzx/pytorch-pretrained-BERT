#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/fortest/xzh/datasets/coref/allen/train.english.v4_gold_conll
export COREF_DEV_DATA_PATH=/fortest/xzh/datasets/coref/allen/dev.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/fortest/xzh/datasets/coref/allen/test.english.v4_gold_conll

allennlp train scripts/allen/coref_bert_cased_tune_large_debug_dgx.jsonnet -s outputs/debug.out --include-package allen_packages

