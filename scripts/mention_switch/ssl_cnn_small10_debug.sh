#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/train10.english.v4_gold_conll
export COREF_DEV_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/dev.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/test.english.v4_gold_conll
export COREF_UNL_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/ssl10_90.english.v4_gold_conll

allennlp train scripts/allen/debug_bert_ssl_small10.jsonnet -s outputs/fix_debug_cnn_small10_1124_7 --include-package allen_packages

