#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/debug.english.v4_gold_conll
export COREF_DEV_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/debug.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/debug.english.v4_gold_conll
export COREF_UNL_DATA_PATH=/ssd-playpen/home/xzh/datasets/unlabeled_news/debug.tokenized.json

allennlp train scripts/allen/tune_4_coref_bert_ssl.jsonnet -s outputs/debug_tune_4_ssl_cnn_baseline_1024 --include-package allen_packages

