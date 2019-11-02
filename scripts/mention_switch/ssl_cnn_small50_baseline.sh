#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/train50.english.v4_gold_conll
export COREF_DEV_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/dev.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/test.english.v4_gold_conll
export COREF_UNL_DATA_PATH=/ssd-playpen/home/xzh/datasets/unlabeled_news/cnn.tokenized.json

allennlp train scripts/allen/coref_bert_ssl_small50.jsonnet -s outputs/ssl_cnn_small50_1030 --include-package allen_packages

