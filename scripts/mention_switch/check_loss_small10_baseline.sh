#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/train10.english.v4_gold_conll
export COREF_DEV_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/dev.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/test.english.v4_gold_conll
export COREF_UNL_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/ssl10_90.english.v4_gold_conll

allennlp train scripts/allen/coref_bert_checkloss_small10_mask.jsonnet -s outputs/check_loss_small10_1223_2 --include-package allen_packages

