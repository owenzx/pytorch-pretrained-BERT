#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/train.english.v4_gold_conll
export COREF_DEV_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/dev.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/test.english.v4_gold_conll
#export COREF_TRAIN_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/debug.english.v4_gold_conll
#export COREF_DEV_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/debug.english.v4_gold_conll
#export COREF_TEST_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/debug.english.v4_gold_conll

allennlp train scripts/mention_switch/tune_4_mentionswitch_consist_same.jsonnet -s outputs/tune_4_mentionswitch_consist_same_1025_1 --include-package allen_packages

