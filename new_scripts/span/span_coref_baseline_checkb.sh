#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/train.english.v4_gold_conll
export COREF_DEV_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/dev.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/playpen/home/xzh/datasets/coref/allen/test.english.v4_gold_conll

for b in 32 16 8 4 2
do export BATCH_SIZE=$b
    allennlp train new_scripts/span/coref_c2f.jsonnet -s outputs/baseline_span_c2f_BATCH$BATCH_SIZE
done

