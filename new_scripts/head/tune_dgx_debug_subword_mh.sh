#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/train.min_span
export COREF_DEV_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/dev.min_span
export COREF_TEST_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/test.min_span

for nh in 1 3 5
    do export NH=$nh
    allennlp train new_scripts/head/debug_subword_mh.jsonnet -s outputs/head_debug_subword_mh_NH$NH --include-package new_allen_packages
done

