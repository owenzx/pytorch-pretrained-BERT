#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/train.collins.min_span
export COREF_DEV_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/dev.collins.min_span
export COREF_TEST_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/test.collins.min_span

allennlp train new_scripts/head/debug_subword.jsonnet -s outputs/head_debug_subword_collins --include-package new_allen_packages

