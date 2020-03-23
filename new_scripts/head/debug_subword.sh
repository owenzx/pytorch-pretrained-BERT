#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/train.min_span.longest
export COREF_DEV_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/debug.min_span
export COREF_TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/debug.min_span

allennlp train new_scripts/head/debug_subword.jsonnet -s outputs/head_debug_subword --include-package new_allen_packages

