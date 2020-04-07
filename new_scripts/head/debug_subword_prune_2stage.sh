#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/train.min_span
export COREF_DEV_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev.min_span
export COREF_TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/test.min_span


for mtl_coeff in 100.0 10.0 1.0 0.1 0.01
do
    export HEAD_LOSS_COEFF=$mtl_coeff
    allennlp train new_scripts/head/debug_subword_prune_2stage.jsonnet -s outputs/head_debug_subword_prune_2stage_coeff_$HEAD_LOSS_COEFF --include-package new_allen_packages
done
