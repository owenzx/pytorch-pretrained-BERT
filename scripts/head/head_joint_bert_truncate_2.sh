#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/train.min_span
export COREF_DEV_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev.min_span
export COREF_TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/test.min_span


for lr in 2e-4 2e-5 2e-7
do
    export LR=$lr
    allennlp train scripts/head/head_joint_bert_truncate_2.jsonnet -s outputs/head_joint_bert_truncate_2_$LR --include-package allen_packages
done

