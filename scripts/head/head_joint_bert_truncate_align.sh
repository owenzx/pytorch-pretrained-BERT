#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/train.min_span
export COREF_DEV_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev.min_span
export COREF_TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/test.min_span

for n_epoch in 10 15 30
do
    export NEPOCH=$n_epoch
    allennlp train scripts/head/head_joint_bert_truncate_align.jsonnet -s outputs/head_joint_bert_truncate_align_epoch_$NEPOCH --include-package allen_packages
done
