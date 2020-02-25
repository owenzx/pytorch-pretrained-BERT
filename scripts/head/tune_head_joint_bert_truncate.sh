#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/train.min_span
export COREF_DEV_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev.min_span
export COREF_TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/test.min_span

for para in 0.1 0.2 0.5
do
    export TUNING_PARA=$para
    echo Now running tuning experiments with the parameter set to $TUNING_PARA
    allennlp train scripts/head/head_joint_bert_truncate.jsonnet -s outputs/head_joint_bert_truncate_dropout_$TUNING_PARA --include-package allen_packages
done

