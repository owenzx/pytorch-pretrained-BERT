#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/train.min_span
export COREF_DEV_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev.min_span
export COREF_TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/test.min_span

allennlp train scripts/head/head_joint_bert_attention_truncate.jsonnet -s outputs/head_joint_bert_attention_truncate --include-package allen_packages

