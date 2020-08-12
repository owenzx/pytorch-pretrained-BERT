#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/fortest/xzh/datasets/coref/allen/train.out.parse.english.v4_gold_conll.short
export COREF_DEV_DATA_PATH=/fortest/xzh/datasets/coref/allen/dev.out.parse.english.v4_gold_conll.short
export COREF_TEST_DATA_PATH=/fortest/xzh/datasets/coref/allen/test.out.parse.english.v4_gold_conll.short
export LOAD_MODEL_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/outputs/tune_coref_bert_baseline_0312_maxsen_60/model.tar.gz

for epochs in 10 20 50
do
    for lr in 1e-6 1e-5 1e-4
    do
        export LR=$lr
        export EPOCHS=$epochs
        allennlp train new_scripts/span/tune_bert_finetune.jsonnet -s outputs/span_bert_finetune_s_E{$EPOCHS}_LR{$LR}
    done
done
