#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/fortest/xzh/datasets/coref/allen/train.english.v4_gold_conll
export COREF_DEV_DATA_PATH=/fortest/xzh/datasets/coref/allen/dev.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/fortest/xzh/datasets/coref/allen/test.english.v4_gold_conll

#
## ldp origin 0.5
for maxs in 80 60
do
    for maxa in 100 80 60 50
    do
        export MAXA=$maxa
        export MAXS=$maxs
#        allennlp train new_scripts/head/debug_subword_morereg.jsonnet -s outputs/head_debug_subword_b1_FFDP{$FFDP}_RNNDP{$RNNDP} --include-package new_allen_packages
         allennlp train new_scripts/span/coref_bert_lstm_c2f_mem.jsonnet -s outputs/coref_bert_c2f_mem_MAXA{$MAXA}_MAXS{$MAXS} --include-package new_allen_packages
    done
done
