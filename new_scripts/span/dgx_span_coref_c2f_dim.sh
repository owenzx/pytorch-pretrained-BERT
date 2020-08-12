#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/fortest/xzh/datasets/coref/allen/train.english.v4_gold_conll
export COREF_DEV_DATA_PATH=/fortest/xzh/datasets/coref/allen/dev.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/fortest/xzh/datasets/coref/allen/test.english.v4_gold_conll

#
# spwid origin 25
# dim origiin 150
for spwid in 30 40 50
do
    for dim in 400 700 1000
    do
        export SPWID=$spwid
        export DIM=$dim
#        allennlp train new_scripts/head/debug_subword_morereg.jsonnet -s outputs/head_debug_subword_b1_FFDP{$FFDP}_RNNDP{$RNNDP} --include-package new_allen_packages
         allennlp train new_scripts/span/coref_bert_lstm_c2f_dim.jsonnet -s outputs/coref_bert_c2f_mem_spwid{$SPWID}_dim{$DIM} --include-package new_allen_packages
    done
done
