#!/usr/bin/env bash

#export COREF_TRAIN_DATA_PATH=/fortest/xzh/datasets/coref/allen/long.english.v4_gold_conll
#export COREF_DEV_DATA_PATH=/fortest/xzh/datasets/coref/allen/long.english.v4_gold_conll
#export COREF_TEST_DATA_PATH=/fortest/xzh/datasets/coref/allen/long.english.v4_gold_conll

export COREF_TRAIN_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/long.min_span
export COREF_DEV_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/long.min_span
export COREF_TEST_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/long.min_span

#allennlp evaluate --include-package new_allen_packages --cuda-device 0 --output-file outputs/head_debug_subword_light/mem.txt outputs/head_debug_subword_light /fortest/xzh/work/pytorch-pretrained-BERT/long.min_span

#allennlp evaluate --include-package new_allen_packages --cuda-device 0 --output-file outputs/coref_bert_c2f_0507/mem.txt outputs/coref_bert_c2f_0507 $COREF_DEV_DATA_PATH

#allennlp train new_scripts/span/mem_span.jsonnet -s outputs/mem_check_span --include-package new_allen_packages

#allennlp train new_scripts/head/mem_check_head.jsonnet -s outputs/mem_check_head --include-package new_allen_packages

allennlp train new_scripts/span/mem_span2.jsonnet -s outputs/mem_check_span2 --include-package new_allen_packages


