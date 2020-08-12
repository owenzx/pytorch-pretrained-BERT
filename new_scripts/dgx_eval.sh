#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/fortest/xzh/datasets/coref/allen/train.english.v4_gold_conll
export COREF_DEV_DATA_PATH=/fortest/xzh/datasets/coref/allen/dev.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/fortest/xzh/datasets/coref/allen/test.english.v4_gold_conll

#allennlp evaluate --cuda-device 0 --output-file outputs/coref_bert_baseline_truncate_0312/zero.result outputs/coref_bert_baseline_truncate_0312 /fortest/xzh/datasets/coref/allen/dev.out.parse.english.v4_gold_conll.short




# zero origin
python get_span_metrics_for_head_model.py --model_path ./outputs/head_debug_subword/ --test_set /fortest/xzh/datasets/coref/allen/dev.english.v4_gold_conll --h2s_path ./outputs/h2s_debug_gold/ --head_test_set dev.min_span

#zero transfer
python get_span_metrics_for_head_model.py --model_path ./outputs/head_debug_subword/ --test_set /fortest/xzh/datasets/coref/allen/dev.out.parse.english.v4_gold_conll.short --h2s_path ./outputs/h2s_debug_gold/ --head_test_set dev.out.min_span.short

#finetune transfer
python get_span_metrics_for_head_model.py --model_path ./outputs/head_debug_subword/ --test_set /fortest/xzh/datasets/coref/allen/dev.out.parse.english.v4_gold_conll.short --h2s_path ./outputs/h2s_out_finetune/ --head_test_set dev.out.min_span.short
