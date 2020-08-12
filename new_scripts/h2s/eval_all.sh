#!/usr/bin/env bash

#export TRAIN_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/train_h2s.json
#export DEV_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_h2s.json
#export TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_h2s.json
#
#allennlp train new_scripts/h2s/basic.jsonnet -s outputs/h2s_debug --include-package new_allen_packages

export MODEL_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/new_head_joint_debug
export HEAD_TEST_SET=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev.min_span
export SPAN_TEST_SET=/playpen/home/xzh/datasets/coref/allen/dev.english.v4_gold_conll
#export HEAD_TEST_SET=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev.out.min_span.short
#export SPAN_TEST_SET=/playpen/home/xzh/datasets/coref/allen/dev.out.parse.english.v4_gold_conll

python get_span_metrics_for_head_model.py --model_path $MODEL_PATH --head_test_set $HEAD_TEST_SET --test_set $SPAN_TEST_SET --h2s_path ./outputs/h2s_bert
#python get_span_metrics_for_head_model.py --model_path $MODEL_PATH --head_test_set $HEAD_TEST_SET --test_set $SPAN_TEST_SET --h2s_path ./outputs/h2s_bert_gold
#python get_span_metrics_for_head_model.py --model_path $MODEL_PATH --head_test_set $HEAD_TEST_SET --test_set $SPAN_TEST_SET --h2s_path ./outputs/h2s_bert_gold_noisy

#python get_span_metrics_for_head_model.py --model_path $MODEL_PATH --head_test_set $HEAD_TEST_SET --test_set $SPAN_TEST_SET --h2s_path ./outputs/h2s_out_finetune_bert_nogold

#python get_span_metrics_for_head_model.py --model_path $MODEL_PATH --head_test_set $HEAD_TEST_SET --test_set $SPAN_TEST_SET --h2s_path ./outputs/h2s_bert_gold
#python get_span_metrics_for_head_model.py --model_path $MODEL_PATH --head_test_set $HEAD_TEST_SET --test_set $SPAN_TEST_SET --h2s_path ./outputs/h2s_out_finetune_bert_nogold_E{20}_R{1e-5}




#python get_span_metrics_for_head_model.py --model_path $MODEL_PATH --head_test_set $HEAD_TEST_SET --test_set $SPAN_TEST_SET --h2s_path ./outputs/pred_h2s_out_finetune_bert_nogold_E{20}_R{1e-5}
#python get_span_metrics_for_head_model.py --model_path $MODEL_PATH --head_test_set $HEAD_TEST_SET --test_set $SPAN_TEST_SET --h2s_path ./outputs/joint_h2s_out_finetune_bert_nogold_E{20}_R{1e-5}
