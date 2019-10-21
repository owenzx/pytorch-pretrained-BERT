#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/fortest/xzh/datasets/coref/allen/train.english.v4_gold_conll
export COREF_DEV_DATA_PATH=/fortest/xzh/datasets/coref/allen/dev.english.v4_gold_conll
export COREF_TEST_DATA_PATH=/fortest/xzh/datasets/coref/allen/test.english.v4_gold_conll



python -m mention_tree_gen.auto_mention_switcher \
            --lambda_pen 0.1 \
            --do_train \
            --do_eval \
            --output_dir ./outputs/mention_tree_selfcritic_1021_detection \
            --train_data_path $COREF_TRAIN_DATA_PATH \
            --eval_data_path $COREF_DEV_DATA_PATH \
            --attackee_path ./outputs/mentionswitch_really_baseline_0910 \
            --eval_batch_size 1 \
            --train_batch_size 16 \
            --reward_type detection_f1


#python mention_tree_gen/auto_mention_switcher.py \
#            --do_train \
#            --output_dir ./outputs/mention_tree_debug \
#            --train_data_path $COREF_TRAIN_DATA_PATH \
#            --eval_data_path $COREF_DEV_DATA_PATH \
#            --attackee_path ./outputs/mentionswitch_really_baseline_0910 \
#            --train_batch_size 1 \
#            --eval_batch_size 2 \
#            --num_training_epochs 2 \
