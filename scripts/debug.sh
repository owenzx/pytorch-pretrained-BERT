#!/usr/bin/env bash

python run_classifier_adv.py \
  --task_name SST-2\
  --test_task_name SST-2 \
  --eval_step 10 \
  --use_nonbert \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 128 \
  --real_path ./outputs/aug_debug_main/subsampled_0.01.pkl.augstr_random_swap.augnum3.pkl \
  --output_dir ./outputs/aug_debug_retrain/ > ./outputs/aug_debug_retrain.out
