#!/usr/bin/env bash

python train_controller_new.py \
  --task_name yelp-2 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 128 \
  --meta_train_size 100 \
  --meta_val_size 1000 \
  --max_meta_epoch 10000 \
  --save_epoch 1 \
  --log_all_policy \
  --reward_c 10 \
  --output_dir ./outputs/auto_baseline_yelp_tunec/ > ./outputs/auto_baseline_yelp_tunec.out


