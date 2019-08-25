#!/usr/bin/env bash

python train_controller_new.py \
  --task_name mtl-overfit \
  --bert_model bert-base-uncased \
  --do_train \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --max_meta_epoch 1000 \
  --save_epoch 1 \
  --adv_pol \
  --pol_type_num 3 \
  --pg_algo ppo \
  --log_all_policy \
  --overfit_setting \
  --cache_dir ./berts/ \
  --output_dir ./outputs/new_verify_small_overfit/


