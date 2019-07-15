#!/usr/bin/env bash

python train_controller_new.py \
  --task_name amazon-2 \
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
  --adv_pol \
  --pol_type_num 3 \
  --pg_algo ppo \
  --log_all_policy \
  --cache_dir ./berts/ \
  --output_dir ./outputs/auto_baseline_amazon_ppo_advpol/ > ./outputs/auto_baseline_amazon_ppo_advpol.out


