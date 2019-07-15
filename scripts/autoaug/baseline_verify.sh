#!/usr/bin/env bash

#python train_controller_new.py \
#  --task_name yelp-2 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 3.0 \
#  --max_seq_length 128 \
#  --meta_train_size 100 \
#  --meta_val_size 1000 \
#  --max_meta_epoch 10000 \
#  --save_epoch 1 \
#  --log_all_policy \
#  --cache_dir ./berts/ \
#  --output_dir ./outputs/auto_baseline_verify_3/ > ./outputs/auto_baseline_verify_3.out
#
#
#
#python train_controller_new.py \
#  --task_name yelp-2 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 10.0 \
#  --max_seq_length 128 \
#  --meta_train_size 100 \
#  --meta_val_size 1000 \
#  --max_meta_epoch 10000 \
#  --save_epoch 1 \
#  --log_all_policy \
#  --cache_dir ./berts/ \
#  --output_dir ./outputs/auto_baseline_verify_10/ > ./outputs/auto_baseline_verify_10.out
#
#
#
#python train_controller_new.py \
#  --task_name yelp-2 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 50.0 \
#  --max_seq_length 128 \
#  --meta_train_size 100 \
#  --meta_val_size 1000 \
#  --max_meta_epoch 10000 \
#  --save_epoch 1 \
#  --log_all_policy \
#  --cache_dir ./berts/ \
#  --output_dir ./outputs/auto_baseline_verify_50/ > ./outputs/auto_baseline_verify_50.out
#
#
#
#python train_controller_new.py \
#  --task_name yelp-2 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 100.0 \
#  --max_seq_length 128 \
#  --meta_train_size 100 \
#  --meta_val_size 1000 \
#  --max_meta_epoch 10000 \
#  --save_epoch 1 \
#  --log_all_policy \
#  --cache_dir ./berts/ \
#  --output_dir ./outputs/auto_baseline_verify_100/ > ./outputs/auto_baseline_verify_100.out



python train_controller_new.py \
  --task_name yelp-2 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 200.0 \
  --max_seq_length 128 \
  --meta_train_size 100 \
  --meta_val_size 1000 \
  --max_meta_epoch 10000 \
  --save_epoch 1 \
  --log_all_policy \
  --cache_dir ./berts/ \
  --output_dir ./outputs/auto_baseline_verify_200/ > ./outputs/auto_baseline_verify_200.out



python train_controller_new.py \
  --task_name yelp-2 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 400.0 \
  --max_seq_length 128 \
  --meta_train_size 100 \
  --meta_val_size 1000 \
  --max_meta_epoch 10000 \
  --save_epoch 1 \
  --log_all_policy \
  --cache_dir ./berts/ \
  --output_dir ./outputs/auto_baseline_verify_400/ > ./outputs/auto_baseline_verify_400.out


