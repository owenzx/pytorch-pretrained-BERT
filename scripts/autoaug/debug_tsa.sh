#!/usr/bin/env bash

python train_controller_new.py \
  --task_name amazon-2 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 100.0 \
  --max_seq_length 128 \
  --meta_train_size 100 \
  --meta_val_size 10000 \
  --max_meta_epoch 10000 \
  --save_epoch 1 \
  --load_subsample \
  --fix_subsample_path ./outputs/vrf_subsample \
  --tsa linear \
  --log_all_policy \
  --cache_dir ./berts/ \
  --output_dir ./outputs/debug_tsa_0/ > ./outputs/debug_tsa_0.out
#
#
#
#python train_controller_new.py \
#  --task_name amazon-2 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 100.0 \
#  --max_seq_length 128 \
#  --meta_train_size 100 \
#  --meta_val_size 10000 \
#  --max_meta_epoch 10000 \
#  --save_epoch 1 \
#  --load_subsample \
#  --fix_subsample_path ./outputs/vrf_subsample \
#  --log_all_policy \
#  --seed 1 \
#  --cache_dir ./berts/ \
#  --output_dir ./outputs/auto_amazon_vrf_random_1/ > ./outputs/auto_amazon_vrf_random_1.out
#
#
#
#python train_controller_new.py \
#  --task_name amazon-2 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 100.0 \
#  --max_seq_length 128 \
#  --meta_train_size 100 \
#  --meta_val_size 10000 \
#  --max_meta_epoch 10000 \
#  --save_epoch 1 \
#  --load_subsample \
#  --fix_subsample_path ./outputs/vrf_subsample \
#  --log_all_policy \
#  --seed 2 \
#  --cache_dir ./berts/ \
#  --output_dir ./outputs/auto_amazon_vrf_random_2/ > ./outputs/auto_amazon_vrf_random_2.out
#
#
#
#python train_controller_new.py \
#  --task_name amazon-2 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 100.0 \
#  --max_seq_length 128 \
#  --meta_train_size 100 \
#  --meta_val_size 10000 \
#  --max_meta_epoch 10000 \
#  --save_epoch 1 \
#  --load_subsample \
#  --fix_subsample_path ./outputs/vrf_subsample \
#  --log_all_policy \
#  --seed 3 \
#  --cache_dir ./berts/ \
#  --output_dir ./outputs/auto_amazon_vrf_random_3/ > ./outputs/auto_amazon_vrf_random_3.out
#
#
#
#python train_controller_new.py \
#  --task_name amazon-2 \
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
#  --load_subsample \
#  --fix_subsample_path ./outputs/vrf_subsample \
#  --log_all_policy \
#  --seed 4 \
#  --cache_dir ./berts/ \
#  --output_dir ./outputs/auto_amazon_vrf_random_4/ > ./outputs/auto_amazon_vrf_random_4.out
#

