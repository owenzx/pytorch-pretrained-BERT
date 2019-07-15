#!/usr/bin/env bash

#
#python train_controller_new.py \
#  --task_name amazon-2 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 10.0 \
#  --max_seq_length 128 \
#  --meta_train_size 100 \
#  --meta_val_size 10000 \
#  --max_meta_epoch 10000 \
#  --save_epoch 1 \
#  --load_subsample \
#  --fix_subsample_path ./outputs/vrf_subsample \
#  --log_all_policy \
#  --cache_dir ./berts/ \
#  --output_dir ./outputs/auto_amazon_vrf_random_0_tsa/ > ./outputs/auto_amazon_vrf_random_0_tsa.out



python train_controller_new.py \
  --task_name amazon-2 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --meta_train_size 100 \
  --meta_val_size 10000 \
  --max_meta_epoch 10000 \
  --save_epoch 1 \
  --load_subsample \
  --fix_subsample_path ./outputs/vrf_subsample \
  --log_all_policy \
  --seed 11 \
  --cache_dir ./berts/ \
  --output_dir ./outputs/auto_amazon_vrf_random_1_tsa/ > ./outputs/auto_amazon_vrf_random_1_tsa.out



python train_controller_new.py \
  --task_name amazon-2 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --meta_train_size 100 \
  --meta_val_size 10000 \
  --max_meta_epoch 10000 \
  --save_epoch 1 \
  --load_subsample  \
  --fix_subsample_path ./outputs/vrf_subsample \
  --log_all_policy \
  --seed 22 \
  --cache_dir ./berts/ \
  --output_dir ./outputs/auto_amazon_vrf_random_2_tsa/ > ./outputs/auto_amazon_vrf_random_2_tsa.out



python train_controller_new.py \
  --task_name amazon-2 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --meta_train_size 100 \
  --meta_val_size 10000 \
  --max_meta_epoch 10000 \
  --save_epoch 1 \
  --load_subsample \
  --fix_subsample_path ./outputs/vrf_subsample \
  --log_all_policy \
  --seed 33 \
  --cache_dir ./berts/ \
  --output_dir ./outputs/auto_amazon_vrf_random_3_tsa/ > ./outputs/auto_amazon_vrf_random_3_tsa.out



python train_controller_new.py \
  --task_name amazon-2 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --meta_train_size 100 \
  --meta_val_size 10000 \
  --max_meta_epoch 10000 \
  --save_epoch 1 \
  --load_subsample \
  --fix_subsample_path ./outputs/vrf_subsample \
  --log_all_policy \
  --seed 44 \
  --cache_dir ./berts/ \
  --output_dir ./outputs/auto_amazon_vrf_random_4_tsa/ > ./outputs/auto_amazon_vrf_random_4_tsa.out


