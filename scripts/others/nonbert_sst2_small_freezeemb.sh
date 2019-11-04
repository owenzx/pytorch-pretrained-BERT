#!/usr/bin/env bash

#
#python run_classifier_adv.py \
#  --task_name SST-2\
#  --test_task_name SST-2 \
#  --eval_step 1000 \
#  --data_portion 1.0 \
#  --use_nonbert \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 20.0 \
#  --max_seq_length 128 \
#  --output_dir ./outputs/nonbert_sst2_freezeemb_100p/ > ./outputs/nonbert_sst2_freezeemb_100p.out
#
#python run_classifier_adv.py \
#  --task_name SST-2\
#  --test_task_name SST-2 \
#  --eval_step 1000 \
#  --data_portion 0.3 \
#  --use_nonbert \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 50.0 \
#  --max_seq_length 128 \
#  --output_dir ./outputs/nonbert_sst2_freezeemb_30p/ > ./outputs/nonbert_sst2_freezeemb_30p.out
#
#
#python run_classifier_adv.py \
#  --task_name SST-2\
#  --test_task_name SST-2 \
#  --eval_step 1000 \
#  --data_portion 0.1 \
#  --use_nonbert \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 100.0 \
#  --max_seq_length 128 \
#  --output_dir ./outputs/nonbert_sst2_freezeemb_10p/ > ./outputs/nonbert_sst2_freezeemb_10p.out
#
#
#python run_classifier_adv.py \
#  --task_name SST-2\
#  --test_task_name SST-2 \
#  --eval_step 1000 \
#  --data_portion 0.01 \
#  --use_nonbert \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 300.0 \
#  --max_seq_length 128 \
#  --output_dir ./outputs/nonbert_sst2_freezeemb_1p/ > ./outputs/nonbert_sst2_freezeemb_1p.out
#
python run_classifier_adv.py \
  --task_name SST-2\
  --test_task_name SST-2 \
  --eval_step 1000 \
  --data_portion 0.001 \
  --use_nonbert \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5000.0 \
  --max_seq_length 128 \
  --output_dir ./outputs/nonbert_sst2_freezeemb_01p/ > ./outputs/nonbert_sst2_freezeemb_01p.out

python run_classifier_adv.py \
  --task_name SST-2\
  --test_task_name SST-2 \
  --eval_step 1000 \
  --data_portion 0.0001 \
  --use_nonbert \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 50000.0 \
  --max_seq_length 128 \
  --output_dir ./outputs/nonbert_sst2_freezeemb_001p/ > ./outputs/nonbert_sst2_freezeemb_001p.out

