#!/usr/bin/env bash


python run_classifier_adv.py \
  --task_name mtl-MR \
  --test_task_name mtl-MR \
  --eval_step 1000 \
  --data_portion 1.0 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --output_dir ./outputs/bert_mtl_MR_100p/ > ./outputs/bert_mtl_MR_100p.out


python run_classifier_adv.py \
  --task_name mtl-MR \
  --test_task_name mtl-MR \
  --eval_step 1000 \
  --data_portion 0.1 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --output_dir ./outputs/bert_mtl_MR_10p/ > ./outputs/bert_mtl_MR_10p.out
#
#
#python run_classifier_adv.py \
#  --task_name mtl-toys_games \
#  --test_task_name mtl-toys_games \
#  --eval_step 1000 \
#  --data_portion 1.0 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 30.0 \
#  --max_seq_length 128 \
#  --output_dir ./outputs/bert_mtl_toys_games_100p/ > ./outputs/bert_mtl_toys_games_100p.out
#
#
#python run_classifier_adv.py \
#  --task_name mtl-toys_games\
#  --test_task_name mtl-toys_games \
#  --eval_step 1000 \
#  --data_portion 0.1 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 30.0 \
#  --max_seq_length 128 \
#  --output_dir ./outputs/bert_mtl_toys_games_10p/ > ./outputs/bert_mtl_toys_games_10p.out

