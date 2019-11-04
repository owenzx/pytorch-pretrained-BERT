#!/usr/bin/env bash

python run_classifier_adv.py \
  --task_name amazon-2 \
  --test_task_name amazon-2 \
  --eval_step 1000 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_steps 3000 \
  --max_seq_length 128 \
  --cache_dir ./berts/ \
  --output_dir ./outputs/bert_amazon2_3k_step/ > ./outputs/bert_amazon2_3k_step.out

python run_classifier_adv.py \
  --task_name amazon-2 \
  --test_task_name amazon-2 \
  --eval_step 1000 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_steps 6000 \
  --max_seq_length 128 \
  --cache_dir ./berts/ \
  --output_dir ./outputs/bert_amazon2_6k_step/ > ./outputs/bert_amazon2_6k_step.out

#
#python run_classifier_adv.py \
#  --task_name amazon-2 \
#  --test_task_name amazon-2 \
#  --eval_step 1000 \
#  --num_per_label 10 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 30.0 \
#  --max_seq_length 128 \
#  --output_dir ./outputs/bert_amazon2_10npl/ > ./outputs/bert_amazon2_10npl.out
#
#
#python run_classifier_adv.py \
#  --task_name amazon-5 \
#  --test_task_name amazon-5 \
#  --eval_step 1000 \
#  --num_per_label 10 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 30.0 \
#  --max_seq_length 128 \
#  --output_dir ./outputs/bert_amazon5_10npl/ > ./outputs/bert_amazon5_10nlp.out
#
#
#python run_classifier_adv.py \
#  --task_name amazon-5 \
#  --test_task_name amazon-5 \
#  --eval_step 1000 \
#  --data_portion 1.0 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 2.0 \
#  --max_seq_length 128 \
#  --output_dir ./outputs/bert_amazon_5_100p/ > ./outputs/bert_amazon5_100p.out
#
#python run_classifier_adv.py \
#  --task_name amazon-2 \
#  --test_task_name amazon-2 \
#  --eval_step 1000 \
#  --data_portion 1.0 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 2.0 \
#  --max_seq_length 128 \
#  --output_dir ./outputs/bert_amazon_2_100p/ > ./outputs/bert_amazon_2_100p.out
#

