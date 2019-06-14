#!/usr/bin/env bash


#python run_classifier_adv.py \
#  --task_name SST-2\
#  --test_task_name SST-2 \
#  --eval_step 1000 \
#  --data_portion 1.0 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 20.0 \
#  --max_seq_length 128 \
#  --output_dir ./outputs/bert_sst2_100p/ > ./outputs/bert_sst2_100p.out
#
#python run_classifier_adv.py \
#  --task_name SST-2\
#  --test_task_name SST-2 \
#  --eval_step 1000 \
#  --data_portion 0.3 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 50.0 \
#  --max_seq_length 128 \
#  --output_dir ./outputs/bert_sst2_30p/ > ./outputs/bert_sst2_30p.out
#
#
#python run_classifier_adv.py \
#  --task_name SST-2\
#  --test_task_name SST-2 \
#  --eval_step 1000 \
#  --data_portion 0.1 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 100.0 \
#  --max_seq_length 128 \
#  --output_dir ./outputs/bert_sst2_10p/ > ./outputs/bert_sst2_10p.out
#
#
#python run_classifier_adv.py \
#  --task_name SST-2\
#  --test_task_name SST-2 \
#  --eval_step 1000 \
#  --data_portion 0.01 \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 300.0 \
#  --max_seq_length 128 \
#  --output_dir ./outputs/bert_sst2_1p/ > ./outputs/bert_sst2_1p.out


python run_classifier_adv.py \
  --task_name SST-2\
  --test_task_name SST-2 \
  --eval_step 1000 \
  --data_portion 0.001 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 300.0 \
  --max_seq_length 128 \
  --output_dir ./outputs/bert_sst2_01p/ > ./outputs/bert_sst2_01p.out

