#!/usr/bin/env bash

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
#  --output_dir ./outputs/bert_stable_1/ > ./outputs/bert_stable_1.out

python run_classifier_adv.py \
  --task_name SST-2\
  --test_task_name SST-2 \
  --eval_step 1000 \
  --data_portion 0.01 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 300.0 \
  --max_seq_length 128 \
  --seed 43 \
  --output_dir ./outputs/bert_stable_2/ > ./outputs/bert_stable_2.out

python run_classifier_adv.py \
  --task_name SST-2\
  --test_task_name SST-2 \
  --eval_step 1000 \
  --data_portion 0.01 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 300.0 \
  --max_seq_length 128 \
  --seed 44 \
  --output_dir ./outputs/bert_stable_3/ > ./outputs/bert_stable_3.out
