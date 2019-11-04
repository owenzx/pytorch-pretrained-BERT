#!/usr/bin/env bash

python run_classifier_adv.py \
  --task_name MNLI\
  --test_task_name MNLI,RTE,WNLI \
  --eval_step 1000 \
  --data_portion 0.001 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3000 \
  --max_seq_length 128 \
  --output_dir ./outputs/mnli_check_01p/
