#!/usr/bin/env bash



python run_classifier_adv.py \
  --task_name yelp-5 \
  --test_task_name yelp-5 \
  --eval_step 1000 \
  --data_portion 1.0 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 128 \
  --output_dir ./outputs/bert_yelp_5_100p/ > ./outputs/bert_yelp5_100p.out

python run_classifier_adv.py \
  --task_name yelp-2 \
  --test_task_name yelp-2 \
  --eval_step 1000 \
  --data_portion 1.0 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 128 \
  --output_dir ./outputs/bert_yelp_2_100p/ > ./outputs/bert_yelp_2_100p.out


