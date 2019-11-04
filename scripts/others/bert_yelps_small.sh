#!/usr/bin/env bash


python run_classifier_adv.py \
  --task_name yelp-2 \
  --test_task_name yelp-2 \
  --eval_step 1000 \
  --num_per_label 10 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --output_dir ./outputs/bert_yelp2_10npl/ > ./outputs/bert_yelp2_10npl.out


python run_classifier_adv.py \
  --task_name yelp-5 \
  --test_task_name yelp-5 \
  --eval_step 1000 \
  --num_per_label 10 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --output_dir ./outputs/bert_yelp5_10npl/ > ./outputs/bert_yelp5_10nlp.out

