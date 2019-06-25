#!/usr/bin/env bash

python run_classifier_adv.py \
  --task_name mtl-apparel \
  --test_task_name mtl-apparel \
  --eval_step 100 \
  --data_portion 0.1 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --real_path ./outputs/bert_mtl_app_10p/subsampled_0.1.pkl.augstr_para_int_basic.augnum3.pkl \
  --output_dir ./outputs/aug_para_int_app/ > ./outputs/aug_para_int_app.out



python run_classifier_adv.py \
  --task_name mtl-baby \
  --test_task_name mtl-baby \
  --eval_step 100 \
  --data_portion 0.1 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --real_path ./outputs/bert_mtl_baby_10p/subsampled_0.1.pkl.augstr_para_int_basic.augnum3.pkl \
  --output_dir ./outputs/aug_para_int_baby/ > ./outputs/aug_para_int_baby.out



python run_classifier_adv.py \
  --task_name mtl-kitchen_housewares \
  --test_task_name mtl-kitchen_housewares \
  --eval_step 100 \
  --data_portion 0.1 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --real_path ./outputs/bert_mtl_kit_10p/subsampled_0.1.pkl.augstr_para_int_basic.augnum3.pkl \
  --output_dir ./outputs/aug_para_int_kit/ > ./outputs/aug_para_int_kit.out



python run_classifier_adv.py \
  --task_name mtl-software \
  --test_task_name mtl-software \
  --eval_step 100 \
  --data_portion 0.1 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --real_path ./outputs/bert_mtl_soft_10p/subsampled_0.1.pkl.augstr_para_int_basic.augnum3.pkl \
  --output_dir ./outputs/aug_para_int_soft/ > ./outputs/aug_para_int_soft.out

python run_classifier_adv.py \
  --task_name mtl-dvd \
  --test_task_name mtl-dvd \
  --eval_step 100 \
  --data_portion 0.1 \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --real_path ./outputs/bert_mtl_dvd/subsampled_0.1.pkl.augstr_para_int_basic.augnum3.pkl \
  --output_dir ./outputs/aug_para_int_dvd/ > ./outputs/aug_para_int_dvd.out


