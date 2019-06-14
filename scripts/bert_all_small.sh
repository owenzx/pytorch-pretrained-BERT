#!/usr/bin/env bash


python run_classifier_adv.py \
  --task_name mtl-apparel \
  --test_task_name mtl-apparel \
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
  --output_dir ./outputs/bert_mtl_app_100p/ > ./outputs/bert_mtl_app_100p.out


python run_classifier_adv.py \
  --task_name mtl-apparel \
  --test_task_name mtl-apparel \
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
  --output_dir ./outputs/bert_mtl_app_10p/ > ./outputs/bert_mtl_app_10p.out


python run_classifier_adv.py \
  --task_name mtl-dvd \
  --test_task_name mtl-dvd \
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
  --output_dir ./outputs/bert_mtl_dvd_100p/ > ./outputs/bert_mtl_dvd_100p.out


python run_classifier_adv.py \
  --task_name mtl-dvd\
  --test_task_name mtl-dvd \
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
  --output_dir ./outputs/bert_mtl_dvd_10p/ > ./outputs/bert_mtl_dvd_10p.out

python run_classifier_adv.py \
  --task_name mtl-kitchen_housewares \
  --test_task_name mtl-kitchen_housewares \
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
  --output_dir ./outputs/bert_mtl_kit_100p/ > ./outputs/bert_mtl_kit_100p.out


python run_classifier_adv.py \
  --task_name mtl-kitchen_housewares \
  --test_task_name mtl-kitchen_housewares \
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
  --output_dir ./outputs/bert_mtl_kit_10p/ > ./outputs/bert_mtl_kit_10p.out

python run_classifier_adv.py \
  --task_name mtl-software \
  --test_task_name mtl-software \
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
  --output_dir ./outputs/bert_mtl_soft_100p/ > ./outputs/bert_mtl_soft_100p.out


python run_classifier_adv.py \
  --task_name mtl-software \
  --test_task_name mtl-software \
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
  --output_dir ./outputs/bert_mtl_soft_10p/ > ./outputs/bert_mtl_soft_10p.out

python run_classifier_adv.py \
  --task_name mtl-baby \
  --test_task_name mtl-baby \
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
  --output_dir ./outputs/bert_mtl_baby_100p/ > ./outputs/bert_mtl_baby_100p.out


python run_classifier_adv.py \
  --task_name mtl-baby \
  --test_task_name mtl-baby \
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
  --output_dir ./outputs/bert_mtl_baby_10p/ > ./outputs/bert_mtl_baby_10p.out

