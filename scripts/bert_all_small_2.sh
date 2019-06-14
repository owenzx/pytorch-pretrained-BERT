#!/usr/bin/env bash


python run_classifier_adv.py \
  --task_name mtl-electronics \
  --test_task_name mtl-electronics \
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
  --output_dir ./outputs/bert_mtl_elec_100p/ > ./outputs/bert_mtl_elec_100p.out


python run_classifier_adv.py \
  --task_name mtl-electronics \
  --test_task_name mtl-electronics \
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
  --output_dir ./outputs/bert_mtl_elec_10p/ > ./outputs/bert_mtl_elec_10p.out


python run_classifier_adv.py \
  --task_name mtl-magazines \
  --test_task_name mtl-magazines \
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
  --output_dir ./outputs/bert_mtl_magazines_100p/ > ./outputs/bert_mtl_magazines_100p.out


python run_classifier_adv.py \
  --task_name mtl-magazines\
  --test_task_name mtl-magazines \
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
  --output_dir ./outputs/bert_mtl_magazines_10p/ > ./outputs/bert_mtl_magazines_10p.out

python run_classifier_adv.py \
  --task_name mtl-sports_outdoors \
  --test_task_name mtl-sports_outdoors \
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
  --output_dir ./outputs/bert_mtl_sport_100p/ > ./outputs/bert_mtl_sport_100p.out


python run_classifier_adv.py \
  --task_name mtl-sports_outdoors \
  --test_task_name mtl-sports_outdoors \
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
  --output_dir ./outputs/bert_mtl_sport_10p/ > ./outputs/bert_mtl_sport_10p.out

python run_classifier_adv.py \
  --task_name mtl-books \
  --test_task_name mtl-books \
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
  --output_dir ./outputs/bert_mtl_books_100p/ > ./outputs/bert_mtl_books_100p.out


python run_classifier_adv.py \
  --task_name mtl-books \
  --test_task_name mtl-books \
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
  --output_dir ./outputs/bert_mtl_books_10p/ > ./outputs/bert_mtl_books_10p.out

python run_classifier_adv.py \
  --task_name mtl-health_personal_care \
  --test_task_name mtl-health_personal_care \
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
  --output_dir ./outputs/bert_mtl_health_personal_care_100p/ > ./outputs/bert_mtl_health_personal_care_100p.out


python run_classifier_adv.py \
  --task_name mtl-health_personal_care \
  --test_task_name mtl-health_personal_care \
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
  --output_dir ./outputs/bert_mtl_health_personal_care_10p/ > ./outputs/bert_mtl_health_personal_care_10p.out

