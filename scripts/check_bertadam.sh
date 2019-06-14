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
#  --num_train_epochs 3.0 \
#  --max_seq_length 128 \
#  --real_path ./outputs/aug_debug_main/subsampled_0.01.pkl \
#  --output_dir ./outputs/check_bertadam_epo3/ > ./outputs/check_bertadam_epo3.out


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
#  --num_train_epochs 30.0 \
#  --max_seq_length 128 \
#  --real_path ./outputs/aug_debug_main/subsampled_0.01.pkl \
#  --output_dir ./outputs/check_bertadam_epo30/ > ./outputs/check_bertadam_epo30.out



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
  --num_train_epochs 10.0 \
  --max_seq_length 128 \
  --real_path ./outputs/aug_debug_main/subsampled_0.01.pkl \
  --output_dir ./outputs/check_bertadam_epo10/ > ./outputs/check_bertadam_epo10.out


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
  --num_train_epochs 50.0 \
  --max_seq_length 128 \
  --real_path ./outputs/aug_debug_main/subsampled_0.01.pkl \
  --output_dir ./outputs/check_bertadam_epo50/ > ./outputs/check_bertadam_epo50.out


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
  --learning_rate 1e-4 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --real_path ./outputs/aug_debug_main/subsampled_0.01.pkl \
  --output_dir ./outputs/check_bertadam_epo30_lr1e-4/ > ./outputs/check_bertadam_epo30_lr1e-4.out


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
  --learning_rate 5e-6 \
  --num_train_epochs 30.0 \
  --max_seq_length 128 \
  --real_path ./outputs/aug_debug_main/subsampled_0.01.pkl \
  --output_dir ./outputs/check_bertadam_epo30_lr5e-6/ > ./outputs/check_bertadam_epo30_lr5e-6.out

