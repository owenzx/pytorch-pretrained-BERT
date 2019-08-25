#!/usr/bin/env bash

python train_coreference.py \
  --bert_model ./outputs/allen_test_bert_tune_large/tuned_bert/ \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file ./datasets/coref/cleaned/coref/train.json \
  --predict_file ./datasets/coref/cleaned/coref/development.json \
  --train_batch_size 12 \
  --learning_rate 2e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --output_dir ./outputs/train_bert_coref_save_debug_reverse/ > ./outputs/train_bert_coref_save_debug_reverse.out