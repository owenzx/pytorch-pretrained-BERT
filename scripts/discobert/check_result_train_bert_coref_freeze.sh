#!/usr/bin/env bash

  python train_coreference.py \
  --bert_model ./outputs/train_bert_coref_freeze_1_seed1/ \
  --freeze_bert \
  --do_predict \
  --save_predictions \
  --do_lower_case \
  --train_file ./datasets/coref/cleaned/coref/train.json \
  --predict_file ./datasets/coref/cleaned/coref/development.json \
  --train_batch_size 12 \
  --learning_rate 0.0001 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --output_dir ./outputs/check_train_bert_coref_freeze_1_seed1/ > ./outputs/check_train_bert_coref_freeze_1_seed1.out

  python train_coreference.py \
  --bert_model ./outputs/train_bert_coref_freeze_1_seed2/ \
  --freeze_bert \
  --do_predict \
  --save_predictions \
  --do_lower_case \
  --train_file ./datasets/coref/cleaned/coref/train.json \
  --predict_file ./datasets/coref/cleaned/coref/development.json \
  --train_batch_size 12 \
  --learning_rate 0.0001 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --output_dir ./outputs/check_train_bert_coref_freeze_1_seed2/ > ./outputs/check_train_bert_coref_freeze_1_seed2.out


