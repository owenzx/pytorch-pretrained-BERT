#!/usr/bin/env bash

#python train_coreference.py \
#  --bert_model bert-base-uncased \
#  --do_train \
#  --do_predict \
#  --do_lower_case \
#  --train_file ./datasets/coref/cleaned/coref/train.json \
#  --predict_file ./datasets/coref/cleaned/coref/development.json \
#  --train_batch_size 12 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 3.0 \
#  --max_seq_length 384 \
#  --output_dir ./outputs/train_bert_coref_2/ > ./outputs/train_bert_coref_2.out




python train_coreference.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file ./datasets/coref/cleaned/coref/train.json \
  --predict_file ./datasets/coref/cleaned/coref/development.json \
  --train_batch_size 12 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --seed 1 \
  --output_dir ./outputs/train_bert_coref_2_seed1/ > ./outputs/train_bert_coref_2_seed1.out




python train_coreference.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file ./datasets/coref/cleaned/coref/train.json \
  --predict_file ./datasets/coref/cleaned/coref/development.json \
  --train_batch_size 12 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --seed 2 \
  --output_dir ./outputs/train_bert_coref_2_seed2/ > ./outputs/train_bert_coref_2_seed2.out
