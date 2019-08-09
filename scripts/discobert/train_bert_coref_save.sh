#!/usr/bin/env bash

python train_coreference.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file ./datasets/coref/cleaned/coref/train.json \
  --predict_file ./datasets/coref/cleaned/coref/development.json \
  --train_batch_size 12 \
  --learning_rate 2e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --save_iter 3000 \
  --output_dir ./outputs/train_bert_coref_save_1/ > ./outputs/train_bert_coref_save_1.out