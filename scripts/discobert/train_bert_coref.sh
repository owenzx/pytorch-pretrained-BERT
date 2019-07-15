#!/usr/bin/env bash

python run_squad_ins.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file ./datasets/coref/cleaned/coref/train.json \
  --predict_file ./datasets/coref/cleaned/coref/development.json \
  --train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --output_dir ./outputs/train_bert_coref_1/ > ./outputs/train_bert_coref_1.out