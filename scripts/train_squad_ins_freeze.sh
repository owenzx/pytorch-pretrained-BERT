#!/usr/bin/env bash

python run_squad_ins.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file ./datasets/ins/dbpedia/train.json \
  --predict_file ./datasets/ins/dbpedia/dev.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --freeze_bert \
  --output_dir ./outputs/run_squad_ins_dbpedia_freeze/ > ./outputs/run_squad_ins_dbpedia_freeze.out