#!/usr/bin/env bash

python run_squad_adv.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file ./datasets/QA/squad/train-v1.1.json \
  --predict_file ./datasets/QA/squad/dev-v1.1.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/