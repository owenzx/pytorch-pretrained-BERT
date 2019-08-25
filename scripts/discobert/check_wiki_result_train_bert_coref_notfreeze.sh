#!/usr/bin/env bash

python train_coreference.py \
  --bert_model ./outputs/train_bert_coref_2/ \
  --do_predict \
  --save_predictions \
  --do_lower_case \
  --train_file ./datasets/coref/cleaned/coref/train.json \
  --predict_file ./datasets/coref/cleaned/coref/dev_wiki_short.json \
  --train_batch_size 12 \
  --learning_rate 0.0001 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --output_dir ./outputs/check_wiki_train_bert_coref_2/ > ./outputs/check_wiki_train_bert_coref_2.out