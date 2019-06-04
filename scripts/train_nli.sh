#!/usr/bin/env bash

python run_classifier_adv.py \
  --task_name MNLI\
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ./datasets/glue_data/MNLI/ \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 128 \
  --output_dir /tmp/mrpc_output/