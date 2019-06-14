#!/usr/bin/env bash

python run_classifier_adv.py \
  --task_name SST-2\
  --test_task_name SST-2 \
  --eval_step 1000 \
  --use_nonbert \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 20.0 \
  --max_seq_length 128 \
  --train_glove_embs \
  --output_dir ./outputs/nonbert_sst2_trainemb/

