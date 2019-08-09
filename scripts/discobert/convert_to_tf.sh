#!/usr/bin/env bash

declare -a iters=("3000" "6000" "9000" "12000" "15000" "18000" "21000" "24000" "27000" "30000" "33000")


for i in "${iters[@]}"
do
    python my_bert/convert_pytorch_checkpoint_to_tf.py \
        --model_name bert-base-uncased \
        --pytorch_model_path ./outputs/train_bert_coref_save_1_int_$i/pytorch_model.bin \
        --tf_cache_dir ./outputs/converted/convert_to_tf_1_$i
done
