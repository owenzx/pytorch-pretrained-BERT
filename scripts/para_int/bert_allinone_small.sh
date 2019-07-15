#!/usr/bin/env bash

#declare -a domains=("apparel" "dvd" "kitchen_housewares" "software" "baby" "electronics" "magazines" "sports_outdoors" "books" "health_personal_care" "mr" "toys_games" "camera_photo" "imdb" "music" "video")

declare -a domains=("apparel" "dvd")

for i in "${domains[@]}"
do
    python run_classifier_adv.py \
      --task_name mtl-$i \
      --test_task_name mtl-$i \
      --eval_step 1000 \
      --data_portion 1.0 \
      --bert_model bert-base-uncased \
      --do_train \
      --do_eval \
      --do_lower_case \
      --train_batch_size 32 \
      --learning_rate 2e-5 \
      --num_train_epochs 3.0 \
      --max_seq_length 128 \
      --output_dir "./outputs/test_"$i"_test/" > "./outputs/test_"$i"_test.out"
done


