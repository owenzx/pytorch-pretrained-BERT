#!/usr/bin/env bash

declare -a domains=("apparel" "dvd" "kitchen_housewares" "software" "baby" "electronics" "magazines" "sports_outdoors" "books" "health_personal_care" "mr" "toys_games" "camera_photo" "imdb" "music" "video")


for i in "${domains[@]}"
do
    python data_augmentation.py \
      --augment_strategy para_int_basic \
      --origin_path "./outputs/aug_para_int_"$i"_all/subsampled_0.1.pkl" \
      --aug_num 3
done


