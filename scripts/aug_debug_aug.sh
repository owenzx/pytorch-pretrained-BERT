#!/usr/bin/env bash

python data_augmentation.py \
  --augment_strategy add_mask \
  --origin_path ./outputs/aug_debug_main/subsampled_0.01.pkl \
  --aug_num 3


