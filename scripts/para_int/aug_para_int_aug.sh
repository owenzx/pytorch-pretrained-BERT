#!/usr/bin/env bash

python data_augmentation.py \
  --augment_strategy para_int_basic \
  --origin_path ./outputs/bert_mtl_app_10p/subsampled_0.1.pkl \
  --aug_num 3

python data_augmentation.py \
  --augment_strategy para_int_basic \
  --origin_path ./outputs/bert_mtl_baby_10p/subsampled_0.1.pkl \
  --aug_num 3

python data_augmentation.py \
  --augment_strategy para_int_basic \
  --origin_path ./outputs/bert_mtl_kit_10p/subsampled_0.1.pkl \
  --aug_num 3

python data_augmentation.py \
  --augment_strategy para_int_basic \
  --origin_path ./outputs/bert_mtl_soft_10p/subsampled_0.1.pkl \
  --aug_num 3

python data_augmentation.py \
  --augment_strategy para_int_basic \
  --origin_path ./outputs/bert_mtl_dvd_10p/subsampled_0.1.pkl \
  --aug_num 3


