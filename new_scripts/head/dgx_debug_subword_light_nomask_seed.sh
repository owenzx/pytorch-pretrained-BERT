#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/train.min_span
export COREF_DEV_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/dev.min_span
export COREF_TEST_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/test.min_span

for seed in 100 200
do
    export SEED=$seed
    allennlp train new_scripts/head/debug_subword_light_seed.jsonnet -s outputs/head_debug_subword_light_nomask_b2_seed_$SEED --include-package new_allen_packages
done

