#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/train.min_span
export COREF_DEV_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/dev.min_span
export COREF_TEST_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/test.min_span

#lr origin 1e-5
for lr in 1e-6 1e-4 2e-5 5e-5
do export LR_TRANS=$lr
    allennlp train new_scripts/head/tune_debug_subword2.jsonnet -s outputs/head_debug_subword_LR$LR_TRANS --include-package new_allen_packages
done
#
#
## ldp origin 0.5
#for ldp in 0.0 0.2 0.7
#    do export LDP=$ldp
#    allennlp train new_scripts/head/tune_debug_subword.jsonnet -s outputs/head_debug_subword_LDP$LDP --include-package new_allen_packages
#done

