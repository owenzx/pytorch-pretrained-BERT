#!/usr/bin/env bash

export COREF_TRAIN_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/train.min_span
export COREF_DEV_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/dev.min_span
export COREF_TEST_DATA_PATH=/fortest/xzh/work/pytorch-pretrained-BERT/test.min_span

#lr origin 1e-5
#for lr in 1e-6 1e-4 2e-5 5e-5
#do export LR_TRANS=$lr
#    allennlp train new_scripts/head/debug_subword.jsonnet -s outputs/head_debug_subword_LR$LR_TRANS --include-package new_allen_packages
#done

#note batch size is set to 1
#
## ldp origin 0.5
for tlr in 1e-4 3e-4 1e-5 5e-4
do
    export TLR=$tlr
    allennlp train new_scripts/head/debug_subword_light_tlr.jsonnet -s outputs/head_debug_subword_light_TLR{$TLR} --include-package new_allen_packages
done

#
## ldp origin 0.5
#for ffdp in 0.0
#do
#    for rnndp in 0.0
#    do
#        export FFDP=$ffdp
#        export RNNDP=$rnndp
#        allennlp train new_scripts/head/debug_subword_morereg.jsonnet -s outputs/head_debug_subword_b1_FFDP{$FFDP}_RNNDP{$RNNDP} --include-package new_allen_packages
#    done
#done

