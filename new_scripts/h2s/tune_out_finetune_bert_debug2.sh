#!/usr/bin/env bash



export LR=1e-5

export TRAIN_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/pred_train_out_short_h2s.json
export DEV_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_out_short_h2s.json
export TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_out_short_h2s.json
export LOAD_MODEL_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/h2s_bert


for epc in 10 20 50
do export EPC=$epc
        allennlp train new_scripts/h2s/tune_out_finetune_bert.jsonnet -s outputs/pred_h2s_out_finetune_bert_nogold_E{$EPC}_R{$LR} --include-package new_allen_packages
done





export LR=1e-5

export TRAIN_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/joint_train_out_short_h2s.json
export DEV_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_out_short_h2s.json
export TEST_DATA_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev_out_short_h2s.json
export LOAD_MODEL_PATH=/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/h2s_bert


for epc in 10 20 50
do export EPC=$epc
        allennlp train new_scripts/h2s/tune_out_finetune_bert.jsonnet -s outputs/joint_h2s_out_finetune_bert_nogold_E{$EPC}_R{$LR} --include-package new_allen_packages
done

