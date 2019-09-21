#!/usr/bin/env bash

allennlp predict --use-dataset-reader --predictor coreference-resolution --silent --cuda-device 0 --include-package allen_packages --output-file ./outputs/debug.out ./outputs/mentionswitch_really_baseline_0917/model.tar.gz /ssd-playpen/home/xzh/datasets/unlabeled_news/cnn.tokenized.json.debug






# /ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/datasets/coref/allen/dev.english.v4_gold_conll







