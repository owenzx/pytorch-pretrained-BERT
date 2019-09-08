#!/usr/bin/env bash

#python coref_adv.py
allennlp evaluate --cuda-device 0 --overrides='{"dataset_reader":{"cached_instance_path":"./switch.ist.3"}}' --include-package allen_packages --output-file ./outputs/debug.out ./outputs/allen_test_bert_tune_large/model.tar.gz /fortest/xzh/datasets/coref/allen/dev.english.v4_gold_conll

