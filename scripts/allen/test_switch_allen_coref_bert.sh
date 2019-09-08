#!/usr/bin/env bash

allennlp evaluate --cuda-device 0 --overrides='{"dataset_reader":{"cached_instance_path":"./debug_same.ist.2"}}' --output-file ./outputs/debug.out --include-package allen_packages ./outputs/allen_test_bert_tune_large/model.tar.gz /fortest/xzh/datasets/coref/allen/dev.english.v4_gold_conll


#allennlp evaluate --cuda-device 0 --output-file ./outputs/debug.out.2 --include-package allen_packages ./outputs/allen_test_bert_tune_large/model.tar.gz ./datasets/coref/allen/dev.english.v4_gold_conll

