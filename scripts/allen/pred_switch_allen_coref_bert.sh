#!/usr/bin/env bash

allennlp predict --use-dataset-reader --predictor coreference-resolution --cuda-device 0 --silent --overrides='{"dataset_reader":{"cached_instance_path":"./debug_same.ist.2"}}' --output-file ./outputs/debug.out --include-package allen_packages ./outputs/allen_test_bert_tune_large/model.tar.gz ./datasets/coref/allen/dev.english.v4_gold_conll


#allennlp predict --use-dataset-reader --predictor coreference-resolution --cuda-device 0 --silent --output-file ./outputs/debug.out.3 --include-package allen_packages ./outputs/allen_test_bert_tune_large/model.tar.gz ./datasets/coref/allen/dev.english.v4_gold_conll

