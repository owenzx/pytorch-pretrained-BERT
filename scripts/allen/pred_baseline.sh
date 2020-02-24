#!/usr/bin/env bash

allennlp predict --use-dataset-reader --predictor coreference-resolution --cuda-device 0 --silent --output-file ./outputs/allen_test/pred_on_dev.out --include-package allen_packages ./outputs/allen_test/model.tar.gz ./datasets/coref/allen/dev.english.v4_gold_conll
