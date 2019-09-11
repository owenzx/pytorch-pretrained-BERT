#!/usr/bin/env bash

allennlp evaluate --cuda-device 0 --overrides='{"dataset_reader":{"cached_instance_path":null}}' --include-package allen_packages --output-file ./outputs/debug.out ./outputs/xxxxx_modeldirpath_xxxx/model.tar.gz /fortest/xzh/datasets/coref/allen/dev.english.v4_gold_conll



