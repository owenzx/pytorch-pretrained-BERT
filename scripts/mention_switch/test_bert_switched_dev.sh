#!/usr/bin/env bash

allennlp evaluate --cuda-device 0 --overrides='{"dataset_reader":{"cached_instance_path":"./cache/conll_dev_gt_switch_dev.ist"}}' --include-package allen_packages --output-file ./outputs/test_bert_sw_dev_dev.out ./outputs/mentionswitch_really_baseline_0910 /playpen/home/xzh/datasets/coref/allen/dev.english.v4_gold_conll

allennlp evaluate --cuda-device 0 --overrides='{"dataset_reader":{"cached_instance_path":"./cache/conll_dev_gt_switch_train.ist"}}' --include-package allen_packages --output-file ./outputs/test_bert_sw_dev_train.out ./outputs/mentionswitch_really_baseline_0910 /playpen/home/xzh/datasets/coref/allen/dev.english.v4_gold_conll

allennlp evaluate --cuda-device 0 --overrides='{"dataset_reader":{"cached_instance_path":"./cache/conll_dev_gt_switch_find.ist"}}' --include-package allen_packages --output-file ./outputs/test_bert_sw_dev_find.out ./outputs/mentionswitch_really_baseline_0910 /playpen/home/xzh/datasets/coref/allen/dev.english.v4_gold_conll



