#!/usr/bin/env bash

allennlp evaluate --cuda-device 0 --overrides='{"dataset_reader":{"cached_instance_path":null}}' --include-package allen_packages --output-file ./outputs/debug.out ./outputs/mentionswitch_consist_same_0910 /playpen/home/xzh/datasets/WikiCoref/Evaluation/key-OntoNotesScheme



