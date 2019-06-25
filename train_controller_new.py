# codint=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME, CONTROLLER_WEIGHTS_NAME, WEIGHTS_CPT_NAME, CONTROLLER_WEIGHTS_CPT_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer, SimpleTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from simple_models import SimpleSequenceClassification

from model_configs import *

from utils import *

from data_processing import *

from trainer_main import Trainer_Main
from controller import Controller
from trainer_controller import Trainer_Controller
from data_augmentation import generate_aug_examples

from copy import deepcopy

logger = logging.getLogger(__name__)

handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    # parser.add_argument("--data_dir",
    #                    default=None,
    #                    type=str,
    #                    required=True,
    #                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--bert_model", default=None, type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--do_test',
                        action='store_true',
                        help='Whether to run eval on the test set.')
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    # Additional parameters
    parser.add_argument("--test_task_names",
                        default=None,
                        type=str,
                        help="The name of out-of-domain test tasks, separate by comma, remain None to only test on in-domain test sets")
    parser.add_argument("--eval_step",
                        default=1000,
                        type=int,
                        help="Eval the model very eval_step steps")
    parser.add_argument("--data_portion",
                        default=1.0,
                        type=float,
                        help="The portion of training dataset to use")
    parser.add_argument("--use_nonbert",
                        default=False,
                        action='store_true',
                        help="Whether to use Bert and the corresponding tokenizer")
    parser.add_argument("--model_path",
                        default=None,
                        type=str,
                        help="The location of saved non-bert model")
    parser.add_argument("--glove_path",
                        default="./datasets/glove.840B.300d.txt",
                        type=str,
                        help="The location of GloVe pre-trained embeddings")
    parser.add_argument("--glove_topn",
                        default=100000,
                        type=int,
                        help="only use top n word embeddings from glove")
    parser.add_argument("--train_glove_embs",
                        default=False,
                        action='store_true',
                        help='Add this argument to train the embeddings')
    parser.add_argument("--real_path",
                        default=None,
                        type=str,
                        help='The path to the real dataset to use for training')
    parser.add_argument('--normal_adam',
                        default=False,
                        action='store_true',
                        help='If set, use normal')
    parser.add_argument('--mtl_domain',
                        default=None,
                        type=str,
                        help='The domain for mtl datasets')
    parser.add_argument('--lm_first',
                        default=False,
                        action='store_true',
                        help='If set, first fine-tune by LM before training')
    parser.add_argument('--num_per_label',
                        default=-1,
                        type=int,
                        help='If >0, make sure # samples are equal for every label')
    parser.add_argument('--controller_hid',
                        type=int,
                        default=100)
    parser.add_argument('--controller_lr',
                        type=float,
                        default=3.5e-4)
    parser.add_argument('--softmax_temperature',
                        type=float,
                        default=5.0)
    parser.add_argument('--ways_aug',
                        type=eval,
                        default="['swap_sen', 'del_sen', 'concat_para']"
                        )
    parser.add_argument('--num_mix',
                        type=int,
                        default=3)
    parser.add_argument('--num_aug_prob',  # 0 0.25 0.5 1 2 4
                        type=int,
                        default=6)
    parser.add_argument('--num_aug_mag',  # wo 0 (1 2 3 4 5)
                        type=int,
                        default=5)
    parser.add_argument('--tanh_c',
                        type=float,
                        default=2.5)
    parser.add_argument('--ema_baseline_decay',
                        type=float,
                        default=0.95)
    parser.add_argument('--entropy_mode',
                        type=str,
                        default='reward',
                        choices=['reward', 'regularizer'])
    parser.add_argument('--entropy_coeff',
                        type=float,
                        default=1e-4)
    parser.add_argument('--controller_grad_clip',
                        type=float,
                        default=0)

    parser.add_argument('--save_main_model',
                        default=False,
                        action='store_true',
                        help='If set save all the main model, only for debugging')

    parser.add_argument('--meta_train_size',
                        default=100,
                        type=int)

    parser.add_argument('--meta_val_size',
                        default=300,
                        type=int)

    parser.add_argument('--max_meta_epoch',
                        default=10,
                        type=int)

    parser.add_argument('--debug',
                        default=False,
                        action='store_true')
    parser.add_argument('--discount',
                        type=float,
                        default=1.0)

    parser.add_argument('--save_epoch',
                        default=1,
                        type=int)

    parser.add_argument('--log_all_policy',
                        default=False,
                        action='store_true')

    parser.add_argument('--alter_datasets',
                        default=False,
                        action='store_true')

    parser.add_argument('--reward_c',
                        default=1,
                        type=int)

    args = parser.parse_args()

    if not args.use_nonbert:
        print(args.use_nonbert)
        assert (args.bert_model is not None)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    args.policy_log_path = os.path.join(args.output_dir, 'policies.log')

    # Finish checking args

    # Start cleaning datasets
    task_name = args.task_name.lower()
    task_main = task_name
    data_dir_main = data_dirs[task_name]

    if task_main not in processors:
        raise ValueError("Task not found: %s" % (task_main))

    processor = processors[task_main](args.output_dir)
    output_mode = output_modes[task_main]

    if args.test_task_names is None:
        test_task_names = args.task_name.lower()
    else:
        test_task_names = args.test_task_names.lower().split(',')

    #test_processors = [processors[t]() for t in test_task_names]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    #test_num_labels = [len(p.get_labels()) for p in test_processors]

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        if args.real_path is None:
            full_examples = processor.get_train_examples(data_dir_main, args.data_portion, args.num_per_label)
        else:
            full_examples = processor.get_augmented_train_examples(args.real_path)

    # Finish cleaning datasets

    # Build models

    trainer_main = Trainer_Main(args, logger, device, num_labels, output_mode, label_list, task_name)

    trainer_controller = Trainer_Controller(args, logger, device, num_labels)

    train_examples, val_examples = sample_meta_trainval(full_examples, args.meta_train_size, args.meta_val_size)

    if args.do_train:
    #    trainer_main.build_optimizer(train_examples)
        trainer_controller.build_optimizer()

    if args.do_eval:
        val_train_examples = processor.get_val_train_examples(data_dir_main)
        val_train_examples = val_train_examples[:args.meta_train_size]

    ## Start Training

    for epoch_idx in range(0, args.max_meta_epoch):
        # Sub-sample data to get train and dev

        if args.alter_datasets:
            train_examples, val_examples = sample_meta_trainval(full_examples, args.meta_train_size, args.meta_val_size)

        if epoch_idx == 0 or epoch_idx == args.max_meta_epoch-1:
            trainer_main.reset_model()
            trainer_main.build_optimizer(train_examples)
            trainer_main.train(train_examples)

            val_metrics = trainer_main.eval(val_examples)
            logger.info("BASELINE_ACC on epoch {}: {}".format(epoch_idx, val_metrics["acc"]))


        aug_policy, log_probs, entropies = trainer_controller.get_policy(with_details=True)
        aug_examples = generate_aug_examples(train_examples, aug_policy)
        trainer_main.reset_model()
        trainer_main.build_optimizer(aug_examples)

        # get data
        trainer_main.train(aug_examples)

        val_metrics = trainer_main.eval(val_examples)
        logger.info("VAL_ACC on epoch {}: {}".format(epoch_idx, val_metrics["acc"]))

        reward = trainer_controller.calc_reward(val_metrics)

        trainer_controller.train(reward, log_probs, entropies)

        if epoch_idx % args.save_epoch == 0:
            aug_policy = trainer_controller.get_policy()
            aug_examples = generate_aug_examples(val_train_examples, aug_policy)
            trainer_main.reset_model()
            trainer_main.build_optimizer(aug_examples)
            trainer_main.train(aug_examples)
            val_metrics = trainer_main.eval(val_examples)
            logger.info("VAL_TRAIN_ACC on epoch {}: {}".format(epoch_idx, val_metrics["acc"]))
            #
            #
            # trainer_cpt_path =  os.path.join(args.output_dir, CONTROLLER_WEIGHTS_CPT_NAME%epoch_idx)
            # trainer_controller.save_controller(trainer_cpt_path)
            # if args.save_main_model:
            #     main_cpt_path = os.path.join(args.output_dir, WEIGHTS_CPT_NAME%epoch_idx)
            #     output_config_path = os.path.join(args.output_dir, CONFIG_NAME)
            #     trainer_main.save_model(main_cpt_path, output_config_path)

    #results_json_file = os.path.join(args.output_dir, "results.json")
    #with open(results_json_file, 'w') as fw:
    #    json.dump(results_dict, fw)


if __name__ == "__main__":
    main()
