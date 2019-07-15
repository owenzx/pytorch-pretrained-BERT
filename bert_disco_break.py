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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer, SimpleTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from simple_models import SimpleSequenceClassification

from model_configs import *

from utils import *

from data_processing import *

from copy import deepcopy

logger = logging.getLogger(__name__)

handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

def compute_accuracy(preds, labels, task_main, task_comp):
    if output_num[task_main] == output_num[task_comp]:
        return simple_accuracy(preds, labels)
    elif output_num[task_main] > output_num[task_comp]:
        return three2two_accuracy(preds, labels)
    elif output_num[task_main] < output_num[task_comp]:
        return two2three_accuracy(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def three2two_accuracy(preds, labels):
    # 3: contradiction, entailment, neutral
    # 2: entailment, non_entailment
    mapped_preds = (preds!=1).astype(int)
    return (mapped_preds == labels).mean()


def two2three_accuracy(preds, labels):
    # 3: contradiction, entailment, neutral
    # 2: entailment, non_entailment
    mapped_labels = (labels!=1).astype(int)
    return (preds == mapped_labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels, task_main):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "yelp-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "yelp-5":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "amazon-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "amazon-5":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "dbpedia":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": compute_accuracy(preds, labels, task_main, task_name)}
    elif task_name == "mnli-mm":
        return {"acc": compute_accuracy(preds, labels, task_main, "mnli")}
    elif task_name == "qnli":
        return {"acc": compute_accuracy(preds, labels, task_main, task_name)}
    elif task_name == "rte":
        return {"acc": compute_accuracy(preds, labels, task_main, task_name)}
    elif task_name == "wnli":
        return {"acc": compute_accuracy(preds, labels, task_main, task_name)}
    elif task_name[:3] == "mtl":
        return {"acc": compute_accuracy(preds, labels, task_main, task_name)}
    else:
        raise KeyError(task_name)

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "yelp-2": Yelp2Processor,
    "yelp-5": Yelp5Processor,
    "amazon-2": Yelp2Processor,
    "amazon-5": Yelp5Processor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "mtl": "classification",
    "yelp-2": "classification",
    "yelp-5": "classification",
    "amazon-2": "classification",
    "amazon-5": "classification",
    "dbpedia": "classification",
}

output_num = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 0,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "yelp-2": 2,
    "yelp-5": 5,
    "amazon-2": 2,
    "amazon-5": 5,
    "dbpedia": 14,
}


data_dirs = {
    "cola": './datasets/glue_data/CoLA/',
    "mnli": './datasets/glue_data/MNLI/',
    "mrpc": './datasets/glue_data/MRPC/',
    "sst-2": './datasets/glue_data/SST-2/',
    "sts-b": './datasets/glue_data/STS-B/',
    "qqp": './datasets/glue_data/QQP/',
    "qnli": './datasets/glue_data/QNLI/',
    "rte": './datasets/glue_data/RTE/',
    "wnli": './datasets/glue_data/WNLI/',
    "yelp-2": './datasets/yelp-2/yelp_review_polarity_csv/',
    "yelp-5": './datasets/yelp-5/yelp_review_full_csv/',
    "amazon-2": './datasets/amazon-2/amazon_review_polarity_csv/',
    "amazon-5": './datasets/amazon-5/amazon_review_full_csv/',
    "dbpedia": './datasets/dbpedia/dbpedia_csv/',
}

mtl_root_path = './datasets/mtl-dataset/subdatasets/'
mtl_domains = ['apparel', 'dvd', 'kitchen_housewares', 'software', 'baby', 'electronics', 'magazines', 'sports_outdoors', 'books', 'health_personal_care', 'mr', 'toys_games', 'camera_photo', 'imdb', 'music', 'video']

for dom in mtl_domains:
    data_dirs['mtl-%s'%dom] = mtl_root_path + dom + '/'
    processors['mtl-%s'%dom] = MtlProcessor
    output_modes['mtl-%s'%dom] = 'classification'
    output_num['mtl-%s'%dom] = 2




def eval_model(task_name, tokenizer, model, args, device, task_main, results_dict, write2file=False, write_predictions=False):
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    task_data_dir = data_dirs[task_name]

    main_processor = processors[task_main]()
    main_label_list = main_processor.get_labels()
    main_num_labels = len(main_label_list)

    eval_examples = processor.get_dev_examples(task_data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, logger)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, main_num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    if write_predictions:
        output_pred_file = os.path.join(args.output_dir, "pred_results_on_%s.txt"%task_name)
        assert(len(preds) == len(eval_examples))
        with open(output_pred_file, "w") as fw:
            for i in range(len(preds)):
                example_dict = {'guid': eval_examples[i].guid,
                                'text_a': eval_examples[i].text_a,
                                'text_b': eval_examples[i].text_b,
                                'pred': preds[i].item(),
                                'label': eval_examples[i].label}
                json.dump(example_dict, fw)
                fw.write('\n')

    result = compute_metrics(task_name, preds, all_label_ids.numpy(), task_main)
    #loss = tr_loss/global_step if args.do_train else None

    result['eval_loss'] = eval_loss
    #result['global_step'] = global_step
    #result['loss'] = loss

    output_eval_file = os.path.join(args.output_dir, "eval_results_on_%s.txt"%task_name)

    for key in sorted(result.keys()):
        if key not in results_dict.keys():
            results_dict[key] = []
        results_dict[key].append(result[key])

    if write2file:
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    else:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    # hack for MNLI-MM
    if task_name == "mnli":
        task_name = "mnli-mm"
        processor = processors[task_name]()

        if os.path.exists(args.output_dir + '-MM') and os.listdir(args.output_dir + '-MM') and args.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        if not os.path.exists(args.output_dir + '-MM'):
            os.makedirs(args.output_dir + '-MM')

        eval_examples = processor.get_dev_examples(task_data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, logger)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, main_num_labels), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(task_name, preds, all_label_ids.numpy(), task_main)
    #    loss = tr_loss/global_step if args.do_train else None

        result['eval_loss'] = eval_loss
    #    result['global_step'] = global_step
    #    result['loss'] = loss

        for key in sorted(result.keys()):
            if key + '-MM' not in results_dict.keys():
                results_dict[key + '-MM'] = []
            results_dict[key + '-MM'].append(result[key])

        if write_predictions:
            output_pred_file = os.path.join(args.output_dir, "pred_results_on_%s-MM.txt"%task_name)
            assert(len(preds) == len(eval_examples))
            with open(output_pred_file, "w") as fw:
                for i in range(len(preds)):
                    example_dict = {'guid': eval_examples[i].guid,
                                    'text_a': eval_examples[i].text_a,
                                    'text_b': eval_examples[i].text_b,
                                    'pred': preds[i].item(),
                                    'label': eval_examples[i].label}
                    json.dump(example_dict, fw)
                    fw.write('\n')

        output_eval_file = os.path.join(args.output_dir, "eval_results_on_%s-MM.txt"%task_name)
        if write2file:
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
        else:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    #parser.add_argument("--data_dir",
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
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")


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




    args = parser.parse_args()

    if not args.use_nonbert:
        print(args.use_nonbert)
        assert(args.bert_model is not None)


    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

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

    test_processors = [processors[t]() for t in test_task_names]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    test_num_labels = [len(p.get_labels()) for p in test_processors]


    # Assert all tasks have same format (should still manually check the content)
    # assert(all_same(test_num_labels+[num_labels]))

    if not args.use_nonbert:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    else:
        tokenizer = SimpleTokenizer(vocab_file=args.glove_path, do_lower_case=args.do_lower_case, topn=args.glove_topn)
        glove_embs = load_glove_embs(args.glove_path, tokenizer.vocab)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        if args.real_path is None:
            train_examples = processor.get_train_examples(data_dir_main, args.data_portion, args.num_per_label)
        else:
            train_examples = processor.get_augmented_train_examples(args.real_path)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    if not args.use_nonbert:
        model = BertForSequenceClassification.from_pretrained(args.bert_model,
                  cache_dir=cache_dir,
                  num_labels=num_labels)
    else:
        simple_config = SimpleConfig(tokenizer.vocab, config_sglsen_1)
        model = SimpleSequenceClassification(simple_config, num_labels=num_labels, glove_embs=glove_embs, freeze_emb=not args.train_glove_embs)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)

        else:
            if not args.normal_adam:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=args.learning_rate,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)
            else:
                optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)


    results_dict = {'train':{'loss':[]},'dev':{}}

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, logger)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                if global_step % args.eval_step == 0:
                    for t_name in test_task_names:
                        if t_name not in results_dict['dev'].keys():
                            results_dict['dev'][t_name] = {}
                        logger.info("Eval on %s"%t_name)
                        eval_model(t_name, tokenizer, model, args, device, task_main, results_dict['dev'][t_name], write2file=False)
                    model.train()

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        if not args.use_nonbert:
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        if not args.use_nonbert:
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        if not args.use_nonbert:
            model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
            tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        else:
            # Not re-initializing GloVe here
            model = SimpleSequenceClassification(simple_config, num_labels=num_labels)
            model.load_state_dict(torch.load(output_model_file))

    else:
        if not args.use_nonbert:
            model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
        else:
            model = SimpleSequenceClassification(simple_config, num_labels=num_labels)
            model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        for t_name in test_task_names:
            if t_name not in results_dict['dev'].keys():
                results_dict['dev'][t_name] = {}
            logger.info("Eval on %s"%t_name)
            eval_model(t_name, tokenizer, model, args, device, task_main, results_dict['dev'][t_name], write2file=True, write_predictions=True)

    results_json_file = os.path.join(args.output_dir, "results.json")
    with open(results_json_file, 'w') as fw:
        json.dump(results_dict, fw)




if __name__ == "__main__":
    main()
