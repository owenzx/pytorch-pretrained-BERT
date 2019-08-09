# coding=utf-8
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
"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from my_bert.file_utils import my_bert_CACHE, WEIGHTS_NAME, CONFIG_NAME
from my_bert.modeling import BertForQuestionAnswering, BertConfig, BertForCoreference
from my_bert.optimization import BertAdam, WarmupLinearSchedule
from my_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import matthews_corrcoef, f1_score
from data_processing import CorefExample

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

class CorefInputFeatures(object):

    def __init__(self,
                 unique_id,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 example_index,
                 tokens,
                 tok_to_orig_index,
                 span1_l_position,
                 span1_r_position,
                 span2_l_position,
                 span2_r_position
                 ):
        self.unique_id = unique_id
        self.input_ids = input_ids
        self.tokens = tokens
        self.tok_to_orig_index = tok_to_orig_index
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.example_index = example_index
        self.span1_l_position = span1_l_position
        self.span1_r_position = span1_r_position
        self.span2_l_position = span2_l_position
        self.span2_r_position = span2_r_position

def read_ontonotes_examples(input_file):
    #read from jiant processed input json file
    with open(input_file, "r", encoding='utf-8')  as reader:
        lines = reader.readlines()

    examples = []
    for line in lines:
        raw_example = json.loads(line)
        #filter out all the examples without spans

        if len(raw_example["targets"]) < 1:
            continue

        for target_id, target in enumerate(raw_example["targets"]):
            guid = raw_example["info"]["document_id"] + "_" + str(raw_example["info"]["sentence_id"]) + "_" + str(target_id)

            text_a = raw_example["text"]

            text_tokens = text_a.split(' ')

            span1_l, span1_r, span2_l, span2_r = target["span1"][0], target["span1"][1], target["span2"][0], target["span2"][1]

            label = target["label"]


            example = CorefExample(guid, text_a, text_tokens, span1_l, span1_r, span2_l, span2_r, label)
            examples.append(example)

    return examples


def write_predictions(out_path, pred_dicts):
    with open(out_path, 'w') as fw:
        for p_dict in pred_dicts:
            if type(p_dict) is not dict:
                p_dict = p_dict._asdict()
            fw.write(json.dumps(p_dict))
            fw.write('\n')


def convert_examples_to_features(examples, tokenizer, max_seq_length):

    label_map = {"0":0, "1":1}

    features = []

    for (example_index, example) in enumerate(examples):
        #text_tokens = tokenizer.tokenize(example.text_a)

        unique_id = example.guid
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_tokens = []
        for (i, token) in enumerate(example.text_tokens):
            orig_to_tok_index.append(len(all_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_tokens.append(sub_token)
        #TODO check boundary
        span1_l_position = orig_to_tok_index[example.span1_l]
        span1_r_position = orig_to_tok_index[example.span1_r-1]+1
        span2_l_position = orig_to_tok_index[example.span2_l]
        span2_r_position = orig_to_tok_index[example.span2_r-1]+1


        tokens = ["[CLS]"] + all_tokens + ["[SEP]"]
        # Offset because of CLS
        span1_l_position += 1
        span1_r_position += 1
        span2_l_position += 1
        span2_r_position += 1

        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert(len(input_ids) == max_seq_length)
        assert(len(input_mask) == max_seq_length)
        assert(len(segment_ids) == max_seq_length)

        label_id = label_map[example.label]

        features.append(
            CorefInputFeatures(unique_id = unique_id,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          example_index=example_index,
                          tokens = all_tokens,
                          tok_to_orig_index = tok_to_orig_index,
                          span1_l_position=span1_l_position,
                          span1_r_position=span1_r_position,
                          span2_l_position=span2_l_position,
                          span2_r_position=span2_r_position)
        )

    return features


#RawResult = collections.namedtuple("RawResult",
#                                   ["unique_id", "start_logits", "end_logits"])

RawResult = collections.namedtuple("RawResult", ["unique_id", "pred", "is_correct"])

def eval_model(model, num_labels, eval_dataloader, eval_features, device, n_gpu, args, write_predicstions=False):
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    all_results = []
    preds = []
    logger.info("Start evaluating")
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0])):
        if n_gpu == 1:
            batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, example_indices, label_ids, span1_l_positions, span1_r_positions, span2_l_positions, span2_r_positions = batch
    #for input_ids, input_mask, segment_ids, example_indices, label_ids, span1_l_positions, span1_r_positions, span2_l_positions, span2_r_positions in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        #input_ids = input_ids.to(device)
        #input_mask = input_mask.to(device)
        #segment_ids = segment_ids.to(device)
        #label_ids = label_ids.to(device)
        with torch.no_grad():
            _, logits = model(input_ids, segment_ids, input_mask, label_ids, span1_l_positions, span1_r_positions, span2_l_positions, span2_r_positions)
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)


        for i, example_index in enumerate(example_indices):
            logits_i = logits[i].detach().cpu()
            pred_i = int(np.argmax(logits_i))
            #logits_i = logits_i.tolist()
            #pred_i = int(preds[i])
            label_i = int(label_ids[i].detach().cpu())
            is_correct = (pred_i == label_i)

            eval_feature = eval_features[example_index.item()]
            unique_id = eval_feature.unique_id
            all_results.append(RawResult(unique_id=unique_id,
                                         pred=pred_i,
                                         is_correct=is_correct))

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)
    #print(preds.shape)
    #print(all_label_ids.numpy().shape)
    #exit()
    all_labels = all_label_ids.numpy()
    result = {"acc": simple_accuracy(preds, all_labels)}
    result['eval_loss'] = eval_loss
    #
    #
    # for i, example_index in enumerate(example_indices):
    #     #logits_i = logits[i].detach().cpu()
    #     #pred_i = np.argmax(logits_i)
    #     #logits_i = logits_i.tolist()
    #     pred_i = int(preds[i])
    #     label_i = int(all_labels[i])
    #     is_correct = (pred_i == label_i)
    #
    #     eval_feature = eval_features[example_index.item()]
    #     unique_id = eval_feature.unique_id
    #     all_results.append(RawResult(unique_id=unique_id,
    #                                  pred=pred_i,
    #                                  is_correct=is_correct))

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

        if write_predicstions:
            output_prediction_file = os.path.join(args.output_dir, "predictions.json")
            write_predictions(output_prediction_file, all_results)
    model.train()
    return result



def save_model(model, tokenizer, save_dir):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(save_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(save_dir)




def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    parser.add_argument('--freeze_bert', default=False, action='store_true', help='If true, freeze bert when training')
    parser.add_argument('--save_predictions', default=False, action='store_true', help='If true, save prediction on valid data')

    parser.add_argument('--save_iter', default=-1, type=int, help='if not -1, then save model every save_iter iters')


    #parser.add_argument('--lr', default=0.0001, type=float)

    args = parser.parse_args()
    print(args)

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

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    num_labels = 2
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = read_ontonotes_examples(input_file=args.train_file)
        # train_examples = read_squad_examples(
        #     input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = BertForCoreference.from_pretrained(args.bert_model, cache_dir=os.path.join(str(my_bert_CACHE), 'distributed_{}'.format(args.local_rank)))
    # model = BertForQuestionAnswering.from_pretrained(args.bert_model,
    #             cache_dir=os.path.join(str(my_bert_CACHE), 'distributed_{}'.format(args.local_rank)))

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

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

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
            if args.freeze_bert:
                optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr = args.learning_rate)
            else:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=args.learning_rate,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)

    global_step = 0

    # Prepare train & eval data
    if args.do_train:
        cached_train_features_file = args.train_file+'_{0}_{1}_{2}_{3}'.format(
                list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))
        train_features = None
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length)
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)
        logger.info("***** Training Data *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_span1_l_positions = torch.tensor([f.span1_l_position for f in train_features], dtype=torch.long)
        all_span1_r_positions = torch.tensor([f.span1_r_position for f in train_features], dtype=torch.long)
        all_span2_l_positions = torch.tensor([f.span2_l_position for f in train_features], dtype=torch.long)
        all_span2_r_positions = torch.tensor([f.span2_r_position for f in train_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index, all_label_ids,
                                   all_span1_l_positions, all_span1_r_positions, all_span2_l_positions, all_span2_r_positions)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    #Preparing evaluation data
    eval_examples = read_ontonotes_examples(
        input_file=args.predict_file)
    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)

    logger.info("***** Evaluation Data *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.predict_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_span1_l_positions = torch.tensor([f.span1_l_position for f in eval_features], dtype=torch.long)
    all_span1_r_positions = torch.tensor([f.span1_r_position for f in eval_features], dtype=torch.long)
    all_span2_l_positions = torch.tensor([f.span2_l_position for f in eval_features], dtype=torch.long)
    all_span2_r_positions = torch.tensor([f.span2_r_position for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    # print(all_input_ids.shape)
    # print(all_input_mask.shape)
    # print(all_segment_ids.shape)
    # print(all_label_ids.shape)
    # print(all_span1_l_positions.shape)
    # print(all_span1_r_positions.shape)
    # print(all_span2_l_positions.shape)
    # print(all_span2_r_positions.shape)
    # print(all_example_index.shape)
    # exit()
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index, all_label_ids, all_span1_l_positions, all_span1_r_positions, all_span2_l_positions, all_span2_r_positions)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)



    if args.do_train:
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, example_index,  label_ids, span1_l_positions, span1_r_positions, span2_l_positions, span2_r_positions = batch
                if args.freeze_bert:
                    loss, _ = model(input_ids, segment_ids, input_mask, label_ids, span1_l_positions, span1_r_positions, span2_l_positions, span2_r_positions, detach_bert=True)
                else:
                    loss, _ = model(input_ids, segment_ids, input_mask, label_ids, span1_l_positions, span1_r_positions, span2_l_positions, span2_r_positions, detach_bert=False)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used and handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if args.save_iter != -1  and (global_step % args.save_iter == 0):
                    if args.output_dir[-1] == '/':
                        save_dir = args.output_dir[:-1] + '_int_%d'%global_step
                    else:
                        save_dir = args.output_dir + '_int_%d'%global_step
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_model(model, tokenizer, save_dir)

                    val_results = eval_model(model, num_labels=num_labels, device=device, eval_dataloader=eval_dataloader, args=args, eval_features=eval_features, n_gpu=n_gpu, write_predicstions=False)

                    logger.info('**********Val results at step %d: '%step + str(val_results['acc']))




    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForCoreference.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    else:
        model = BertForCoreference.from_pretrained(args.bert_model)

    model.to(device)


    # Start final evaluation
    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        #TODO replace read_squad_examples
       eval_model(model, num_labels=num_labels, device=device, eval_dataloader=eval_dataloader, args=args, eval_features=eval_features, n_gpu=n_gpu, write_predicstions=args.save_predictions)



if __name__ == "__main__":
    main()
