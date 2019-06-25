import torch
from utils import *
from data_processing import *
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss
import json
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import BertTokenizer, SimpleTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME

from model_configs import *
from simple_models import SimpleSequenceClassification

from tqdm import tqdm, trange


class Trainer_Main(object):
    def __init__(self, args, logger, device, num_labels, output_mode, label_list, task_name):
        self.args = args
        self.logger = logger
        self.device = device
        self.num_labels = num_labels
        self.output_mode = output_mode
        self.label_list = label_list
        self.task_name = task_name

        self.tokenizer, pretrain_embs = self._get_tokenizer()

        self.model = self._get_model(pretrain_embs=pretrain_embs)

        if self.args.fp16:
            self.model.half()
        self.model.to(self.device)
        if self.args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            self.model = DDP(self.model)
        elif args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

    def reset_model(self):
        self.tokenizer, pretrain_embs = self._get_tokenizer()
        self.model = self._get_model(pretrain_embs=pretrain_embs)
        self.model.to(self.device)

    def _get_tokenizer(self):
        if not self.args.use_nonbert:
            tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, do_lower_case=self.args.do_lower_case)
            pretrain_embs = None
        else:
            tokenizer = SimpleTokenizer(vocab_file=self.args.glove_path, do_lower_case=self.args.do_lower_case,
                                        topn=self.args.glove_topn)
            pretrain_embs = load_glove_embs(self.args.glove_path, tokenizer.vocab)
        return tokenizer, pretrain_embs




    def _get_model(self, pretrain_embs):
        cache_dir = self.args.cache_dir if self.args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                                 'distributed_{}'.format(
                                                                                     self.args.local_rank))
        if not self.args.use_nonbert:
            model = BertForSequenceClassification.from_pretrained(self.args.bert_model,
                                                                  cache_dir=cache_dir,
                                                                  num_labels=self.num_labels)
        else:
            simple_config = SimpleConfig(self.tokenizer.vocab, config_sglsen_1)
            model = SimpleSequenceClassification(simple_config, num_labels=self.num_labels, glove_embs=pretrain_embs,
                                                 freeze_emb=not self.args.train_glove_embs)

        print(type(model))

        return model

    def build_optimizer(self, train_examples):
        num_train_optimization_steps = int(
            len(
                train_examples) / self.args.train_batch_size / self.args.gradient_accumulation_steps) * self.args.num_train_epochs
        if self.args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if self.args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=self.args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if self.args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=self.args.loss_scale)

        else:
            if not self.args.normal_adam:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=self.args.learning_rate,
                                     warmup=self.args.warmup_proportion,
                                     t_total=num_train_optimization_steps)
            else:
                optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args.learning_rate)
        self.optimizer = optimizer
        self.warmup_linear = WarmupLinearSchedule(warmup=self.args.warmup_proportion,
                                                  t_total=num_train_optimization_steps)

    def train(self, train_examples):
        label_list = self.label_list
        global_step = 0
        train_features = convert_examples_to_features(
            train_examples, label_list, self.args.max_seq_length, self.tokenizer, self.output_mode, self.logger)
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(train_examples))
        self.logger.info("  Batch size = %d", self.args.train_batch_size)
        # self.logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if self.output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif self.output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if self.args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.train_batch_size)

        self.model.train()


        for _ in trange(int(self.args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = self.model(input_ids, segment_ids, input_mask, labels=None)

                if self.output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                elif self.output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                if self.args.fp16:
                    self.optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = self.args.learning_rate * self.warmup_linear.get_lr(global_step,
                                                                                      self.args.warmup_proportion)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    if self.args.debug:
                        break
                # if global_step % self.args.eval_step == 0:
                #     for t_name in test_task_names:
                #         if t_name not in results_dict['dev'].keys():
                #             results_dict['dev'][t_name] = {}
                #         self.logger.info("Eval on %s" % t_name)
                #         #eval_model(t_name, self.tokenizer, self.model, self.args, self.device, task_main,
                        #           results_dict['dev'][t_name], write2file=False)
                    #self.model.train()

    def eval(self, eval_examples):
        eval_features = convert_examples_to_features(
            eval_examples, self.label_list, self.args.max_seq_length, self.tokenizer, self.output_mode, self.logger)
        self.logger.info("***** Running evaluation *****")
        self.logger.info("  Num examples = %d", len(eval_examples))
        self.logger.info("  Batch size = %d", self.args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        if self.output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        elif self.output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        self.model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask, labels=None)

            # create eval loss and other metric required by the task
            if self.output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
            elif self.output_mode == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)
            if self.args.debug:
                break

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        if self.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.output_mode == "regression":
            preds = np.squeeze(preds)

        result = compute_metrics(self.task_name, preds, all_label_ids.numpy(), self.task_name)
        #result = {"acc":0.5}
        # loss = tr_loss/global_step if args.do_train else None

        result['eval_loss'] = eval_loss
        # result['global_step'] = global_step
        # result['loss'] = loss

        #output_eval_file = os.path.join(self.args.output_dir, "eval_results_on_%s.txt" % task_name)

        #for key in sorted(result.keys()):
        #    if key not in results_dict.keys():
        #        results_dict[key] = []
        #    results_dict[key].append(result[key])

        #if write2file:
        #    with open(output_eval_file, "w") as writer:
        #        self.logger.info("***** Eval results *****")
        #        for key in sorted(result.keys()):
        #            self.logger.info("  %s = %s", key, str(result[key]))
        #            writer.write("%s = %s\n" % (key, str(result[key])))
        #else:
        #    self.logger.info("***** Eval results *****")
        #    for key in sorted(result.keys()):
        #        self.logger.info("  %s = %s", key, str(result[key]))
        return result

    def save_model(self, path, config_path=None):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), path)
        if not self.args.use_nonbert:
            model_to_save.config.to_json_file(config_path)
            self.tokenizer.save_vocabulary(self.args.output_dir)

    def load_model(self, path):
        # You have to change the name of the weight file before
        if not self.args.use_nonbert:
            self.tokenizer = BertTokenizer.from_pretrained(self.args.output_dir)
            self.model = BertForSequenceClassification(self.args.output_dir, num_labels=self.num_labels)
        else:
            self.model.load_state_dict(torch.load(path))
