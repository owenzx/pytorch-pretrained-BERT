import torch
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
from controller import Controller
import scipy
import utils


def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]

class Trainer_Controller(object):
    def __init__(self, args, logger, device, num_labels):
        self.args = args
        self.logger = logger
        self.device = device
        self.num_labels = num_labels

        self.controller = Controller(args)
        self.controller.to(device)

        self.baseline = None



    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=self.args.controller_lr)

    def calc_reward(self, val_results):
        if "acc" in val_results.keys():
            return (-1*self.args.reward_c)/val_results["acc"]



    def train(self, reward, log_probs, entropy):
        model = self.controller

        model.train()

        avg_reward_base = None
        adv_history = []
        entropy_history = []
        reward_history = []

        if 1 > self.args.discount > 0:
            reward = discount(reward, self.args.discount)

        np_entropy = entropy.data.cpu().numpy()
        reward_history.append(reward)
        entropy_history.append(np_entropy)

        if self.baseline is None:
            self.baseline = reward
        else:
            decay = self.args.ema_baseline_decay
            self.baseline = decay * self.baseline + (1-decay) * reward

        adv = reward - self.baseline
        adv_history.append(adv)

        loss = -log_probs*utils.get_variable([adv], self.args.no_cuda, requires_grad=False)

        if self.args.entropy_mode == 'regularizer':
            loss -= self.args.entropy_coeff * entropy

        loss = loss.sum()

        self.optimizer.zero_grad()
        loss.backward()

        if self.args.controller_grad_clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(),
                                          self.args.controller_grad_clip)

        self.optimizer.step()


    def save_controller(self, path):
        torch.save(self.controller.state_dict(), path)



    def load_controller(self, path):
        self.controller.load_state_dict(torch.load(path))


    def eval(self):
        pass

    def get_policy(self, with_details=False):
        return self.controller.sample(with_details=with_details)

