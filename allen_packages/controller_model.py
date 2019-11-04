"""A module with NAS controller-related code."""
import collections
import os

import torch
import torch.nn.functional as F
import json

import utils
from collections import defaultdict
import numpy as np

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret




class Controller(torch.nn.Module):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py

    TODO(brendan): RL controllers do not necessarily have much to do with
    language models.

    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    def __init__(self, ways_aug, num_aug_prob, num_aug_magn, controller_hid, softmax_temperature, num_mix):
        super(Controller, self).__init__()

        # NOTE(brendan): `num_tokens` here is just the activation function
        # for every even step,
        self.num_tokens = []
        self.ways_aug = ways_aug
        self.num_aug_prob = num_aug_prob
        self.num_aug_magn = num_aug_magn
        self.controller_hid = controller_hid
        self.softmax_temperature = softmax_temperature
        self.num_mix = num_mix

        self.aug_probs = np.linspace(0,1,self.num_aug_prob)
        self.aug_magns = np.linspace(0,1, self.num_aug_magn)

        for idx in range(self.num_mix):
            self.num_tokens += [len(self.ways_aug), self.num_aug_prob, self.num_aug_magn]
        self.func_names = self.ways_aug

        num_total_tokens = sum(self.num_tokens)

        self.encoder = torch.nn.Embedding(num_total_tokens,
                                          controller_hid)
        self.lstm = torch.nn.LSTMCell(controller_hid, controller_hid)

        # TODO(brendan): Perhaps these weights in the decoder should be
        # shared? At least for the activation functions, which all have the
        # same size.
        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = torch.nn.Linear(controller_hid, size)
            self.decoders.append(decoder)

        self._decoders = torch.nn.ModuleList(self.decoders)

        #self.reset_parameters()
        self.static_init_hidden = keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return torch.zeros(key, self.controller_hid).cuda()

        self.static_inputs = keydefaultdict(_get_default_hidden)

    def _form_policy(self, aug_methods, aug_probs, aug_mgns, func_names):
        assert(len(aug_methods) == len(aug_probs) == len(aug_mgns))
        policy = []
        subp_num = len(aug_methods)
        step_num = len(aug_methods[0])
        for i in range(subp_num):
            policy.append([])
            for j in range(step_num):
                # Convert Aug_probs
                aug_prob = aug_probs[i][j].cpu().numpy()
                aug_prob = self.aug_probs[aug_prob]
                #if aug_prob == 0:
                #    aug_prob = 0
                #else:
                #    aug_prob = 2**(float(aug_prob)-3)

                #aug_mgn = aug_mgns[i][j].cpu().numpy() + 1
                aug_mgn = self.aug_magns[aug_mgns[i][j].cpu().numpy()]

                policy[i].append((func_names[aug_methods[i][j].data.cpu().numpy()], aug_prob, aug_mgn))
        return policy

    def forward_step(self,  # pylint:disable=arguments-differ
                inputs,
                hidden,
                block_idx,
                is_embed):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[block_idx](hx)

        logits /= self.softmax_temperature

        # exploration
        #if self.args.mode == 'train':
        #    logits = (self.args.tanh_c*F.tanh(logits))

        return logits, (hx, cx)


    def sample(self, batch_size=1, with_details=False, save_dir=None, greedy=False):
        """Samples a set of `args.num_mix` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        # [B, L, H]
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        aug_methods = []
        aug_probs = []
        aug_magns = []
        entropies = []
        log_probs = []
        aug_types = []
        # NOTE(brendan): The RNN controller alternately outputs an activation,
        # followed by a previous node, for each block except the last one,
        # which only gets an activation function. The last node is the output
        # node, and its previous node is the average of all leaf nodes.

        for block_idx in range(3*self.num_mix):
            logits, hidden = self.forward_step(inputs,
                                          hidden,
                                          block_idx,
                                          is_embed=(block_idx == 0))

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            # TODO(brendan): .mean() for entropy?
            entropy = -(log_prob * probs).sum(1, keepdim=False)

            if greedy:
                _, action = torch.max(probs, 1)
                action = action.unsqueeze(-1)
            else:
                action = probs.multinomial(num_samples=1).data

            selected_log_prob = log_prob.gather(
                1, action)

            action = action[:, 0]

            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            # 0: function, 1: previous node
            mode = block_idx % 3
            inputs = action + sum(self.num_tokens[:mode])

            if mode == 0:
                aug_methods.append(action)
            elif mode == 1:
                aug_probs.append(action)
            elif mode == 2:
                aug_magns.append(action)


        aug_methods= torch.stack(aug_methods).transpose(0, 1)
        aug_probs = torch.stack(aug_probs).transpose(0, 1)
        aug_magns = torch.stack(aug_magns).transpose(0, 1)


        aug_policy = self._form_policy(aug_methods, aug_probs, aug_magns, self.func_names)

        if save_dir is not None:
            with open(os.path.join(save_dir, 'aug_policy.json'), 'w') as fw:
                json.dump(aug_policy, fw)
            #for idx, dag in enumerate(dags):
            #    utils.draw_network(dag,
            #                       os.path.join(save_dir, f'graph{idx}.png'))

        if with_details:
            return aug_policy, torch.cat(log_probs), torch.cat(entropies)

        return aug_policy

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return zeros.cuda(), zeros.clone().cuda()