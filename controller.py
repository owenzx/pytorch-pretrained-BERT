"""A module with NAS controller-related code."""
import collections
import os

import torch
import torch.nn.functional as F
import json

import utils


def _form_policy(aug_methods, aug_probs, aug_mgns, func_names, args):
    assert(len(aug_methods) == len(aug_probs) == len(aug_mgns))
    policy = []
    subp_num = len(aug_methods)
    step_num = len(aug_methods[0])
    for i in range(subp_num):
        policy.append([])
        for j in range(step_num):
            # Convert Aug_probs
            aug_prob = aug_probs[i][j].data.cpu().numpy()
            if aug_prob == 0:
                aug_prob = 0
            else:
                aug_prob = 2**(float(aug_prob)-3)
            # Aug_mgns start from 1 so have to add 1
            aug_mgn = aug_mgns[i][j].data.cpu().numpy() + 1
            policy[i].append((func_names[aug_methods[i][j].data.cpu().numpy()], aug_prob, aug_mgn))
    if args.log_all_policy:
        with open(args.policy_log_path, 'a') as fw:
            fw.write(str(policy) + '\n')
    return policy


class Controller(torch.nn.Module):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py

    TODO(brendan): RL controllers do not necessarily have much to do with
    language models.

    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args

        # NOTE(brendan): `num_tokens` here is just the activation function
        # for every even step,
        self.num_tokens = []
        for idx in range(self.args.num_mix):
            self.num_tokens += [len(args.ways_aug), args.num_aug_prob, args.num_aug_mag]
        self.func_names = args.ways_aug

        num_total_tokens = sum(self.num_tokens)

        self.encoder = torch.nn.Embedding(num_total_tokens,
                                          args.controller_hid)
        self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)

        # TODO(brendan): Perhaps these weights in the decoder should be
        # shared? At least for the activation functions, which all have the
        # same size.
        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = torch.nn.Linear(args.controller_hid, size)
            self.decoders.append(decoder)

        self._decoders = torch.nn.ModuleList(self.decoders)

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, self.args.controller_hid),
                self.args.no_cuda,
                requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self,  # pylint:disable=arguments-differ
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

        logits /= self.args.softmax_temperature

        # exploration
        #if self.args.mode == 'train':
        #    logits = (self.args.tanh_c*F.tanh(logits))

        return logits, (hx, cx)

    def sample(self, batch_size=1, with_details=False, save_dir=None):
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
        # NOTE(brendan): The RNN controller alternately outputs an activation,
        # followed by a previous node, for each block except the last one,
        # which only gets an activation function. The last node is the output
        # node, and its previous node is the average of all leaf nodes.
        for block_idx in range(3*self.args.num_mix):
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          block_idx,
                                          is_embed=(block_idx == 0))

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            # TODO(brendan): .mean() for entropy?
            entropy = -(log_prob * probs).sum(1, keepdim=False)

            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(
                1, utils.get_variable(action, requires_grad=False))

            # TODO(brendan): why the [:, 0] here? Should it be .squeeze(), or
            # .view()? Same below with `action`.
            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            # 0: function, 1: previous node
            mode = block_idx % 3
            inputs = utils.get_variable(
                action[:, 0] + sum(self.num_tokens[:mode]),
                requires_grad=False)

            if mode == 0:
                aug_methods.append(action[:, 0])
            elif mode == 1:
                aug_probs.append(action[:, 0])
            elif mode == 2:
                aug_magns.append(action[:,0])


        aug_methods= torch.stack(aug_methods).transpose(0, 1)
        aug_probs = torch.stack(aug_probs).transpose(0, 1)
        aug_magns = torch.stack(aug_magns).transpose(0,1)


        aug_policy = _form_policy(aug_methods, aug_probs, aug_magns, self.func_names, self.args)

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
        zeros = torch.zeros(batch_size, self.args.controller_hid)
        return (utils.get_variable(zeros, self.args.no_cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.args.no_cuda, requires_grad=False))