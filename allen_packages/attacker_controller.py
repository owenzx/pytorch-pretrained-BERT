import numpy as np
import torch
from stanfordnlp.server import CoreNLPClient
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import SimpleSeq2Seq
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from .controller_model import Controller
from allennlp.models.model import Model


class AttackerController(torch.nn.Module):
    def __init__(self,
                 vocab: Vocabulary,
                 ways_aug,
                 num_aug_prob,
                 num_aug_magn,
                 controller_hid,
                 softmax_temperature,
                 num_mix,
                 input_aware: bool = False,
                 entropy_regularize: bool = True,
                 entropy_coeff: float = 1e-4):
        super(AttackerController, self).__init__()
        # init client
        self.entropy_regularize = entropy_regularize
        self.entropy_coeff = entropy_coeff
        CUSTOM_PROPS = {'tokenize.whitespace':True}
        self.client = CoreNLPClient(annotators=['tokenize', 'ssplit', 'coref'], timeout=6000000, memory='16G', be_quiet=True, properties=CUSTOM_PROPS)

        self.input_aware = input_aware


        # init controller

        self.controller = Controller(ways_aug, num_aug_prob, num_aug_magn, controller_hid, softmax_temperature, num_mix)



    def sample_policy(self, instance):
        #self.training = False
        if self.input_aware:
            controller_output = self.controller.forward(instance[0]['text'], instance[0]['selected_mentions'], sample=True, get_prediction=True)
            controller_output_2 = self.controller.decode(controller_output)
        else:
            policy, log_probs, entropies = self.controller.sample(with_details=True)
        #self.controller.train()
        return policy, log_probs, entropies

    def sample_baseline_policy(self, instance):
        """Samples the greedy policy as baseline"""
        if self.input_aware:
            controller_output = self.controller.forward(instance[0]['text'], instance[0]['selected_mentions'], get_prediction=True)
            controller_output_2 = self.controller.decode(controller_output)
        else:
            policy, log_probs, entropies = self.controller.sample(with_details=True, greedy=True)

        return policy, log_probs, entropies


    def train_controller_w_reward(self,  log_probs, reward, entropies):
        controller_loss = -log_probs * reward
        if self.entropy_regularize:
            controller_loss -= self.entropy_coeff * entropies
        #TODO try sum and mean
        controller_loss = controller_loss.sum()
        return controller_loss


    #def get_new_mention_text(self, decoder_outdict, whole_sentence, np_head):
    def policy_to_dict(self, policy):
        """no use now"""
        return policy

    def apply_policy_dict(self, tokenized_text, switch_clusters, policy_dict):
        #raise NotImplementedError
        return None


    def get_complexity_penalty(self, decoder_outdict):
        cfgs = decoder_outdict['predicted_tokens'][0]
        complexity = 0
        for cfg in cfgs:
            if cfg == 'STOP':
                return complexity
            complexity -= 1
        return complexity

