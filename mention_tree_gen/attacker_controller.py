import numpy as np
import torch
from .lexicon_filler import LexiconFiller
from .seq2seq_tree_generator import Controller
from stanfordnlp.server import CoreNLPClient
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import SimpleSeq2Seq
from .PGseq2seq import PGSeq2Seq as seq2seq
from .PGseq2seq_wparse import PGSeq2Seq_wparse as seq2seq_parse
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder


class AttackerController(object):
    def __init__(self, args, vocab: Vocabulary):
        # init client
        CUSTOM_PROPS = {'tokenize.whitespace':True}
        self.client = CoreNLPClient(annotators=['tokenize', 'ssplit', 'coref'], timeout=6000000, memory='16G', be_quiet=True, properties=CUSTOM_PROPS)


        #Tree cfg vocab
        #TODO add delete/switch rules
        #TODO check if tree can be formed using these rules
        #self.valid_cfgs = ['STOP', 'NP->N', 'N->AP,N', 'N->N,PP', 'AP->AP,AP', 'AP->A', 'A->Adv,A', 'PP->PP,PP', 'PP->P,NP']
        self.valid_cfgs = ['STOP', 'NP->N', 'NP->AP,N', 'N->N,PP', 'NP->AP,N,PP', 'AP->AP,AP', 'AP->A', 'AP->Adv,A', 'PP->PP,PP', 'PP->P,NP']
        self.valid_cfgs = ['@start@', '@end@', 'STOP', 'NP->N', 'NP->AP,N', 'N->N,PP', 'NP->AP,N,PP', 'AP->AP,AP', 'AP->A', 'AP->Adv,A', 'PP->PP,PP', 'PP->P,NP']

        # init lexicon filler

        vocab._extend(non_padded_namespaces=['cfgs'], tokens_to_add={'cfgs':self.valid_cfgs})
        #vocab._extend(tokens_to_add={'cfgs':self.valid_cfgs})
        self.lexicon_filler = LexiconFiller(args, vocab)



        # init controller

        embeddings = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=args.embedding_dim)
        embedder = BasicTextFieldEmbedder({'tokens': embeddings})
        #TODO load pre-trained GloVe Embeddings

        encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=args.embedding_dim+1,
                                hidden_size=args.encoder_dim,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True
                                ))


        self.controller = seq2seq(vocab=vocab,
                                  source_embedder=embedder,
                                  encoder=encoder,
                                  max_decoding_steps=args.max_decoding_steps,
                                  beam_size=args.beam_size,
                                  target_namespace='cfgs',
                                  target_embedding_dim=args.cfg_dim,
                                  use_bleu=False,
                                  action_mask_getter=self.lexicon_filler.get_available_action
                                  )



    def sample_policy(self, instance):
        #self.training = False
        controller_output = self.controller.forward(instance[0]['text'], instance[0]['selected_mentions'], sample=True, get_prediction=True)
        controller_output_2 = self.controller.decode(controller_output)
        #self.controller.train()
        return controller_output_2

    def sample_baseline_policy(self, instance):
        """Samples the greedy policy as baseline"""
        controller_output = self.controller.forward(instance[0]['text'], instance[0]['selected_mentions'], get_prediction=True)
        controller_output_2 = self.controller.decode(controller_output)
        return controller_output_2


    def train_controller_w_reward(self, instance, prediction, reward):
        #self.controller.eval()
        target_tokens = {'tokens': torch.cat( (torch.unsqueeze(prediction['predictions'][0], 0), torch.unsqueeze(torch.Tensor([self.controller._end_index]).long().cuda(), 0)), -1)}
        controller_loss = self.controller.forward(instance[0]['text'], instance[0]['selected_mentions'], target_tokens, loss_weights = reward, get_prediction=False)
        #self.controller.train()
        return controller_loss

    def get_loss_entropy(self, instance, prediction):
        target_tokens = {'tokens': torch.cat( (torch.unsqueeze(prediction['predictions'][0], 0), torch.unsqueeze(torch.Tensor([self.controller._end_index]).long().cuda(), 0)), -1)}
        controller_output = self.controller.forward(instance[0]['text'], instance[0]['selected_mentions'], target_tokens, get_prediction=False)
        return controller_output



    #def get_new_mention_text(self, decoder_outdict, whole_sentence, np_head):
    def policy_to_dict(self, decoder_outdict, metadata):
        whole_sentence = metadata['original_text']
        np_head = metadata['np_head']
        cfgs = decoder_outdict['predicted_tokens'][0]
        new_mention_text = self.lexicon_filler.fill_lexicon(cfgs, whole_sentence, np_head)
        return new_mention_text

    def apply_policy_dict(self, tokenized_text, switch_clusters, policy_dict):
        raise NotImplementedError


    def get_complexity_penalty(self, decoder_outdict):
        cfgs = decoder_outdict['predicted_tokens'][0]
        complexity = 0
        for cfg in cfgs:
            if cfg == 'STOP':
                return complexity
            complexity -= 1
        return complexity

