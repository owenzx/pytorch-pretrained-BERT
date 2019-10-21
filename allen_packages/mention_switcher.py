from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer
from stanfordnlp.server import CoreNLPClient
from utils import bert_simple_detokenize, wordpiece_tokenize_input, get_span_label_from_clusters, bert_detokenize_with_index_map
import pickle
from allennlp.data.fields import Field, ListField, TextField, SpanField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
import json
import numpy as np
import warnings
import csv
from coref_adv import overlapping_span, get_mention_features, get_mention_text_w_properties, switch_new_mentions, map_clusters, get_mention_text_w_properties_glove, get_sentence_vec
from utils import make_text_tensors
import torch
from gensim.models import KeyedVectors as Word2Vec

from mention_tree_gen.attacker_tree import AttackerTree
from mention_tree_gen.attacker_reader import AttackerCorefReader
from allennlp.data.vocabulary import Vocabulary
import os
from collections import namedtuple
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.common.util import lazy_groups_of
from allennlp.nn.util import move_to_device



class RNNMentionSwitcher(object):
    def __init__(self, bert_model_name, vocab, max_span_width, model_path=None, vocab_path=None):
        #TODO clean all the parameters here
        #assert(model_path is not None)
        #assert(vocab_path is not None)
        args = self.get_model_args()
        self.model_vocab = Vocabulary.from_files(args.vocab_path)
        self.model = AttackerTree(args, self.model_vocab)
        self.model.controller.cuda()
        model_file = 'epoch%d.bin'% args.load_epoch_num
        model_save_path = os.path.join(args.output_dir, model_file)
        self.model.controller.load_state_dict(torch.load(model_save_path))
        self.lowercase = 'uncased' in bert_model_name
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.max_pieces = 512
        self.max_span_width = max_span_width
        self.model_data_reader = AttackerCorefReader()
        self.data_iterator = BucketIterator(sorting_keys=[("text", "num_tokens")], padding_noise=0.0, batch_size=1)
        self.data_iterator.index_with(self.model_vocab)
        self.fake_token_indexer = {"tokens": SingleIdTokenIndexer()}
        self.vocab = vocab


    def get_model_args(self):
        model_args = namedtuple('model_args', 'glove_path glove_gensim_path vocab_path output_dir load_epoch_num beam_size max_decoding_steps embedding_dim encoder_dim cfg_dim dropout')
        switcher_model_args = model_args(glove_path="./datasets/glove.840B.300d.txt",
                                         glove_gensim_path="./datasets/glove.840B.300d.w2vformat.txt",
                                         vocab_path= "./outputs/mention_tree_selfcritic_1015_link/epoch9.vocab",
                                         output_dir="./outputs/mention_tree_selfcritic_1015_link",
                                         load_epoch_num=9,
                                         beam_size=None,
                                         max_decoding_steps=20,
                                         embedding_dim=300,
                                         encoder_dim=300,
                                         cfg_dim=50,
                                         dropout=0.0)
        return switcher_model_args

    def _fix_overlapping_span_mapping(self, span_mapping, max_len, new_max_len):
        # FIXME more clever mapping and length-mismatch treatment
        full_span_mapping = {}
        # for i, j in span_mapping.items():
        #    full_span_mapping[i] = j
        left_neighbor_map = (0, 0)
        right_neighbor_map = (max_len, new_max_len)
        left_neighbor_maps = []
        right_neighbor_maps = []
        for i in range(max_len + 1):
            if i in span_mapping.keys():
                left_neighbor_map = (i, span_mapping[i])
            left_neighbor_maps.append(left_neighbor_map)

        for i in range(max_len, -1, -1):
            if i in span_mapping.keys():
                right_neighbor_map = (i, span_mapping[i])
            right_neighbor_maps.insert(0, right_neighbor_map)

        for i in range(0, max_len + 1):
            if left_neighbor_maps[i][0] == i:
                full_span_mapping[i] = left_neighbor_maps[i][1]
            elif right_neighbor_maps[i][0] == i:
                full_span_mapping[i] = right_neighbor_maps[i][1]
            else:
                full_span_mapping[i] = int(np.interp(i, [left_neighbor_maps[i][0], right_neighbor_maps[i][0]],
                                                     [left_neighbor_maps[i][1], right_neighbor_maps[i][1]]))
        # for i in range(0, max_len):
        #    try:
        #        assert(full_span_mapping[i] <= full_span_mapping[i+1])
        #    except:
        #        print("ERROR `MAPPING")
        #        print(i)
        #        print(full_span_mapping)
        #        print(span_mapping)
        #        print(max_len)
        #        print(new_max_len)
        #        exit()

        return full_span_mapping

    def get_switch_mention_text_and_spans(self, output_dict, old_spans):
        """Given old text and old span, return new text and new span"""
        tokenized_text = output_dict['tokenized_text'][0]
        old_spans = old_spans[0]
        switch_clusters = output_dict['clusters'][0]
        mentions_to_switch = []




        #Randomly select one span
        target_spans = []
        for i_c, cluster in enumerate(switch_clusters):
            #TODO do it offline and correctly
            for i_s, span in enumerate(cluster):
                l, r = span

                #filter out pronouns
                if (r-l)<=1:
                    continue

                #also check non-overlapping
                non_overlapping = True
                for j_c, cluster2 in enumerate(switch_clusters):
                    for j_s, span2 in enumerate(cluster2):
                        if j_c == i_c and j_s == i_s:
                            continue
                        l2, r2 = span2
                        if l<= l2 <=r or l<=r2<=r:
                            non_overlapping = False
                            break
                    if not non_overlapping:
                        break
                if not non_overlapping:
                    continue

                target_spans.append(span)

        if len(switch_clusters) != 0:
            chosen_cluster = switch_clusters[np.random.choice(len(switch_clusters), 1)[0]]
            target_cluster = [chosen_cluster[np.random.choice(len(chosen_cluster), 1)[0]]]
            #target_cluster = [target_spans[np.random.choice(len(target_spans), 1)[0]]]

            #Detokenize text
            #TODO Delete first last bert token
            detokenized_text, idx_map = bert_detokenize_with_index_map(tokenized_text)
            converted_cluster = []
            for span in target_cluster:
                l, r = span
                c_l, c_r = idx_map[l], idx_map[r]
                converted_cluster.append((c_l, c_r))


            #Wrap into instance format
            #TODO fix here using wrong switch_clusters as gold_clusters placeholder
            instance = self.model_data_reader.text_to_instance(sentences = [detokenized_text], gold_clusters=switch_clusters, target_cluster=converted_cluster)
            dataloader = self.data_iterator([instance])
            dataloader = lazy_groups_of(dataloader, 1)
            example = next(dataloader)
            example = move_to_device(example, 0)
            new_mention_cfgs = self.model.sample_new_mention(example)
            new_mention_text = self.model.get_new_mention_text(new_mention_cfgs, example[0]['metadata'][0]['original_text'], example[0]['metadata'][0]['np_head'])
            if new_mention_text is not None:
                mentions_to_switch.append((target_cluster[0], new_mention_text))



        # Reusing codes, only needs to specify the mentions_to_switch
        new_tokenized_text, span_mapping = switch_new_mentions(tokenized_text, mentions_to_switch, self.bert_tokenizer, self.lowercase)
        #new_clusters = map_clusters(switch_clusters, span_mapping)

        full_span_mapping = self._fix_overlapping_span_mapping(span_mapping, len(tokenized_text), len(new_tokenized_text))


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        new_tokenized_text = new_tokenized_text[:self.max_pieces]
        new_text = make_text_tensors(new_tokenized_text, self.max_pieces, self.bert_tokenizer, self.fake_token_indexer)
        new_text.index(self.vocab)
        pad_len = new_text.get_padding_lengths()
        new_text = new_text.as_tensor(pad_len)
        new_text['tokens'] = new_text['tokens'].unsqueeze(0).to(device)

        #Fix empty span locations
        old_spans_np = old_spans.cpu().numpy()
        new_spans_np = np.zeros(old_spans_np.shape)
        for i in range(old_spans_np.shape[0]):
            new_spans_np[i][0] = full_span_mapping[old_spans_np[i][0]]
            new_spans_np[i][1] = max(full_span_mapping[old_spans_np[i][1]+1] - 1, full_span_mapping[old_spans_np[i][1]])
            new_spans_np[i][0] = min(new_spans_np[i][0], len(new_tokenized_text)-1)
            new_spans_np[i][1] = min(new_spans_np[i][1], len(new_tokenized_text)-1)
            if new_spans_np[i][1] - new_spans_np[i][0] >= self.max_span_width:
                overflow_size = (new_spans_np[i][1] - new_spans_np[i][0] - self.max_span_width)
                move_len_l = overflow_size // 2
                move_len_r = overflow_size - move_len_l
                new_spans_np[i][0] += move_len_l
                new_spans_np[i][1] -= (move_len_r+1)
            try:
                assert((new_spans_np[i][1]-new_spans_np[i][0])<self.max_span_width)
                assert((new_spans_np[i][1] >= new_spans_np[i][0]))
            except:
                print("ERROR!")
                print(new_spans_np[i])
                print(old_spans_np[i])
                print(full_span_mapping)
                exit()
        #print(np.max(new_spans_np))
        #new_spans_np[1][1] = new_spans_np[1][0] + (self.max_span_width - 1)

        new_spans = torch.LongTensor([new_spans_np]).to(device)

        #Back to new text and new span
        return new_text, new_spans

class MentionSwitcher(object):
    def __init__(self, bert_model_name, mention_dict_path, vocab, max_span_width):
        assert(type(bert_model_name) is str)
        self.lowercase = 'uncased' in bert_model_name
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        CUSTOM_PROPS = {'tokenize.whitespace':True}
        self.client = CoreNLPClient(annotators=['tokenize', 'ssplit', 'coref'], timeout=6000000, memory='16G', be_quiet=True, properties=CUSTOM_PROPS)
        # TODO pass paramter
        self.max_pieces = 512

        self.vocab = vocab

        with open(mention_dict_path, 'rb') as fr:
            self.mention_dict = pickle.load(fr)
        self.fake_token_indexer = {"tokens": SingleIdTokenIndexer()}

        self.max_span_width = max_span_width
        glove_gensim_path = './datasets/glove.840B.300d.w2vformat.txt'
        self.w2v_model = Word2Vec.load_word2vec_format(glove_gensim_path)
        #self.switch_type = 'glove_mention'


    def _fix_overlapping_span_mapping(self, span_mapping, max_len, new_max_len):
        #FIXME more clever mapping and length-mismatch treatment
        full_span_mapping = {}
        #for i, j in span_mapping.items():
        #    full_span_mapping[i] = j
        left_neighbor_map = (0, 0)
        right_neighbor_map = (max_len, new_max_len)
        left_neighbor_maps = []
        right_neighbor_maps = []
        for i in range(max_len+1):
            if i in span_mapping.keys():
                left_neighbor_map = (i, span_mapping[i])
            left_neighbor_maps.append(left_neighbor_map)

        for i in range(max_len, -1, -1):
            if i in span_mapping.keys():
                right_neighbor_map = (i, span_mapping[i])
            right_neighbor_maps.insert(0, right_neighbor_map)

        for i in range(0, max_len+1):
            if left_neighbor_maps[i][0] == i:
                full_span_mapping[i] = left_neighbor_maps[i][1]
            elif right_neighbor_maps[i][0] == i:
                full_span_mapping[i] = right_neighbor_maps[i][1]
            else:
                full_span_mapping[i] = int(np.interp(i, [left_neighbor_maps[i][0], right_neighbor_maps[i][0]], [left_neighbor_maps[i][1], right_neighbor_maps[i][1]]))
        #for i in range(0, max_len):
        #    try:
        #        assert(full_span_mapping[i] <= full_span_mapping[i+1])
        #    except:
        #        print("ERROR `MAPPING")
        #        print(i)
        #        print(full_span_mapping)
        #        print(span_mapping)
        #        print(max_len)
        #        print(new_max_len)
        #        exit()

        return full_span_mapping


    def get_switch_mention_text_and_spans(self, output_dict, old_spans, switch_type='glove_mention'):
        assert switch_type in ['simple', 'switch_pron', 'glove_mention']
        # if simple, only switching new mentions to non-pronouns, if switch_pron, also randomly change some pronoun (actually 50%)
        switch_clusters = output_dict['clusters'][0]
        #print(switch_clusters)
        #print(output_dict)
        #exit()

        tokenized_text = output_dict['tokenized_text'][0]
        if switch_type in ['glove_mention']:
            whitespace_tokens = bert_simple_detokenize(tokenized_text).split(' ')
            sentence_vec = get_sentence_vec(whitespace_tokens, self.w2v_model)

        old_spans = old_spans[0]
        #print(old_spans.shape)
        #exit()

        mentions_to_switch = []


        for idx_c1, c in enumerate(switch_clusters):
            mention_vec = None
            #filter out the mentions overlapped with other mentions
            valid_cluster = True

            #check internal overlapping
            for i, span in enumerate(c):
                for j, span2 in enumerate(c):
                    if i==j:
                        continue
                    if overlapping_span(span, span2):
                        valid_cluster = False
                        break
                if not valid_cluster:
                    break
            if not valid_cluster:
                continue

            #check cross-cluster overlapping
            for span in c:
                for idx_c2, c2 in enumerate(switch_clusters):
                    if idx_c1==idx_c2:
                        continue
                    for span2 in c2:
                        if overlapping_span(span, span2):
                            valid_cluster = False
                            break
                    if not valid_cluster:
                        break
                if not valid_cluster:
                    break
            if not valid_cluster:
                continue

            common_features = {'mentionType': {},
                               'animacy': {},
                               'number': {},
                               'person': {},
                               'nerString':{},
                               'gender':{}}
            switchable_mentions = []
            non_empty = False
            for span in c:
                try:
                    l, r =span
                except:
                    print(span)
                    print(c)
                    exit()
                span_text = tokenized_text[l:r+1]
                m_features = get_mention_features(span_text, self.client)
                if m_features['mentionType'] == 'PRONOMINAL':
                    if switch_type == 'switch_pron':
                        rand = np.random.randint(2)
                        if rand == 1:
                            switchable_mentions.append((l,r))
                        continue
                    else:
                        continue
                if switch_type in ['glove_mention']:
                    if mention_vec is None:
                        whitespace_tokens = bert_simple_detokenize(span_text).split(' ')
                        mention_vec = get_sentence_vec(whitespace_tokens, self.w2v_model)

                switchable_mentions.append((l,r))
                non_empty = True
                for k in m_features.keys():
                    if m_features[k] in common_features[k].keys():
                        common_features[k][m_features[k]] += 1
                    else:
                        common_features[k][m_features[k]] = 1
            if not non_empty:
                continue
            # set the feature value to the majority value
            common_features['mentionType']['pronominal'] = 0 # Ignore all the pronominal mention
            common_feature_values = {}
            for k in common_features.keys():
                common_feature_values[k] = max(common_features[k].items(), key=lambda x:x[1])[0]
            if switch_type in ['glove_mention'] and sentence_vec is not None:
                if mention_vec is None:
                    new_mention_text = get_mention_text_w_properties_glove(self.mention_dict, common_feature_values, self.w2v_model, sentence_vec)
                else:
                    new_mention_text = get_mention_text_w_properties_glove(self.mention_dict, common_feature_values, self.w2v_model, mention_vec)
            else:
                new_mention_text = get_mention_text_w_properties(self.mention_dict, common_feature_values)
            if new_mention_text is not None:
                for m in switchable_mentions:
                    mentions_to_switch.append((m, new_mention_text))
        new_tokenized_text, span_mapping = switch_new_mentions(tokenized_text, mentions_to_switch, self.bert_tokenizer, self.lowercase)
        #new_clusters = map_clusters(switch_clusters, span_mapping)

        full_span_mapping = self._fix_overlapping_span_mapping(span_mapping, len(tokenized_text), len(new_tokenized_text))


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        new_tokenized_text = new_tokenized_text[:self.max_pieces]
        new_text = make_text_tensors(new_tokenized_text, self.max_pieces, self.bert_tokenizer, self.fake_token_indexer)
        new_text.index(self.vocab)
        pad_len = new_text.get_padding_lengths()
        new_text = new_text.as_tensor(pad_len)
        new_text['tokens'] = new_text['tokens'].unsqueeze(0).to(device)

        #Fix empty span locations
        old_spans_np = old_spans.cpu().numpy()
        new_spans_np = np.zeros(old_spans_np.shape)
        for i in range(old_spans_np.shape[0]):
            new_spans_np[i][0] = full_span_mapping[old_spans_np[i][0]]
            new_spans_np[i][1] = max(full_span_mapping[old_spans_np[i][1]+1] - 1, full_span_mapping[old_spans_np[i][1]])
            new_spans_np[i][0] = min(new_spans_np[i][0], len(new_tokenized_text)-1)
            new_spans_np[i][1] = min(new_spans_np[i][1], len(new_tokenized_text)-1)
            if new_spans_np[i][1] - new_spans_np[i][0] >= self.max_span_width:
                overflow_size = (new_spans_np[i][1] - new_spans_np[i][0] - self.max_span_width)
                move_len_l = overflow_size // 2
                move_len_r = overflow_size - move_len_l
                new_spans_np[i][0] += move_len_l
                new_spans_np[i][1] -= (move_len_r+1)
            try:
                assert((new_spans_np[i][1]-new_spans_np[i][0])<self.max_span_width)
                assert((new_spans_np[i][1] >= new_spans_np[i][0]))
            except:
                print("ERROR!")
                print(new_spans_np[i])
                print(old_spans_np[i])
                print(full_span_mapping)
                exit()
        #print(np.max(new_spans_np))
        #new_spans_np[1][1] = new_spans_np[1][0] + (self.max_span_width - 1)

        new_spans = torch.LongTensor([new_spans_np]).to(device)


        return new_text, new_spans


