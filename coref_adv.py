import collections

from stanfordnlp.server import CoreNLPClient

import pickle
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer
from utils import bert_simple_detokenize, wordpiece_tokenize_input, get_span_label_from_clusters

from allennlp.data.fields import Field, ListField, TextField, SpanField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
import json
import numpy as np
import warnings
import csv


def extract_mentions_from_findmypast(tsv_path):
    #extract mentions from the FindMyPast entities list
    with open(tsv_path, 'r') as fr:
        reader = csv.reader(fr, delimiter='\t', quotechar=None)
        lines = []
        for line in reader:
            print(line)
            exit()




def extract_mentions_from_corpus(corpus):
    mentions = []
    unique_mentions = set()
    CUSTOM_PROPS = {'tokenize.whitespace': True}
    #FIXME
    #corpus = corpus[:10]
    with CoreNLPClient(annotators=['tokenize', 'ssplit', 'coref'], timeout=6000000, memory='16G', be_quiet=False, properties=CUSTOM_PROPS) as client:
        for sentence in tqdm(corpus):
            ann = client.annotate(' '.join(sentence))
            ann_mentions = ann.mentionsForCoref
            ann_sentences = ann.sentence
            sentences = [[tok.word for tok in s.token] for s in ann_sentences]

            for m in ann_mentions:
                if m.mentionType == 'PRONOMINAL':
                    continue
                text= sentences[m.sentNum][m.startIndex:m.endIndex]
                if ' '.join(text) not in unique_mentions:
                    unique_mentions.add(' '.join(text))
                    mentions.append({'text':text, 'mentionType': m.mentionType, 'number':m.number, 'animacy':m.animacy, 'person':m.person, 'nerString':m.nerString, 'gender':m.gender})

    return mentions


def get_mention_key(d):
    mention_key = ""
    for k in ['mentionType', 'number', 'animacy', 'person', 'nerString', 'gender']:
        assert(type(d[k]) is str)
        mention_key += (k + ":" + d[k] + "__")
    return mention_key




def organize_mention_list_to_dict(mentions):
    mention_dict = {}
    for m in mentions:
        new_k = get_mention_key(m)
        if new_k not in mention_dict:
            mention_dict[new_k] = [m['text']]
        else:
            mention_dict[new_k].append(m['text'])
    return mention_dict







def get_mention_text_w_properties(mentions, p_dict, choose_one=True):
    assert(type(mentions) is dict)
    if p_dict['mentionType'] in ['LIST', 'NONE']:
        return None
    query_key = get_mention_key(p_dict)

    if query_key not in mentions.keys():
        warnings.warn("!!! WARNING !!!: no matched mentions, not switching.")
        print(p_dict)
        return None
    results = mentions[query_key]
    #for m in mentions:
    #    valid_mention = True
    #    for k in p_dict.keys():
    #        if m[k] != p_dict[k]:
    #            valid_mention = False
    #            break
    #    if valid_mention:
    #        results.append(m)

    if choose_one:
        return results[np.random.choice(len(results), 1)[0]]

    else:
        return results



def get_seq_text_label(text, clusters):
    # we need tokenzied text here!
    text_labels = []
    for tok in text:
        text_labels.append({'token': tok, 'labels':[]})
    for i, c in enumerate(clusters):
        for span in c:
            l, r = span
            text_labels[l]['labels'].append('s%d'%i)
            text_labels[r]['labels'].append('e%d'%i)
    return text_labels


def get_mention_features(text, client):
    # Most likely there is only one span
    ann = client.annotate(' '.join(text))

    if len(ann.mentionsForCoref) == 0:
        return {'mentionType': 'NONE', 'number':'NONE', 'animacy':'NONE', 'person':'NONE', 'gender': 'NONE', 'nerString':'NONE'}
    m = ann.mentionsForCoref[0]
    return {'mentionType': m.mentionType, 'number':m.number, 'animacy':m.animacy, 'person':m.person, 'nerString':m.nerString, 'gender':m.gender}


def switch_new_mentions(text, switch_list, bert_tokenizer=None, lowercase=True):
    switch_l_dict = {x[0][0]:x for x in switch_list}
    idx_map = {}
    new_text = []
    i = 0
    while i<len(text):
        tok = text[i]
        idx_map[i] = len(new_text)
        if i not in switch_l_dict:
            new_text.append(tok)
            i += 1
        else:
            new_mention_text = switch_l_dict[i][1]
            if bert_tokenizer is not None:
                if lowercase:
                    new_mention = bert_tokenizer.wordpiece_tokenizer.tokenize(' '.join(new_mention_text).lower())
                else:
                    new_mention = bert_tokenizer.wordpiece_tokenizer.tokenize(' '.join(new_mention_text))
                #assert('[UNK]' not in new_mention)
            new_text.extend(new_mention)
            #idx_map[switch_l_dict[i][0][1]] = len(new_text) - 1
            idx_map[switch_l_dict[i][0][1] + 1] = len(new_text)
            i = switch_l_dict[i][0][1] + 1

    return new_text, idx_map


def map_clusters(old_clusters, idx_map):
    new_clusters = []
    for c in old_clusters:
        new_c = []
        for span in c:
            new_c.append((idx_map[span[0]], idx_map[span[1]+1]-1))
        new_clusters.append(new_c)
    return new_clusters




def overlapping_span(span, span2):
    l1, r1 = span
    l2, r2 = span2
    if l1 < l2:
        return not (r1 < l2)
    if l1 == l2:
        return True
    if l1 > l2:
        return not (r2 < l1)



def split_sentences(client, tokens):
    splitted = []
    ann = client.annotate(' '.join(tokens))
    sentences = ann.sentence
    for s in sentences:
        splitted.append([t.word for t in s.token])
    return splitted


def switch_mentions(examples, new_mention_dict, bert_tokenizer, lowercase=True, cluster_key="clusters"):
    #input and output are both dicts of data
    #This function switch mentions accoridng to one cluster key

    CUSTOM_PROPS = {'tokenize.whitespace':True}

    new_examples = []

    with CoreNLPClient(annotators=['tokenize', 'ssplit', 'coref'], timeout=6000000, memory='16G', be_quiet=True, properties=CUSTOM_PROPS) as client:
        for example in tqdm(examples):
            tokenized_text = example['tokenized_text']
            switch_clusters = example[cluster_key]
            # gold_clusters = example['gold_clusters']
            # predicted_clusters = example['clusters']

            mentions_to_switch = []

            for idx_c1, c in enumerate(switch_clusters):
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
                    l, r =span
                    span_text = tokenized_text[l:r+1]
                    m_features = get_mention_features(span_text, client)
                    if m_features['mentionType'] == 'PRONOMINAL':
                        continue
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
                new_mention_text = get_mention_text_w_properties(new_mention_dict, common_feature_values)
                if new_mention_text is not None:
                    for m in switchable_mentions:
                        mentions_to_switch.append((m, new_mention_text))
            new_tokenized_text, span_mapping = switch_new_mentions(tokenized_text, mentions_to_switch, bert_tokenizer, lowercase)
            new_document = bert_simple_detokenize(new_tokenized_text)
            new_clusters = map_clusters(switch_clusters, span_mapping)

            #ssplit the tokenized_text for later convenience
            unflattened_text = split_sentences(client, new_tokenized_text)

            new_example = {'document':new_document,
                           'tokenized_text': new_tokenized_text,
                           'unflattened_text': unflattened_text,
                           cluster_key: new_clusters}
            for k in example.keys():
                if k not in new_example.keys():
                    new_example[k] = example[k]
            new_examples.append(new_example)

    return new_examples


def convert_instances_to_dict(instances):
    # dict must have key: tokenized_text, gold_clusters, document
    dicts = []
    for ins in instances:
        metadata = ins['metadata']
        dicts.append({'tokenized_text'  : metadata['tokenized_text'],
                      'gold_clusters'   : convert_clusters_to_tuples(metadata['clusters']),
                      'document'        : metadata['original_text']})
    return dicts






def convert_dict_to_instances(dict_data, bert_tokenizer, golden_clusters=True, max_pieces=512):
    """convert dict (from json) to instances that can be used for allennlp training"""

    fake_token_indexer = {"tokens": SingleIdTokenIndexer()}

    instances = []
    CUSTOM_PROPS = {'tokenize.whitespace':True}
    with CoreNLPClient(annotators=['tokenize', 'ssplit'], timeout=6000000, memory='16G', be_quiet=True,
                       properties=CUSTOM_PROPS) as client:
        for d in dict_data:

            flatten_sentences = d['tokenized_text']
            if 'unflattened_sentences' not in d.keys():
                    unflattened_sentences = split_sentences(client, d['tokenized_text'])
            else:
                unflattened_sentences = d['unflattened_text']
            tmp_flatten = [word for s in unflattened_sentences for word in s]
            assert(tmp_flatten == flatten_sentences)

            if golden_clusters:
                clusters = convert_clusters_to_tuples(d['gold_clusters'])
            else:
                clusters = convert_clusters_to_tuples(d['clusters'])

            cluster_dict = {}
            for cluster_id, cluster in enumerate(clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

            text_field = TextField([Token(word, text_id=bert_tokenizer.vocab[word]) for word in flatten_sentences][:max_pieces], fake_token_indexer)
            spans, span_labels = get_span_label_from_clusters(unflattened_sentences, cluster_dict, text_field, max_span_width=25, max_pieces=max_pieces, flattened=flatten_sentences)
            span_field = ListField(spans)
            span_label_field = SequenceLabelField(span_labels, span_field)

            metadata = {'clusters':clusters,
                        'tokenized_text':d['tokenized_text'],
                        'original_text':d['document'],
                        'span': spans,
                        'span_labels': span_labels}
            #optional attributes
            if 'id' in d.keys():
                metadata['sen_id'] = d['id']

            metadata_field = MetadataField(metadata)

            fields: Dict[str, Field] = {'text': text_field,
                                        'spans': span_field,
                                        'span_labels': span_label_field,
                                        'metadata': metadata_field}

            instance = Instance(fields)
            instances.append(instance)
    return instances


def extract_corpus_from_json(json_path):
    sentences = []
    with open(json_path, 'r') as fr:
        lines = fr.readlines()

    for line in lines:
        example = json.loads(line)
        sentences.append(example['document'])

    return sentences


def convert_clusters_to_tuples(clusters):
    new_clusters = []
    for c in clusters:
        new_clusters.append([])
        for span in c:
            new_clusters[-1].append(tuple(span))
    return new_clusters


def get_debug_ist():
    predict_file = './tmp.out.2'
    #predict file is in the format of json
    with open(predict_file, 'r') as fr:
        lines = fr.readlines()

    examples = []
    for line in lines:
        example = json.loads(line)
        examples.append(example)

    bert_model_name = 'bert-base-uncased'
    #bert_model_name = './saved_bert'
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)


    instance_out_file = './debug_same.ist.2'
    new_instances = convert_dict_to_instances(examples, bert_tokenizer, False)

    with open(instance_out_file, 'wb') as fw:
        pickle.dump(new_instances, fw)

def main(load_mentions_path=None, lowercase=True):
    if load_mentions_path is not None:
        with open(load_mentions_path, 'rb') as fr:
            new_mention_dict = pickle.load(fr)
    else:
        corpus_file = './tmp.out.2'
        corpus = extract_corpus_from_json(corpus_file)
        new_mention_list = extract_mentions_from_corpus(corpus)
        new_mention_dict = organize_mention_list_to_dict(new_mention_list)
        mention_save_path = './cache/conll_dev_mentions.dict'
        with open(mention_save_path, 'wb') as fw:
            pickle.dump(new_mention_dict, fw)

    predict_file = './tmp.out.2'
    #predict file is in the format of json
    with open(predict_file, 'r') as fr:
        lines = fr.readlines()

    examples = []
    for line in lines:
        example = json.loads(line)
        examples.append(example)

    json_out_file = './switch.json.3'
    instance_out_file = './switch.ist.3'

    bert_model_name = 'bert-base-uncased'
    #bert_model_name = './saved_bert'
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    new_examples = switch_mentions(examples, new_mention_dict, bert_tokenizer, lowercase)

    with open(json_out_file, 'w') as fw:
        for example in new_examples:
            json.dump(example, fw)
            fw.write('\n')


    new_instances = convert_dict_to_instances(new_examples, bert_tokenizer, False)

    with open(instance_out_file, 'wb') as fw:
        pickle.dump(new_instances, fw)


def check_mask_difficulty(json_file_path):
    print("CHECKING MASK DIFFICULTY...")
    with open(json_file_path, 'r') as fr:
        lines = fr.readlines()

    examples = []
    for line in lines:
        example = json.loads(line)
        examples.append(example)

    ana_file_path = './mentions_mask.ana.txt'

    with open(ana_file_path, 'w') as fw:
        for exp_i, example in enumerate(tqdm(examples)):
            tokenized_text = example['tokenized_text']
            gold_clusters = example['gold_clusters']
            predicted_clusters = example['clusters']


            for idx_c1, c in enumerate(gold_clusters):
                masked_l_dict= {} # map l to r
                if len(c) <= 1:
                    continue
                for span in c:
                    l, r =span
                    masked_l_dict[l] = r
                masked_text = []
                i = 0
                while i<len(tokenized_text):
                    tok = tokenized_text[i]
                    if i not in masked_l_dict:
                        masked_text.append(tok)
                        i = i + 1
                    else:
                        masked_text.append('[MASK]')
                        i = masked_l_dict[i] + 1
                print("============================ PASSAGE {}, CLUSTER {} ===============================".format(exp_i, idx_c1), file=fw)
                print(masked_l_dict, file=fw)
                print(' '.join(tokenized_text)+ '\n', file=fw)
                print(' '.join(masked_text), file=fw)
                print('====================================================================================' + '\n\n\n', file=fw)



def dump_train_instances(max_span_width, bert_model_name, max_pieces, train_data_path, dump_path):
    from allen_packages.allen_reader import  MyConllCorefReader

    train_reader = MyConllCorefReader(max_span_width=max_span_width,
                                      bert_model_name=bert_model_name,
                                      max_pieces=max_pieces,
                                      save_instance_path=dump_path)
    train_reader.dump_instances(file_path=train_data_path)



def get_augmented_labeled_data(labeled_instance_path=None, load_mentions_path=None, result_json_path=None, result_path=None, self_augment=True):

    bert_model_name = 'bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    lowercase = True

    with open(labeled_instance_path, 'rb') as fr:
        labeled_instances = pickle.load(fr)

    labeled_examples = convert_instances_to_dict(labeled_instances)

    if load_mentions_path is not None:
        with open(load_mentions_path, 'rb') as fr:
            mention_dict = pickle.load(fr)
    else:
        mention_save_path = './cache/debug_conll_train.corpus'
        assert(self_augment is True)
        corpus = [exp['document'] for exp in labeled_examples]
        mention_list = extract_mentions_from_corpus(corpus)
        mention_dict = organize_mention_list_to_dict(mention_list)
        with open(mention_save_path, 'wb') as fw:
            pickle.dump(mention_dict, fw)

    print("STARTING SWITCHING MENTIONS...")
    augmented_examples = switch_mentions(labeled_examples, mention_dict, bert_tokenizer, lowercase, cluster_key="gold_clusters")

    total_examples = labeled_examples + augmented_examples

    with open(result_json_path, 'w') as fw:
        for example in total_examples:
            json.dump(example, fw)
            fw.write('\n')

    new_instances = convert_dict_to_instances(total_examples, bert_tokenizer, True)

    with open(result_path, 'wb') as fw:
        pickle.dump(new_instances, fw)


def tmp_fix():
    #ONLY USED FOR TEMP FIX!!!

    bert_model_name = 'bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    lowercase = True

    result_json_path = './cache/conll_train_aug_same.json'
    result_path = './cache/conll_train_aug_same_fix.ins'

    total_examples = []
    with open(result_json_path, 'r') as fr:
        lines = fr.readlines()
    for line in lines:
        example = json.loads(line)
        total_examples.append(example)

    new_instances = convert_dict_to_instances(total_examples, bert_tokenizer, True)

    with open(result_path, 'wb') as fw:
        pickle.dump(new_instances, fw)



def run_debug_func(labeled_instance_path=None, load_mentions_path=None, result_json_path=None, result_path=None, self_augment=True):
    bert_model_name = 'bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    lowercase = True

    with open(labeled_instance_path, 'rb') as fr:
        labeled_instances = pickle.load(fr)

    labeled_examples = convert_instances_to_dict(labeled_instances)

    if load_mentions_path is not None:
        with open(load_mentions_path, 'rb') as fr:
            mention_dict = pickle.load(fr)
    else:
        mention_save_path = './cache/debug_conll_train.corpus'
        assert(self_augment is True)
        corpus = [exp['document'] for exp in labeled_examples]
        mention_list = extract_mentions_from_corpus(corpus)
        mention_dict = organize_mention_list_to_dict(mention_list)
        with open(mention_save_path, 'wb') as fw:
            pickle.dump(mention_dict, fw)

    for k, mentions in mention_dict.items():
        for m in mentions:
            new_mention = bert_tokenizer.wordpiece_tokenizer.tokenize(' '.join(m).lower())
            try:
                assert('[UNK]' not in new_mention)
            except:
                print("UNK appeared!")
                print(' '.join(m).lower())
                print(' '.join(new_mention).lower())
                exit()



if __name__ == '__main__':
    #main(load_mentions_path='./debug.corpus')
    #main(load_mentions_path='./debug.corpus')
    #extract_mentions_from_findmypast(tsv_path='./entities.tsv')
    #get_debug_ist()
    #check_mask_difficulty(json_file_path='./tmp.out.2')
    #dump_train_instances(max_span_width=25, max_pieces=512, bert_model_name='bert-base-uncased', train_data_path='./datasets/coref/allen/train.english.v4_gold_conll', dump_path = './cache/conll_train.ins')
    #get_augmented_labeled_data(labeled_instance_path='./cache/conll_train.ins', load_mentions_path='./cache/debug_conll_train.corpus', result_json_path='./cache/conll_train_aug_same.json', result_path= './cache/conll_train_aug_same.ins', self_augment=True)
    tmp_fix()
    #run_debug_func(labeled_instance_path='./cache/conll_train.ins', load_mentions_path='./cache/debug_conll_train.corpus', result_json_path='./cache/conll_train_aug_same.json', result_path= './cache/conll_train_aug_same.ins', self_augment=True)
