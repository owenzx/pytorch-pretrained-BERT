import sys
import os
import argparse
from conll import reader
from tqdm import tqdm, trange
from coref_metric_helper import write_lines_back_to_file
import numpy as np


original_path = '/playpen/home/xzh/datasets/WikiCoref/Evaluation/key-OntoNotesScheme'

new_path = '/playpen/home/xzh/datasets/coref/allen/out.parse.english.v4_gold_conll'
# new_path = '/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/out.min_span'


def convert2conll_simple():

    with open(original_path, 'r') as fr:
        lines = fr.readlines()

    new_lines  = []

    count = 0

    for line in lines:
        if line[:3]!='NaN':
            if line[0] == '#':
                count = 0
            if count < 600:
                new_lines.append(line)
            continue

        columns = line.strip().split()

        new_columns = columns[:4] + ['XX', '*', '-', '-', '-', 'Speaker#1', '*'] + columns[4:]
        # new_columns = columns[:4] + ['NN', '*', '-', '-', '-', 'Speaker#1', '*'] + ['-']
        count += 1

        if count < 600:
            new_lines.append('   '.join(new_columns) + '\n')


    with open(new_path, 'w') as fw:
        for line in new_lines:
            fw.write(line)

def format_new_sentence(old_sentence, parse_labs, pos_labs):
    new_sentence = []
    for i, old_line in enumerate(old_sentence):
        columns = old_line.strip().split()
        new_columns = columns[:4] + [pos_labs[i], parse_labs[i]] + ['-', '-', '-', 'Speaker#1', '*'] + [columns[-1]]
        new_line = '   '.join(new_columns)
        new_sentence.append(new_line)
    return new_sentence


def get_linearized_parse(a_sentence):
    parse_str = ""
    pos_labs = []

    def is_terminal_node(node):
        if len(node.child) == 1 and len(node.child[0].child) == 0:
            return True
        else:
            return False

    def preorder_traverse(node):
        nonlocal parse_str
        nonlocal pos_labs

        if is_terminal_node(node):
            parse_str += '*'
            pos_labs.append(node.value)
            return

        parse_str += '('
        if node.value == 'ROOT':
            parse_str += 'TOP'
        else:
            parse_str += node.value
        if len(node.child) > 0:
            if not all([is_terminal_node(c) for c in node.child]):
                #                 parse_str += '('
                for child in node.child:
                    preorder_traverse(child)
            #                 parse_str += ')'
            else:
                for child in node.child:
                    # parse_str += '[*%s]'%child.child[0].value
                    preorder_traverse(child)
        parse_str += ')'

    preorder_traverse(a_sentence.parseTree)
    return parse_str, pos_labs


def split_parse(parse_str):
    annotations = []
    cur_idx = 0
    while cur_idx < len(parse_str):
        annotations.append('')
        while (cur_idx < len(parse_str) and parse_str[cur_idx] != '*'):
            annotations[-1] += parse_str[cur_idx]
            cur_idx += 1
        annotations[-1] += parse_str[cur_idx]
        cur_idx += 1
        while (cur_idx < len(parse_str) and parse_str[cur_idx] == ')'):
            annotations[-1] += parse_str[cur_idx]
            cur_idx += 1
    return annotations



def convert2conll_wparse(original_path, new_path):
    import stanza
    from stanza.server import CoreNLPClient


    doc_lines = reader.get_doc_lines(original_path)
    print(type(doc_lines))
    print(list(doc_lines.keys()))

    #intialize corenlp server
    CUSTOM_RPOPS = {'tokenize.whitespace': True}
    with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'], timeout=30000, memory='16G', properties=CUSTOM_RPOPS) as client:

        new_doc_lines = {}
        num_docs = len(doc_lines.keys())
        # i = 0
        for k, sentences in tqdm(doc_lines.items(), total=num_docs):
            # i+=1
            # if i <13:
            #     continue
            new_doc_lines[k] = []

            for sentence in sentences:

                words = [columns.split()[3] for columns in sentence]
                ws_words = ' '.join(words)

                ann = client.annotate(ws_words)

                total_parse_labs = []
                total_pos_labs = []
                for sen_annotation in ann.sentence:
                    parse_str, pos_labs =  get_linearized_parse(sen_annotation)
                    parse_labs = split_parse(parse_str)

                    total_parse_labs.extend(parse_labs)
                    total_pos_labs.extend(pos_labs)

                try:
                    assert(len(total_parse_labs) == len(total_pos_labs) == len(sentence))
                except:
                    print(len(sentence))
                    print(ws_words)
                    print(len(total_parse_labs))
                    print(len(total_pos_labs))
                    exit()

                new_sentence = format_new_sentence(sentence, parse_labs, pos_labs)
                new_doc_lines[k].append(new_sentence)

    write_lines_back_to_file(new_doc_lines, new_path)




def split_dataset(full_data_path):
    full_doc_lines = reader.get_doc_lines(full_data_path)
    doc_keys = list(full_doc_lines.keys())
    np.random.seed(416)
    permuted_keys = np.random.permutation(doc_keys)

    train_keys = permuted_keys[:len(permuted_keys)//3]
    dev_keys = permuted_keys[len(permuted_keys)//3:len(permuted_keys)//3*2]
    test_keys = permuted_keys[len(permuted_keys)//3*2:]

    train_lines = {k:full_doc_lines[k] for k in train_keys}
    dev_lines = {k:full_doc_lines[k] for k in dev_keys}
    test_lines = {k:full_doc_lines[k] for k in test_keys}

    write_lines_back_to_file(train_lines, full_data_path+'.train')
    write_lines_back_to_file(dev_lines, full_data_path + '.dev')
    write_lines_back_to_file(test_lines, full_data_path + '.test')


def check_longest_document(data_path):
    doc_lines = reader.get_doc_lines(data_path)
    max_len = 0
    for k, doc in doc_lines.items():
        l = sum([len(s) for s in doc])

        if l > 4000:
            print(l)
            print(k)


def filter_long_sentences(data_path):
    doc_lines = reader.get_doc_lines(data_path)

    new_doc_lines = {}

    for k, doc in doc_lines.items():
        l = sum([len(s) for s in doc])
        if l < 4000:
            new_doc_lines[k] = doc
        else:
            num_part = l // 4000 + 1
            part_len =  l // num_part
            start_sen_idx = 0
            for part_num  in range(num_part):
                total_len = 0
                for end_sen_idx in range(start_sen_idx, len(doc) - 1):
                    total_len += len(doc[end_sen_idx])
                    if total_len >= part_len:
                        break
                new_doc_lines[k[:-1]+'%d\n'%part_num] = doc[start_sen_idx:end_sen_idx+1]
                start_sen_idx = end_sen_idx + 1
    write_lines_back_to_file(new_doc_lines, data_path+'.short')











if __name__ == '__main__':
    # convert2conll_wparse(original_path, new_path)
    # split_dataset('/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/out.min_span')
    # split_dataset('/playpen/home/xzh/datasets/coref/allen/out.parse.english.v4_gold_conll')
    # filter_long_sentences('/playpen/home/xzh/datasets/coref/allen/test.out.parse.english.v4_gold_conll')
    # check_longest_document('/playpen/home/xzh/datasets/coref/allen/test.out.parse.english.v4_gold_conll')
    check_longest_document('/playpen/home/xzh/datasets/coref/allen/test.out.parse.english.v4_gold_conll.short')
    # filter_long_sentences('/playpen/home/xzh/datasets/coref/allen/dev.out.parse.english.v4_gold_conll')
    # filter_long_sentences('/playpen/home/xzh/datasets/coref/allen/train.out.parse.english.v4_gold_conll')
    # filter_long_sentences('/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/train.out.min_span')
    # filter_long_sentences('/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/dev.out.min_span')
    # filter_long_sentences('/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/test.out.min_span')
