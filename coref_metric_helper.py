import numpy as np
import json
import sys
import os

from conll import reader
from conll import util
from conll import mention

from pprint import pprint
from tqdm import tqdm, trange





def convert_pred_to_conll_format(pred_file, key_file, out_file):
    with open(pred_file, 'r') as fpred, open(key_file, 'r') as fkey:
        pred_lines = fpred.readlines()
        key_lines = fkey.readlines()


    doc_lines = []
    # get document lines for key lines
    for k_line in key_lines:
        if k_line[:6] == "#begin":
            doc_lines.append([])
        doc_lines[-1].append(k_line)

    #assertion
    # for d in doc_lines:
    #     try:
    #         assert(d[-1][:4] == '#end')
    #     except:
    #         print(d)
    #         print(doc_lines)
    #         exit()
    # try:
    #     assert(len(doc_lines) == len(pred_lines))
    # except:
    #     print(len(doc_lines))
    #     print(len(pred_lines))
    #     exit()


    new_lines = []

    for i, p_line in enumerate(pred_lines):
        preds = json.loads(p_line)
        predicted_antecedents = preds["predicted_antecedents"]
        clusters = preds["clusters"]
        if "reverse_map" in preds.keys():
            reverse_map = preds["reverse_map"]
        else:
            reverse_map = {str(x): x for x in range(99999)}

        wordid2label = {}
        for c_idx, cluster in enumerate(clusters):
            for span in cluster:
                assert(span[0] == span[1])
                if str(span[0]) not in reverse_map:
                    continue
                wordid = reverse_map[str(span[0])]
                if wordid in wordid2label:
                    wordid2label[wordid].append(c_idx)
                else:
                    wordid2label[wordid] = [c_idx]

        document = doc_lines[i]

        word_count = 0
        for k_line in document:
            if k_line[0] == '#' or len(k_line) < 5:
                # start, end and empty lines
                new_lines.append(k_line)
                continue
            columns = k_line.strip().split()
            if word_count not in wordid2label:
                new_label = '-'
            else:
                new_label = '|'.join(['(%d)'%c_id for c_id in wordid2label[word_count]])
            new_columns = columns[:-1] + [new_label]
            new_line = '   '.join(new_columns) + '\n'
            word_count += 1
            new_lines.append(new_line)

    with open(out_file, 'w') as fw:
        for line in new_lines:
            fw.write(line)


def write_lines_back_to_file(doc_lines, path):
    with open(path, 'w') as fw:
        for k, v in doc_lines.items():
            fw.write('#begin document '+ k)
            for sentence in v:
                for token_line in sentence:
                    fw.write(token_line)
                    if token_line[-1]!='\n':
                        fw.write('\n')
                if sentence:
                    fw.write('\n')
            fw.write('#end document\n')



def set_new_cluster_annotation_woldspan(doc_lines, key_mention_sys_cluster):
    # get cluster annotation dict cluster_id_dict
    all_keys = list(key_mention_sys_cluster.keys())
    all_mentions = [[(m, c_id) for (m, c_id) in key_mention_sys_cluster[k].items()] for k in all_keys]
    all_mentions = [(m, c_id) for sublist in all_mentions for (m, c_id) in sublist]
    cluster_id_dict = {}
    for m, c_id in all_mentions:
        words = m.words
        min_spans = m.min_spans
        last_min_span = sorted(list(min_spans), key=lambda x: x[1])[-1]
        m_token_identifier = m.doc_name.strip() + '_senidx%d' % m.sent_num + '_tokenidx%d' % last_min_span[1]
        if m_token_identifier not in cluster_id_dict:
            cluster_id_dict[m_token_identifier] = '(%d, %d, %d)' % (c_id, m.start, m.end)
        else:
            cluster_id_dict[m_token_identifier] += '|(%d, %d, %d)' % (c_id, m.start, m.end)

    new_doc_lines = {}
    for k, v in doc_lines.items():
        new_doc_lines[k] = []
        for sen_idx, sentence in enumerate(v):
            new_doc_lines[k].append([])
            for token_idx, token_line in enumerate(sentence):
                split_columns = token_line.split()
                # last column is coref cluster annotation
                token_identifier = k.strip() + '_senidx%d' % sen_idx + '_tokenidx%d' % token_idx
                if token_identifier in cluster_id_dict:
                    split_columns[-1] = cluster_id_dict[token_identifier]
                else:
                    split_columns[-1] = '-'
                # print(len(token_line.split()))
                new_columns = '   '.join(split_columns)
                new_doc_lines[k][-1].append(new_columns)
    # print(new_doc_lines)
    return new_doc_lines


def set_new_cluster_annotation(doc_lines, key_mention_sys_cluster):
    # get cluster annotation dict cluster_id_dict
    all_keys = list(key_mention_sys_cluster.keys())
    all_mentions = [[(m, c_id) for (m, c_id) in key_mention_sys_cluster[k].items()] for k in all_keys]
    all_mentions = [(m, c_id) for sublist in all_mentions for (m, c_id) in sublist]
    cluster_id_dict = {}
    for m, c_id in all_mentions:
        words = m.words
        min_spans = m.min_spans
        last_min_span = sorted(list(min_spans), key=lambda x: x[1])[-1]
        m_token_identifier = m.doc_name.strip() + '_senidx%d' % m.sent_num + '_tokenidx%d' % last_min_span[1]
        if m_token_identifier not in cluster_id_dict:
            cluster_id_dict[m_token_identifier] = '(%d)' % c_id
        else:
            cluster_id_dict[m_token_identifier] += '|(%d)' % c_id

    new_doc_lines = {}
    for k, v in doc_lines.items():
        new_doc_lines[k] = []
        for sen_idx, sentence in enumerate(v):
            new_doc_lines[k].append([])
            for token_idx, token_line in enumerate(sentence):
                split_columns = token_line.split()
                # last column is coref cluster annotation
                token_identifier = k.strip() + '_senidx%d' % sen_idx + '_tokenidx%d' % token_idx
                if token_identifier in cluster_id_dict:
                    split_columns[-1] = cluster_id_dict[token_identifier]
                else:
                    split_columns[-1] = '-'
                # print(len(token_line.split()))
                new_columns = '   '.join(split_columns)
                new_doc_lines[k][-1].append(new_columns)
    # print(new_doc_lines)
    return new_doc_lines



def set_collins_new_cluster_annotation(doc_lines, key_mention_sys_cluster):
    # get cluster annotation dict cluster_id_dict
    all_keys = list(key_mention_sys_cluster.keys())
    all_mentions = [[(m, c_id) for (m, c_id) in key_mention_sys_cluster[k].items()] for k in all_keys]
    all_mentions = [(m, c_id) for sublist in all_mentions for (m, c_id) in sublist]
    cluster_id_dict = {}
    for m, c_id in all_mentions:
        head = m.head
        if head is None:
            m.set_head()
            head = m.head
        assert(head is not None)
        m_token_identifier = m.doc_name.strip() + '_senidx%d' % m.sent_num + '_tokenidx%d' % head[1]
        if m_token_identifier not in cluster_id_dict:
            cluster_id_dict[m_token_identifier] = '(%d)' % c_id
        else:
            cluster_id_dict[m_token_identifier] += '|(%d)' % c_id

    new_doc_lines = {}
    for k, v in doc_lines.items():
        new_doc_lines[k] = []
        for sen_idx, sentence in enumerate(v):
            new_doc_lines[k].append([])
            for token_idx, token_line in enumerate(sentence):
                split_columns = token_line.split()
                # last column is coref cluster annotation
                token_identifier = k.strip() + '_senidx%d' % sen_idx + '_tokenidx%d' % token_idx
                if token_identifier in cluster_id_dict:
                    split_columns[-1] = cluster_id_dict[token_identifier]
                else:
                    split_columns[-1] = '-'
                # print(len(token_line.split()))
                new_columns = '   '.join(split_columns)
                new_doc_lines[k][-1].append(new_columns)
    # print(new_doc_lines)
    return new_doc_lines


def set_pred_cluster_annotation(doc_lines, pred_examples, span_loc_dict):
    # get cluster annotation dict cluster_id_dict
    cluster_id_dict = {}

    for e_id, example in enumerate(pred_examples):
        pred_clusters = example['clusters']
        for c_id, cluster in enumerate(pred_clusters):
            for span in cluster:
                l, r = span
                # first do l
                doc_name = span_loc_dict[e_id]['doc_name']
                sen_idx = span_loc_dict[e_id][l]["sen_idx"]
                token_idx = span_loc_dict[e_id][l]["token_idx"]
                m_token_identifier = doc_name.strip() + '_senidx%d' % sen_idx + '_tokenidx%d' % token_idx
                if m_token_identifier not in cluster_id_dict:
                    cluster_id_dict[m_token_identifier] = '(%d' % c_id
                else:
                    cluster_id_dict[m_token_identifier] += '|(%d' % c_id

                # then do r
                doc_name = span_loc_dict[e_id]['doc_name']
                sen_idx = span_loc_dict[e_id][r]["sen_idx"]
                token_idx = span_loc_dict[e_id][r]["token_idx"]
                m_token_identifier = doc_name.strip() + '_senidx%d' % sen_idx + '_tokenidx%d' % token_idx
                if l == r:
                    cluster_id_dict[m_token_identifier] += ')'
                else:
                    if m_token_identifier not in cluster_id_dict:
                        cluster_id_dict[m_token_identifier] = '%d)' % c_id
                    else:
                        cluster_id_dict[m_token_identifier] += '|%d)' % c_id

    #Reuse this part
    new_doc_lines = {}
    for k, v in doc_lines.items():
        new_doc_lines[k] = []
        for sen_idx, sentence in enumerate(v):
            new_doc_lines[k].append([])
            for token_idx, token_line in enumerate(sentence):
                split_columns = token_line.split()
                # last column is coref cluster annotation
                token_identifier = k.strip() + '_senidx%d' % sen_idx + '_tokenidx%d' % token_idx
                if token_identifier in cluster_id_dict:
                    split_columns[-1] = cluster_id_dict[token_identifier]
                else:
                    split_columns[-1] = '-'
                # print(len(token_line.split()))
                new_columns = '   '.join(split_columns)
                new_doc_lines[k][-1].append(new_columns)
    # print(new_doc_lines)
    return new_doc_lines


def read_pred_file(pred_file):
    examples = []
    with open(pred_file, 'r') as fr:
        lines = fr.readlines()
    for line in lines:
        example = json.loads(line)
        examples.append(example)
    return examples


def check_token_match(pred_examples, doc_lines):
    i = 0
    span_loc_dict = {}
    for k, v in doc_lines.items():
        span_loc_dict[i]= {'doc_name': k}
        pred_example = pred_examples[i]

        token_id = 0
        for sen_id, sentence in enumerate(v):
            for sen_token_id, token_line in enumerate(sentence):
                real_token = token_line.strip().split()[3]
                pred_token = pred_example["document"][token_id]
                try:
                    assert(real_token == pred_token or real_token == ('/' + pred_token))
                except:
                    print(i)
                    print(k)
                    print(token_id)
                    print(real_token)
                    print(pred_token)
                    exit()
                span_loc_dict[i][token_id] = {"sen_idx": sen_id, "token_idx": sen_token_id}
                token_id += 1
        i += 1

    return span_loc_dict


def get_min_span_file(file_path, NP_only=False, remove_nested=False, keep_singletons=True, min_span=True):
    """input max span file in conll format, output min span file in conll format"""
    key_file = sys_file = file_path
    doc_coref_infos = reader.get_coref_infos(key_file, sys_file, NP_only, remove_nested, keep_singletons, min_span)
    key_mention_sys_cluster = {k: v[2] for k, v in doc_coref_infos.items()}
    doc_lines = reader.get_doc_lines(key_file)
    minspan_doc_lines = set_new_cluster_annotation(doc_lines, key_mention_sys_cluster)
    write_lines_back_to_file(minspan_doc_lines, file_path + '.min_span')



def get_collins_min_span_file(file_path, NP_only=False, remove_nested=False, keep_singletons=True, min_span=True):
    """input max span file in conll format, output min span file in conll format"""
    key_file = sys_file = file_path
    doc_coref_infos = reader.get_coref_infos(key_file, sys_file, NP_only, remove_nested, keep_singletons, min_span)
    key_mention_sys_cluster = {k: v[2] for k, v in doc_coref_infos.items()}
    doc_lines = reader.get_doc_lines(key_file)
    minspan_doc_lines = set_collins_new_cluster_annotation(doc_lines, key_mention_sys_cluster)
    write_lines_back_to_file(minspan_doc_lines, file_path + '.collins.min_span')


def get_min_span_head_mapping(file_path, NP_only=False, remove_nested=False, keep_singletons=True, min_span=True):
    """input max span file in conll format, output min span file in conll format"""
    key_file = sys_file = file_path
    doc_coref_infos = reader.get_coref_infos(key_file, sys_file, NP_only, remove_nested, keep_singletons, min_span)
    key_mention_sys_cluster = {k: v[2] for k, v in doc_coref_infos.items()}
    doc_lines = reader.get_doc_lines(key_file)
    minspan_doc_lines = set_new_cluster_annotation_woldspan(doc_lines, key_mention_sys_cluster)
    write_lines_back_to_file(minspan_doc_lines, file_path + '.hs_map')


def map_pred_to_conll_file(pred_file, conll_file):
    doc_lines = reader.get_doc_lines(conll_file)

    pred_examples = read_pred_file(pred_file)

    # check if all the tokens match
    span_loc_dict = check_token_match(pred_examples, doc_lines)

    pred_doc_lines = set_pred_cluster_annotation(doc_lines, pred_examples, span_loc_dict)
    write_lines_back_to_file(pred_doc_lines, './pred_on_dev.conll_span')


### Code for min2max conversion


def get_node_index_range(node):
    if hasattr(node, 'l_range'):
        assert(hasattr(node, 'r_range'))
        return node.l_range, node.r_range


    l = 99999
    r = -99999
    if node.isTerminal:
        r = node.index
        # print(node.tag)
        l = node.index - len(node.tag.split(' ')) + 1
        # l = r = node.index
        # print(node.tag+str(node.index))
    else:
        for c in node.children:
            l = min(l, get_node_index_range(c)[0])
            r = max(r, get_node_index_range(c)[1])
    node.l_range = l
    node.r_range = r
    return l, r


def find_the_end(node):
    if hasattr(node, 'end'):
        return node.end

    pos = node.index
    for c in node.children:
        pos = max(pos, find_the_end(c))
    node.end =  pos
    return pos


def get_terminals_single(node):
    if hasattr(node, 'single_token_terminals'):
        return node.single_token_terminals


    terminals = []
    node.get_terminals(terminals)

    # make sure all separated by single space
    single_tokens = ' '.join(terminals).split(' ')
    node.single_token_terminals = single_tokens
    return single_tokens


def verify_is_min_span(node, min_mention, key_doc_lines):
    """verify is the word is the min span of the node"""
    assert(min_mention.start == min_mention.end)

    guess_start = node.index
    guess_end = find_the_end(node)
    guess_words = get_terminals_single(node)

    #Make this node a mention
    max_mention = mention.Mention(doc_name=min_mention.doc_name, sent_num=min_mention.sent_num, start= guess_start, end= guess_end, words=guess_words)

    # fix start and end
    check_mention_attributes(key_doc_lines, max_mention)

    tree = reader.extract_annotated_parse(key_doc_lines[max_mention.sent_num][max_mention.start:max_mention.end+1], max_mention.start)
    max_mention.set_gold_parse(tree)
    max_mention.set_min_span()
    last_min_span = sorted(list(max_mention.min_spans), key=lambda x:x[1])[-1]
    #head = last_min_span[0].split(' ')[-1]
    head_loc = last_min_span[1]
    return (head_loc == min_mention.start)









def get_max_span_for_mention(mention, sentence_tree, key_doc_lines):
    # print(sentence_tree)
    assert (mention.start == mention.end)
    m_index = mention.start
    target_word = mention.words
    # print(target_word)
    # print(sentence_tree)
    # Locate child
    all_parents = []
    p = root = sentence_tree
    path = []
    children_stack = []
    # print(p.index)
    # print(p.tag)
    # print(mention.words)
    if not (any(c.isalpha() for c in mention.words[0]) or \
            any(c.isdigit() for c in mention.words[0])):
        mention.max_span = True
        return
    while len(p.children) > 0:
        # print("WHILE111")
        all_parents.append(p)
        childrens = p.children
        for c in childrens:
            l, r = get_node_index_range(c)
            if (l <= m_index) and (r >= m_index):
                # print("%d\t%d"%(l,r))
                p = c
                break
        if p != c:
            # Invalid node here?
            mention.max_span = True
            return
    all_parents.append(p)
    final_tag = all_parents[-2].tag

    # get valid node
    if final_tag[0:2] in ["NP", "NM", "QP", "NX"]:
        valid_tags = ["NP", "NM", "QP", "NX"]
    elif final_tag[0:2] in ["VP"]:
        valid_tags = ["VP"]
    else:
        valid_tags = ["NP", "NM", "QP", "NX"]

    # print([p.tag for p in all_parents])
    # find the maximum canidate by traversing back the link
    max_parent = all_parents[-2]
    for node in all_parents[:-2][::-1]:
        if node.tag[0:2] in valid_tags:
            if verify_is_min_span(node, mention, key_doc_lines):
                max_parent = node
            else:
                break
        else:
            break
    # print(max_parent.tag)
    # print(get_terminals_single(max_parent))

    #     print(max_parent.index)
    #     print(get_terminals(max_parent))
    #     print(max_parent.index + len(get_terminals(max_parent)) - 1)

    # Change attributes accordingly
    mention.start = max_parent.index
    #mention.end = max_parent.index + len(get_terminals_single(max_parent)) - 1
    mention.end = find_the_end(max_parent)
    mention.words = get_terminals_single(max_parent)
    mention.max_span = True


def check_mention_attributes(doc_lines, mention):
    add_length = 0
    sent = doc_lines[mention.sent_num]
    real_end_idx = mention.end
    real_start_idx = mention.end
    covered_start_idx = len(mention.words) - 1
    matched = True
    while covered_start_idx >= 0 and real_start_idx >= 0:
        #print("WHILE222")
        token = sent[real_start_idx].split()[3]
        if token.isalpha() or token.isdigit():
            if token == mention.words[covered_start_idx]:
                covered_start_idx -= 1
            else:
                break
        else:
            if token == mention.words[covered_start_idx]:
                covered_start_idx -= 1
        if covered_start_idx < 0:
            break
        real_start_idx -= 1
    selected_lines = sent[real_start_idx:mention.end+1]
    all_tokens = [s.split()[3] for s in selected_lines]
    if covered_start_idx >= 0:
        print("START DEBUG")
        print(token)
        print(real_start_idx)
        print(covered_start_idx)
        full_token = [s.split()[3] for s in sent]
        print(full_token)
        print(real_start_idx)
        print(all_tokens)
        print(mention.words)
        print(mention.start)
        print(mention.end)
        exit()
    else:
        mention.start = real_start_idx
        mention.words = all_tokens

def set_annotated_parse_trees_and_max_span(clusters, key_doc_lines, NP_only,
                                           partial_vp_chain_pruning=True, print_debug=False):
    pruned_cluster_indices = set()
    pruned_clusters = {}

    for i, c in enumerate(clusters):
        pruned_cluster = list(c)
        for m in c:
            # Modify m.start, m.end and m.words based on the annotated tree
            try:
                whole_sentence_tree = reader.extract_annotated_parse(key_doc_lines[m.sent_num], 0)
            except IndexError as err:
                print(err, len(key_doc_lines), m.sent_num)
            # pprint(key_doc_lines[m.sent_num])
            # print("ORIGIN")
            # print(m.words)
            # print("1...")
            if not (hasattr(m, 'max_span')) or m.max_span == False:
                get_max_span_for_mention(m, whole_sentence_tree, key_doc_lines)
                # print("BEFORE FIX")
                # print(m.words)
                # Re-align mention, to include all the comma, period, etc.
                # print("2...")
                check_mention_attributes(key_doc_lines, m)

            ##If the conll file does not have words
            # print(m.words)
            if not m.words[0]:
                terminals = []
                m.gold_parse.get_terminals(terminals)
                m.words = []
                for t in terminals:
                    for w in t.split():
                        m.words.append(w)

        #             m.set_max_span()

        pruned_clusters[i] = pruned_cluster

    return [pruned_clusters[k] for k in pruned_clusters]



def get_max_doc_coref_infos(min_doc_lines, NP_only=False, remove_nested=True, keep_singletons=False):
    doc_coref_infos = {}

    key_nested_coref_num = 0
    key_removed_nested_clusters = 0
    key_singletons_num = 0
    for doc in tqdm(min_doc_lines):
        try:
            key_clusters, singletons_num = reader.get_doc_mentions(
                    doc, min_doc_lines[doc], keep_singletons)
            key_singletons_num += singletons_num
            key_clusters = set_annotated_parse_trees_and_max_span(key_clusters,
                    min_doc_lines[doc],
                    NP_only)
        except:
            print("DEBUG")
            for sent in min_doc_lines[doc]:
                for line in sent:
                    pprint(line)
                    pprint(reader.extract_coref_annotation(line))
            raise

        if remove_nested:
            nested_mentions, removed_clusters = reader.remove_nested_coref_mentions(
                    key_clusters, keep_singletons)
            key_nested_coref_num += nested_mentions
            key_removed_nested_clusters += removed_clusters



        #key_mention_sys_cluster = get_mention2cluster_dict(key_clusters)

        doc_coref_infos[doc] = (key_clusters, )

    if remove_nested:
        print('Number of removed nested coreferring mentions in the annotation: %s' % (
                key_nested_coref_num))
        print('Number of resulting singleton clusters in the annotation: %s' % (
                key_removed_nested_clusters))

    if not keep_singletons:
        print('%d singletons are removed from the file' % (
                key_singletons_num))

    return doc_coref_infos


def set_new_maxspan_annotation(doc_lines, key_mention_cluster):
    # get cluster annotation dict cluster_id_dict
    all_keys = list(key_mention_cluster.keys())
    all_mentions = []
    for doc_k in all_keys:
        doc_mention_clusters = key_mention_cluster[doc_k]
        for c_id, cluster in enumerate(doc_mention_clusters):
            for m in cluster:
                all_mentions.append((m, c_id))
    # all_mentions = [[(m, c_id) for (m, c_id) in key_mention_cluster[k].items()] for k in all_keys]
    # all_mentions = [(m, c_id) for sublist in all_mentions for (m, c_id) in sublist]
    cluster_id_dict = {}
    for m, c_id in all_mentions:
        words = m.words
        min_spans = m.min_spans
        start = m.start
        end = m.end
        # do l and r saparately

        m_token_identifier = m.doc_name.strip() + '_senidx%d' % m.sent_num + '_tokenidx%d' % start
        if m_token_identifier not in cluster_id_dict:
            cluster_id_dict[m_token_identifier] = '(%d' % c_id
        else:
            cluster_id_dict[m_token_identifier] += '|(%d' % c_id

        m_token_identifier = m.doc_name.strip() + '_senidx%d' % m.sent_num + '_tokenidx%d' % end
        if start == end:
            cluster_id_dict[m_token_identifier] += ')'
        else:
            if m_token_identifier not in cluster_id_dict:
                cluster_id_dict[m_token_identifier] = '%d)' % c_id
            else:
                cluster_id_dict[m_token_identifier] += '|%d)' % c_id

    new_doc_lines = {}
    for k, v in doc_lines.items():
        new_doc_lines[k] = []
        for sen_idx, sentence in enumerate(v):
            new_doc_lines[k].append([])
            for token_idx, token_line in enumerate(sentence):
                split_columns = token_line.split()
                # last column is coref cluster annotation
                token_identifier = k.strip() + '_senidx%d' % sen_idx + '_tokenidx%d' % token_idx
                if token_identifier in cluster_id_dict:
                    split_columns[-1] = cluster_id_dict[token_identifier]
                else:
                    split_columns[-1] = '-'
                # print(len(token_line.split()))
                new_columns = '   '.join(split_columns)
                new_doc_lines[k][-1].append(new_columns)
    # print(new_doc_lines)
    return new_doc_lines



def get_max_span_file(input_file):
    """Input file is min span"""
    min_doc_lines = reader.get_doc_lines(input_file)
    doc_coref_infos = get_max_doc_coref_infos(min_doc_lines)
    key_mention_cluster = {k:v[0] for k, v in doc_coref_infos.items()}
    maxspan_doc_lines = set_new_maxspan_annotation(min_doc_lines, key_mention_cluster)
    write_lines_back_to_file(maxspan_doc_lines, input_file + '.max_span')




def find_root(p, mapper):
    if p not in mapper:
        return p
    while(mapper[p]!=p):
        p = mapper[p]
    return p



def get_upper_bound(input_file):
    """This function is used for predicting the upper bound for head-coref. It's used for evaluating the quality of head-finding/extraction. The metric is optimistic based sorely on the uniqueness of heads."""



    with open(input_file, 'r') as fr:
        gt_lines = fr.readlines()


    passage_lines = []
    ub_lines = []
    for line in tqdm(gt_lines):
        columns = line.strip().split()
        if len(columns)==0:
            passage_lines.append(line)
            continue
        elif columns[0] == '#begin':
            id_mapper = {}
            passage_lines.append(line)
            continue
        elif columns[0] == '#end':
            passage_lines.append(line)
            #compressing path in id_mapper
            for k in id_mapper.keys():
                p = id_mapper[k]
                if p==k or id_mapper[p] == p:
                    continue

                path = [k]
                while p!=id_mapper[p]:
                    path.append(p)
                    p = id_mapper[p]
                for u in path:
                    id_mapper[u] = p

            # start copying lines to ub_lines
            for p_line in passage_lines:
                columns = p_line.strip().split()
                if len(columns) == 0 or columns[0] == '#begin' or columns[0]=='#end':
                    ub_lines.append(p_line)
                    continue
                cluster_ids = columns[-1]
                if cluster_ids == '-':
                    ub_lines.append(p_line)
                    continue
                else:
                    id_list = cluster_ids.split('|')
                    for i in range(len(id_list)):
                        id_list[i] = int(id_list[i][1:-1])
                    uni_id = id_mapper[id_list[0]]
                    for id in id_list:
                        try:
                            assert(id_mapper[id] == uni_id)
                        except:
                            print(id)
                            print(uni_id)
                            print(id_mapper)
                            exit()
                    columns[-1] = '(%d)'%uni_id
                    new_line = '   '.join(columns)+ '\n'
                    ub_lines.append(new_line)
                    continue
            passage_lines = []
            continue
        else:
            cluster_ids = columns[-1]
            if cluster_ids == '-':
                passage_lines.append(line)
                continue
            else:
                id_list = cluster_ids.split('|')
                #remove brackets and convert to int
                for i in range(len(id_list)):
                    id_list[i] = int(id_list[i][1:-1])
                #map id
                uni_id = find_root(id_list[0], id_mapper)
                for i in range(len(id_list)):
                    #TODO find root, link the root instead of the leaf
                    p = find_root(id_list[i], id_mapper)
                    id_mapper[p] = uni_id
                passage_lines.append(line)

                continue





    ub_file = input_file + '.ub'
    with open(ub_file, 'w') as fw:
        for line in ub_lines:
            fw.write(line)


    print("Please run outer scripts to evaluate the conll metrics for input_file and the generated file.")


def check_max_len(file_path):
    doc_lines = reader.get_doc_lines(file_path)
    max_len = 0
    longest_k = ""
    for k,v in doc_lines.items():
        if sum([len(s) for s in v]) > max_len:
            max_len = sum([len(s) for s in v])
            longest_k = k


    long_doc_lines = {longest_k: doc_lines[longest_k]}
    write_lines_back_to_file(long_doc_lines,'/playpen/home/xzh/datasets/coref/allen/long.english.v4_gold_conll')



    print(max_len)



if __name__ == '__main__':
    #map_pred_to_conll_file(pred_file='./outputs/allen_test/pred_on_dev.out', conll_file='./dev.min_span')
    # get_min_span_file(file_path='./pred_on_dev.conll_span')
    # get_min_span_file(file_path='/playpen/home/xzh/datasets/coref/allen/train.english.v4_gold_conll')
    # get_min_span_file(file_path='/playpen/home/xzh/datasets/coref/allen/debug.english.v4_gold_conll')
    # get_min_span_file(file_path='/playpen/home/xzh/datasets/coref/allen/long.english.v4_gold_conll')
    # get_min_span_head_mapping(file_path='/playpen/home/xzh/datasets/coref/allen/train.out.parse.english.v4_gold_conll.short')
    # get_min_span_head_mapping(file_path='/playpen/home/xzh/datasets/coref/allen/dev.out.parse.english.v4_gold_conll.short')
    # get_min_span_head_mapping(file_path='/playpen/home/xzh/datasets/coref/allen/test.out.parse.english.v4_gold_conll.short')
    # get_min_span_file(file_path='/playpen/home/xzh/datasets/coref/allen/out.parse.english.v4_gold_conll')
    #get_max_span_file('train.min_span')
    #get_upper_bound('train.min_span')
    # get_collins_min_span_file(file_path='/playpen/home/xzh/datasets/coref/allen/debug.english.v4_gold_conll')
    # get_collins_min_span_file(file_path='/playpen/home/xzh/datasets/coref/allen/dev.english.v4_gold_conll')
    # get_collins_min_span_file(file_path='/playpen/home/xzh/datasets/coref/allen/test.english.v4_gold_conll')
    # get_collins_min_span_file(file_path='/playpen/home/xzh/datasets/coref/allen/train.english.v4_gold_conll')
    # print("DONE!")

    # check_max_len('/playpen/home/xzh/datasets/coref/allen/dev.english.v4_gold_conll')
    pass
