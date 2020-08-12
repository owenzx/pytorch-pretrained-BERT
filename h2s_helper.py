import os
import sys
import json
from conll import reader, mention, util
from pprint import pprint
from coref_metric_helper import write_lines_back_to_file
import numpy as np
from coref_metric_helper import convert_pred_to_conll_format

# the total length will be 2 * MAX_CONTEXT + 1 (plus 1 because of head)
MAX_CONTEXT = 64


def make_h2s_example(text, local_head_id, global_head_id, passage_id, sentence_id, answer_span):
    example = {"text":text, "local_head_id": local_head_id, "global_head_id":global_head_id, "passage_id": passage_id, "sentence_id": sentence_id, "answer_span": answer_span}
    return example





def h2s2conll(h2s_pred_file, head_file, out_file):
    with open(h2s_pred_file, 'r') as fspred:
        spred_lines = fspred.readlines()
    key_docs = reader.get_doc_lines(head_file)
    print([list(key_docs.keys())[0]])


    cluster_id_dict = {}
    for line in spred_lines:
        spred_dict = json.loads(line)
        passage_id = spred_dict["passage_ids"]
        assert(passage_id in key_docs)

        sentence_id = spred_dict["sentence_ids"]
        local_head_id = spred_dict["local_head_ids"]
        global_head_id = spred_dict["global_head_ids"]
        local_span_id = spred_dict["best_span"]
        local_span_l, local_span_r = local_span_id
        offset = global_head_id -  local_head_id
        global_span_l, global_span_r = local_span_l + offset, local_span_r + offset

        str_c_ids = key_docs[passage_id][sentence_id][global_head_id].strip().split()[-1].split('|')
        c_ids = [int(s[1:-1]) for s in str_c_ids]

        for c_id in c_ids:
            #do l
            m_token_identifier = passage_id.strip() + '_senidx%d' % sentence_id + '_tokenidx%d' % global_span_l
            if m_token_identifier not in cluster_id_dict:
                cluster_id_dict[m_token_identifier] = '(%d' % c_id
            else:
                cluster_id_dict[m_token_identifier] += '|(%d' % c_id

            #do r
            m_token_identifier = passage_id.strip() + '_senidx%d' % sentence_id + '_tokenidx%d' % global_span_r
            if m_token_identifier not in cluster_id_dict:
                cluster_id_dict[m_token_identifier] = '%d)' % c_id
            else:
                cluster_id_dict[m_token_identifier] += '|%d)' % c_id

    #Reuse this part
    new_doc_lines = {}
    for k, v in key_docs.items():
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



    write_lines_back_to_file(new_doc_lines, out_file)


def get_all_spans(sentence):
    all_spans = []
    span_start_idxs = {}
    for line_i, line in enumerate(sentence):
        columns = line.strip().split()
        coref_labels = columns[-1]
        if coref_labels == '-':
            continue
        coref_labels_list = coref_labels.split('|')
        for lab_i, lab in enumerate(coref_labels_list):
            # print(lab)
            if lab[-1] == ')':
                lab = lab[:-1]
                rm_label = True
            else:
                rm_label = False

            if lab[0] == '(':
                lab = lab[1:]
                add_label = True
            else:
                add_label = False

            lab = int(lab)
            if lab not in span_start_idxs.keys():
                span_start_idxs[lab] = []

            if add_label:
                span_start_idxs[lab].append(line_i)

            if rm_label:
                all_spans.append([span_start_idxs[lab][-1], line_i])
                span_start_idxs[lab] = span_start_idxs[lab][:-1]
    # print(all_spans)

    # map span back to idxs
    idx2span = {}
    for i, line in enumerate(sentence):
        idx2span[i] = []

    for span in all_spans:
        l, r = span
        for i in range(l, r+1):
            idx2span[i].append([l, r])

    # just select the span with minimal length
    for i, line in enumerate(sentence):
        if len(idx2span[i]) > 0:
            sorted_spans = sorted(idx2span[i], key=lambda x: x[1]-x[0])
            idx2span[i] = sorted_spans[0]

    return all_spans, idx2span





def get_h2s_file(max_span_file, min_span_file, output_file):

    max_lines = reader.get_doc_lines(max_span_file)
    min_lines = reader.get_doc_lines(min_span_file)

    h2s_dataset = []

    for k in max_lines.keys():
        assert(k) in min_lines.keys()
        assert(len(max_lines[k])==len(min_lines[k]))

        for i_s, max_s in enumerate(max_lines[k]):
            min_s = min_lines[k][i_s]
            all_spans, idx2span = get_all_spans(max_s)

            words = [min_line.strip().split()[3] for min_line in min_s]



            for i_line, min_line in enumerate(min_s):
                columns = min_line.strip().split()
                coref_labels = columns[-1]
                if coref_labels == '-':
                    continue
                coref_labels_list = coref_labels.split('|')
                int_label_list = [int(lab[1:-1]) for lab in coref_labels_list]
                for lab in int_label_list:
                    first_word_index = max(0, i_line - MAX_CONTEXT)
                    last_word_index = min(len(min_s) - 1, i_line + MAX_CONTEXT)

                    truncated_words = words[first_word_index: last_word_index+1]
                    local_head_index = i_line - first_word_index

                    original_span = idx2span[i_line]
                    print(len(original_span))

                    #Relax assert for generating pred_out file
                    if len(original_span) == 0:
                        continue
                    #
                    # try:
                    #     assert(len(original_span)) > 0
                    # except:
                    #     print(all_spans)
                    #     print(idx2span)
                    #     print(k)
                    #     print(i_s)
                    #     print(i_line)
                    #     exit()
                    #TODO check what's happening here
                    if len(original_span) > 0:
                        if original_span[0] < first_word_index or original_span[1] > last_word_index:
                            continue

                        answer_span = [original_span[0] - first_word_index, original_span[1] - first_word_index]

                        example = make_h2s_example(text=truncated_words, local_head_id = local_head_index, global_head_id=i_line, passage_id=k, sentence_id=i_s, answer_span = answer_span)
                        h2s_dataset.append(example)


    with open(output_file, 'w') as fw:
        for e in h2s_dataset:
            json.dump(e, fw)
            fw.write('\n')



def get_h2s_from_hs_map(hs_map, output_file, noisy=False):
    hs_lines = reader.get_doc_lines(hs_map)
    h2s_dataset = []

    for k in hs_lines.keys():
        for i_s, sentence in enumerate(hs_lines[k]):

            words = [line.strip().split()[3] for line in sentence]

            for i_line, line in enumerate(sentence):
                columns = line.strip().split('   ')
                coref_labels = columns[-1]
                if coref_labels == '-':
                    continue
                coref_labels_list = coref_labels.split('|')
                tuple_label_list = [eval(lab[1:-1]) for lab in coref_labels_list]

                for tuple_lab in tuple_label_list:
                    lab, span_start, span_end = tuple_lab
                    if not noisy:
                        global_head_id = i_line
                    else:
                        global_head_id = np.random.randint(max(i_line-2, span_start), min(i_line+2+1, span_end+1))


                    first_word_index = max(0, global_head_id - MAX_CONTEXT)
                    last_word_index = min(len(sentence) - 1, global_head_id + MAX_CONTEXT)

                    truncated_words = words[first_word_index: last_word_index+1]

                    local_head_index = global_head_id - first_word_index

                    # original_span = [span_start, span_end]

                    answer_span = [span_start - first_word_index, span_end - first_word_index]

                    if span_start < first_word_index or span_end > last_word_index:
                        continue

                    example = make_h2s_example(text=truncated_words, local_head_id=local_head_index, global_head_id=global_head_id, passage_id=k, sentence_id=i_s, answer_span=answer_span)
                    h2s_dataset.append(example)

    with open(output_file, 'w') as fw:
        for e in h2s_dataset:
            json.dump(e, fw)
            fw.write('\n')




def get_predicted_head_conll(model_path, head_test_set, span_test_set):
    # Step 1 Get prediction on testing set

    head_pred_path = os.path.join(model_path, 'head_pred.out')

    #TODO only for debug
    pred_command = """allennlp predict --cuda-device 0 --use-dataset-reader --predictor coreference-resolution --silent --include-package new_allen_packages --output-file {0} {1} {2}""".format(head_pred_path , model_path, head_test_set)

    os.system(pred_command)



    # Step 2 Map the prediction file to h2s format
    head_conll_file = os.path.join(model_path, 'head_pred.conll')
    convert_pred_to_conll_format(head_pred_path, span_test_set, head_conll_file)
    return head_conll_file





if __name__ == '__main__':
    # get_h2s_file(max_span_file='/playpen/home/xzh/datasets/coref/allen/train.out.parse.english.v4_gold_conll.short', min_span_file='train.out.min_span.short', output_file='train_out_short_h2s.json')
    # get_h2s_file(max_span_file='/playpen/home/xzh/datasets/coref/allen/dev.out.parse.english.v4_gold_conll.short', min_span_file='dev.out.min_span.short', output_file='dev_out_short_h2s.json')

    #nlp7
    # head_conll_file = get_predicted_head_conll(model_path="./outputs/new_head_joint_debug", head_test_set="./train.out.min_span.short", span_test_set="/playpen/home/xzh/datasets/coref/allen/train.out.parse.english.v4_gold_conll.short")
    # get_h2s_file(max_span_file='/playpen/home/xzh/datasets/coref/allen/train.out.parse.english.v4_gold_conll.short', min_span_file=head_conll_file, output_file='pred_train_out_short_h2s.json')

    #dgx
    head_conll_file = get_predicted_head_conll(model_path="./outputs/head_debug_subword_light", head_test_set="./train.out.min_span.short", span_test_set="/fortest/xzh/datasets/coref/allen/train.out.parse.english.v4_gold_conll.short")
    get_h2s_file(max_span_file='/fortest/xzh/datasets/coref/allen/train.out.parse.english.v4_gold_conll.short', min_span_file=head_conll_file, output_file='pred_train_out_short_h2s.json')



    # get_h2s_file(max_span_file='/playpen/home/xzh/datasets/coref/allen/test.english.v4_gold_conll', min_span_file='test.min_span', output_file='test_h2s.json')
    # get_h2s_from_hs_map(hs_map='train.out.short.hs_map', output_file='train_out_short_h2s_gold.json')
    # get_h2s_from_hs_map(hs_map='dev.out.short.hs_map', output_file='dev_out_short_h2s_gold.json')
    # get_h2s_from_hs_map(hs_map='test.out.short.hs_map', output_file='test_out_short_h2s_gold.json')
    # np.random.seed(521)
    # get_h2s_from_hs_map(hs_map='train.hs_map', output_file='train_h2s_gold.noisy.json', noisy=True)
    # get_h2s_from_hs_map(hs_map='dev.hs_map', output_file='dev_h2s_gold.noisy.json', noisy=True)
    # get_h2s_from_hs_map(hs_map='test.hs_map', output_file='test_h2s_gold.noisy.json', noisy=True)
