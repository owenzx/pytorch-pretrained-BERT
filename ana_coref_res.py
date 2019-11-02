import sys
import json
import os
import numpy as np
from copy import deepcopy
from utils import bert_simple_detokenize



def is_overlap(l1, r1, l2, r2):
    if (r1 < l2) or (r2 < l1):
        return False
    else:
        return True


def get_mistakes(pred_path = None, data_path=None):
    import json
    import numpy as np
    if pred_path is None:
        pred_path = '/playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/check_train_bert_coref_freeze_1/predictions.json'
    #pred_path = '/playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/check_train_bert_coref_2/predictions.json'
    if data_path is None:
        data_path = '/playpen/home/xzh/datasets/coref/cleaned/coref/development.json'
    #data_path = '/playpen/home/xzh/datasets/coref/cleaned/coref/dev_wiki_short.json'
    error_path = '/playpen/home/xzh/work/pytorch-pretrained-BERT/error.txt'
    correct_path = '/playpen/home/xzh/work/pytorch-pretrained-BERT/correct.txt'



    result_dict= {}
    mistakes = []
    correct_samples = []
    with open(pred_path, 'r') as fr:
        lines =  fr.readlines()
    for line in lines:
        result = json.loads(line)
        result_dict[result['unique_id']] = {'is_correct': result['is_correct']}

    with open(data_path, 'r') as fr:
        lines = fr.readlines()
    for line in lines:
        raw_example = json.loads(line)
        #filter out all the examples without spans

        if len(raw_example["targets"]) < 1:
            continue


        for target_id, target in enumerate(raw_example["targets"]):
            guid = raw_example["info"]["document_id"] + "_" + str(raw_example["info"]["sentence_id"]) + "_" + str(target_id)

            span1_l, span1_r, span2_l, span2_r = target["span1"][0], target["span1"][1], target["span2"][0], target["span2"][1]

            label = target["label"]

            raw_tokens = raw_example["text"].split(' ')

            raw_tokens[span1_l] = "[[SPAN1 "+ raw_tokens[span1_l]
            raw_tokens[span1_r-1] = raw_tokens[span1_r-1] + " SPAN1]]"
            raw_tokens[span2_l] = "[[SPAN2 "+ raw_tokens[span2_l]
            raw_tokens[span2_r-1] = raw_tokens[span2_r-1] + " SPAN2]]"

            mistake_text ='LABEL: %s\n'%str(label) + ' '.join(raw_tokens)
            if not result_dict[guid]['is_correct']:
                mistakes.append(mistake_text)
            else:
                correct_samples.append(mistake_text)


    with open(error_path, 'w') as fw:
        for m in mistakes:
            fw.write(m + '\n')


    with open(correct_path, 'w') as fw:
        for c in correct_samples:
            fw.write(c + '\n')

    return mistakes, correct_samples


def get_difficulty_acc_figure():
    import json
    import numpy as np
    from collections import Counter
    import matplotlib
    import matplotlib.pyplot as plt
    pred_path = '/playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/check_wiki_train_bert_coref_2/predictions.json'
    #pred_path = '/playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/check_train_bert_coref_2/predictions.json'
    #data_path = '/playpen/home/xzh/datasets/coref/cleaned/coref/development.json'
    data_path = '/playpen/home/xzh/datasets/coref/cleaned/coref/dev_wiki_short.json'

    dist_option = 'word'

    assert(dist_option in ['word', 'entity'])

    result_dict= {}
    with open(pred_path, 'r') as fr:
        lines =  fr.readlines()
    for line in lines:
        result = json.loads(line)
        result_dict[result['unique_id']] = {'is_correct': result['is_correct']}


    with open(data_path, 'r') as fr:
        lines = fr.readlines()
    for line in lines:
        raw_example = json.loads(line)
        #filter out all the examples without spans

        if len(raw_example["targets"]) < 1:
            continue

        entities = []

        if dist_option == 'entity':
            for target_id, target in enumerate(raw_example["targets"]):
                span1_l, span1_r, span2_l, span2_r = target["span1"][0], target["span1"][1], target["span2"][0], target["span2"][1]
                entities.append((span1_l, span1_r))
                entities.append((span2_l, span2_r))
        entities = set(entities)
        #print(len(entities))


        for target_id, target in enumerate(raw_example["targets"]):
            guid = raw_example["info"]["document_id"] + "_" + str(raw_example["info"]["sentence_id"]) + "_" + str(target_id)

            span1_l, span1_r, span2_l, span2_r = target["span1"][0], target["span1"][1], target["span2"][0], target["span2"][1]

            label = target["label"]

            #if label == '0':
            #    continue

            min_r = min(span1_r, span2_r)
            max_l = max(span1_l, span2_l)

            if dist_option == 'word':
                dist = max_l - min_r
                if dist < 0:
                    dist = -1
            else:
                dist = 0
                for e in entities:
                    if e[0]>=min_r and e[1]<=max_l:
                        dist += 1
            result_dict[guid]['dist'] = dist

    dist_dict = {}
    for id, res_dict in result_dict.items():
        if 'dist' not in res_dict:
            continue
        if dist_option == 'word':
            if res_dict['dist']//10 not in dist_dict:
                dist_dict[res_dict['dist']//10] = {'total':0, 'correct':0}
            dist_dict[res_dict['dist']//10]['total'] += 1
            dist_dict[res_dict['dist']//10]['correct'] += res_dict['is_correct']
        elif dist_option =='entity':
            if res_dict['dist'] not in dist_dict:
                dist_dict[res_dict['dist']] = {'total':0, 'correct':0}
            dist_dict[res_dict['dist']]['total'] += 1
            dist_dict[res_dict['dist']]['correct'] += res_dict['is_correct']

    for k in dist_dict.keys():
        dist_dict[k]['acc'] = dist_dict[k]['correct'] / dist_dict[k]['total']



    #Merge small buckets so that the size of each bucket is at least 100
    raw_labels = sorted(dist_dict.keys(), reverse=True)
    bucket_count = 0
    merge_labels = []
    for l in raw_labels:
        if dist_dict[l]['total'] >= 100:
            break
        bucket_count += dist_dict[l]['total']
        merge_labels.append(l)
        if bucket_count >= 100:
            new_label = '{}-{}'.format(merge_labels[-1], merge_labels[0])
            dist_dict[new_label] = {'total': bucket_count, 'correct': sum([dist_dict[k]['correct'] for k in merge_labels])}
            dist_dict[new_label]['acc'] = dist_dict[new_label]['correct'] / dist_dict[new_label]['total']
            for m_l in merge_labels:
                del dist_dict[m_l]
            bucket_count = 0
            merge_labels = []

    labels = sorted(dist_dict.keys(), key=lambda x:x if type(x) is int else int(x.split('-')[0]))
    #labels = sorted(dist_dict.keys())
    values = [dist_dict[k]['acc'] for k in labels]
    #values = [dist_dict[k]['total'] for k in labels]

    #labels, values = zip(*sorted(dist_dict))
    indexes = np.arange(len(labels))
    width = 1


    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.savefig('tmp.pdf')


def get_common_mistakes():
    import json
    #pred_paths = ['/playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/check_train_bert_coref_freeze_1/predictions.json', '/playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/check_train_bert_coref_freeze_1_seed1/predictions.json', '/playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/check_train_bert_coref_freeze_1_seed2/predictions.json']
    pred_paths = ['/playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/check_train_bert_coref_2/predictions.json', '/playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/check_train_bert_coref_2_seed1/predictions.json', '/playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/check_train_bert_coref_2_seed2/predictions.json']
    common_mistakes = None
    common_correct = None

    #common_error_path = '/playpen/home/xzh/work/pytorch-pretrained-BERT/common_freeze.error'
    #common_correct_path = '/playpen/home/xzh/work/pytorch-pretrained-BERT/common_freeze.correct'
    common_error_path = '/playpen/home/xzh/work/pytorch-pretrained-BERT/common.error'
    common_correct_path = '/playpen/home/xzh/work/pytorch-pretrained-BERT/common.correct'

    for path in pred_paths:
        mistakes, correct = get_mistakes(pred_path=path)
        if common_mistakes is None:
            common_mistakes = set(mistakes)
            common_correct = set(correct)
        else:
            common_mistakes = common_mistakes.intersection(mistakes)
            common_correct = common_correct.intersection(correct)

    with open(common_error_path, 'w') as fw:
        for m in common_mistakes:
            fw.write(m+'\n')

    with open(common_correct_path, 'w') as fw:
        for c in common_correct:
            fw.write(c+'\n')



def check_coref_doc_stats():
    coref_file = '/playpen/home/xzh/datasets/coref/allen/dev.english.v4_gold_conll'

    with open(coref_file, 'r') as fr:
        lines = fr.readlines()

    doc_lens = []

    for line in lines:
        if len(line) <= 1:
            continue
        if line[0] == '#':
            if line[:6] == '#begin':
                doc_len = 0
            elif line[:4] == '#end':
                doc_lens.append(doc_len)
        else:
            doc_len += 1

    print(doc_lens)
    print(sorted(doc_lens))
    print(max(doc_lens))

    return doc_lens






def cluster_error_ana(pred_file, ana_type='pair', sample_num=10, out_file=None):
    prediction_file = pred_file

    assert(ana_type in ['mention', 'pair', 'mention_overlap'])
    md_strs = []
    wrong_pairs = []
    miss_pairs = []

    with open(prediction_file, 'r') as fr:
        lines = fr.readlines()
    #sample some cases just for the convenience of checking
    if sample_num!=-1:
        samples = []
        for i in range(sample_num):
            samples.append(lines[int(len(lines)/sample_num*i)-1])
        lines = samples

    for line in lines:
        predictions = json.loads(line)

        #top_spans predicted_antecedents, loss, document, tokenized_text, clusters, gold_clusters
        #if "Udai" not in predictions['document']:
        #    continue

        tokenized_text = predictions['tokenized_text']

        totuple = lambda t: tuple(totuple(tt) for tt in t) if isinstance(t, list) else t

        predicted_clusters = totuple(predictions['clusters'])
        predicted_clusters = [set(c) for c in predicted_clusters]
        predicted_mentions = set().union(*predicted_clusters)

        gold_clusters = totuple(predictions['gold_clusters'])
        gold_clusters = [set(c) for c in gold_clusters]

        gold_mentions = set().union(*gold_clusters)


        if ana_type == 'mention_overlap':
            wrong_boundary = []
            wrong_predictions =  predicted_mentions -  gold_mentions
            not_predicted = gold_mentions - predicted_mentions
            for mention in wrong_predictions:
                l, r = mention
                for g_mention in gold_mentions:
                    gl, gr = g_mention
                    if is_overlap(l, r, gl, gr):
                        wrong_boundary.append(mention)

            attributes = [[] for _ in tokenized_text]
            for mention in wrong_boundary:
                l, r = mention
                for i in range(l, r+1):
                    attributes[i].append('red')
                #tokenized_text[l] = '<span style="color:red"> ' + tokenized_text[l]
                #tokenized_text[r] = tokenized_text[r] + ' </span>'
            for mention in gold_mentions:
                l, r = mention
                for i in range(l, r+1):
                    attributes[i].append('b+i')
                #tokenized_text[l] = '***' + tokenized_text[l]
                #tokenized_text[r] = tokenized_text[r] + '***'

            for i in range(len(tokenized_text)):
                if 'b+i' in attributes[i]:
                    tokenized_text[i] = '***' + tokenized_text[i] + '***'
                if 'red' in attributes[i]:
                    tokenized_text[i] = '<span style="color:red"> ' + tokenized_text[i] + '</span>'



            #md_str = bert_simple_detokenize(tokenized_text)
            md_str = ' '.join(tokenized_text)

            md_str = md_str.replace('##', '^',)
            md_strs.append(md_str)

        if ana_type == 'mention':
            for mention in predicted_mentions - gold_mentions:
                l, r = mention
                tokenized_text[l] = '<span style="color:red">' + tokenized_text[l]
                tokenized_text[r] = tokenized_text[r] + '</span>'

            for mention in gold_mentions - predicted_mentions:
                l, r = mention
                tokenized_text[l] = '***' + tokenized_text[l]
                tokenized_text[r] = tokenized_text[r] + '***'


            md_str = bert_simple_detokenize(tokenized_text)
            md_strs.append(md_str)

        if ana_type == 'pair':
            #This block print the missing link and wrongly predicted link in a cluster


            miss_words = set()

            common_mentions = gold_mentions.intersection(predicted_mentions)
            for c in gold_clusters:

                text_copy = deepcopy(tokenized_text)

                c = c.intersection(common_mentions)
                if len(c) <= 1:
                    continue
                max_int = 0
                for c2 in predicted_clusters:
                    if len(c2.intersection(c)) > max_int:
                        max_int = len(c2.intersection(c))
                        miss_words = c - c2
                        wrong_words = c2 - c
                #miss_words = miss_words.union(miss_word)



                for (l, r) in miss_words:
                    text_copy[l] = '<span style="color:yellow">' + text_copy[l]
                    text_copy[r] = text_copy[r] + '</span>'

                for (l, r) in wrong_words:
                    text_copy[l] = '<span style="color:green">' + text_copy[l]
                    text_copy[r] = text_copy[r] + '</span>'

                for (l,r) in c:
                    text_copy[l] = '<span style="color:red">' + text_copy[l]
                    text_copy[r] = text_copy[r] + '</span>'

                md_str = bert_simple_detokenize(text_copy)
                md_str = md_str.replace('`', "'")
                if 'color:yellow' in md_str or 'color:green' in md_str:
                    md_strs.append(md_str)

    if out_file is None:
        file_name = './get_back/cluster_ana%s.md'%ana_type
    else:
        file_name = out_file
    with open(file_name, 'w') as fw:
        for s in md_strs:
            fw.write(s + '\n\n')





def main():
    pred_file = sys.argv[1]
    ana_type = sys.argv[2]
    out_file = sys.argv[3]
    if ana_type == 'full':
        for a_type in ['mention', 'pair', 'mention_overlap']:
            o_file = out_file + a_type + '.md'
            cluster_error_ana(pred_file, a_type, 10, o_file)
    else:
        cluster_error_ana(pred_file, ana_type, 10, out_file)





if __name__ == '__main__':
    main()



