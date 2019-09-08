

class A(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, output_dir = None):
        self.output_dir = output_dir



class B(A):
    def look(self):
        print('AAA')


    def look2(self):
        print(self.output_dir)



def C():
    print('c')



def main():
    bb = B()
    bb.look()
    bb2 = B(output_dir='123')
    bb2.look2()


def check_data():
    import csv
    import nltk
    import numpy as np
    from nltk.tokenize import sent_tokenize
    import json
    np.random.seed(1)
    #path = './datasets/amazon-2/amazon_review_polarity_csv/train.csv'
    path = './datasets/dbpedia/dbpedia_csv/train.csv'
    json_result_path = './datasets/ins/dbpedia/train.json'
    with open(path, 'r', encoding='utf-8') as fr:
        reader = csv.reader(fr, delimiter=',', quotechar='"')
        selected_paras = []
        for i, line in enumerate(reader):
            #if i==200:
            #    break
            title = line[0]
            para = line[-1]
            id = "ins_dbpedia_%d"%i
            sens = sent_tokenize(para)
            if len(sens)>4:
                selected_paras.append((para,title, id))
    print(len(selected_paras))
    exit()


    ins_sen = []

    for para, title, id in selected_paras:
        sens = sent_tokenize(para)
        # TODO: remove the -1 after len(sens)
        miss_idx = np.random.randint(0, len(sens) - 1)
        miss_sen = sens[miss_idx]
        #rem_para = '<INS> ' + ' <INS> '.join(sens[:miss_idx] + sens[miss_idx+1:]) + ' <INS>'
        prev_str = ' '.join(sens[:miss_idx])
        if miss_idx == 0:
            answer_start = 0
        else:
            answer_start = len(prev_str) + len(' ')
        answer_text = sens[miss_idx+1].split(' ')[0]
        rem_para = ' '.join(sens[:miss_idx] + sens[miss_idx+1:])
        assert(rem_para[answer_start] == answer_text[0])
        ins_sen.append((title, rem_para, miss_sen, miss_idx, id, answer_start, answer_text))

    #for title, para, sen, miss_idx, id, answer_start, answer_text in ins_sen:
    #    print("PARA: \n{}\nSEN: \n{}\n\n".format(para, sen))

    json_dict = {"data":[]}
    for title, para, sen, miss_idx, id, answer_start, answer_text in ins_sen:
        context = para
        question = sen
        #answer_text = ""
        #answer_start = 0

        article_dict = dict()
        article_dict['title'] = title
        qas_dict = [{"id":id, "question": question, "answers":[{"text":answer_text, "answer_start":answer_start}]}]
        article_dict["paragraphs"] = [{"context": context, "qas":qas_dict}]


        json_dict["data"].append(article_dict)
    with open(json_result_path, 'w') as fw:
        json.dump(json_dict, fw)


def read_coref_data():
    import json
    data_path = '/playpen/home/xzh/datasets/coref/cleaned/coref/train.json'
    with open(data_path, 'r') as fr:
        lines = fr.readlines()
    for line in lines:
        exp = json.loads(line)
        print(exp)



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



def get_analysis_datasets():
    #Random function used to pring some analysis stuff
    dataset_path = '/playpen/home/xzh/datasets/coref/cleaned/coref/development.json'

    analysis_path = ''


    with open(dataset_path, 'r') as fr:
        lines = fr.readlines()
    for line in lines:
        pass



def clean_wikicoref():
    import json
    wiki_path = '/playpen/home/xzh/datasets/WikiCoref/Evaluation/key-OntoNotesScheme'

    clean_path = '/playpen/home/xzh/datasets/coref/cleaned/coref/dev_wiki_short.json'

    with open(wiki_path, 'r') as fr:
        lines = fr.readlines()

    clean_dicts =[]

    in_document = False
    document_count = 0

    for i, line in enumerate(lines):
        if line[-1] == '\n':
            line = line[:-1]
        if line[:6] == '#begin':
            in_document = True
            document_text = []
            span_dict = dict()
        elif line[:4] == '#end':
            in_document = False


            #TODO remove 300/290 limit
            clean_dict = {"info":{"document_id":"wiki/%d"%document_count, "sentence_id":0}, "text": " ".join(document_text[:290]), "targets": []}
            spans_list = []
            for span_idx, spans in span_dict.items():
                for span in spans:
                    if span[0]>=290 or span[1]>=290:
                        continue
                    spans_list.append({"span":span, "span_idx":span_idx})

            for i in range(len(spans_list)):
                for j in range(i+1, len(spans_list)):
                    clean_dict["targets"].append({"span1":spans_list[i]["span"], "span2":spans_list[j]["span"], "label":"1" if spans_list[i]["span_idx"]==spans_list[j]["span_idx"] else "0"})

            document_count += 1
            clean_dicts.append(clean_dict)


        elif len(line) == 0:
            continue
        elif line[:4] == 'null':
            continue
        else:
            assert(in_document)
            assert(len(line.split('\t')) == 5)
            _, _, idx, word, span_label = line.split('\t')
            cur_word_idx = len(document_text)
            document_text.append(word)
            if span_label == '-':
                pass
            else:
                spans = span_label.split('|')
                for s in spans:
                    if s[0] == '(' and s[-1] == ')':
                        span_idx = int(s[1:-1])
                        if span_idx not in span_dict:
                            span_dict[span_idx] = []
                        span_dict[span_idx].append([cur_word_idx, cur_word_idx+1])
                    elif s[0] == '(':
                        span_idx = int(s[1:])
                        if span_idx not in span_dict:
                            span_dict[span_idx] = []
                        span_dict[span_idx].append([cur_word_idx, -1])
                    elif s[-1] == ')':
                        span_idx = int(s[:-1])
                        span_dict[span_idx][-1][1] = cur_word_idx + 1
                    else:
                        print(len(span_label))
                        print(span_label[-1]=='\n')
                        print(span_label)
                        print(s)
                        exit()




    with open(clean_path, 'w') as fw:
        for d in clean_dicts:
            json.dump(d, fw)
            fw.write('\n')


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








def get_bert_ckpt_out():
    import os
    import torch
    from collections import OrderedDict
    from shutil import copyfile

    config_option = 'bert_base_uncased'
    allen_dir = '/playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/allen_test_bert_tune_large/'
    if config_option == 'bert_base_uncased':
        bert_config_file =  '/playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/train_bert_coref_save_1_int_33000/config.json'
    else:
        raise NotImplementedError

    new_bert_dir = os.path.join(allen_dir, 'tuned_bert')
    if not os.path.exists(new_bert_dir):
        os.makedirs(new_bert_dir)
    new_bert_config_path = os.path.join(new_bert_dir, 'config.json')
    copyfile(bert_config_file, new_bert_config_path)
    new_bert_state_dict_path = os.path.join(new_bert_dir, 'pytorch_model.bin')


    allen_state_dict_path = os.path.join(allen_dir, 'best.th')

    allen_state_dict = torch.load(allen_state_dict_path)

    new_state_dict = OrderedDict()

    for k in allen_state_dict.keys():
        if 'bert' not in k:
            continue
        new_key = k.replace('bert_model', 'bert')
        new_state_dict[new_key] = allen_state_dict[k]

    torch.save(new_state_dict, new_bert_state_dict_path)




def cluster_error_ana():
    import json
    prediction_file = './tmp.out'

    ana_type = 'pair'
    assert(ana_type in ['mention', 'pair'])
    md_strs = []
    wrong_pairs = []
    miss_pairs = []

    with open(prediction_file, 'r') as fr:
        lines = fr.readlines()
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

        if ana_type == 'mention':
            for mention in predicted_mentions - gold_mentions:
                l, r = mention
                tokenized_text[l] = '<span style="color:red">' + tokenized_text[l]
                tokenized_text[r] = tokenized_text[r] + '</span>'

            for mention in gold_mentions - predicted_mentions:
                l, r = mention
                tokenized_text[l] = '<span style="color:yellow">' + tokenized_text[l]
                tokenized_text[r] = tokenized_text[r] + '</span>'

        if ana_type == 'pair':

            miss_words = set()

            common_mentions = gold_mentions.intersection(predicted_mentions)
            for c in gold_clusters:
                c = c.intersection(common_mentions)
                if len(c) <= 1:
                    continue
                max_int = 0
                for c2 in predicted_clusters:
                    if len(c2.intersection(c)) > max_int:
                        max_int = len(c2.intersection(c))
                        miss_word = c - c2
                miss_words = miss_words.union(miss_word)

            for (l, r) in miss_words:
                tokenized_text[l] = '<span style="color:green">' + tokenized_text[l]
                tokenized_text[r] = tokenized_text[r] + '</span>'

            wrong_words =  set()
            for c2 in predicted_clusters:
                c2 = c2.intersection(common_mentions)
                if len(c) <= 1:
                    continue
                max_int = 0
                for c in gold_clusters:
                    if len(c.intersection(c2)) > max_int:
                        max_int = len(c.intersection(c2))
                        wrong_word = c2 - c
                wrong_words = wrong_words.union(wrong_word)

            for (l, r) in wrong_words:
                tokenized_text[l] = '<span style="color:blue">' + tokenized_text[l]
                tokenized_text[r] = tokenized_text[r] + '</span>'


        md_str = ' '.join(tokenized_text)
        if len(miss_words) + len(wrong_words) > 0:
            md_strs.append(md_str)

    with open('./get_back/cluster_ana%s.md'%ana_type, 'w') as fw:
        for s in md_strs:
            fw.write(s + '\n\n')



def check_if_json():
    file_path = './tmp.out'
    import json
    with open(file_path, 'r') as fr:
        lines = fr.readlines()

    for line in lines:
        pred = json.loads(line)
        print(pred)
        print(type(pred))
        print(pred.keys())
        exit()


def check_mentions():
    import pickle
    from pytorch_pretrained_bert import BertTokenizer
    bert_model_name = 'bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    mention_save_path = './debug.corpus'
    with open(mention_save_path, 'rb') as fr:
        mention_list = pickle.load(fr)

    for m in mention_list:
        text = m['text']
        new_mention = bert_tokenizer.wordpiece_tokenizer.tokenize(' '.join(text))
        if '[UNK]' in new_mention:
            print(text)
            print(new_mention)





if __name__ == '__main__':
    #check_data()
    #read_coref_data()
    #get_difficulty_acc_figure()
    #get_mistakes()
    #get_common_mistakes()
    #clean_wikicoref()
    #check_coref_doc_stats()
    #get_bert_ckpt_out()
    #cluster_error_ana()
    #check_if_json()
    check_mentions()

