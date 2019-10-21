import numpy as np
import torch
import sys
import json



def is_overlap(l1, r1, l2, r2):
    if (r1 < l2) or (r2 < l1):
        return False
    else:
        return True

def check_boundary_prediction(prediction_file, ana_type):
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
            for mention in wrong_boundary:
                l, r= mention
                tokenized_text[l] = '<span style="color:red">' + tokenized_text[l]
                tokenized_text[r] = tokenized_text[r] + '</span>'
            for mention in gold_mentions:
                l, r = mention
                tokenized_text[l] = '***' + tokenized_text[l]
                tokenized_text[r] = tokenized_text[r] + '***'


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





def main():
    pred_file = sys.argv[1]
    check_boundary_prediction(pred_file)



if __name__ == '__main__':
    main()