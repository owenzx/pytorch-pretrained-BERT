import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set
from allen_packages.ontonotes_reader import Ontonotes
from allen_packages.custom_coref_reader import SimpleCoref
from utils import get_chunk_sentences
from allen_packages.allen_reader import canonicalize_clusters

def load_text_from_dataset(path):
    """This function reads the dataset and returns pure text (in order for different tokenization for different models)"""
    raw_datasets = []
    dataset_reader = Ontonotes()
    for long_sentences in dataset_reader.dataset_document_iterator(path):

        chunk_sentences = get_chunk_sentences(long_sentences, max_num_tokens=400)


        for sentences in chunk_sentences:
            total_tokens = 0
            clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

            for sentence in sentences:
                for typed_span in sentence.coref_spans:
                    # Coref annotations are on a _per sentence_
                    # basis, so we need to adjust them to be relative
                    # to the length of the document.
                    span_id, (start, end) = typed_span
                    clusters[span_id].append((start + total_tokens,
                                              end + total_tokens))
                total_tokens += len(sentence.words)

            canonical_clusters = canonicalize_clusters(clusters)
            #_ =  self.text_to_instance([s.words for s in sentences], canonical_clusters)
            sen_id = str(sentences[0].document_id) + ':' + str(sentences[0].sentence_id)
            text = [s.words for s in sentences]
            untoken_data = {'text': text,
                            'clusters': canonical_clusters,
                            'sen_id': sen_id}
            raw_datasets.append(untoken_data)
            #instance = self.text_to_instance([s.words for s in sentences], canonical_clusters, sen_id)
            #yield instance
    return raw_datasets


def attacker2attackee_input(er_instance):
    ee_instance = None
    return ee_instance



def switch_new_mentions(text, switch_list, new_mention_text):
    assert(len(switch_list) == 1)
    switch_l_dict = {x[0]:x for x in switch_list}
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
            #new_mention_text = switch_l_dict[i][1]
            new_text.extend(new_mention_text)
            #idx_map[switch_l_dict[i][0][1]] = len(new_text) - 1
            idx_map[switch_l_dict[i][1] + 1] = len(new_text)
            i = switch_l_dict[i][1] + 1
    # add pseudo-map for the end of text
    idx_map[i] = len(new_text)

    return new_text, idx_map

def map_clusters(old_clusters, idx_map):
    new_clusters = []
    for c in old_clusters:
        new_c = []
        for span in c:
            new_c.append((idx_map[span[0]], idx_map[span[1]+1]-1))
        new_clusters.append(new_c)
    return new_clusters

def get_switched_ee_example(new_mention_text: str, old_metadict):
    """return in the same format of old_metadict just for consistency"""
    #TODO check this again
    new_mention_text = new_mention_text.split(' ')
    sen_id = old_metadict['input_sen_id']
    old_sentences = old_metadict['original_text'][0]
    old_gold_clusters = old_metadict['input_gold_clusters']

    selected_span = old_metadict['target_cluster']
    #TODO support multiple selected spans

    #print(selected_span)
    #print(old_gold_clusters)

    new_sentences, idx_map = switch_new_mentions(old_sentences, selected_span, new_mention_text)

    #print(new_sentences)
    #print(old_sentences)
    #print(idx_map)

    # Assume no overlapping spans
    new_clusters = map_clusters(old_gold_clusters, idx_map)

    #print(old_gold_clusters)
    #print(new_clusters)

    new_target_span = (idx_map[selected_span[0][0]], idx_map[selected_span[0][1]+1]-1)

    return {'input_sentences':[new_sentences],
            'input_gold_clusters':new_clusters,
            'input_sen_id':sen_id}, new_target_span
