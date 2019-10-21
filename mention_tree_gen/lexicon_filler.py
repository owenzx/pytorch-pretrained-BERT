import numpy as np
from anytree import Node, RenderTree, PreOrderIter
#from gensim.models import Word2Vec
from gensim.models import KeyedVectors as Word2Vec
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set
import nltk
from collections import namedtuple
import torch


def _translate_tree_to_str(tree_root):
    str_result = []
    for node in PreOrderIter(tree_root):
        if node.is_leaf:
            str_result.append(node.name)
    return ' '.join(str_result)


def _build_empty_tree():
    root_node = Node("NP")
    #n = Node("N", parent = root_node)
    return root_node


def _load_glove_vecs(glove_gensim_path):
    #TODO delete 1000 after debugging
    return Word2Vec.load_word2vec_format(glove_gensim_path, limit=10000)


def get_singleword_tag(word, wordtags):
    brown_tags = wordtags[word.lower()]
    if len(brown_tags) != 0:
        return list(brown_tags)
    else:
        nltk_tags = nltk.pos_tag([word])
        if 'N' in nltk_tags[0][1]:
            return ['NOUN']
        elif 'V' in nltk_tags[0][1]:
            return ['VERB']
        elif 'J' in nltk_tags[0][1]:
            return ['ADJ']
        elif 'RB' in nltk_tags[0][1]:
            return ['ADV']
        else:
            return [nltk_tags[0][1]]


def get_closest_word_by_pos_sentvec(pos, sentvec, wordtags, w2v):

    words = w2v.similar_by_vector(sentvec, topn = 300, restrict_vocab=10000)
    words_valid_pos = []
    for w in words:
        w_pos = get_singleword_tag(w[0], wordtags)[0]
        if w_pos == 'NOUN' and pos == 'N':
            words_valid_pos.append(w)
        elif w_pos == 'ADJ' and pos == 'A':
            words_valid_pos.append(w)
        elif w_pos == 'ADV' and pos == 'Adv':
            words_valid_pos.append(w)

    if len(words_valid_pos) == 0:
        words_valid_pos = words
    selected_idx = np.random.choice(len(words_valid_pos), 1)[0]
    selected_w = words_valid_pos[selected_idx][0]
    #selected_w = np.random.choice(words_valid_pos, 1)[0][0]
    return selected_w









def get_sentence_vec(sentence, w2v):
    word_vecs = []
    sentence = sentence[0 ]
    for word in sentence:
        if word not in w2v.vocab:
            continue
        word_vecs.append(w2v[word])
    if len(word_vecs) == 0:
        # TODO delete after debugging
        #return None
        return w2v['the']
    else:
        return np.mean(word_vecs, axis=0)





class LexiconFiller(object):
    def __init__(self, args, vocab):
        #self.cfg2idx = {}
        self.args = args
        self.vocab = vocab
        self.idx2cfg = vocab._index_to_token['cfgs']
        print(self.idx2cfg)

        self.cfg2idx = {v:k for k,v in self.idx2cfg.items()}

        #Load GloVe dictionary
        self.glove_words = _load_glove_vecs(args.glove_gensim_path)
        self.wordtags = nltk.ConditionalFreqDist((w.lower(), t)
                                                 for w, t in nltk.corpus.brown.tagged_words(tagset="universal"))

    def check_completion(self, root) -> bool:
        """this function checks if the NP tree is complete (No non-terminal tokens at leaf level"""
        complete = True
        for node in PreOrderIter(root):
            if node.is_leaf:
                if node.name in ['NP', 'AP', 'PP']:
                    complete = False
                    break
        return complete


    def apply_cfg_rules_to_tree(self, cfg_rule, root):
        parent, children = cfg_rule.split('->')
        children_list = children.split(',')
        children_node = []
        for c in children_list:
            children_node.append(Node(c))
        applied = False
        for node in PreOrderIter(root):
            if node.is_leaf:
                if node.name == parent:
                    node.children = tuple(children_node)
                    applied = True
                    break

        if not applied:
            return None
        else:
            return root


    def get_word_by_pos(self, pos, sent_vec):
        if pos == 'P':
            #TODO better prep list
            return np.random.choice(['at', 'in', 'near', 'on'], 1)[0]
        elif pos in ['A', 'Adv', 'N']:
            return get_closest_word_by_pos_sentvec(pos, sent_vec, self.wordtags, self.glove_words)


    def fill_leaf_word(self, root, sent_vec, head_filled, np_head):
        for node in PreOrderIter(root):
            if node.is_leaf:
                if not head_filled and node.name in ['N']:
                    node.name = np_head
                    head_filled = True
                elif node.name in ['N', 'A', 'Adv', 'P']:
                    node.name = self.get_word_by_pos(node.name, sent_vec)
        return root, head_filled


    def trim_tree(self, root):
        """This function trims the tree and make all the non-terminal node terminal"""
        for node in PreOrderIter(root):
            if node.is_leaf:
                if node.name in ['NP', 'AP', 'PP']:
                    node.name = node.name[:-1]
        return root



    def fill_lexicon(self, mention_cfgs, sent, np_head) -> str:
        sent_vec = get_sentence_vec(sent, self.glove_words)
        print(mention_cfgs)
        root = _build_empty_tree()
        stopped = False

        head_filled = False

        for cfg in mention_cfgs:
            #cfg = self.idx2cfg[cfg_idx]
            #print(cfg)
            if cfg[0] == '@':
                return None
            if cfg == 'STOP':
                stopped = True
                break
            root = self.apply_cfg_rules_to_tree(cfg, root)

            if root is None:
                return None

            root, head_filled = self.fill_leaf_word(root, sent_vec, head_filled, np_head)

            if root is None:
                return None



        if not stopped:
            return None

        is_complete = self.check_completion(root)
        if is_complete:
            new_mention_text = _translate_tree_to_str(root)
            return new_mention_text
        else:
            root = self.trim_tree(root)
            root, head_filled = self.fill_leaf_word(root, sent_vec, head_filled, np_head)
            #return None
            new_mention_text = _translate_tree_to_str(root)
            return new_mention_text




    def _translate_cfg_(self, cfg_idx):
        pass


    def get_available_action(self, action_history):
        action_history = [a.cpu().numpy() for a in action_history]
        #print(action_history[0].shape)
        cfgs = [self.idx2cfg[idx[0]] for idx in action_history]
        #cfgs = get_cfg_from_idx(action_history)
        root = _build_empty_tree()
        if len(cfgs) > 0 and cfgs[-1] in ['STOP', '@end@']:
            avail = ['@end@']
        else:
            print(cfgs)
            for cfg in cfgs:
                root = self.apply_cfg_rules_to_tree(cfg, root)
            expandable = None
            for node in PreOrderIter(root):
                if node.is_leaf:
                    if node.name in ['NP', 'AP', 'PP']:
                        expandable = node.name
                        break
            if expandable is None:
                avail = ['STOP']
            else:
                avail = []
                for cfg in self.cfg2idx.keys():
                    if cfg[:len(expandable)] == expandable:
                        avail.append(cfg)

        empty_mask = np.zeros((1, len(self.idx2cfg.keys())))
        for c in avail:
            empty_mask[0][self.cfg2idx[c]] = 1

        torch_mask = torch.Tensor(empty_mask).cuda()
        return torch_mask











def check_oracle_policy_results():
    glove_struct = namedtuple('GlovePath', 'glove_gensim_path')
    simple_args = glove_struct(glove_gensim_path="./datasets/glove.840B.300d.w2vformat.txt")
    lexicon_filler =LexiconFiller(simple_args, None)
    head_noun = "park"
    context_sentence = "This is a beautiful park. It's located at the south part of Chapel hill."
    oracle_grammar = ['NP->AP,N,PP', 'AP->Adv,A', 'PP->P,NP', 'NP->N', 'STOP']
    generated_text = lexicon_filler.fill_lexicon(oracle_grammar, context_sentence, head_noun)
    print(generated_text)

def debug():
    glove_struct = namedtuple('GlovePath', 'glove_gensim_path')
    simple_args = glove_struct(glove_gensim_path="./datasets/glove.840B.300d.w2vformat.txt")
    lexicon_filler =LexiconFiller(simple_args, None)


if __name__ == '__main__':
    check_oracle_policy_results()
    debug()
