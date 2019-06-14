import numpy as np
import nltk
import argparse
import pickle

from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from data_processing import InputExample


def random_swap(input_sen, swap_ratio=3):
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    words = tokenizer.tokenize(input_sen)

    tot_len = len(words)

    swap_num = tot_len // swap_ratio
    if swap_num == 0:
        return input_sen

    selected_words_idx = np.random.choice(tot_len-1, swap_num, replace=False)

    for idx in selected_words_idx:
        words[idx], words[idx+1] = words[idx+1], words[idx]


    output_sen = detokenizer.detokenize(words)

    return output_sen


def drop_stopword(input_sen):
    stopword_keep_lst = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
                         "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
                         'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
                         'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                         'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
                         "not"]

    useless_stopwords = list(set(stopwords.words("english")).difference(set(stopword_keep_lst)))

    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    words = tokenizer.tokenize(input_sen)

    new_words = [w for w in words if w not in useless_stopwords]

    output_sen = detokenizer.detokenize(new_words)

    return output_sen


def add_mask(input_sen, mask_ratio=3):
    mask_token = "[MASK]"

    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    words = tokenizer.tokenize(input_sen)

    tot_len = len(words)

    mask_num = tot_len // mask_ratio
    if mask_num == 0:
        return input_sen

    selected_words_idx = np.random.choice(tot_len, mask_num, replace=False)

    for idx in selected_words_idx:
        words[idx] = mask_token

    output_sen = detokenizer.detokenize(words)

    return output_sen





def get_augmented_inputs(examples, aug_strategy, aug_num=1):
    """The augmented examples contain the original examples."""
    aug_func_dict = {'random_swap':random_swap,
                     'drop_stopword':drop_stopword,
                     'add_mask':add_mask}

    aug_func = aug_func_dict[aug_strategy]

    aug_examples = []
    for example in examples:
        aug_examples.append(example)
        for i in range(aug_num):
            new_guid = example.guid+'aug%d'%i
            new_text_a = aug_func(example.text_a)
            new_text_b = aug_func(example.text_b) if example.text_b is not None else None
            new_label = example.label
            new_example = InputExample(new_guid, new_text_a, new_text_b, new_label)
            aug_examples.append(new_example)

    return aug_examples


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--augment_strategy",
                        default=None,
                        type=str,
                        help='The strategy used to augment the data')
    parser.add_argument("--origin_path",
                        default=None,
                        type=str,
                        help='The path to the real dataset to use for training')
    parser.add_argument("--aug_num",
                        default=1,
                        type=int,
                        help='The number of augmented samples generated for every origin sample')

    args = parser.parse_args()

    assert(args.augment_strategy is not None)

    with open(args.origin_path, 'rb') as fr:
        origin_examples = pickle.load(fr)

    aug_examples = get_augmented_inputs(origin_examples, args.augment_strategy, args.aug_num)

    aug_path = args.origin_path + '.augstr_%s'%args.augment_strategy + '.augnum%d' % args.aug_num + '.pkl'

    print(aug_path)

    with open(aug_path, 'wb') as fw:
        pickle.dump(aug_examples, fw)

    aug_text_path = args.origin_path + '.augstr_%s'%args.augment_strategy + '.augnum%d' % args.aug_num + '.txt'

    with open(aug_text_path, 'w') as fw:
        for example in aug_examples:
            fw.write(str(example.text_a))
            fw.write('\n')
            fw.write(str(example.text_b))
            fw.write('\n')


if __name__ == '__main__':
    main()





