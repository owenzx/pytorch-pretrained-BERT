import numpy as np
import nltk
import argparse
import pickle
import random
import os

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from data_processing import InputExample
from copy import deepcopy


from utils import prepare4translation, extract_source_from_middle, clean_back_trans_results



def get_paraphrase_backtrans(raw_file_path):
    trans_en_fr_path = '../ref_repos/paraphrase_translation/para_trans_enfr.sh'
    trans_fr_en_path = '../ref_repos/paraphrase_translation/para_trans_fren.sh'
    mid_path = raw_file_path + '.mid'
    final_path = raw_file_path + '.fin'
    print("Preparing source file...")
    if 'csv' in raw_file_path:
        quotechar='"'
        delimiter=','
    else:
        quotechar=None,
        delimiter='\t'
    trans_source_file = prepare4translation(raw_file_path, quotechar=quotechar, delimiter=delimiter)
    print("Translating...")
    os.system("""cat {} | bash {} | egrep 'H-|S-' > {}""".format(trans_source_file, trans_en_fr_path, mid_path))
    print("Preparing source file for back translation...")
    trans_back_source_file = extract_source_from_middle(mid_path)
    print("Back translating...")
    os.system("""cat {} | bash {} | egrep 'H-|S-' > {}""".format(trans_back_source_file, trans_fr_en_path, final_path))
    print("Cleaning translation results...")
    cleaned_file = clean_back_trans_results(final_path)
    print("Finish!")
    return cleaned_file



def para_interpolation_basic(input_para1, input_para2, magn_ratio=3):
    sens1 = sent_tokenize(input_para1)
    sens2 = sent_tokenize(input_para2)

    len_sens1 = len(sens1)
    len_sens2 = len(sens2)

    sens1_split_point = np.random.randint(1, len_sens1+1)
    sens2_split_point = np.random.randint(0, len_sens2)

    new_sens = sens1[:sens1_split_point] + sens2[sens2_split_point:]

    new_para = ' '.join(new_sens)

    return new_para




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


def swap_sen(input_sen, magn_ratio=3):
    sens = sent_tokenize(input_sen)

    tot_len = len(sens)

    swap_num = tot_len // (magn_ratio + 1)

    if swap_num == 0:
        return input_sen

    selected_words_idx = np.random.choice(tot_len-1, swap_num, replace=False)

    for idx in selected_words_idx:
        sens[idx], sens[idx+1] = sens[idx+1], sens[idx]

    output_sen = ' '.join(sens)

    return output_sen

def del_sen(input_sen, magn_ratio=3):
    sens = sent_tokenize(input_sen)
    tot_len = len(sens)

    del_num = tot_len // (magn_ratio + 1)

    if del_num == 0:
        return input_sen

    print(tot_len)
    print(del_num)
    selected_words_idx = np.random.choice(tot_len, del_num, replace=False)

    result_sens = [sens[i] for i in range(tot_len) if  i not in selected_words_idx]

    output_sen = ' '.join(result_sens)

    return output_sen

def concat_para(input_sena, input_senb, magn_ratio=3):
    sens1 = sent_tokenize(input_sena)
    sens2 = sent_tokenize(input_senb)

    len_sens1 = len(sens1)
    len_sens2 = len(sens2)

    sens1_split_point = np.random.randint(1, len_sens1+1)
    sens2_split_point = np.random.randint(0, len_sens2)

    new_sens = sens1[:sens1_split_point] + sens2[sens2_split_point:]

    new_para = ' '.join(new_sens)

    return new_para





aug_func_dict = {'random_swap':random_swap,
                 'drop_stopword':drop_stopword,
                 'add_mask':add_mask,
                 'para_int_basic':para_interpolation_basic,
                 'swap_sen': swap_sen,
                 'del_sen': del_sen,
                 'concat_para':concat_para,}




def generate_aug_examples(examples, aug_policy):
    """Since we're doing low resource setting. For each sample, we're using all the aug_policies"""
    aug_examples = []
    inp_examples = deepcopy(examples)

    assert(len(aug_policy) == 1)
    aug_policy = aug_policy[0]

    for sub_policy in aug_policy:
        aug_method, aug_num, aug_magn = sub_policy
        aug_func = aug_func_dict[aug_method]
        if aug_method in ['para_int_basic', 'concat_para']:
            input_num = 2
        else:
            input_num = 1
        if 0 < aug_num < 1:
            epoch_num = 1
        else:
            epoch_num = int(aug_num)
        for i in range(epoch_num):
            for j, example in enumerate(inp_examples):
                new_guid = example.guid+'aug{}{}'.format(i, j)
                if input_num == 1:
                    new_text_a = aug_func(example.text_a, aug_magn)
                    new_text_b = aug_func(example.text_b, aug_magn) if example.text_b is not None else None
                elif input_num == 2:
                    exampleb = inp_examples[np.random.randint(0,len(inp_examples))]
                    while exampleb.label != example.label:
                        exampleb = inp_examples[np.random.randint(0,len(inp_examples))]
                    new_text_a = aug_func(example.text_a, exampleb.text_a, aug_magn)
                    new_text_b = aug_func(example.text_b, exampleb.text_b, aug_magn) if example.text_b is not None else None
                new_label = example.label
                new_example = InputExample(new_guid, new_text_a, new_text_b, new_label)
                aug_examples.append(new_example)
        if 0 < aug_num < 1:
            inp_examples.extend(aug_examples[:int(len(inp_examples)*aug_num)])
        else:
            inp_examples.extend(aug_examples)
        aug_examples = []
    #aug_examples.extend(examples)
    return inp_examples


def get_augmented_inputs(examples, aug_strategy, aug_num=1):
    """The augmented examples contain the original examples."""

    aug_func = aug_func_dict[aug_strategy]

    if aug_strategy in ['para_int_basic']:
        input_num = 2
    else:
        input_num = 1

    aug_examples = []
    for example in examples:
        aug_examples.append(example)
        for i in range(aug_num):
            new_guid = example.guid+'aug%d'%i
            if input_num == 1:
                new_text_a = aug_func(example.text_a)
                new_text_b = aug_func(example.text_b) if example.text_b is not None else None
            elif input_num == 2:
                exampleb = examples[np.random.randint(0,len(examples))]
                while exampleb.label != example.label:
                    exampleb = examples[np.random.randint(0,len(examples))]
                new_text_a = aug_func(example.text_a, exampleb.text_a)
                new_text_b = aug_func(example.text_b, exampleb.text_b) if example.text_b is not None else None

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
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    assert(args.augment_strategy is not None)

    random.seed(args.seed)
    np.random.seed(args.seed)

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


def test():
    para1 = '''very disappointing . this product has less control than an average pair of pantyhose . the `` high waist '' was always rolling down . i was n't expecting miracles ; i just wanted
 to smooth out my silouette while i lost the final ten pounds of pregnancy weight . this product was completely useless . product has a clever name but that 's it . i really felt like a
sucker for spending so much money on this item . look elsewhere .'''
    para2 = '''these slippers are without a doubt the most uncomfortable things i 've ever put my feet into . the concept is good , but the shearling lining is so thick that you would definatel
y want to order the next size larger . that might solve the length discomfort , but there still is n't sufficient heighth comfort . they felt like an all encompassing vise , squeezing my
 feet so tightly that i had to remove them to gain relief . very uncomfortable .'''
    print(para_interpolation_basic(para1, para2))

if __name__ == '__main__':
    #main()
    #test()
    #raw_file_path = './datasets/mtl-dataset/subdatasets/apparel/train.tsv'
    raw_file_path = './datasets/amazon-2/amazon_review_polarity_csv/train.csv'
    get_paraphrase_backtrans(raw_file_path)





