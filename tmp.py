# import os
# import sys
#
# vocab2 = "/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/h2s_out_finetune/vocabulary/tokens.txt"
# vocab1 = "/ssd-playpen/home/xzh/work/pytorch-pretrained-BERT/outputs/h2s_debug_gold/vocabulary/tokens.txt"
#
# with open(vocab1, 'r') as fv:
#     lines = fv.readlines()
#
# vocab = []
# for line in lines:
#     vocab.append(line)
#
# vset = set(vocab)
#
# with open(vocab2, 'r') as fv:
#     new_lines = fv.readlines()
#
# for line in new_lines:
#     if line not in vset:
#         vocab.append(line)
#
#
# print(len(vocab))
#
# with open('tokens.txt', 'w') as fw:
#     for line in vocab:
#         fw.write(line)

def all_same(items):
    return all(x == items[0] for x in items)

from conll import reader
file = '/playpen/home/xzh/datasets/coref/allen/train.english.v4_gold_conll'
doc_lines = reader.get_doc_lines(file)
for doc_key, doc in doc_lines.items():
    for sentence in doc:
        speakers = [token_line.strip().split()[9] for token_line in sentence]
        try:
            assert(all_same(speakers))
        except:
            print(doc_key)
            exit()
print("COOL!")
