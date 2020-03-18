import numpy as np
import json
import sys
import os

from conll import reader
from conll import util
from conll import mention

from pprint import pprint
from tqdm import tqdm, trange

from coref_metric_helper import write_lines_back_to_file



def get_max_mem_debug_set(file_path, sen_num=110, dataset_size=50):
    doc_lines = reader.get_doc_lines(file_path)


    k2l_list = []

    for k in doc_lines.keys():
        sens = doc_lines[k][:sen_num]
        total_len = np.sum([len(s) for s in sens])
        k2l_list.append((k, total_len))


    k2l_list = sorted(k2l_list, key=lambda x: x[1], reverse=True)


    selected_doc_lines = {}
    for k, _ in k2l_list[:dataset_size]:
        selected_doc_lines[k] = doc_lines[k]



    write_lines_back_to_file(selected_doc_lines, file_path+'.longest')




get_max_mem_debug_set('./train.min_span')