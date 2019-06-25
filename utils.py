import numpy as np
import collections
from collections import defaultdict
import torch
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr
from data_processing import *



def sample_meta_trainval(examples, train_size, val_size):
    total_size = train_size + val_size
    examples = np.array(examples)
    selected_idx = np.random.choice(len(examples), total_size, replace=False)
    train_idx = selected_idx[:train_size]
    val_idx = selected_idx[train_size:]
    train_examples = examples[train_idx].tolist()
    val_examples = examples[val_idx].tolist()
    examples = examples.tolist()

    return train_examples, val_examples




def compute_accuracy(preds, labels, task_main, task_comp):
    if output_num[task_main] == output_num[task_comp]:
        return simple_accuracy(preds, labels)
    elif output_num[task_main] > output_num[task_comp]:
        return three2two_accuracy(preds, labels)
    elif output_num[task_main] < output_num[task_comp]:
        return two2three_accuracy(preds, labels)


def compute_metrics(task_name, preds, labels, task_main):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "yelp-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "yelp-5":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "amazon-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "amazon-5":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "dbpedia":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": compute_accuracy(preds, labels, task_main, task_name)}
    elif task_name == "mnli-mm":
        return {"acc": compute_accuracy(preds, labels, task_main, "mnli")}
    elif task_name == "qnli":
        return {"acc": compute_accuracy(preds, labels, task_main, task_name)}
    elif task_name == "rte":
        return {"acc": compute_accuracy(preds, labels, task_main, task_name)}
    elif task_name == "wnli":
        return {"acc": compute_accuracy(preds, labels, task_main, task_name)}
    elif task_name[:3] == "mtl":
        return {"acc": compute_accuracy(preds, labels, task_main, task_name)}
    else:
        raise KeyError(task_name)

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "yelp-2": Yelp2Processor,
    "yelp-5": Yelp5Processor,
    "amazon-2": Yelp2Processor,
    "amazon-5": Yelp5Processor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "mtl": "classification",
    "yelp-2": "classification",
    "yelp-5": "classification",
    "amazon-2": "classification",
    "amazon-5": "classification",
    "dbpedia": "classification",
}

output_num = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 0,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "yelp-2": 2,
    "yelp-5": 5,
    "amazon-2": 2,
    "amazon-5": 5,
    "dbpedia": 14,
}


data_dirs = {
    "cola": './datasets/glue_data/CoLA/',
    "mnli": './datasets/glue_data/MNLI/',
    "mrpc": './datasets/glue_data/MRPC/',
    "sst-2": './datasets/glue_data/SST-2/',
    "sts-b": './datasets/glue_data/STS-B/',
    "qqp": './datasets/glue_data/QQP/',
    "qnli": './datasets/glue_data/QNLI/',
    "rte": './datasets/glue_data/RTE/',
    "wnli": './datasets/glue_data/WNLI/',
    "yelp-2": './datasets/yelp-2/yelp_review_polarity_csv/',
    "yelp-5": './datasets/yelp-5/yelp_review_full_csv/',
    "amazon-2": './datasets/amazon-2/amazon_review_polarity_csv/',
    "amazon-5": './datasets/amazon-5/amazon_review_full_csv/',
    "dbpedia": './datasets/dbpedia/dbpedia_csv/',
}

mtl_root_path = './datasets/mtl-dataset/subdatasets/'
mtl_domains = ['apparel', 'dvd', 'kitchen_housewares', 'software', 'baby', 'electronics', 'magazines', 'sports_outdoors', 'books', 'health_personal_care', 'mr', 'toys_games', 'camera_photo', 'imdb', 'music', 'video']

for dom in mtl_domains:
    data_dirs['mtl-%s'%dom] = mtl_root_path + dom + '/'
    processors['mtl-%s'%dom] = MtlProcessor
    output_modes['mtl-%s'%dom] = 'classification'
    output_num['mtl-%s'%dom] = 2




def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def three2two_accuracy(preds, labels):
    # 3: contradiction, entailment, neutral
    # 2: entailment, non_entailment
    mapped_preds = (preds!=1).astype(int)
    return (mapped_preds == labels).mean()


def two2three_accuracy(preds, labels):
    # 3: contradiction, entailment, neutral
    # 2: entailment, non_entailment
    mapped_labels = (labels!=1).astype(int)
    return (preds == mapped_labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def get_variable(inputs, no_cuda=True, **kwargs):
    cuda = not no_cuda
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        #out = Variable(inputs.cuda(), **kwargs)
        out = inputs.cuda()
    else:
        #out = Variable(inputs, **kwargs)
        out = inputs
    return out


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def all_same(items):
    return all(x == items[0] for x in items)


def load_glove_embs(file_path, vocab):
    """
    Load GloVe embeddings. Doing a random normal initialization for OOV words.
    """
    n = len(vocab)
    m = 300
    emb = np.empty((n, m), dtype=np.float32)

    emb[:, :] = np.random.normal(size=(n, m))

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0:1, :] = np.zeros((1, m), dtype="float32")

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):

            s = line.split()
            if s[0] in vocab:
                emb[vocab[s[0]], :] = np.asarray(s[-m:])

    return emb


def compare_preds(patha, pathb, path_report = './report_compare.txt'):
    import json

    win_a, win_b = [], []
    both_correct, both_wrong = [], []

    preds_a = []
    with open(patha, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            preds_a.append(json.loads(line))

    preds_b = []
    with open(pathb, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            preds_b.append(json.loads(line))

    assert(len(preds_a) == len(preds_b))
    preds_a = sorted(preds_a, key=lambda x:x['guid'])
    preds_b = sorted(preds_b, key=lambda x:x['guid'])

    for i in range(len(preds_a)):
        assert(preds_a[i]['guid'] == preds_b[i]['guid'])
        if preds_a[i]['pred'] == preds_b[i]['pred']:
            if preds_a[i]['pred'] == int(preds_a[i]['label']):
                both_correct.append(preds_a[i])
            else:
                both_wrong.append(preds_a[i])
        elif preds_a[i]['pred'] == int(preds_a[i]['label']):
            win_a.append(preds_a[i])
        elif preds_b[i]['pred'] == int(preds_a[i]['label']):
            win_b.append(preds_b[i])
        else:
            print(preds_a[i]['pred'])
            print(preds_b[i]['pred'])
            print(preds_a[i]['label'])
            exit()


    with open(path_report, 'w') as fw:
        fw.write('========================= Winner: %s ===========================\n'%patha)
        for example in win_a:
            fw.write(str(example['label']) + '\t' + str(example['text_a']) + '\n')

        fw.write('========================= Winner: %s ===========================\n'%pathb)
        for example in win_b:
            fw.write(str(example['label']) + '\t' + str(example['text_a']) + '\n')


        fw.write('========================= Both Correct: %s ===========================\n'%patha)
        for example in both_correct:
            fw.write(str(example['label']) + '\t' + str(example['text_a']) + '\n')

        fw.write('========================= Both Wrong: %s ===========================\n'%pathb)
        for example in both_wrong:
            fw.write(str(example['label']) + '\t' + str(example['text_a']) + '\n')


def prepare4translation(raw_file, quotechar=None, delimiter='\t', tokenize_sens = True):
    import csv
    import nltk
    from nltk.tokenize import sent_tokenize
    with open(raw_file, 'r', encoding='utf-8') as fr:
        reader = csv.reader(fr, delimiter=delimiter, quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)

    trans_source_file = raw_file + '.source'
    with open(trans_source_file, 'w') as fw:
        for line in lines:
            if tokenize_sens:
                sens = sent_tokenize(line[1])
                for sen in sens:
                    fw.write(sen+'\n')
                fw.write('\n')
            else:
                fw.write(line[1] + '\n')
    return trans_source_file



def extract_source_from_middle(middle_file):
    import csv
    with open(middle_file, 'r', encoding='utf-8') as fr:
        reader = csv.reader(fr, delimiter='\t', quotechar=None)
        lines = []
        for line in reader:
            lines.append(line)

    back_source_file = middle_file + '.back_source'
    with open(back_source_file, 'w') as fw:
        for line in lines:
            if line[0][0] == 'S':
                continue
            fw.write(line[-1] + '\n')

    return back_source_file


def clean_back_trans_results(final_file):
    import csv
    with open(final_file, 'r', encoding='utf-8') as fr:
        reader = csv.reader(fr, delimiter='\t', quotechar=None)
        lines = []
        for line in reader:
            lines.append(line)

    cleaned_file = final_file + '.cleaned'
    with open(cleaned_file, 'w') as fw:
        for line in lines:
            if line[0][0] == 'S':
                continue
            fw.write(line[-1] + '\n')

    return cleaned_file





if __name__ == '__main__':
    #patha = "./outputs/bert_mtl_toys_games_100p/pred_results_on_mtl-toys_games.txt"
    #pathb = "./outputs/bert_mtl_toys_games_10p/pred_results_on_mtl-toys_games.txt"

    #compare_preds(patha, pathb)

    #prepare4translation(raw_file='./datasets/mtl-dataset/subdatasets/apparel/train.tsv')
    extract_source_from_middle(middle_file='../ref_repos/paraphrase_translation/middle.out')

