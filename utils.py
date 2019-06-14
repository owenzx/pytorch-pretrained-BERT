import numpy as np

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




if __name__ == '__main__':
    patha = "./outputs/bert_mtl_toys_games_100p/pred_results_on_mtl-toys_games.txt"
    pathb = "./outputs/bert_mtl_toys_games_10p/pred_results_on_mtl-toys_games.txt"

    compare_preds(patha, pathb)
