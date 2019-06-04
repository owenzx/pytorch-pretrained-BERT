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
