from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME

MRQA_FILE = './datasets/QA/dev_in/SQuAD.jsonl.gz'

import gzip,json

zip_handle = gzip.open(MRQA_FILE, 'rb')

header = zip_handle.readline()


for example in zip_handle:
    print(json.loads(example))
    print(json.loads(example).keys())
    exit()

print(zip_handle[0])