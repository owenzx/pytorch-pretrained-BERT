from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allen_packages.coref_model import MyCoreferenceResolver

#from mention_tree_gen.data_processing_tree import load_text_from_dataset

from . data_processing_tree import load_text_from_dataset



attackee_path = './outputs/mentionswitch_really_baseline_0910'
dev_set_path = '/playpen/home/xzh/datasets/coref/allen/debug.english.v4_gold_conll'


archive = load_archive(attackee_path)
attackee = Predictor.from_archive(archive, 'coreference-resolution')



dev_examples = load_text_from_dataset(dev_set_path)
dev_examples = dev_examples[:2]

instance = attackee._dataset_reader.text_to_instance(dev_examples[1]['text'], dev_examples[1]['clusters'], dev_examples[1]['sen_id'])

#tmp = attackee.predict_instance(dev_examples[0])
tmp = attackee.predict_instance(instance)
metrics = attackee._model.get_metrics(True)

print(tmp)
#print(metrics)



