import sys
import os
import argparse
from coref_metric_helper import convert_pred_to_conll_format
from h2s_helper import get_h2s_file, h2s2conll




def main_test(opt):
    head_pred_path = os.path.join(opt.model_path, 'head_pred.out')
    head_conll_file = os.path.join(opt.model_path, 'head_pred.conll')

    h2s_test_file = os.path.join(opt.model_path, 'head_pred.h2stest')

    h2s_pred_path = os.path.join(opt.h2s_path, 'pred.out')
    head_span_conll_file = os.path.join(opt.model_path, 'head_pred.span.conll')
    h2s2conll(h2s_pred_path, head_conll_file, head_span_conll_file)
    conll_eval_command = """python scorer.py {0} {1}""".format(opt.test_set, head_span_conll_file)
    os.system(conll_eval_command)

def main(opt):
    # Step 1 Get prediction on testing set

    head_pred_path = os.path.join(opt.model_path, 'head_pred.out')

    #TODO only for debug
    #pred_command = """allennlp predict --cuda-device 0  --overrides='{{"validation_dataset_reader":{{"truncation":true}}}}' --use-dataset-reader --predictor coreference-resolution --silent --include-package new_allen_packages --output-file {0} {1} {2}""".format(head_pred_path , opt.model_path, opt.head_test_set)
    pred_command = """allennlp predict --cuda-device 0 --use-dataset-reader --predictor coreference-resolution --silent --include-package new_allen_packages --output-file {0} {1} {2}""".format(head_pred_path , opt.model_path, opt.head_test_set)

    os.system(pred_command)



    # Step 2 Map the prediction file to h2s format
    head_conll_file = os.path.join(opt.model_path, 'head_pred.conll')
    convert_pred_to_conll_format(head_pred_path, opt.test_set, head_conll_file)

    h2s_test_file = os.path.join(opt.model_path, 'head_pred.h2stest')
    #TODO see if set max span to itself is right
    get_h2s_file(max_span_file=head_conll_file, min_span_file=head_conll_file, output_file=h2s_test_file)



    # Step 3 Get h2s results

    h2s_pred_path = os.path.join(opt.h2s_path, 'pred.out')

    h2s_pred_command = """allennlp predict --cuda-device 0 --use-dataset-reader --batch-size 40 --silent --include-package new_allen_packages --output-file {0} {1} {2}""".format(h2s_pred_path, opt.h2s_path, h2s_test_file)

    os.system(h2s_pred_command)

    # Step 4 Convert h2s results back to conll format
    head_span_conll_file = os.path.join(opt.model_path, 'head_pred.span.conll')
    h2s2conll(h2s_pred_path, head_conll_file, head_span_conll_file)


    # Step 5 Compare conll results using coval toolkit

    conll_eval_command = """python scorer.py {0} {1}""".format(opt.test_set, head_span_conll_file)
    os.system(conll_eval_command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-model_path', type=str, help='the model output path')
    parser.add_argument('--head_test_set', '-head_test_set', type=str, help='the testing set for coref with head labels')
    parser.add_argument('--test_set', '-test_set', type=str, help='the testing set for coref with ground truth labels')
    parser.add_argument('--h2s_path', '-h2s_path', type=str, help='the h2s model path')
    opt = parser.parse_args()
    main(opt)
