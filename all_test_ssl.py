import os
import sys
import random


CONLL_SET_PATH = '/playpen/home/xzh/datasets/coref/allen/'
WIKICOREF_PATH = '/playpen/home/xzh/datasets/WikiCoref/'


def get_test_command_from_train(comm, final_output_path, model_path=None):

    # Part 0 parse information from training command
    if model_path is None:
        main_comm = comm['main']
        comm_parts = main_comm.split(' ')

        model_path= comm_parts[comm_parts.index('-s')+1]


    randint = str(int(random.random()*1000000000000000000))

    dev_output_path = "./.ats_dev_output" + randint
    test_output_path = "./.ats_test_output" + randint
    gena_output_path = "./.ats_gena_output" + randint
    switch_output_path = "./.ats_switch_output" + randint

    dev_pred_path = './.ats_dev_pred.txt' + randint
    switch_mention_cache_path = './cache/.ats_dev_switch' + randint


    dev_set_path = CONLL_SET_PATH + "dev.english.v4_gold_conll"
    test_set_path = CONLL_SET_PATH + 'test.english.v4_gold_conll'
    gena_set_path = WIKICOREF_PATH + 'Evaluation/key-OntoNotesScheme'




    test_command = ""

    # Part I testing on the dev/test set
    test_command_template = """allennlp evaluate --cuda-device 0 --overrides='{{"dataset_reader":{{"cached_instance_path":null}}}}' --include-package allen_packages --output-file {0} {1} {2}"""
    # {0} output_dir, {1} model path {2} test set


    test_command_1 = test_command_template.format(dev_output_path, model_path, dev_set_path) + '\n'
    test_command += test_command_1

    test_command_2 = test_command_template.format(test_output_path, model_path, test_set_path) + '\n'
    test_command += test_command_2

    # Part II testing on wiki testing set
    test_command_1 = test_command_template.format(gena_output_path, model_path, gena_set_path) + '\n'
    test_command += test_command_1

    # Part III check mention switch attack performance

    pred_dev_command = """allennlp predict --use-dataset-reader --predictor coreference-resolution --silent --cuda-device 0 --include-package allen_packages --output-file {} {} {}""".format(dev_pred_path, model_path, dev_set_path)
    test_command += (pred_dev_command + '\n')

    #TODO add option for switch type
    #gen_switched_data_command = "python coref_adv.py --mode switch_mention_pred --mentions_path ./cache/conll_dev_mentions.dict --pred_path {0} --output_path {1} --switch_type simple".format(dev_pred_path, switch_mention_cache_path)
    gen_switched_data_command = "python coref_adv.py --mode switch_mention_pred --mentions_path ./cache/conll_dev_mentions.dict --pred_path {0} --output_path {1} --switch_type glove_mention".format(dev_pred_path, switch_mention_cache_path)
    test_command += (gen_switched_data_command + '\n')

    f1_change_test_command = """allennlp evaluate --cuda-device 0 --overrides='{{"dataset_reader":{{"cached_instance_path":"{0}.ist"}}}}' --include-package allen_packages --output-file {1} {2} {3}""".format(switch_mention_cache_path,switch_output_path, model_path, dev_set_path) # here the dev_set_path is not used
    test_command += (f1_change_test_command + '\n')


    # Part IV Merge testing results
    merge_command_template = """tail -n +1 {0} {1} {2} {3} > {4}"""
    merge_comand = merge_command_template.format(dev_output_path, test_output_path, gena_output_path, switch_output_path, final_output_path) + '\n'
    test_command += merge_comand

    return test_command




def main():
    train_file = sys.argv[1]
    final_output_path = sys.argv[2]
    if train_file[-3:] == '.sh':
        with open(train_file, 'r') as fr:
            train_command_lines = fr.readlines()

        main_command = ""
        for line in train_command_lines:
            if line.split()[:2] == ['allennlp', 'train']:
                main_command = line


        commands = {'main': main_command,
                    'all': train_command_lines}


        test_command = get_test_command_from_train(commands, final_output_path)
    else:
        test_command = get_test_command_from_train(None, final_output_path, train_file)

    print(test_command)

    os.system(test_command)








if __name__ == '__main__':
    main()