import os
import sys
import random


CONLL_SET_PATH = '/playpen/home/xzh/datasets/coref/allen/'
WIKICOREF_PATH = '/playpen/home/xzh/datasets/WikiCoref/'
BIASCOREF_PATH = '/ssd-playpen/home/xzh/datasets/WinoBias/wino/data/conll_format/'



def get_bias_test_command(comm, final_output_path, model_path=None):
    switch_type = 'simple'

    # Part 0 parse information from training command
    if model_path is None:
        main_comm = comm['main']
        comm_parts = main_comm.split(' ')

        model_path = comm_parts[comm_parts.index('-s') + 1]

    randint = str(int(random.random() * 1000000000000000000))

    dev_output_path = "./.ats_dev_output" + randint
    bias_dev_output_path_1a = "./.ats_biasdev_output_1a" + randint
    bias_dev_output_path_1p = "./.ats_biasdev_output_1p" + randint
    bias_dev_output_path_2a = "./.ats_biasdev_output_2a" + randint
    bias_dev_output_path_2p = "./.ats_biasdev_output_2p" + randint

    dev_set_path = CONLL_SET_PATH + "dev.english.v4_gold_conll"
    bias_dev_path_1a = BIASCOREF_PATH + "dev_type1_anti_stereotype.v4_auto_conll"
    bias_dev_path_1p = BIASCOREF_PATH + "dev_type1_pro_stereotype.v4_auto_conll"
    bias_dev_path_2a = BIASCOREF_PATH + "dev_type2_anti_stereotype.v4_auto_conll"
    bias_dev_path_2p = BIASCOREF_PATH + "dev_type2_pro_stereotype.v4_auto_conll"

    light_model_config_str = '"model":{"consistency_loss":false,"semi_supervise":false}'

    test_command = ""

    # Part I testing on the dev/test set
    test_command_template = """allennlp evaluate --cuda-device 0 --overrides='{{"dataset_reader":{{"cached_instance_path":null}},{0}}}' --include-package allen_packages --output-file {1} {2} {3}"""
    # {0} model config, {1} output_dir, {2} model path {3} test set

    test_command_1 = test_command_template.format(light_model_config_str, dev_output_path, model_path,
                                                  dev_set_path) + '\n'
    test_command += test_command_1

    test_command_1 = test_command_template.format(light_model_config_str, dev_output_path, model_path,
                                                  dev_set_path) + '\n'
    test_command += test_command_1

    test_command_1 = test_command_template.format(light_model_config_str, bias_dev_output_path_1a, model_path, bias_dev_path_1a) + '\n'
    test_command += test_command_1

    test_command_1 = test_command_template.format(light_model_config_str, bias_dev_output_path_1p, model_path, bias_dev_path_1p) + '\n'
    test_command += test_command_1

    test_command_1 = test_command_template.format(light_model_config_str, bias_dev_output_path_2a, model_path, bias_dev_path_2a) + '\n'
    test_command += test_command_1

    test_command_1 = test_command_template.format(light_model_config_str, bias_dev_output_path_2p, model_path, bias_dev_path_2p) + '\n'
    test_command += test_command_1


    # Part IV Merge testing results
    merge_command_template = """tail -n +1 {0} {1} {2} {3} {4}> {5}"""
    merge_comand = merge_command_template.format(dev_output_path, bias_dev_output_path_1a, bias_dev_output_path_1p, bias_dev_output_path_2a, bias_dev_output_path_2p,
                                                 final_output_path) + '\n'
    test_command += merge_comand

    return test_command


def get_test_command_from_train_simple_switch_only(comm, final_output_path, model_path=None, switch_type=None):
    assert(switch_type == 'only_simple')
    switch_type = 'simple'

    # Part 0 parse information from training command
    if model_path is None:
        main_comm = comm['main']
        comm_parts = main_comm.split(' ')

        model_path= comm_parts[comm_parts.index('-s')+1]


    randint = str(int(random.random()*1000000000000000000))

    dev_output_path = "./.ats_dev_output" + randint
    switch_output_path = "./.ats_switch_output" + randint
    switch_output_pred_path = "./.ats_switch_output_pred" + randint

    dev_pred_path = './.ats_dev_pred.txt' + randint
    switch_mention_cache_path = './cache/.ats_dev_switch' + randint

    switch_pag_output_path = "./.ats_switch_pag_output" + randint
    switch_pag_output_pred_path = "./.ats_switch_pag_output_pred" + randint
    switch_pag_mention_cache_path = './cache/.ats_dev_switch_pag' + randint



    dev_set_path = CONLL_SET_PATH + "dev.english.v4_gold_conll"


    light_model_config_str = '"model":{"consistency_loss":false,"semi_supervise":false}'


    test_command = ""

    # Part I testing on the dev/test set
    test_command_template = """allennlp evaluate --cuda-device 0 --overrides='{{"dataset_reader":{{"cached_instance_path":null}},{0}}}' --include-package allen_packages --output-file {1} {2} {3}"""
    # {0} model config, {1} output_dir, {2} model path {3} test set


    test_command_1 = test_command_template.format(light_model_config_str, dev_output_path, model_path, dev_set_path) + '\n'
    test_command += test_command_1

    # Part III check mention switch attack performance

    pred_dev_command = """allennlp predict --overrides='{{ {0} }}' --use-dataset-reader --predictor coreference-resolution --silent --cuda-device 0 --include-package allen_packages --output-file {1} {2} {3}""".format(light_model_config_str, dev_pred_path, model_path, dev_set_path)
    test_command += (pred_dev_command + '\n')

    gen_switched_data_command = "python coref_adv.py --mode switch_mention_pred --mentions_path ./cache/conll_dev_mentions.dict --pred_path {0} --output_path {1} --switch_type {2}".format(dev_pred_path, switch_mention_cache_path, switch_type)
    test_command += (gen_switched_data_command + '\n')

    f1_change_test_command = """allennlp evaluate --cuda-device 0 --overrides='{{"dataset_reader":{{"cached_instance_path":"{0}.ist"}}, {1} }}' --include-package allen_packages --output-file {2} {3} {4}""".format(switch_mention_cache_path,light_model_config_str,switch_output_path, model_path, dev_set_path) # here the dev_set_path is not used

    test_command += (f1_change_test_command + '\n')



    f1_change_predict_command = """allennlp predict --use-dataset-reader --predictor coreference-resolution --silent --cuda-device 0  --overrides='{{"dataset_reader":{{"cached_instance_path":"{0}.ist"}}, {1} }}' --include-package allen_packages --output-file {2} {3} {4}""".format(switch_mention_cache_path, light_model_config_str, switch_output_pred_path, model_path, dev_set_path) #dev_set_path is not used

    test_command += (f1_change_predict_command + '\n')

    # pred as golden analysis
    gen_pag_switched_data_command = "python coref_adv.py --mode switch_mention_pred --mentions_path ./cache/conll_dev_mentions.dict --pred_path {0} --output_path {1} --switch_type {2} --pred_as_golden".format(dev_pred_path, switch_pag_mention_cache_path, switch_type)
    test_command += (gen_pag_switched_data_command + '\n')

    f1_pag_change_test_command = """allennlp evaluate --cuda-device 0 --overrides='{{"dataset_reader":{{"cached_instance_path":"{0}.ist"}}, {1} }}' --include-package allen_packages --output-file {2} {3} {4}""".format(switch_pag_mention_cache_path, light_model_config_str, switch_pag_output_path, model_path, dev_set_path) # here the dev_set_path is not used

    test_command += (f1_pag_change_test_command + '\n')


    f1_pag_change_predict_command = """allennlp predict --use-dataset-reader --predictor coreference-resolution --silent --cuda-device 0  --overrides='{{"dataset_reader":{{"cached_instance_path":"{0}.ist"}}, {1} }}' --include-package allen_packages --output-file {2} {3} {4}""".format(switch_pag_mention_cache_path, light_model_config_str, switch_pag_output_pred_path, model_path, dev_set_path) #dev_set_path is not used

    test_command += (f1_pag_change_predict_command + '\n')


    # Part IV Merge testing results
    merge_command_template = """tail -n +1 {0} {1} {2}> {3}"""
    merge_comand = merge_command_template.format(dev_output_path, switch_output_path, switch_pag_output_path, final_output_path) + '\n'
    test_command += merge_comand

    return test_command

def get_test_command_from_train(comm, final_output_path, model_path=None, switch_type='simple'):

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
    switch_output_pred_path = "./.ats_switch_output_pred" + randint

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
    gen_switched_data_command = "python coref_adv.py --mode switch_mention_pred --mentions_path ./cache/conll_dev_mentions.dict --pred_path {0} --output_path {1} --switch_type {2}".format(dev_pred_path, switch_mention_cache_path, switch_type)
    test_command += (gen_switched_data_command + '\n')

    f1_change_test_command = """allennlp evaluate --cuda-device 0 --overrides='{{"dataset_reader":{{"cached_instance_path":"{0}.ist"}}}}' --include-package allen_packages --output-file {1} {2} {3}""".format(switch_mention_cache_path,switch_output_path, model_path, dev_set_path) # here the dev_set_path is not used

    test_command += (f1_change_test_command + '\n')


    f1_change_predict_command = """allennlp predict --use-dataset-reader --predictor coreference-resolution --silent --cuda-device 0  --overrides='{{"dataset_reader":{{"cached_instance_path":"{}.ist"}}}}' --include-package allen_packages --output-file {} {} {}""".format(switch_mention_cache_path, switch_output_pred_path, model_path, dev_set_path) #dev_set_path is not used

    test_command += (f1_change_predict_command + '\n')



    # Part IV Merge testing results
    merge_command_template = """tail -n +1 {0} {1} {2} {3} > {4}"""
    merge_comand = merge_command_template.format(dev_output_path, test_output_path, gena_output_path, switch_output_path, final_output_path) + '\n'
    test_command += merge_comand

    return test_command




def main():
    switch_type = sys.argv[1]
    train_file = sys.argv[2]
    final_output_path = sys.argv[3]
    assert switch_type in ['simple', 'switch_pron', 'add_clause', 'glove_close', 'glove_mention', 'full_switch', 'only_simple', 'bias']
    if train_file[-3:] == '.sh':
        with open(train_file, 'r') as fr:
            train_command_lines = fr.readlines()

        main_command = ""
        for line in train_command_lines:
            if line.split()[:2] == ['allennlp', 'train']:
                main_command = line


        commands = {'main': main_command,
                    'all': train_command_lines}

        if switch_type == 'full_switch':
            raise NotImplementedError
        elif switch_type == 'bias':
            test_command = get_bias_test_command(commands, final_output_path, None)
        elif switch_type == 'only_simple':
            test_command = get_test_command_from_train_simple_switch_only(commands, final_output_path, None, switch_type)
        else:
            test_command = get_test_command_from_train(commands, final_output_path, None, switch_type)

    else:
        if switch_type == 'full_switch':
            raise NotImplementedError
        elif switch_type == 'bias':
            test_command = get_bias_test_command(None, final_output_path, train_file)
        elif switch_type == 'only_simple':
            test_command = get_test_command_from_train_simple_switch_only(None, final_output_path, train_file, switch_type)
        else:
            test_command = get_test_command_from_train(None, final_output_path, train_file, switch_type)

    print(test_command)

    os.system(test_command)








if __name__ == '__main__':
    main()