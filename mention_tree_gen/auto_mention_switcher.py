
from __future__ import absolute_import, division, print_function
import argparse
import random
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm
from . attacker_tree import AttackerTree

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.util import lazy_groups_of
from allennlp.data.vocabulary import Vocabulary
# Don't comment out the next line
from allen_packages.coref_model import MyCoreferenceResolver
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.iterators.bucket_iterator import BucketIterator

from . data_processing_tree import load_text_from_dataset, attacker2attackee_input, get_switched_ee_example
from .attacker_reader import AttackerCorefReader
from allennlp.nn.util import move_to_device
import sys
import logging
from gensim.models import KeyedVectors as Word2Vec
from coref_adv import get_sentence_vec

WEIGHTS_NAME = "pytorch_model.bin"
LOG_NAME = "stdout.log"
EXAMPLE_LOG_NAME = "examples.log"

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)




def is_overlap(l1, r1, l2, r2):
    if (r1 < l2) or (r2 < l1):
        return False
    else:
        return True


def get_link_f1(ee_results, target_span):
    gold_clusters = ee_results['gold_clusters']
    pred_clusters = ee_results['clusters']

    target_cluster = None
    for cluster in gold_clusters:
        for mention in cluster:
            if mention[0] == target_span[0] and mention[1] == target_span[1]:
                target_cluster = cluster
                break
    assert(target_cluster is not None)

    pred_t_cluster = None
    for cluster in pred_clusters:
        for mention in cluster:
            if mention[0] == target_span[0] and mention[1] == target_span[1]:
                pred_t_cluster = cluster
                break
    if pred_t_cluster is None:
        return 0, 0, 0

    #make everything tuple
    pred_t_cluster = set([tuple(m) for m in pred_t_cluster])
    target_cluster = set([tuple(m) for m in target_cluster])
    inter = pred_t_cluster.intersection(target_cluster)

    prec = len(inter) / len(pred_t_cluster)
    reca = len(inter) / len(target_cluster)
    f1 = 2 / ( (1 / prec) + (1 / reca) )
    return prec, reca, f1






def get_detection_f1(ee_results, target_span):
    gold_clusters = ee_results['gold_clusters']
    pred_clusters = ee_results['clusters']

    prec, reca, f1 = 0, 0, 0

    for cluster in pred_clusters:
        for mention in cluster:
            if is_overlap(mention[0], mention[1], target_span[0], target_span[1]):
                overlap_len = min(mention[1], target_span[1]) - max(mention[0], target_span[0]) + 1
                tmp_prec = overlap_len / (mention[1] - mention[0] + 1)
                tmp_reca = overlap_len / (target_span[1] - target_span[0] + 1)
                tmp_f1 = 2 / ( (1 / tmp_prec) + (1 / tmp_reca) )
                if tmp_f1 > f1:
                    prec, reca, f1 = tmp_prec, tmp_reca, tmp_f1

    return prec, reca, f1


def get_sem_reward(old_mention_text, new_mention_text, sem_type, w2v_model):
    if sem_type == 'glove_simple':
        old_mention_vec = get_sentence_vec(old_mention_text, w2v_model)
        new_mention_vec = get_sentence_vec(new_mention_text, w2v_model)
        #TODO check the scale of dist
        dist = np.linalg.norm(old_mention_vec - new_mention_vec)
        return dist
    else:
        raise NotImplementedError


def get_reward(gen_mention_cfg, gen_mention_text, model, attackee, old_metrics, old_example, old_ee_results, old_ee_inst, w2v_model, args):
    base_reward = 0
    old_target_text_span = old_example[0]['metadata'][0]['target_cluster'][0]
    old_target_span = (old_ee_inst['metadata']['start_idx_maps'][old_target_text_span[0]], old_ee_inst['metadata']['end_idx_maps'][old_target_text_span[1]])
    # TODO also fix new

    if gen_mention_text is None:
        base_reward = -100
    else:
        new_ee_example_dict, new_target_text_span = get_switched_ee_example(gen_mention_text, old_example[0]['metadata'][0])
        #new_ee_example_dict = er_example[0]['metadata'][0]

        attackee_new_inst = attackee._dataset_reader.text_to_instance(new_ee_example_dict['input_sentences'], new_ee_example_dict['input_gold_clusters'], new_ee_example_dict['input_sen_id'])
        new_target_span = (attackee_new_inst['metadata']['start_idx_maps'][new_target_text_span[0]], attackee_new_inst['metadata']['end_idx_maps'][new_target_text_span[1]])
        new_ee_results = attackee.predict_instance(attackee_new_inst)
        new_metrics = attackee._model.get_metrics(True)

        if args.reward_type == 'normal_f1':
            #TODO implement multiple kinds of rewards (Always try to maximize the reward)
            base_reward = old_metrics['coref_f1'] - new_metrics['coref_f1']
        elif args.reward_type == 'link_f1':
            _, _, old_link_f1 = get_link_f1(old_ee_results, old_target_span)
            _, _, new_link_f1 = get_link_f1(new_ee_results, new_target_span)
            base_reward = old_link_f1 - new_link_f1
        elif args.reward_type == 'detection_f1':
            _, _, old_detection_f1 = get_detection_f1(old_ee_results, old_target_span)
            _, _, new_detection_f1 = get_detection_f1(new_ee_results, new_target_span)
            base_reward = old_detection_f1 - new_detection_f1
        elif args.reward_type == 'link_detection':
            _, _, old_link_f1 = get_link_f1(old_ee_results, old_target_span)
            _, _, new_link_f1 = get_link_f1(new_ee_results, new_target_span)
            _, _, old_detection_f1 = get_detection_f1(old_ee_results, old_target_span)
            _, _, new_detection_f1 = get_detection_f1(new_ee_results, new_target_span)
            base_reward = args.lambda_linkf1 * (old_link_f1 - new_link_f1) + args.lambda_detectionf1 * (old_detection_f1 - new_detection_f1)


    if args.sem_reward != 'none':
        old_mention_span = old_example[0]['metadata'][0]['target_cluster'][0][0]
        old_mention_text = old_example[0]['metadata'][0]['original_text'][0][old_mention_span[0]:old_mention_span[1]+1]
        new_mention_text = gen_mention_text
        sem_reward = get_sem_reward(old_mention_text, new_mention_text, args.sem_reward, w2v_model)
        base_reward = base_reward + args.lambda_sem * sem_reward




    complexity_penalty = model.get_complexity_penalty(gen_mention_cfg)

    total_reward = base_reward + args.lambda_pen * complexity_penalty

    return total_reward




def train_model(model, attackee, train_examples, vocab, optimizer, args, logger, w2v_model, eval_examples =None):
    num_train_epochs = args.num_train_epochs

    train_stats = {}


    #Implementing batch training using gradient accumulation
    #train_iterator = BucketIterator(sorting_keys=[("text", "num_tokens")], padding_noise=0.0, batch_size=args.train_batch_size)
    train_iterator = BucketIterator(sorting_keys=[("text", "num_tokens")], padding_noise=0.0, batch_size=1)
    train_iterator.index_with(vocab)

    gradient_accumulation_step = args.train_batch_size


    avg_reward = None
    logger.info("***** Start training *****")
    logger.info("  Num epoch = %d", num_train_epochs)
    logger.info("  Batch size=%d", args.train_batch_size)
    logger.info("  Length penalty=%f", args.lambda_pen)
    for epoch_i in range(num_train_epochs):

        epoch_model_path = os.path.join(args.output_dir, "epoch%s.bin"%epoch_i)
        epoch_vocab_path = os.path.join(args.output_dir, "epoch%s.vocab"%epoch_i)
        with open(epoch_model_path, 'wb') as f:
            torch.save(model.controller.state_dict(), f)
        vocab.save_to_files(epoch_vocab_path)


        #model.train()
        model.controller.train()

        train_dataloader = train_iterator(train_examples, num_epochs=1, shuffle = True)
        train_dataloader = lazy_groups_of(train_dataloader, 1)

        for i, er_example in enumerate(tqdm(train_dataloader, desc="Training")):

            er_example = move_to_device(er_example, 0)

            attackee_old_inst = attackee._dataset_reader.text_to_instance(er_example[0]['metadata'][0]['input_sentences'], er_example[0]['metadata'][0]['input_gold_clusters'], er_example[0]['metadata'][0]['input_sen_id'])
            old_ee_results = attackee.predict_instance(attackee_old_inst)
            old_metrics = attackee._model.get_metrics(True)

            new_mention_cfgs = model.sample_new_mention(er_example)
            new_mention_text = model.get_new_mention_text(new_mention_cfgs, er_example[0]['metadata'][0]['original_text'], er_example[0]['metadata'][0]['np_head'])

            total_reward = get_reward(new_mention_cfgs, new_mention_text, model, attackee, old_metrics, er_example, old_ee_results, attackee_old_inst, args)


            # get baseline reward using beam search
            if args.baseline == 'beam_search':
                baseline_mention_cfgs = model.beam_search_new_mention(er_example)
                baseline_mention_text = model.get_new_mention_text(new_mention_cfgs, er_example[0]['metadata'][0]['original_text'], er_example[0]['metadata'][0]['np_head'])

                baseline_reward = get_reward(baseline_mention_cfgs, baseline_mention_text, model, attackee, old_metrics, er_example, old_ee_results, attackee_old_inst, w2v_model, args)
            else:
                baseline_reward = 0

            #total_reward = reward + args.lambda_pen * complexity_penalty

            final_reward = total_reward - baseline_reward
            if avg_reward is None:
                avg_reward = total_reward
            else:
                avg_reward = avg_reward * args.exp_avg_w + final_reward * (1 - args.exp_avg_w)
            #print("TOTAL REWARD: ")
            #print(total_reward)
            #print("FINAL REWARD: ")
            #print(final_reward)

            loss = model.train_controller_w_reward(er_example, new_mention_cfgs, final_reward)['loss']
            if gradient_accumulation_step is not None:
                loss = loss / gradient_accumulation_step
            loss.backward()
            if i % gradient_accumulation_step == 0:
                optimizer.step()


        if eval_examples is not None:
            eval_model(model, attackee, eval_examples, vocab, args, logger, epoch_num=epoch_i)
    return train_stats










def eval_model(model, attackee, eval_examples, vocab, args, logger, epoch_num=None):
    model.controller.eval()
    #model.eval()

    #eval_sampler  = SequentialSampler(eval_examples)
    #eval_dataloader = DataLoader(eval_examples, sampler = eval_sampler)

    assert(args.eval_batch_size == 1)
    # TODO batch evaluation

    eval_iterator = BucketIterator(sorting_keys=[("text", "num_tokens")], padding_noise=0.0, batch_size=args.eval_batch_size)
    eval_iterator.index_with(vocab)
    eval_dataloader = eval_iterator(eval_examples, num_epochs=1, shuffle = False)
    eval_dataloader = lazy_groups_of(eval_dataloader, 1)

    logger.info("***** Start Evaluating *****")
    if epoch_num is not None:
        logger.info("  Epoch num=%d", epoch_num)

    eval_metrics = {'f1_drop': 0, 'total_num': 0, 'new_f1': 0, 'fail_num': 0}
    for instance in tqdm(eval_dataloader, desc="Evaluating"):
        instance = move_to_device(instance, 0)

        with torch.no_grad():
            attackee_old_inst = attackee._dataset_reader.text_to_instance(instance[0]['metadata'][0]['input_sentences'], instance[0]['metadata'][0]['input_gold_clusters'], instance[0]['metadata'][0]['input_sen_id'])
            old_results = attackee.predict_instance(attackee_old_inst)

            old_metrics = attackee._model.get_metrics(True)

            #new_mention = model(...)
            new_mention_cfgs = model.sample_new_mention(instance)
            new_mention_text = model.get_new_mention_text(new_mention_cfgs, instance[0]['metadata'][0]['original_text'], instance[0]['metadata'][0]['np_head'])
            if new_mention_text is not None:
                new_ee_example_dict, _ = get_switched_ee_example(new_mention_text, instance[0]['metadata'][0])
                attackee_new_inst = attackee._dataset_reader.text_to_instance(new_ee_example_dict['input_sentences'], new_ee_example_dict['input_gold_clusters'], new_ee_example_dict['input_sen_id'])

                # FIXEDTODO apply some real transformation
                #attackee_new_inst = attackee_old_inst

                new_results = attackee.predict_instance(attackee_new_inst)

                new_metrics = attackee._model.get_metrics(True)

                eval_metrics['f1_drop'] += (new_metrics['coref_f1'] - old_metrics['coref_f1'])
                eval_metrics['new_f1'] += (new_metrics['coref_f1'])
                eval_metrics['total_num'] += 1
            else:
                eval_metrics['fail_num'] += 1

    if eval_metrics['total_num']!=0:
        eval_metrics['f1_drop'] /= eval_metrics['total_num']
        eval_metrics['new_f1'] /= eval_metrics['total_num']

    #if epoch_num is not None:
    #    print("EPOCH NO. " + str(epoch_num))
    #print(eval_metrics)
    logging.info(eval_metrics)
    get_samples_log(model, vocab, eval_examples, args, epoch_num=epoch_num, sample_num=10)

    return eval_metrics



def get_samples_log(model, vocab, eval_examples, args, epoch_num=None, sample_num=10):
    """Sample from the model and save some policies/text examples to log for later debugging"""
    sample_log_path = os.path.join(args.output_dir, EXAMPLE_LOG_NAME)

    eval_iterator = BucketIterator(sorting_keys=[("text", "num_tokens")], padding_noise=0.0, batch_size=args.eval_batch_size)
    eval_iterator.index_with(vocab)
    eval_dataloader = eval_iterator(eval_examples, num_epochs=1, shuffle = False)
    eval_dataloader = lazy_groups_of(eval_dataloader, 1)

    with open(sample_log_path, 'a') as fw:
        if epoch_num is not None:
            fw.write("EVAL EXAMPLES FOR EPOCH %d\n"%epoch_num)
        else:
            fw.write("EVAL EXAMPLES\n")

        for i, instance in enumerate(tqdm(eval_dataloader, desc="Logging samples")):
            if i > sample_num:
                break
            instance = move_to_device(instance, 0)
            with torch.no_grad():
                new_mention_cfgs = model.sample_new_mention(instance)
                new_mention_text = model.get_new_mention_text(new_mention_cfgs, instance[0]['metadata'][0]['original_text'], instance[0]['metadata'][0]['np_head'])
            fw.write("Example %d\n"%i)
            fw.write("CFGS: " + str(new_mention_cfgs) + '\n')
            fw.write("TEXT: " + str(new_mention_text) + '\n')
            fw.write("Context: " + str(instance[0]['metadata'][0]['original_text']) + '\n')
            fw.write('\n')







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_data_path",
                        default=None,
                        type=str,
                        help="The location of train file in Ontonotes fomrat")
    parser.add_argument("--eval_data_path",
                        default=None,
                        type=str,
                        help="The location of eval file in Ontonotes fomrat")

    parser.add_argument("--attackee_path",
                        default=None,
                        required=True,
                        type=str,
                        help="The location of the attackee model")


    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int)
    parser.add_argument("--eval_batch_size",
                       default=1,
                       type=int)
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float)
    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=int)

    parser.add_argument("--glove_path",
                        default="./datasets/glove.840B.300d.txt",
                        type=str,
                        help="The location of GloVe pre-trained embeddings")
    parser.add_argument("--glove_gensim_path",
                        default="./datasets/glove.840B.300d.w2vformat.txt",
                        type=str,
                        help="The location of GloVe pre-trained embeddings in gensim format")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--load_pretrained',
                        action='store_true',
                        help='Whether to load pretrained attacker model')
    parser.add_argument('--vocab_path',
                        type=str,
                        default=None,
                        help='path to save/load the allennlp vocab')
    parser.add_argument('--load_epoch_num',
                        type=int,
                        default=None,
                        help='determine which epoch to load')
    parser.add_argument('--beam_size',
                        type=int,
                        default=None,
                        help='beam size, set to None for greedy decoding')
    parser.add_argument('--max_decoding_steps',
                        type=int,
                        default=20,
                        help='max decoding steps')
    parser.add_argument('--exp_avg_w',
                        type=float,
                        default = 0.99,
                        help='the weight for exponential moving average')


    # Loss hyper-parameters
    parser.add_argument('--sem_reward',
                        type=str,
                        default='none',
                        help='determine which kinds of semantic validity reward to use')
    parser.add_argument('--lambda_sem',
                        type=float,
                        default=1.0,
                        help='weight for semantic reward')
    parser.add_argument('--reward_type',
                        type=str,
                        default='normal_f1',
                        help='determine which kinds of reward to use')
    parser.add_argument('--lambda_pen',
                        type=float,
                        default=1.0,
                        help='weight for complexity penalty, this value is positive and the penalty is negative')
    parser.add_argument('--lambda_linkf1',
                        type=float,
                        default=1.0,
                        help='weight for mention-link-f1')
    parser.add_argument('--lambda_detectionf1',
                        type=float,
                        default=1.0,
                        help='weight for mention-detection-f1')


    # Model structure hyper-parameteres
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=300)
    parser.add_argument('--encoder_dim',
                        type=int,
                        default=300)
    parser.add_argument('--cfg_dim',
                        type=int,
                        default=50)

    #optimization hyper-parameters
    parser.add_argument('--dropout',
                        type=float,
                        default=0.0)
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3)

    parser.add_argument('--len_pen',
                        type=float,
                        default=1.0)
    parser.add_argument('--baseline',
                        type=str,
                        default='beam_search')



    args = parser.parse_args()

    assert(args.do_train or args.do_eval)

    assert(args.reward_type in ['normal_f1', 'link_f1', 'detection_f1', 'link_detection_f1'])

    if 'glove' in args.sem_reward:
        w2v_model = Word2Vec.load_word2vec_format(args.glove_gensim_path)
    else:
        w2v_model = None

    if args.load_pretrained:
        assert(args.load_epoch_num is not None)


    vocab = None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.vocab_path is not None and os.path.exists(args.vocab_path):
        vocab = Vocabulary.from_files(args.vocab_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load attackee model
    #attackee = None
    archive = load_archive(args.attackee_path, cuda_device=0)
    attackee = Predictor.from_archive(archive, 'coreference-resolution')

    attacker_reader = AttackerCorefReader()

    log_file = os.path.join(args.output_dir, LOG_NAME)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level = logging.INFO,
                        filename=log_file,
                        filemode='a')


    if args.do_train:
        #Construct training data
        #train_examples_text = load_text_from_dataset(args.train_data_path)
        train_examples_attacker = attacker_reader.read(args.train_data_path)
        if vocab is None:
            vocab = Vocabulary.from_instances(train_examples_attacker)


    if args.do_eval:
        assert(vocab is not None)
        # Construct validation data
        #eval_examples = load_text_from_dataset(args.eval_data_path)
        eval_examples_attacker = attacker_reader.read(args.eval_data_path)

    # Define/Load model
    model = AttackerTree(args, vocab)
    model.controller.cuda()

    model_file = 'epoch%d.bin'% args.load_epoch_num
    model_save_path = os.path.join(args.output_dir, model_file)
    if os.path.exists(model_save_path):
        assert(args.load_pretrained)
        model.controller.load_state_dict(torch.load(model_save_path))


    if args.do_train:
        optimizer = torch.optim.Adam(model.controller.parameters(), lr=args.lr)
        train_model(model, attackee, train_examples_attacker, vocab, optimizer, args, logger, w2v_model, eval_examples_attacker)

    #Final Evaluation on dev/test sets
    if args.do_eval:
        eval_model(model, attackee, eval_examples_attacker, vocab, args, logger)





if __name__ == '__main__':
    main()