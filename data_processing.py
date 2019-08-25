import csv
import os
import sys
#import logging
import pickle

import numpy as np

#logger = logging.getLogger(__name__)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()




def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, logger):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

class CorefExample(object):
    def __init__(self, guid, text_a, text_tokens, span1_l, span1_r, span2_l, span2_r, label):
        self.guid = guid
        self.text_a = text_a
        self.text_tokens = text_tokens
        self.span1_l = span1_l
        self.span1_r = span1_r
        self.span2_l = span2_l
        self.span2_r = span2_r
        self.label = label

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.label = label

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id



def basicSubsample(examples, data_portion):
    if data_portion == 1.0:
        return examples
    examples = np.array(examples)
    examples = examples[np.random.choice(len(examples), int(len(examples) * data_portion), replace=False)]
    examples = examples.tolist()
    return examples


def balanceSubsample(examples, num_per_label):
    labels_dict = {}
    for i, example in enumerate(examples):
        if example.label not in labels_dict:
            labels_dict[example.label] = [i]
        else:
            labels_dict[example.label].append(i)
    selected_idx = []
    for l in labels_dict.keys():
        labels_dict[l] = np.array(labels_dict[l])
        labels_dict[l] = labels_dict[l][np.random.choice(len(labels_dict[l]), num_per_label, replace=False)]
        labels_dict[l] = labels_dict[l].tolist()
        selected_idx += labels_dict[l]
    selected_idx = np.array(selected_idx)
    examples = np.array(examples)
    examples = examples[selected_idx].tolist()
    return examples

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, output_dir=None):
        self.output_dir = output_dir

    def get_augmented_train_examples(self, aug_path):
        with open(aug_path, 'rb') as fr:
            aug_examples = pickle.load(fr)
        return aug_examples



    def get_train_examples(self, data_dir, data_portion, num_per_label):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, delimiter="\t"):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class Yelp2Processor(DataProcessor):
    """Processor for the Yelp data set."""

    def get_train_examples(self, data_dir, data_portion, num_per_label):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv"), quotechar='"', delimiter=","), "train", data_portion, num_per_label)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.csv"), quotechar='"', delimiter=","), "dev")

    def get_val_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv.val_train"), quotechar='"', delimiter=","), "val_train")

    def get_labels(self):
        """See base class."""
        return ["1", "2"]

    def _create_examples(self, lines, set_type, data_portion=1.0, num_per_label=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[-1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        if num_per_label<0:
            examples = basicSubsample(examples, data_portion)
        else:
            examples = balanceSubsample(examples, num_per_label)
        if set_type == 'train' and (data_portion!=1.0 or num_per_label!=-1):
            self._save_subsampled_examples(examples, lines, data_portion, num_per_label)
        return examples

    def _save_subsampled_examples(self, examples, lines, data_portion, num_per_label):
        selected_lineids = [int(example.guid.split('-')[-1]) for example in examples]

        selected_lines = [lines[i] for i in selected_lineids]
        if data_portion != 1.0:
            subsample_file = os.path.join(self.output_dir, "subsampled_{0}.txt".format(data_portion))
            subsample_pkl_file = os.path.join(self.output_dir, "subsampled_{0}.pkl".format(data_portion))
        else:
            subsample_file = os.path.join(self.output_dir, "subsampled_{0}.txt".format(num_per_label))
            subsample_pkl_file = os.path.join(self.output_dir, "subsampled_{0}.pkl".format(num_per_label))

        with open(subsample_pkl_file, 'wb') as fw:
            pickle.dump(examples, fw)

        with open(subsample_file, 'w') as fw:
            for line in selected_lines:
                fw.write('\t'.join(line))
                fw.write('\n')


class Yelp5Processor(Yelp2Processor):
    def get_labels(self):
        return ["1", "2", "3", "4", "5"]


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, data_portion, num_per_label):
        """See base class."""
        #logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_portion, num_per_label)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, data_portion=1.0, num_per_label=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if num_per_label<0:
            examples = basicSubsample(examples, data_portion)
        else:
            examples = balanceSubsample(examples, num_per_label)
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, data_portion, num_per_label):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_portion, num_per_label)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type, data_portion=1.0, num_per_label=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if num_per_label<0:
            examples = basicSubsample(examples, data_portion)
        else:
            examples = balanceSubsample(examples, num_per_label)
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir, data_portion, num_per_label):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_portion, num_per_label)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, data_portion=1.0, num_per_label=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        if num_per_label<0:
            examples = basicSubsample(examples, data_portion)
        else:
            examples = balanceSubsample(examples, num_per_label)
        return examples


class MtlProcessor(DataProcessor):
    """Processor for the mtl-dataset for sentence classification"""
    def get_train_examples(self, data_dir, data_portion, num_per_label):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_portion, num_per_label)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_val_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv.val_train")), "val_train")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, data_portion=1.0, num_per_label=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        if num_per_label<0:
            examples = basicSubsample(examples, data_portion)
        else:
            examples = balanceSubsample(examples, num_per_label)
        if set_type == 'train' and data_portion!=1.0:
            self._save_subsampled_examples(examples, lines, data_portion)
        return examples

    def _save_subsampled_examples(self, examples, lines, data_portion):
        selected_lineids = [int(example.guid.split('-')[-1]) for example in examples]

        selected_lines = [lines[i] for i in selected_lineids]

        subsample_file = os.path.join(self.output_dir, "subsampled_{0}.txt".format(data_portion))
        subsample_pkl_file = os.path.join(self.output_dir, "subsampled_{0}.pkl".format(data_portion))

        with open(subsample_pkl_file, 'wb') as fw:
            pickle.dump(examples, fw)

        with open(subsample_file, 'w') as fw:
            for line in selected_lines:
                fw.write('\t'.join(line))
                fw.write('\n')




class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir, data_portion, num_per_label):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_portion, num_per_label)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, data_portion=1.0, num_per_label=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        if num_per_label<0:
            examples = basicSubsample(examples, data_portion)
        else:
            examples = balanceSubsample(examples, num_per_label)
        if set_type == 'train' and data_portion!=1.0:
            self._save_subsampled_examples(examples, lines, data_portion)
        return examples

    def _save_subsampled_examples(self, examples, lines, data_portion):
        selected_lineids = [int(example.guid.split('-')[-1]) for example in examples]

        selected_lines = [lines[i] for i in selected_lineids]

        subsample_file = os.path.join(self.output_dir, "subsampled_{0}.txt".format(data_portion))
        subsample_pkl_file = os.path.join(self.output_dir, "subsampled_{0}.pkl".format(data_portion))

        with open(subsample_pkl_file, 'wb') as fw:
            pickle.dump(examples, fw)

        with open(subsample_file, 'w') as fw:
            for line in selected_lines:
                fw.write('\t'.join(line))
                fw.write('\n')






class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir, data_portion, num_per_label):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_portion, num_per_label)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type, data_portion=1.0, num_per_label=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if num_per_label<0:
            examples = basicSubsample(examples, data_portion)
        else:
            examples = balanceSubsample(examples, num_per_label)
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir, data_portion, num_per_label):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_portion, num_per_label)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, data_portion=1.0, num_per_label=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if num_per_label<0:
            examples = basicSubsample(examples, data_portion)
        else:
            examples = balanceSubsample(examples, num_per_label)
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, data_portion, num_per_label):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_portion, num_per_label)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type, data_portion=1.0, num_per_label=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if num_per_label<0:
            examples = basicSubsample(examples, data_portion)
        else:
            examples = balanceSubsample(examples, num_per_label)
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir, data_portion, num_per_label):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_portion, num_per_label)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type, data_portion=1.0, num_per_label=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if num_per_label<0:
            examples = basicSubsample(examples, data_portion)
        else:
            examples = balanceSubsample(examples, num_per_label)
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, data_portion, num_per_label):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_portion, num_per_label)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, data_portion=1.0, num_per_label=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if num_per_label<0:
            examples = basicSubsample(examples, data_portion)
        else:
            examples = balanceSubsample(examples, num_per_label)
        return examples

