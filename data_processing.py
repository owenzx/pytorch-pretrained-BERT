import csv
import os
import sys
#import logging
import pickle

import numpy as np

#logger = logging.getLogger(__name__)

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

