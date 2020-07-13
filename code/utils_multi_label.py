# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
source code for APLC_XLNet

"""


from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
import numpy as np

import pickle
import torch

logger = logging.getLogger(__name__)


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


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:

                    if sys.version_info[0] >= 3:
                        unicode = str

                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MultiLabelTextProcessor(DataProcessor):


    def get_train_examples(self, data_dir, size=-1):
        filename = 'train.csv'
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, filename)))
        if size == -1:
            # data_df = pd.read_csv(os.path.join(data_dir, filename))
            data_df = pd.read_csv(os.path.join(data_dir, filename),na_filter= False)

            return self._create_examples(data_df, "train")
            # return self._create_examples(reader, "train")

        else:
            data_df = pd.read_csv(os.path.join(data_dir, filename))
            #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(data_df.sample(size), "train")

    def get_dev_examples(self, data_dir, size=-1):
        """See base class."""
        filename = 'dev.csv'
        if size == -1:
            # data_df = pd.read_csv(os.path.join(data_dir, filename))
            data_df = pd.read_csv(os.path.join(data_dir, filename),na_filter= False)

            return self._create_examples(data_df, "dev")
        else:
            data_df = pd.read_csv(os.path.join(data_dir, filename))
            #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(data_df.sample(size), "dev")

    def get_test_examples(self, data_dir, data_file_name, size=-1):
        data_df = pd.read_csv(os.path.join(data_dir, data_file_name))
        #         data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
        if size == -1:
            return self._create_examples(data_df, "test")
        else:
            return self._create_examples(data_df.sample(size), "test")

    def get_labels(self,data_dir):
        """See base class."""

        file_name = os.path.join(data_dir, "labels.txt")
        labels = []
        file = open(file_name, 'r')
        lines = file.readlines()
        file.close()
        for line in lines:
            label = line.strip()
            labels.append(label)

        return labels

    def _create_examples(self, df, set_type, labels_available=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, row) in enumerate(df.values):
        # for row in reader:

            guid = row[0]
            text_a = row[1]
            if labels_available:

                if row[2] is '':
                    labels = []
                else:
                    labels = row[2].split(',')
            else:
                labels = []
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=labels))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
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
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
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
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":

            label_id = [label_map[item] for item in example.label]
            #label_id = label_map[example.label]

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
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

            logger.info("label: %s (id = %s)" % (','.join(example.label),','.join([str(item) for item in label_id])))
            # logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


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


processors = {
    'rcv1': MultiLabelTextProcessor,
    'eurlex': MultiLabelTextProcessor,
    'wiki10': MultiLabelTextProcessor,
    'bibtex': MultiLabelTextProcessor,
    'delicious': MultiLabelTextProcessor,
    'wikilshtc': MultiLabelTextProcessor,
    'amazoncat': MultiLabelTextProcessor,
    'wiki500k': MultiLabelTextProcessor,
    'amazon670k': MultiLabelTextProcessor,
    'amazon3m': MultiLabelTextProcessor,

}

output_modes = {
    "rcv1": "classification",
    "eurlex": "classification",
    "wiki10": "classification",
    'bibtex': "classification",
    'delicious': "classification",
    'wikilshtc': "classification",
    'amazoncat': "classification",
    'wiki500k': "classification",
    'amazon670k': "classification",
    'amazon3m': "classification",

}




def eval_precision(hit_list, num_sample):
    num_p1, num_p3, num_p5 = 0, 0, 0
    for item in hit_list:  # k = 1, 3, 5
        num_p1 += item[0]
        num_p3 += item[1]
        num_p5 += item[2]
    p1 = num_p1 / num_sample
    p3 = num_p3 / (num_sample * 3)
    p5 = num_p5 / (num_sample * 5)

    return p1, p3, p5



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_one_hot(label_id, num_label):

    label_index = label_id.tolist()
    batch_size = len(label_index)
    one_hot_label = np.zeros((batch_size, num_label), dtype=int)
    for i in range(batch_size):
        index = label_index[i]
        one_hot_label[i,index] = 1

    return torch.tensor(one_hot_label)


def eval_batch(logits, truth):
    # row, col = predict.shape
    k = 10
    result = torch.sigmoid(logits)
    pre_ind = torch.topk(result, k, dim=1)[1]

    pre_ind = pre_ind.detach().cpu().numpy()
    truth = truth.detach().cpu().numpy()

    row, col = truth.shape

    hit_list = []
    for k in range(1, 6, 2):  # k = 1, 3, 5
        pre_k = pre_ind[:,:k]
        hit_sum = 0
        for i in range(row):

            intersection = np.intersect1d(pre_k[i,:], truth[i,:])
            hit_sum += len(intersection)

        hit_list.append(hit_sum)


    return hit_list, row


def metric_pk(logits,truth, k = 1):

    pre_ind = torch.topk(logits, k, dim=1)[1]

    pre_ind = pre_ind.detach().cpu().numpy()
    truth = truth.detach().cpu().numpy()

    row, col = truth.shape

    hit_sum = 0
    for i in range(row):

        intersection = np.intersect1d(pre_ind[i,:], truth[i,:])
        hit_sum += len(intersection)

    p_k = hit_sum / row


    return p_k