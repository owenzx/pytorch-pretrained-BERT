import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set

from overrides import overrides
from pytorch_pretrained_bert import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ListField, TextField, SpanField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from . ontonotes_reader import Ontonotes

from stanfordnlp.server import CoreNLPClient
from utils import get_sentence_features, get_stanford_tokens, inv_map, get_chunk_sentences
import pickle
import json

import torch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def canonicalize_clusters(clusters: DefaultDict[int, List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    The CONLL 2012 data includes 2 annotated spans which are identical,
    but have different ids. This checks all clusters for spans which are
    identical, and if it finds any, merges the clusters containing the
    identical spans.
    """
    merged_clusters: List[Set[Tuple[int, int]]] = []
    for cluster in clusters.values():
        cluster_with_overlapping_mention = None
        for mention in cluster:
            # Look at clusters we have already processed to
            # see if they contain a mention in the current
            # cluster for comparison.
            for cluster2 in merged_clusters:
                if mention in cluster2:
                    # first cluster in merged clusters
                    # which contains this mention.
                    cluster_with_overlapping_mention = cluster2
                    break
            # Already encountered overlap - no need to keep looking.
            if cluster_with_overlapping_mention is not None:
                break
        if cluster_with_overlapping_mention is not None:
            # Merge cluster we are currently processing into
            # the cluster in the processed list.
            cluster_with_overlapping_mention.update(cluster)
        else:
            merged_clusters.append(set(cluster))
    return [list(c) for c in merged_clusters]


@DatasetReader.register("my_coref")
class MyConllCorefReader(DatasetReader):
    """
    Reads a single CoNLL-formatted file. This is the same file format as used in the
    :class:`~allennlp.data.dataset_readers.semantic_role_labelling.SrlReader`, but is preprocessed
    to dump all documents into a single file per train, dev and test split. See
    scripts/compile_coref_data.sh for more details of how to pre-process the Ontonotes 5.0 data
    into the correct format.

    Returns a ``Dataset`` where the ``Instances`` have four fields: ``text``, a ``TextField``
    containing the full document text, ``spans``, a ``ListField[SpanField]`` of inclusive start and
    end indices for span candidates, and ``metadata``, a ``MetadataField`` that stores the instance's
    original text. For data with gold cluster labels, we also include the original ``clusters``
    (a list of list of index pairs) and a ``SequenceLabelField`` of cluster ids for every span
    candidate.

    Parameters
    ----------
    max_span_width: ``int``, required.
        The maximum width of candidate spans to consider.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 bert_model_name: str=None,
                 max_pieces: int = 512,
                 lowercase_input: bool=None,
                 extract_features: bool=False,
                 cached_instance_path: str=None,
                 save_instance_path: str=None) -> None:
        super().__init__(lazy)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if bert_model_name is not None:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            if lowercase_input is None:
                self.lowercase_input = "uncased" in bert_model_name
            else:
                self.lowercase_input = lowercase_input
            self.max_pieces = max_pieces
        else:
            self.bert_tokenizer = None
            self.lowercase_input = False
            self.max_pieces=9999999999999

        self.save_instance = False
        self.cached_instance = None
        if cached_instance_path is not None:
            self.cached_instance = pickle.load(open(cached_instance_path, 'rb'))
        elif save_instance_path is not None:
            self.save_instance = True
            self.instance_path = save_instance_path

        CUSTOM_PROPS = {'tokenize.whitespace':True}
        if extract_features:
            #self.feature_extractor = CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'], timeout=60000, memory='16G', be_quiet=False)
            self.feature_extractor = CoreNLPClient(annotators=['tokenize', 'ssplit', 'coref' ], timeout=60000, memory='16G', be_quiet=True, properties=CUSTOM_PROPS)
            #self.debug_extractor = CoreNLPClient(annotators=['tokenize'], timeout=60000, memory='16G', be_quiet=False, host=12345)
        else:
            self.feature_extractor = None



    def _read_unlabeled(self, file_path: str):
        instances_list = []

        file_path = cached_path(file_path)
        i = 0

        with open(file_path, 'r') as fr:
            lines = fr.readlines()
        for line in lines:
            passage_example = json.loads(line)
            passage_id = passage_example['file_name']
            passage_tokens = passage_example['tokens']


            chunk_sentences = get_chunk_sentences(passage_tokens, max_num_tokens=400, raw_str=True)

            for sentences in chunk_sentences:
                sen_id = passage_id
                instance = self.text_to_instance(sentences, None, sen_id)
                #instances_list.append(instance)
                i += 1
                yield instance
        #return instances_list




    @overrides
    def _read(self, file_path: str):
        if 'unlabel' in file_path:
            #return self._read_unlabeled(file_path)
            instances_list = []

            file_path = cached_path(file_path)
            i = 0

            with open(file_path, 'r') as fr:
                lines = fr.readlines()
            for line in lines:
                passage_example = json.loads(line)
                passage_id = passage_example['file_name']
                passage_tokens = passage_example['tokens']


                chunk_sentences = get_chunk_sentences(passage_tokens, max_num_tokens=400, raw_str=True)

                for sentences in chunk_sentences:
                    sen_id = passage_id
                    instance = self.text_to_instance(sentences, None, sen_id)
                    #instances_list.append(instance)
                    i += 1
                    yield instance

        if self.cached_instance is not None:
            for instance in self.cached_instance:
                yield instance
        else:
            if self.save_instance is not None:
                instances_list = []
            # if `file_path` is a URL, redirect to the cache
            file_path = cached_path(file_path)

            ontonotes_reader = Ontonotes()
            i = 0
            for long_sentences in ontonotes_reader.dataset_document_iterator(file_path):

                chunk_sentences = get_chunk_sentences(long_sentences, max_num_tokens=400)


                for sentences in chunk_sentences:
                    total_tokens = 0
                    clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

                    for sentence in sentences:
                        for typed_span in sentence.coref_spans:
                            # Coref annotations are on a _per sentence_
                            # basis, so we need to adjust them to be relative
                            # to the length of the document.
                            span_id, (start, end) = typed_span
                            clusters[span_id].append((start + total_tokens,
                                                      end + total_tokens))
                        total_tokens += len(sentence.words)

                    canonical_clusters = canonicalize_clusters(clusters)
                    #_ =  self.text_to_instance([s.words for s in sentences], canonical_clusters)
                    sen_id = str(sentences[0].document_id) + ':' + str(sentences[0].sentence_id)
                    instance = self.text_to_instance([s.words for s in sentences], canonical_clusters, sen_id)
                    instances_list.append(instance)
                    i += 1
                    yield instance

            #yield self.text_to_instance([s.words for s in sentences], canonical_clusters)



    def dump_instances(self, file_path: str):
        import pickle
        instances_list = []
        assert(self.save_instance is not None)
        file_path = cached_path(file_path)

        ontonotes_reader = Ontonotes()
        for long_sentences in ontonotes_reader.dataset_document_iterator(file_path):

            chunk_sentences = get_chunk_sentences(long_sentences, max_num_tokens=400)


            for sentences in chunk_sentences:
                total_tokens = 0
                clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

                for sentence in sentences:
                    for typed_span in sentence.coref_spans:
                        # Coref annotations are on a _per sentence_
                        # basis, so we need to adjust them to be relative
                        # to the length of the document.
                        span_id, (start, end) = typed_span
                        clusters[span_id].append((start + total_tokens,
                                                  end + total_tokens))
                    total_tokens += len(sentence.words)

                canonical_clusters = canonicalize_clusters(clusters)
                #_ =  self.text_to_instance([s.words for s in sentences], canonical_clusters)
                sen_id = str(sentences[0].document_id) + ':' + str(sentences[0].sentence_id)
                instance = self.text_to_instance([s.words for s in sentences], canonical_clusters, sen_id)
                instances_list.append(instance)
        with open(self.instance_path, 'wb') as fw:
            pickle.dump(instances_list, fw)


    def _wordpiece_tokenize_input(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        """
        Convert a list of tokens to wordpiece tokens and offsets, as well as adding
        BERT CLS and SEP tokens to the begining and end of the sentence.
        """
        word_piece_tokens: List[str] = []
        offsets = []
        cumulative = 0
        start_idx_maps, end_idx_maps = dict(), dict()
        for idx, token in enumerate(tokens):
            if self.lowercase_input:
                token = token.lower()
            word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)
            start_idx_maps[idx] = cumulative
            cumulative += len(word_pieces)
            end_idx_maps[idx] = cumulative - 1
            offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)

        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]

        offsets = [x + 1 for x in offsets]
        for k in start_idx_maps.keys():
            start_idx_maps[k] = start_idx_maps[k] + 1
        for k in end_idx_maps.keys():
            end_idx_maps[k] = end_idx_maps[k] + 1
        return wordpieces, offsets, start_idx_maps, end_idx_maps

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentences: List[List[str]],
                         gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
                         sen_id: str = None) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sentences : ``List[List[str]]``, required.
            A list of lists representing the tokenised words and sentences in the document.
        gold_clusters : ``Optional[List[List[Tuple[int, int]]]]``, optional (default = None)
            A list of all clusters in the document, represented as word spans. Each cluster
            contains some number of spans, which can be nested and overlap, but will never
            exactly match between clusters.

        Returns
        -------
        An ``Instance`` containing the following ``Fields``:
            text : ``TextField``
                The text of the full document.
            spans : ``ListField[SpanField]``
                A ListField containing the spans represented as ``SpanFields``
                with respect to the document text.
            span_labels : ``SequenceLabelField``, optional
                The id of the cluster which each possible span belongs to, or -1 if it does
                 not belong to a cluster. As these labels have variable length (it depends on
                 how many spans we are considering), we represent this a as a ``SequenceLabelField``
                 with respect to the ``spans ``ListField``.
        """
        flattened_sentences = [self._normalize_word(word)
                               for sentence in sentences
                               for word in sentence]



        metadata: Dict[str, Any] = {"original_text": flattened_sentences, "sen_id": sen_id}

        if self.feature_extractor is not None:
            sentences_features = get_sentence_features(self.feature_extractor, metadata)
            metadata['features'] = sentences_features

        tokens = [Token(word) for word in flattened_sentences]

        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters

        if self.bert_tokenizer is not None:
            wordpieces, offsets, start_idx_maps, end_idx_maps = self._wordpiece_tokenize_input([t.text for t in tokens])
            metadata["offsets"] = offsets
            metadata['start_idx_maps'] = start_idx_maps
            metadata['end_idx_maps'] = end_idx_maps
            metadata['rev_start_maps'] = inv_map(start_idx_maps)
            #Set text_id attribute to override the indexing mechanism
            metadata['rev_end_maps'] = inv_map(end_idx_maps)
            text_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces][:self.max_pieces], token_indexers=self._token_indexers)

        else:
            text_field = TextField([Token(word) for word in flattened_sentences], self._token_indexers)

        cluster_dict = {}
        if gold_clusters is not None:
            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        spans: List[Field] = []
        span_labels: Optional[List[int]] = [] if gold_clusters is not None else None

        sentence_offset = 0
        for sentence in sentences:
            for start, end in enumerate_spans(sentence,
                                              offset=sentence_offset,
                                              max_span_width=self._max_span_width):
                if self.bert_tokenizer is not None:
                    if end_idx_maps[end] >= self.max_pieces:
                        continue
                    if end_idx_maps[end] - start_idx_maps[start] >= self._max_span_width:
                        continue
                if span_labels is not None:
                    if (start, end) in cluster_dict:
                        span_labels.append(cluster_dict[(start, end)])
                    else:
                        span_labels.append(-1)
                if self.bert_tokenizer is not None:
                    spans.append(SpanField(start_idx_maps[start], end_idx_maps[end], text_field))
                else:
                    spans.append(SpanField(start, end, text_field))
            sentence_offset += len(sentence)

        span_field = ListField(spans)

        metadata['spans'] = spans
        if span_labels is not None:
            metadata['span_labels'] = span_labels

        if self.bert_tokenizer is not None:
            newmetadata: Dict[str, Any] = {"original_text": metadata['original_text']}
            if gold_clusters is not None:
                newclusters: List[Set[Tuple[int, int]]] = []
                for cluster in gold_clusters:
                    newclusters.append(set([(start_idx_maps[start], end_idx_maps[end]) for (start, end) in cluster]))
                newmetadata['clusters'] = newclusters
            newmetadata["tokenized_text"] = wordpieces
            #copy other things from metadata
            for k in metadata.keys():
                if k not in ['original_text', 'clusters', 'tokenized_text']:
                    newmetadata[k] = metadata[k]

            metadata_field = MetadataField(newmetadata)
        else:
            metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {"text": text_field,
                                    "spans": span_field,
                                    "metadata": metadata_field}
        if span_labels is not None:
            fields["span_labels"] = SequenceLabelField(span_labels, span_field)

        return Instance(fields)

    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word
