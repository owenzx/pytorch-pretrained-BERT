import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set
import numpy as np

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    Field,
    ListField,
    TextField,
    SpanField,
    MetadataField,
    SequenceLabelField,
)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, enumerate_spans


logger = logging.getLogger(__name__)


def canonicalize_clusters(
    clusters: DefaultDict[int, List[Tuple[int, int]]]
) -> List[List[Tuple[int, int]]]:
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


#@DatasetReader.register("coref_head_joint_bert_concat")
class ConllCorefReaderHeadBertC(DatasetReader):
    def __init__(self):
        pass

    @overrides
    def _read(self, file_path: str):
        pass

    @overrides
    def text_to_instance(self, *inputs):
        pass


@DatasetReader.register("coref_head_joint_bert_subword")
class ConllCorefReaderHeadBert(DatasetReader):
    """
    Reads a single CoNLL-formatted file. This is the same file format as used in the
    :class:`~allennlp.data.dataset_readers.semantic_role_labelling.SrlReader`, but is preprocessed
    to dump all documents into a single file per train, dev and test split. See
    scripts/compile_coref_data.sh for more details of how to pre-process the Ontonotes 5.0 data
    into the correct format.

    Returns a `Dataset` where the `Instances` have four fields : `text`, a `TextField`
    containing the full document text, `spans`, a `ListField[SpanField]` of inclusive start and
    end indices for span candidates, and `metadata`, a `MetadataField` that stores the instance's
    original text. For data with gold cluster labels, we also include the original `clusters`
    (a list of list of index pairs) and a `SequenceLabelField` of cluster ids for every span
    candidate.

    # Parameters

    max_span_width : `int`, required.
        The maximum width of candidate spans to consider.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    """


    def __init__(
        self,
        lazy: bool = False,
        token_indexers: Dict[str, TokenIndexer] = None,
        wordpiece_modeling_tokenizer: Optional[PretrainedTransformerTokenizer] = None,
        truncation: bool = False,
        max_sentences: int = None,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        assert(wordpiece_modeling_tokenizer is not None)
        self._wordpiece_modeling_tokenizer = wordpiece_modeling_tokenizer
        self._truncation = truncation
        self._max_sentences = max_sentences


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        ontonotes_reader = Ontonotes()
        for sentences in ontonotes_reader.dataset_document_iterator(file_path):



            clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

            total_tokens = 0
            for sentence in sentences:
                for typed_span in sentence.coref_spans:
                    # Coref annotations are on a _per sentence_
                    # basis, so we need to adjust them to be relative
                    # to the length of the document.
                    span_id, (start, end) = typed_span
                    clusters[span_id].append((start + total_tokens, end + total_tokens))
                total_tokens += len(sentence.words)

            canonical_clusters = canonicalize_clusters(clusters)
            if self._truncation or self._max_sentences is None:
                yield self.text_to_instance([s.words for s in sentences], canonical_clusters)
            else:
                if len(sentences) > self._max_sentences:
                    start_idxs = [0]
                    while start_idxs[-1] + self._max_sentences < len(sentences):
                        start_idxs.append(min(start_idxs[-1]+self._max_sentences, len(sentences)-self._max_sentences))
                    for idx in start_idxs:
                        yield self.text_to_instance([s.words for s in sentences], canonical_clusters, idx)
                else:
                    yield self.text_to_instance([s.words for s in sentences], canonical_clusters)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentences: List[List[str]],
        gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
        start: Optional[int] = None,
    ) -> Instance:

        """
        # Parameters

        sentences : `List[List[str]]`, required.
            A list of lists representing the tokenised words and sentences in the document.
        gold_clusters : `Optional[List[List[Tuple[int, int]]]]`, optional (default = None)
            A list of all clusters in the document, represented as word spans. Each cluster
            contains some number of spans, which can be nested and overlap, but will never
            exactly match between clusters.

        # Returns

        An `Instance` containing the following `Fields`:
            text : `TextField`
                The text of the full document.
            spans : `ListField[SpanField]`
                A ListField containing the spans represented as `SpanFields`
                with respect to the document text.
            span_labels : `SequenceLabelField`, optional
                The id of the cluster which each possible span belongs to, or -1 if it does
                 not belong to a cluster. As these labels have variable length (it depends on
                 how many spans we are considering), we represent this a as a `SequenceLabelField`
                 with respect to the `spans `ListField`.
        """
        if self._max_sentences is not None and len(sentences) > self._max_sentences:
            if not self._truncation:
                start_idx = start if start is not None else np.random.randint(0, len(sentences) - self._max_sentences)
                first_sentence = start_idx
                last_sentence = first_sentence + self._max_sentences - 1
                first_token_offset = sum(len(sentence) for sentence in sentences[:first_sentence])
                last_token_offset = (first_token_offset + sum(len(sentence) for sentence in sentences[first_sentence:last_sentence+1]) - 1)
                sentences = sentences[start_idx:start_idx+self._max_sentences]
            else:
                sentences = sentences[:self._max_sentences]
            total_length = sum(len(sentence) for sentence in sentences)

            if gold_clusters is not None:
                new_gold_clusters = []

                for cluster in gold_clusters:
                    new_cluster = []
                    for mention in cluster:
                        start, end = mention
                        if not self._truncation:
                            if first_token_offset <= start <= end <= last_token_offset:
                                new_cluster.append((start-first_token_offset, end-first_token_offset))
                        else:
                            if end < total_length:
                                new_cluster.append(mention)
                    if new_cluster:
                        new_gold_clusters.append(new_cluster)

                gold_clusters = new_gold_clusters

        flattened_sentences = [self._normalize_word(word) for sentence in sentences for word in sentence]


        if self._wordpiece_modeling_tokenizer is not None:
            flat_sentences_tokens, offsets = self._wordpiece_modeling_tokenizer.intra_word_tokenize(
                flattened_sentences
            )
            flattened_sentences = [t.text for t in flat_sentences_tokens]
        else:
            flat_sentences_tokens = [Token(word) for word in flattened_sentences]


        text_field = TextField(flat_sentences_tokens, self._token_indexers)


        metadata: Dict[str, Any] = {"original_text": flattened_sentences}
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters


        cluster_dict = {}
        if gold_clusters is not None:
            if self._wordpiece_modeling_tokenizer is not None:
                for cluster in gold_clusters:
                    for mention_id, mention in enumerate(cluster):
                        assert(mention[0] == mention[1])
                        start = offsets[mention[0]][0]
                        end = offsets[mention[1]][1]
                        # Special treatement for head-based reader
                        cluster[mention_id] = (end, end)

            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        spans: List[Field] = []
        span_labels: Optional[List[int]] = [] if gold_clusters is not None else None



        for start, end in enumerate_spans(flattened_sentences, max_span_width=1):
            if span_labels is not None:
                if (start, end) in cluster_dict:
                    span_labels.append(cluster_dict[(start, end)])
                else:
                    span_labels.append(-1)
                spans.append(SpanField(start, end, text_field))


        # head2cluster_map = {}
        # sentence_offset = 0
        # for sentence in sentences:
        #     for start, end in enumerate_spans(sentence, offset=sentence_offset, max_span_width=1):
        #         if self._wordpiece_modeling_tokenizer is not None:
        #             start = offsets[start][0]
        #             end = offsets[end][1]
        #             # We also don't generate spans that contain special tokens
        #             if start < self._wordpiece_modeling_tokenizer.num_added_start_tokens:
        #                 continue
        #             if (
        #                 end
        #                 >= len(flat_sentences_tokens)
        #                 - self._wordpiece_modeling_tokenizer.num_added_end_tokens
        #             ):
        #                 continue
        #
        #         if span_labels is not None:
        #             if (start, end) in cluster_dict:
        #                 #span_labels.append(cluster_dict[(start, end)])
        #                 head2cluster_map[end] = cluster_dict[(start, end)]
        #             else:
        #                 #span_labels.append(-1)
        #                 head2cluster_map[end] = -1
        #
        #         #only use the end_idx_map to make sure all the converted span is also length 1
        #         spans.append(SpanField(end, end, text_field))

        # #add cls and sep label
        # for i in range(len(text_field)):
        #     if i in head2cluster_map.keys():
        #         span_labels.append(head2cluster_map[i])
        #     else:
        #         span_labels.append(-1)


        # DEBUG
        # print(flattened_sentences)
        # print(sentences)
        # assert(len(text_field) == len(span_labels))
        # for i in range(len(text_field)):
        #     print(text_field.tokens[i].text + "\t" + str(span_labels[i]))
        #
        # exit()



        metadata: Dict[str, Any] = {"original_text": flattened_sentences}
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters
        metadata_field = MetadataField(metadata)


        fields: Dict[str, Field] = {
            "text": text_field,
            #"spans": span_field,
            "metadata": metadata_field,
        }

        if span_labels is not None:
            #fields["span_labels"] = SequenceLabelField(span_labels, span_field)
            fields["span_labels"] = SequenceLabelField(span_labels, text_field)


        return Instance(fields)

    @staticmethod
    def _normalize_word(word):
        if word in ("/.", "/?"):
            return word[1:]
        else:
            return word