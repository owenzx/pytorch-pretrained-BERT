import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set

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
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, enumerate_spans

from pytorch_pretrained_bert import BertTokenizer

from utils import get_chunk_sentences

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


@DatasetReader.register("coref_head_joint_bert_truncate")
class ConllCorefReaderHeadBertT(DatasetReader):
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
        bert_model_name: str = None,
        max_pieces: int = 512,
        lowercase_input: bool = None,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}
        assert(bert_model_name is not None)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        if lowercase_input is None:
            self.lowercase_input = "uncased" in bert_model_name
        else:
            self.lowercase_input = lowercase_input
        self.max_pieces =max_pieces


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
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        ontonotes_reader = Ontonotes()
        for long_sentences in ontonotes_reader.dataset_document_iterator(file_path):

            chunk_sentences = get_chunk_sentences(long_sentences, max_num_tokens=400)

            for sentences in chunk_sentences:

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
                yield self.text_to_instance([s.words for s in sentences], canonical_clusters)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentences: List[List[str]],
        gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
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
        flattened_sentences = [
            self._normalize_word(word) for sentence in sentences for word in sentence
        ]

        metadata: Dict[str, Any] = {"original_text": flattened_sentences}
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters


        wordpieces, offsets, start_idx_maps, end_idx_maps = self._wordpiece_tokenize_input(flattened_sentences)
        metadata['start_idx_maps'] = start_idx_maps
        metadata['end_idx_maps'] = end_idx_maps


        text_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces][:self.max_pieces], token_indexers=self._token_indexers)

        cluster_dict = {}
        if gold_clusters is not None:
            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        spans: List[Field] = []
        span_labels: Optional[List[int]] = [] if gold_clusters is not None else None


        head2cluster_map = {}


        for start, end in enumerate_spans(flattened_sentences, max_span_width=1):

            if end_idx_maps[end] >= self.max_pieces:
                continue

            if span_labels is not None:
                if (start, end) in cluster_dict:
                    #span_labels.append(cluster_dict[(start, end)])
                    head2cluster_map[end_idx_maps[end]] = cluster_dict[(start, end)]
                else:
                    #span_labels.append(-1)
                    head2cluster_map[end_idx_maps[end]] = -1

            #only use the end_idx_map to make sure all the converted span is also length 1
            spans.append(SpanField(end_idx_maps[start], end_idx_maps[end], text_field))
        #add cls and sep label
        for i in range(len(text_field)):
            if i in head2cluster_map.keys():
                span_labels.append(head2cluster_map[i])
            else:
                span_labels.append(-1)




        span_field = ListField(spans)
        #assert(len(span_field) == len(text_field))

        newmetadata: Dict[str, Any] = {"original_text": metadata["original_text"]}
        if gold_clusters is not None:
            newclusters: List[Set[Tuple[int, int]]] = []
            for cluster in gold_clusters:
                newclusters.append(set([(end_idx_maps[start], end_idx_maps[end]) for (start, end) in cluster]))
            newmetadata['clusters'] = newclusters
        newmetadata['tokenized_text'] = wordpieces

        metadata_field = MetadataField(newmetadata)

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