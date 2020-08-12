import logging
import collections
from typing import Dict, List, Optional, Tuple, DefaultDict, Set, Any

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

from allennlp.data.dataset_readers.dataset_utils import Ontonotes, enumerate_spans
from allennlp.data.fields import (
    Field,
    ListField,
    TextField,
    SpanField,
    MetadataField,
    SequenceLabelField,
)

logger = logging.getLogger(__name__)


@DatasetReader.register("coref_af")
class ConllCorefReader(DatasetReader):
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
    wordpiece_modeling_tokenizer: `PretrainedTransformerTokenizer`, optional (default = None)
        If not None, this dataset reader does subword tokenization using the supplied tokenizer
        and distribute the labels to the resulting wordpieces. All the modeling will be based on
        wordpieces. If this is set to `False` (default), the user is expected to use
        `PretrainedTransformerMismatchedIndexer` and `PretrainedTransformerMismatchedEmbedder`,
        and the modeling will be on the word-level.
    max_sentences: int, optional (default = None)
        The maximum number of sentences in each document to keep. By default keeps all sentences.
    remove_singleton_clusters : `bool`, optional (default = False)
        Some datasets contain clusters that are singletons (i.e. no coreferents). This option allows
        the removal of them. Ontonotes shouldn't have these, and this option should be used for
        testing only.
    """

    def __init__(
        self,
        max_span_width: int,
        token_indexers: Dict[str, TokenIndexer] = None,
        speaker_indexers: Dict[str, TokenIndexer] = None,
        genre_indexers: Dict[str, TokenIndexer] = None,
        wordpiece_modeling_tokenizer: Optional[PretrainedTransformerTokenizer] = None,
        max_sentences: int = None,
        remove_singleton_clusters: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._speaker_indexers = speaker_indexers or {"speakers": SingleIdTokenIndexer(namespace='speakers')}
        self._genre_indexers = genre_indexers or {"genres": SingleIdTokenIndexer(namespace='genres')}
        self._wordpiece_modeling_tokenizer = wordpiece_modeling_tokenizer
        self._max_sentences = max_sentences
        self._remove_singleton_clusters = remove_singleton_clusters

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

            yield self.text_to_instance([s for s in sentences], list(clusters.values()))

    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentences: List[List[str]],
        gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> Instance:
        return make_coref_instance(
            sentences,
            self._token_indexers,
            self._speaker_indexers,
            self._genre_indexers,
            self._max_span_width,
            gold_clusters,
            self._wordpiece_modeling_tokenizer,
            self._max_sentences,
            self._remove_singleton_clusters,
        )




def make_coref_instance(
    full_sentences: List[Any],
    token_indexers: Dict[str, TokenIndexer],
    speaker_indexers: Dict[str, TokenIndexer],
    genre_indexers: Dict[str, TokenIndexer],
    max_span_width: int,
    gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
    wordpiece_modeling_tokenizer: PretrainedTransformerTokenizer = None,
    max_sentences: int = None,
    remove_singleton_clusters: bool = True,
) -> Instance:

    """
    # Parameters
    sentences : `List[List[str]]`, required.
        A list of lists representing the tokenised words and sentences in the document.
    token_indexers : `Dict[str, TokenIndexer]`
        This is used to index the words in the document.  See :class:`TokenIndexer`.
    max_span_width : `int`, required.
        The maximum width of candidate spans to consider.
    gold_clusters : `Optional[List[List[Tuple[int, int]]]]`, optional (default = None)
        A list of all clusters in the document, represented as word spans with absolute indices
        in the entire document. Each cluster contains some number of spans, which can be nested
        and overlap. If there are exact matches between clusters, they will be resolved
        using `_canonicalize_clusters`.
    wordpiece_modeling_tokenizer: `PretrainedTransformerTokenizer`, optional (default = None)
        If not None, this dataset reader does subword tokenization using the supplied tokenizer
        and distribute the labels to the resulting wordpieces. All the modeling will be based on
        wordpieces. If this is set to `False` (default), the user is expected to use
        `PretrainedTransformerMismatchedIndexer` and `PretrainedTransformerMismatchedEmbedder`,
        and the modeling will be on the word-level.
    max_sentences: int, optional (default = None)
        The maximum number of sentences in each document to keep. By default keeps all sentences.
    remove_singleton_clusters : `bool`, optional (default = True)
        Some datasets contain clusters that are singletons (i.e. no coreferents). This option allows
        the removal of them.
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
                with respect to the spans `ListField`.
    """
    if max_sentences is not None and len(full_sentences) > max_sentences:
        full_sentences = full_sentences[:max_sentences]
        total_length = sum(len(sentence.words) for sentence in full_sentences)

        if gold_clusters is not None:
            new_gold_clusters = []

            for cluster in gold_clusters:
                new_cluster = []
                for mention in cluster:
                    if mention[1] < total_length:
                        new_cluster.append(mention)
                if new_cluster:
                    new_gold_clusters.append(new_cluster)

            gold_clusters = new_gold_clusters

    speakers = [s.speakers for s in full_sentences]
    genres = [s.document_id.split('/')[0] for s in full_sentences]
    sentences = [s.words for s in full_sentences]



    flattened_sentences, offsets = [], []
    flat_sentences_tokens = []
    flat_speakers, flat_genres = [], []
    for s_idx, sentence in enumerate(sentences):
        normalized_words = [_normalize_word(word) for word in sentence]
        if wordpiece_modeling_tokenizer is not None:
            sentence_tokens, sentence_offsets = wordpiece_modeling_tokenizer.intra_word_tokenize(
                normalized_words
            )
            flat_sentences_tokens.extend(sentence_tokens)
            offsets.extend(sentence_offsets)
            flattened_sentences.extend([t.text for t in sentence_tokens])
            flat_speakers.extend([speakers[s_idx][0] for _ in sentence_tokens])
            flat_genres.extend([genres[s_idx] for _ in sentence_tokens])
        else:
            flat_sentences_tokens.extend([Token(word) for word in sentence])
            flat_speakers.extend([Token(speakers[s_idx][0] if speakers[s_idx][0] is not None else '-') for _ in sentence])
            flat_genres.extend([Token(genres[s_idx]) for _ in sentence])




    # flattened_sentences = [_normalize_word(word) for sentence in sentences for word in sentence]
    #
    # if wordpiece_modeling_tokenizer is not None:
    #     flat_sentences_tokens, offsets = wordpiece_modeling_tokenizer.intra_word_tokenize(
    #         flattened_sentences
    #     )
    #     flattened_sentences = [t.text for t in flat_sentences_tokens]
    # else:
    #     flat_sentences_tokens = [Token(word) for word in flattened_sentences]






    text_field = TextField(flat_sentences_tokens, token_indexers)
    speaker_field = TextField(flat_speakers, speaker_indexers)
    genre_field = TextField(flat_genres, genre_indexers)






    cluster_dict = {}
    if gold_clusters is not None:
        gold_clusters = _canonicalize_clusters(gold_clusters)
        if remove_singleton_clusters:
            gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 1]

        if wordpiece_modeling_tokenizer is not None:
            for cluster in gold_clusters:
                for mention_id, mention in enumerate(cluster):
                    start = offsets[mention[0]][0]
                    end = offsets[mention[1]][1]
                    cluster[mention_id] = (start, end)

        for cluster_id, cluster in enumerate(gold_clusters):
            for mention in cluster:
                cluster_dict[tuple(mention)] = cluster_id

    spans: List[Field] = []
    span_labels: Optional[List[int]] = [] if gold_clusters is not None else None

    sentence_offset = 0
    for sentence in sentences:
        for start, end in enumerate_spans(
            sentence, offset=sentence_offset, max_span_width=max_span_width
        ):
            if wordpiece_modeling_tokenizer is not None:
                start = offsets[start][0]
                end = offsets[end][1]

                # `enumerate_spans` uses word-level width limit; here we apply it to wordpieces
                # We have to do this check here because we use a span width embedding that has
                # only `max_span_width` entries, and since we are doing wordpiece
                # modeling, the span width embedding operates on wordpiece lengths. So a check
                # here is necessary or else we wouldn't know how many entries there would be.
                if end - start + 1 > max_span_width:
                    continue
                # We also don't generate spans that contain special tokens
                if start < wordpiece_modeling_tokenizer.num_added_start_tokens:
                    continue
                if (
                    end
                    >= len(flat_sentences_tokens)
                    - wordpiece_modeling_tokenizer.num_added_end_tokens
                ):
                    continue

            if span_labels is not None:
                if (start, end) in cluster_dict:
                    span_labels.append(cluster_dict[(start, end)])
                else:
                    span_labels.append(-1)

            spans.append(SpanField(start, end, text_field))
        sentence_offset += len(sentence)

    span_field = ListField(spans)

    metadata: Dict[str, Any] = {"original_text": flattened_sentences}
    if gold_clusters is not None:
        metadata["clusters"] = gold_clusters
    metadata_field = MetadataField(metadata)


    assert(len(text_field) == len(speaker_field) == len(genre_field))

    fields: Dict[str, Field] = {
        "text": text_field,
        "speaker": speaker_field,
        "genre": genre_field,
        "spans": span_field,
        "metadata": metadata_field,
    }
    if span_labels is not None:
        fields["span_labels"] = SequenceLabelField(span_labels, span_field)

    return Instance(fields)


def _normalize_word(word):
    if word in ("/.", "/?"):
        return word[1:]
    else:
        return word


def _canonicalize_clusters(clusters: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    The data might include 2 annotated spans which are identical,
    but have different ids. This checks all clusters for spans which are
    identical, and if it finds any, merges the clusters containing the
    identical spans.
    """
    merged_clusters: List[Set[Tuple[int, int]]] = []
    for cluster in clusters:
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