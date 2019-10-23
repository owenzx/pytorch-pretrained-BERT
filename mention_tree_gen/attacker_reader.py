import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field,  TextField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, enumerate_spans
from utils import get_sentence_features, get_stanford_tokens, inv_map, get_chunk_sentences
from stanfordnlp.server import CoreNLPClient

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _convert_mention_to_map(cluster, text_tokens):
    map = [0 for _ in text_tokens]
    for span in cluster:
        l, r = span
        for i in range(l, r+1):
            map[i] = 1
    return map




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


@DatasetReader.register("AttackerCoref")
class AttackerCorefReader(DatasetReader):
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
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        CUSTOM_PROPS = {'tokenize.whitespace':True}
        self.client = CoreNLPClient(annotators=['tokenize', 'ssplit', 'coref'], timeout=6000000, memory='16G', be_quiet=True, properties=CUSTOM_PROPS)

    def _get_phrase_head(self, phrase):
        ann = self.client.annotate(phrase)
        mentions4coref = ann.mentionsForCoref
        if len(mentions4coref) == 0:
            return None
        else:
            return mentions4coref[0].headString

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        ontonotes_reader = Ontonotes()
        # for sentences in ontonotes_reader.dataset_document_iterator(file_path):
        #     clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        #
        #     total_tokens = 0
        #     for sentence in sentences:
        #         for typed_span in sentence.coref_spans:
        #             # Coref annotations are on a _per sentence_
        #             # basis, so we need to adjust them to be relative
        #             # to the length of the document.
        #             span_id, (start, end) = typed_span
        #             clusters[span_id].append((start + total_tokens,
        #                                       end + total_tokens))
        #         total_tokens += len(sentence.words)
        #
        #     canonical_clusters = canonicalize_clusters(clusters)
        #     yield self.text_to_instance([s.words for s in sentences], canonical_clusters)

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
                for i_c, cluster in enumerate(canonical_clusters):
                    #TODO do it offline and correctly
                    for i_s, span in enumerate(cluster):
                        l, r = span

                        #filter out pronouns
                        if (r-l)<=1:
                            continue

                        #also check non-overlapping
                        non_overlapping = True
                        for j_c, cluster2 in enumerate(canonical_clusters):
                            for j_s, span2 in enumerate(cluster2):
                                if j_c == i_c and j_s == i_s:
                                    continue
                                l2, r2 = span2
                                if l<= l2 <=r or l<=r2<=r:
                                    non_overlapping = False
                                    break
                            if not non_overlapping:
                                break
                        if not non_overlapping:
                            continue



                        target_cluster = [span]
                        instance = self.text_to_instance([s.words for s in sentences], canonical_clusters, target_cluster, sen_id)
                        yield instance





    @overrides
    def text_to_instance(self,  # type: ignore
                         sentences: List[List[str]],
                         gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
                         target_cluster: Optional[List[Tuple[int, int]]] = None,
                         sen_id: Optional[str] = None) -> Instance:
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

        #TODO make this part better
        l_s, r_s = target_cluster[0]
        selected_mention = flattened_sentences[l_s: r_s+1]
        phrase_head = self._get_phrase_head(' '.join(selected_mention))
        if phrase_head is None:
            phrase_head = selected_mention[-1]

        metadata: Dict[str, Any] = {"original_text": [flattened_sentences],
                                    "target_cluster": target_cluster,
                                    "np_head": phrase_head,
                                    "input_sentences":sentences,
                                    "input_gold_clusters": gold_clusters,
                                    "input_sen_id": sen_id}
        assert(gold_clusters is not None)
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters

        text_field = TextField([Token(word) for word in flattened_sentences], self._token_indexers)


        mention_map = _convert_mention_to_map(target_cluster, flattened_sentences)

        assert(len(mention_map) == len(flattened_sentences))

        mention_field = SequenceLabelField(mention_map, text_field)

        metadata_field = MetadataField(metadata)


        fields: Dict[str, Field] = {"text": text_field,
                                    "selected_mentions": mention_field,
                                    "metadata": metadata_field}

        return Instance(fields)

    @staticmethod
    def _normalize_word(word):
        if word in ("/.", "/?"):
            return word[1:]
        else:
            return word






@DatasetReader.register("AttackerCoref_edit")
class EditAttackerCorefReader(DatasetReader):
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
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        CUSTOM_PROPS = {'tokenize.whitespace':True}
        self.client = CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'coref', 'parse', 'depparse'], timeout=6000000, memory='16G', be_quiet=True, properties=CUSTOM_PROPS)

    def _get_phrase_head(self, phrase):
        ann = self.client.annotate(phrase)
        mentions4coref = ann.mentionsForCoref
        if len(mentions4coref) == 0:
            return None
        else:
            return mentions4coref[0].headString


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        ontonotes_reader = Ontonotes()
        # for sentences in ontonotes_reader.dataset_document_iterator(file_path):
        #     clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        #
        #     total_tokens = 0
        #     for sentence in sentences:
        #         for typed_span in sentence.coref_spans:
        #             # Coref annotations are on a _per sentence_
        #             # basis, so we need to adjust them to be relative
        #             # to the length of the document.
        #             span_id, (start, end) = typed_span
        #             clusters[span_id].append((start + total_tokens,
        #                                       end + total_tokens))
        #         total_tokens += len(sentence.words)
        #
        #     canonical_clusters = canonicalize_clusters(clusters)
        #     yield self.text_to_instance([s.words for s in sentences], canonical_clusters)

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
                for i_c, cluster in enumerate(canonical_clusters):
                    #TODO do it offline and correctly
                    for i_s, span in enumerate(cluster):
                        l, r = span

                        #filter out pronouns
                        if (r-l)<=1:
                            continue

                        #also check non-overlapping
                        non_overlapping = True
                        for j_c, cluster2 in enumerate(canonical_clusters):
                            for j_s, span2 in enumerate(cluster2):
                                if j_c == i_c and j_s == i_s:
                                    continue
                                l2, r2 = span2
                                if l<= l2 <=r or l<=r2<=r:
                                    non_overlapping = False
                                    break
                            if not non_overlapping:
                                break
                        if not non_overlapping:
                            continue



                        target_cluster = [span]
                        instance = self.text_to_instance([s.words for s in sentences], canonical_clusters, target_cluster, sen_id)
                        yield instance



    def _get_constituency_parse(self, text):
        text_str = ' '.join(text)
        ann = self.client.annotate(text_str)
        sentence = ann.sentence[0]
        tree = sentence.parseTree
        return tree

    def _get_flatten_parse(self, tree, parse_str_tokens):
        if tree is not None:
            # print("value={}".format(tree.value))
            child_num = len(tree.child)
            if child_num != 0:
                parse_str_tokens.append('(')

            # print(tree.value, end=' ')
            parse_str_tokens.append(tree.value)

            # print(len(tree.child))
            for c in tree.child:
                self._get_flatten_parse(c, parse_str_tokens)

            if child_num != 0:
                # print(')', end=' ')
                parse_str_tokens.append(')')
        return parse_str_tokens


    @overrides
    def text_to_instance(self,  # type: ignore
                         sentences: List[List[str]],
                         gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
                         target_cluster: Optional[List[Tuple[int, int]]] = None,
                         sen_id: Optional[str] = None) -> Instance:
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


        #TODO finish this
        l_s, r_s = target_cluster[0]
        selected_mention = flattened_sentences[l_s: r_s+1]
        parse_tree = self._get_constituency_parse(selected_mention)
        flattened_parse = self._get_flatten_parse(parse_tree, [])

        #TODO make this part better
        l_s, r_s = target_cluster[0]
        selected_mention = flattened_sentences[l_s: r_s+1]
        phrase_head = self._get_phrase_head(' '.join(selected_mention))
        if phrase_head is None:
            phrase_head = selected_mention[-1]

        metadata: Dict[str, Any] = {"original_text": [flattened_sentences],
                                    "original_parse": flattened_parse,
                                    "target_cluster": target_cluster,
                                    "np_head": phrase_head,
                                    "input_sentences":sentences,
                                    "input_gold_clusters": gold_clusters,
                                    "input_sen_id": sen_id}
        assert(gold_clusters is not None)
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters

        text_field = TextField([Token(word) for word in flattened_sentences], self._token_indexers)

        parse_field = TextField([Token(word) for word in flattened_parse], self._token_indexers)


        mention_map = _convert_mention_to_map(target_cluster, flattened_sentences)

        assert(len(mention_map) == len(flattened_sentences))

        mention_field = SequenceLabelField(mention_map, text_field)

        metadata_field = MetadataField(metadata)


        fields: Dict[str, Field] = {"text": text_field,
                                    "parse": parse_field,
                                    "selected_mentions": mention_field,
                                    "metadata": metadata_field}

        return Instance(fields)

    @staticmethod
    def _normalize_word(word):
        if word in ("/.", "/?"):
            return word[1:]
        else:
            return word
