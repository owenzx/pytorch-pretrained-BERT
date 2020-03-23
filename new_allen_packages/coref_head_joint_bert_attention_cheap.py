import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from .conll_head_coref_scores import ConllHeadCorefScores


def two_dim_fill(tensor, value, indices):
    assert(indices.shape[-1] == 2)
    max_len = tensor.shape[-2]

    flat_indices = indices[:, :, 0] * max_len + indices[:, :, 1]
    flat_indices = flat_indices.view(1, flat_indices.shape[0], -1, 1)
    flat_indices = flat_indices.repeat(tensor.shape[0], 1, 1, tensor.shape[-1])


    flattened_tensor = tensor.view(tensor.shape[0], tensor.shape[1], -1, tensor.shape[-1])
    print(flattened_tensor.shape)
    print(flat_indices.shape)
    flattened_tensor.scatter_(2, flat_indices, value)

    flattened_tensor = flattened_tensor.view(*tensor.shape)
    print(flattened_tensor.shape)
    return flattened_tensor




def get_total_attention_matrix(total_mask, fold_maps, num_seg):
    """Target is to add block mask to the total_mask"""
    # total_mask has the shape of [#att, b, pruned_len, pruned_len]
    # fold_maps has the shape of [b, pruned_len, 2]

    device = total_mask.device
    print(total_mask.shape)
    print(fold_maps.shape)

    # TODO add two dummy lines for batching
    total_mask = torch.cat([total_mask, torch.zeros((total_mask.shape[0], total_mask.shape[1], total_mask.shape[2], 1), device=device, dtype=torch.bool)], -1)
    total_mask = torch.cat([total_mask, torch.zeros((total_mask.shape[0], total_mask.shape[1], 1, total_mask.shape[3]), device=device, dtype=torch.bool)], -2)

    #DEBUG CODE
    total_size = 0
    batch_size = fold_maps.shape[0]
    for b in range(batch_size):
        seg_dict = {}
        for s in range(num_seg):
            seg_dict[s] = 0
        locs = fold_maps[b, :, :].view(-1, 2)
        all_locs = {}
        for loc in locs:
            k = (loc[0].item(), loc[1].item())
            seg_dict[k[0]] += 1
            if k not in all_locs:
                all_locs[k] = 1
            else:
                all_locs[k] += 1
        for s in seg_dict.keys():
            total_size += seg_dict[s]*seg_dict[s] *144
        print([(k,all_locs[k]) for k in all_locs.keys() if all_locs[k] > 1])
        print("MAX:%d"%(max(all_locs.values())))
    print(total_size)






    count = torch.zeros(fold_maps.shape[0], device=device, dtype=torch.long)
    for i in range(num_seg):
        start_loc = count
        end_loc = count + torch.sum(fold_maps[:,:, 0].eq(i), 1).flatten()
        count = end_loc
        block_idx = torch.zeros((1, total_mask.size(1), torch.max(end_loc-start_loc).item()), device=device, dtype=torch.long) - 1

        # for batch_idx in range(block_idx.shape[1]):
        #     block_idx[:, batch_idx, :end_loc[batch_idx]-start_loc[batch_idx]] = torch.arange(start_loc[batch_idx], end_loc[batch_idx])
        # total_mask[:,:,block_idx.unsqueeze(3), block_idx.unsqueeze(2)] = 1

        for batch_idx in range(block_idx.shape[1]):
            total_mask[:, batch_idx, start_loc[batch_idx]: end_loc[batch_idx], start_loc[batch_idx]:end_loc[batch_idx]] = 1

    total_mask = total_mask[:,:, :-1, :-1]
    print(total_mask.shape)
    print(torch.sum(total_mask.bool()))
    return total_mask



logger = logging.getLogger(__name__)


@Model.register("coref_head_joint_bert_attention_cheap")
class CoreferenceResolver(Model):
    """
    This ``Model`` implements the coreference resolution model described "End-to-end Neural
    Coreference Resolution"
    <https://www.semanticscholar.org/paper/End-to-end-Neural-Coreference-Resolution-Lee-He/3f2114893dc44eacac951f148fbff142ca200e83>
    by Lee et al., 2017.
    The basic outline of this model is to get an embedded representation of each span in the
    document. These span representations are scored and used to prune away spans that are unlikely
    to occur in a coreference cluster. For the remaining spans, the model decides which antecedent
    span (if any) they are coreferent with. The resulting coreference links, after applying
    transitivity, imply a clustering of the spans in the document.
    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``
        This layer incorporates contextual information for each word in the document.
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    antecedent_feedforward: ``FeedForward``
        This feedforward network is applied to pairs of span representation, along with any
        pairwise features, which is then scored by a linear layer.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    max_span_width: ``int``
        The maximum width of candidate spans.
    spans_per_word: float, required.
        A multiplier between zero and one which controls what percentage of candidate mention
        spans we retain with respect to the number of words in the document.
    max_antecedents: int, required.
        For each mention which survives the pruning stage, we consider this many antecedents.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        context_layer: Seq2SeqEncoder,
        text_field_embedder: TextFieldEmbedder,
        heads_per_word: float,
        bert_feedforward: Optional[FeedForward] = None,
        token_mask_feedforward: Optional[FeedForward] = None,
        lexical_dropout: float = 0.2,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        self._context_layer = context_layer

        self._heads_per_word = heads_per_word
        assert(self._heads_per_word < 1.0)

        assert(token_mask_feedforward is not None)

        self._token_mask_ff = TimeDistributed(token_mask_feedforward)

        self._text_field_embedder = text_field_embedder
        if bert_feedforward is not None:
            self._bert_feedforward = TimeDistributed(bert_feedforward)
            self._final_link_attention = BilinearMatrixAttention(self._bert_feedforward.get_output_dim(), self._bert_feedforward.get_output_dim())
        else:
            self._bert_feedforward = None
            #TODO dim as argument
            self._final_link_attention = BilinearMatrixAttention(self._context_layer.get_output_dim(), self._context_layer.get_output_dim())


        #144 attentions in Bert-base
        self.weighted_attention_logits = torch.nn.Parameter(torch.zeros(144))


        # 10 possible distance buckets.
        self._num_distance_buckets = 10


        #TODO: should design some intermediate metric
        #self._mention_recall = MentionRecall()

        self._conll_coref_scores = ConllHeadCorefScores()
        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x
        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        text: Dict[str, torch.LongTensor],
        #spans: torch.IntTensor,
        span_labels: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Parameters
        ----------
        text : ``Dict[str, torch.LongTensor]``, required.
            The output of a ``TextField`` representing the text of
            the document.
        spans : ``torch.IntTensor``, required.
            A tensor of shape (batch_size, num_spans, 2), representing the inclusive start and end
            indices of candidate spans for mentions. Comes from a ``ListField[SpanField]`` of
            indices into the text of the document.
        span_labels : ``torch.IntTensor``, optional (default = None).
            A tensor of shape (batch_size, num_spans), representing the cluster ids
            of each span, or -1 for those which do not appear in any clusters.
        metadata : ``List[Dict[str, Any]]``, optional (default = None).
            A metadata dictionary for each instance in the batch. We use the "original_text" and "clusters" keys
            from this dictionary, which respectively have the original text and the annotated gold coreference
            clusters for that instance.
        Returns
        -------
        An output dictionary consisting of:
        top_spans : ``torch.IntTensor``
            A tensor of shape ``(batch_size, num_spans_to_keep, 2)`` representing
            the start and end word indices of the top spans that survived the pruning stage.
        antecedent_indices : ``torch.IntTensor``
            A tensor of shape ``(num_spans_to_keep, max_antecedents)`` representing for each top span
            the index (with respect to top_spans) of the possible antecedents the model considered.
        predicted_antecedents : ``torch.IntTensor``
            A tensor of shape ``(batch_size, num_spans_to_keep)`` representing, for each top span, the
            index (with respect to antecedent_indices) of the most likely antecedent. -1 means there
            was no predicted link.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        """
        # Shape: (batch_size, document_length, embedding_size)
        mask = util.get_text_field_mask(text)
        text_mask = util.get_text_field_mask(text).float()

        # Shape: (batch_size, document_length, embedding_size)
        #bert_embeddings, _ = self.bert_model(input_ids=text['tokens'], attention_mask=mask, output_all_encoded_layers=False)

        bert_embeddings, all_attentions, unfold_map = self._text_field_embedder(text)
        #Here assume all_attention has the shape of [batch * segment, #attention, max_seq_len, max_seq_len]
        seq_length = bert_embeddings.shape[1]
        batch_size = bert_embeddings.shape[0]
        num_attentions = all_attentions.shape[1]
        max_seg_len = all_attentions.shape[-1]
        all_attentions = all_attentions.view(batch_size, -1, num_attentions, max_seg_len, max_seg_len)
        num_seg = all_attentions.shape[1]
        print(all_attentions.shape)


        all_attentions = all_attentions.permute(2, 0, 1, 3, 4)




        text_embeddings = self._lexical_dropout(bert_embeddings)
        if self._bert_feedforward is not None:
            text_embeddings = self._bert_feedforward(text_embeddings)

        text_embeddings = self._context_layer(text_embeddings, text_mask)

        batch_size = text_embeddings.size(0)
        document_length = text_embeddings.size(1)
        #num_spans = spans.size(1)

        # Shape: (batch_size, document_length)
        text_mask = util.get_text_field_mask(text).float()

        span_labels = span_labels + (-1*(1-text_mask.long()))
        #print(span_labels)

        ## Shape: (batch_size, num_spans)
        #span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()

        # Start pruning heads
        assert(self._heads_per_word < 1)
        num_heads_to_keep = int(math.floor(self._heads_per_word * document_length))

        print("#PRUNED HEADS:")
        print(num_heads_to_keep)

        head_prune_scores = self._token_mask_ff(text_embeddings)


        # TODO check how the mask behaves for additional tokens at the beggining/end
        head_mask = text_mask.bool()
        head_prune_scores = head_prune_scores.squeeze(-1)
        print("IN:")
        print(head_prune_scores.shape)
        print(head_mask.shape)
        top_head_scores, top_head_mask, top_head_indices = util.masked_topk(
            head_prune_scores, head_mask, num_heads_to_keep
        )
        print("OUT:")
        print(top_head_scores.shape)
        print(top_head_mask.shape)
        print(top_head_indices.shape)

        flat_top_head_indices = util.flatten_and_batch_shift_indices(top_head_indices, seq_length)
        top_head_embeddings = util.batched_index_select(text_embeddings.contiguous(), top_head_indices, flat_top_head_indices)
        top_head_fold_maps = util.batched_index_select(unfold_map.contiguous(), top_head_indices, flat_top_head_indices)
        pruned_mask = util.batched_index_select(text_mask.unsqueeze(-1), top_head_indices, flat_top_head_indices).squeeze(-1)
        print(top_head_embeddings.shape)
        print(top_head_fold_maps.shape)

        column_mask = torch.zeros_like(all_attentions, dtype=torch.bool)
        column_mask = two_dim_fill(column_mask, 1, top_head_fold_maps)

        row_mask =column_mask.permute(0, 1, 2, 4, 3)

        attention_prune_mask = column_mask * row_mask

        flat_pruned_attention = torch.masked_select(all_attentions, attention_prune_mask)

        print(flat_pruned_attention.shape)


        head_link_matrix = self._final_link_attention(top_head_embeddings, top_head_embeddings)

        pruned_attention_matrix = torch.zeros_like(head_link_matrix).unsqueeze(0).repeat(all_attentions.size(0), 1, 1, 1)

        pruned_attention_mask = torch.zeros_like(pruned_attention_matrix, dtype=torch.bool)

        pruned_attention_mask = get_total_attention_matrix(pruned_attention_mask, top_head_fold_maps, num_seg=num_seg)

        print(torch.sum(pruned_attention_mask.bool()))
        #flat_pruned_attention = torch.ones(torch.sum(pruned_attention_mask.bool()), device=pruned_attention_mask.device)
        pruned_attention_matrix.masked_scatter_(pruned_attention_mask, flat_pruned_attention)

        pruned_attention_matrix = pruned_attention_matrix.permute(1,0,2,3)

        weighted_attentions_w = F.softmax(self.weighted_attention_logits)
        weighted_all_attention = weighted_attentions_w.view(1, -1, 1, 1) * pruned_attention_matrix
        weighted_all_attention = torch.sum(weighted_all_attention, 1)

        head_link_matrix = head_link_matrix + weighted_all_attention

        if self._token_mask_ff is not None:
            token_mask = self._token_mask_ff(top_head_embeddings)
            token_mask_attention = torch.matmul(token_mask, token_mask.transpose(1, 2))
            head_link_matrix = head_link_matrix * token_mask_attention

        pruned_length = head_link_matrix.shape[-1]
        dummy_score = head_link_matrix.new_ones(batch_size, pruned_length, 1)
        head_link_matrix = torch.cat([dummy_score, head_link_matrix], dim=-1)

        # Shape (doc_length, doc_length)
        unidirection_mask = self._get_unidirectional_mask(pruned_length, dev=text_mask.device)
        # Shape (batch_size, doc_length, doc_length)
        batched_uni_mask = unidirection_mask.expand(batch_size, -1, -1)
        batched_uni_mask = batched_uni_mask * pruned_mask.unsqueeze(-1)

        # Shape (batch_size, doc_length, doc_length+1)
        batched_uni_mask = torch.cat([batched_uni_mask.new_ones(batch_size, pruned_length, 1), batched_uni_mask], dim=-1)

        head_link_matrix = head_link_matrix * batched_uni_mask

        head_link_distribution = util.masked_log_softmax(head_link_matrix, batched_uni_mask, -1)


        # Shape: (batch_size, document_length)
        _, predicted_antecedents = head_link_distribution.max(-1)
        predicted_antecedents -= 1



        output_dict = {
            "predicted_antecedents": predicted_antecedents
        }

        if span_labels is not None:
            # # Find the gold labels for the spans which we kept.
            # pruned_gold_labels = util.batched_index_select(
            #     span_labels.unsqueeze(-1), top_span_indices, flat_top_span_indices
            # )
            #
            # antecedent_labels = util.flattened_index_select(
            #     pruned_gold_labels, valid_antecedent_indices
            # ).squeeze(-1)
            # antecedent_labels += valid_antecedent_log_mask.long()
            #
            # # Compute labels.
            # # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
            # gold_antecedent_labels = self._compute_antecedent_gold_labels(
            #     pruned_gold_labels, antecedent_labels
            # )
            print(span_labels.shape)
            pruned_span_labels = util.batched_index_select(span_labels.unsqueeze(-1), top_head_indices, flat_top_head_indices).squeeze(-1)
            #top_head_embeddings = util.batched_index_select(text_embeddings.contiguous(), top_head_indices, flat_top_head_indices)

            #Shape： (batch_size, seq_len, seq_len+1)
            gold_antecedent_labels= self._compute_antecedent_gold_labels(pruned_span_labels)

            # Now, compute the loss using the negative marginal log-likelihood.
            # This is equal to the log of the sum of the probabilities of all antecedent predictions
            # that would be consistent with the data, in the sense that we are minimising, for a
            # given span, the negative marginal log likelihood of all antecedents which are in the
            # same gold cluster as the span we are currently considering. Each span i predicts a
            # single antecedent j, but there might be several prior mentions k in the same
            # coreference cluster that would be valid antecedents. Our loss is the sum of the
            # probability assigned to all valid antecedents. This is a valid objective for
            # clustering as we don't mind which antecedent is predicted, so long as they are in
            #  the same coreference cluster.
            coreference_log_probs = head_link_distribution
            #coreference_log_probs = util.masked_log_softmax(coreference_scores, top_span_mask)
            # Try min trick
            #min_antecedent_log_probs, _ = (coreference_log_probs * gold_antecedent_labels).min(-1)
            #min_loss = - min_antecedent_log_probs * text_mask
            #min_loss = min_loss.sum()


            correct_antecedent_log_probs = coreference_log_probs + gold_antecedent_labels.log()
            negative_marginal_log_likelihood = -util.logsumexp(correct_antecedent_log_probs)
            negative_marginal_log_likelihood = negative_marginal_log_likelihood * pruned_mask
            negative_marginal_log_likelihood = negative_marginal_log_likelihood.sum()
            #print(negative_marginal_log_likelihood)

            try:
                self._conll_coref_scores(
                    predicted_antecedents, metadata
                )
            except:
                torch.set_printoptions(profile="full")
                print(head_link_matrix)
                print(predicted_antecedents)
                print(head_link_distribution)
                torch.set_printoptions(profile="default")
                exit()


            output_dict["loss"] = negative_marginal_log_likelihood
            #output_dict["loss"] = min_loss


        if metadata is not None:
            output_dict["document"] = [x["original_text"] for x in metadata]
        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]):
        """
        Converts the list of spans and predicted antecedent indices into clusters
        of spans for each element in the batch.
        Parameters
        ----------
        output_dict : ``Dict[str, torch.Tensor]``, required.
            The result of calling :func:`forward` on an instance or batch of instances.
        Returns
        -------
        The same output dictionary, but with an additional ``clusters`` key:
        clusters : ``List[List[List[Tuple[int, int]]]]``
            A nested list, representing, for each instance in the batch, the list of clusters,
            which are in turn comprised of a list of (start, end) inclusive spans into the
            original document.
        """

        # A tensor of shape (batch_size, doc_length) representing, for each span,
        # the index into ``antecedent_indices`` which specifies the antecedent span. Additionally,
        # the index can be -1, specifying that the span has no predicted antecedent.
        batch_predicted_antecedents = output_dict["predicted_antecedents"].detach().cpu()

        batch_clusters: List[List[List[Tuple[int, int]]]] = []

        # Calling zip() on two tensors results in an iterator over their
        # first dimension. This is iterating over instances in the batch.
        for batch_idx, predicted_antecedents in enumerate(batch_predicted_antecedents):
            spans_to_cluster_ids: Dict[Tuple[int, int], int] = {}
            clusters: List[List[Tuple[int, int]]] = []

            for cur_index, predicted_antecedent in enumerate(predicted_antecedents):
                if predicted_antecedent < 0:
                    # We don't care about spans which are
                    # not co-referent with anything.
                    continue

                # Find the right cluster to update with this span.
                # To do this, we find the row in ``antecedent_indices``
                # corresponding to this span we are considering.
                # The predicted antecedent is then an index into this list
                # of indices, denoting the span from ``top_spans`` which is the
                # most likely antecedent.
                predicted_index = predicted_antecedent

                antecedent_span = (
                    predicted_index.item(),
                    predicted_index.item(),
                )

                # Check if we've seen the span before.
                if antecedent_span in spans_to_cluster_ids:
                    predicted_cluster_id: int = spans_to_cluster_ids[antecedent_span]
                else:
                    # We start a new cluster.
                    predicted_cluster_id = len(clusters)
                    # Append a new cluster containing only this span.
                    clusters.append([antecedent_span])
                    # Record the new id of this span.
                    spans_to_cluster_ids[antecedent_span] = predicted_cluster_id

                # Now add the span we are currently considering.
                span_start, span_end = cur_index, cur_index
                clusters[predicted_cluster_id].append((span_start, span_end))
                spans_to_cluster_ids[(span_start, span_end)] = predicted_cluster_id
            batch_clusters.append(clusters)

        output_dict["clusters"] = batch_clusters
        return output_dict


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        #mention_recall = self._mention_recall.get_metric(reset)
        coref_precision, coref_recall, coref_f1 = self._conll_coref_scores.get_metric(reset)

        return {
            "coref_precision": coref_precision,
            "coref_recall": coref_recall,
            "coref_f1": coref_f1,
            #"mention_recall": mention_recall,
        }

    @staticmethod
    def _get_unidirectional_mask(n, dev):
        """return a mask that only the upper-triangler is one"""
        all_ones = torch.ones(n, n).to(dev)
        uni_mask = torch.tril(all_ones) - torch.eye(n, n).to(dev)
        return uni_mask







    @staticmethod
    def _compute_antecedent_gold_labels(
        span_labels: torch.IntTensor
    ):
        """
        Generates a binary indicator for every pair of spans. This label is one if and
        only if the pair of spans belong to the same cluster. The labels are augmented
        with a dummy antecedent at the zeroth position, which represents the prediction
        that a span does not have any antecedent.
        Parameters
        ----------
        top_span_labels : ``torch.IntTensor``, required.
            The cluster id label for every span. The id is arbitrary,
            as we just care about the clustering. Has shape (batch_size, num_spans_to_keep).
        Returns
        -------
        pairwise_labels_with_dummy_label : ``torch.FloatTensor``
            A binary tensor representing whether a given pair of spans belong to
            the same cluster in the gold clustering.
            Has shape (batch_size, num_spans_to_keep, max_antecedents + 1).
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        seq_len = span_labels.shape[-1]
        span_labels_exp = span_labels.unsqueeze(-1).expand(-1, -1, seq_len)
        same_cluster_indicator = (span_labels_exp == span_labels_exp.permute(0, 2, 1)).float()
        non_dummy_indicator = (span_labels_exp >= 0).float()
        pairwise_labels = same_cluster_indicator * non_dummy_indicator

        # Shape: (batch_size, num_spans_to_keep, 1)
        dummy_labels = (1 - pairwise_labels).prod(-1, keepdim=True)

        # Shape: (batch_size, seq_len, seq_len + 1)
        pairwise_labels_with_dummy_label = torch.cat([dummy_labels, pairwise_labels], -1)
        return pairwise_labels_with_dummy_label

    def _compute_coreference_scores(
        self,
        pairwise_embeddings: torch.FloatTensor,
        top_span_mention_scores: torch.FloatTensor,
        antecedent_mention_scores: torch.FloatTensor,
        antecedent_log_mask: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Computes scores for every pair of spans. Additionally, a dummy label is included,
        representing the decision that the span is not coreferent with anything. For the dummy
        label, the score is always zero. For the true antecedent spans, the score consists of
        the pairwise antecedent score and the unary mention scores for the span and its
        antecedent. The factoring allows the model to blame many of the absent links on bad
        spans, enabling the pruning strategy used in the forward pass.
        Parameters
        ----------
        pairwise_embeddings: ``torch.FloatTensor``, required.
            Embedding representations of pairs of spans. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, encoding_dim)
        top_span_mention_scores: ``torch.FloatTensor``, required.
            Mention scores for every span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).
        antecedent_mention_scores: ``torch.FloatTensor``, required.
            Mention scores for every antecedent. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).
        antecedent_log_mask: ``torch.FloatTensor``, required.
            The log of the mask for valid antecedents.
        Returns
        -------
        coreference_scores: ``torch.FloatTensor``
            A tensor of shape (batch_size, num_spans_to_keep, max_antecedents + 1),
            representing the unormalised score for each (span, antecedent) pair
            we considered.
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        antecedent_scores = self._antecedent_scorer(
            self._antecedent_feedforward(pairwise_embeddings)
        ).squeeze(-1)
        antecedent_scores += top_span_mention_scores + antecedent_mention_scores
        antecedent_scores += antecedent_log_mask

        # Shape: (batch_size, num_spans_to_keep, 1)
        shape = [antecedent_scores.size(0), antecedent_scores.size(1), 1]
        dummy_scores = antecedent_scores.new_zeros(*shape)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        coreference_scores = torch.cat([dummy_scores, antecedent_scores], -1)
        return coreference_scores