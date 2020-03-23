from typing import Optional

from overrides import overrides
import torch

from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.nn import util
from .attention_pretrained_transformer_matched_embedder import AttentionPretrainedTransformerEmbedder


@TokenEmbedder.register("attention_pretrained_transformer_mismatched")
class AttentionPretrainedTransformerMismatchedEmbedder(TokenEmbedder):
    """
    Use this embedder to embed wordpieces given by `PretrainedTransformerMismatchedIndexer`
    and to pool the resulting vectors to get word-level representations.
    # Parameters
    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerMismatchedIndexer`.
    max_length : `int`, optional (default = None)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerMismatchedIndexer`.
    """

    def __init__(self, model_name: str, max_length: int = None) -> None:
        super().__init__()
        # The matched version v.s. mismatched
        self._matched_embedder = AttentionPretrainedTransformerEmbedder(model_name, max_length)

    @overrides
    def get_output_dim(self):
        return self._matched_embedder.get_output_dim()


    def _get_match_attention_by_mean_pooling(self, attentions, offsets, unfold_map):
        print(attentions.shape)
        print(offsets.shape)
        print(unfold_map.shape)
        exit()


    def _get_span_by_mean_pooling(self, emb, offsets):
        span_embeddings, span_mask = util.batched_span_select(emb.contiguous(), offsets)
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask

        span_embeddings_sum = span_embeddings.sum(2)
        span_embeddings_len = span_mask.sum(2)

        orig_embeddings = span_embeddings_sum / span_embeddings_len
        return orig_embeddings

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters
        token_ids: torch.LongTensor
            Shape: [batch_size, num_wordpieces] (for exception see `PretrainedTransformerEmbedder`).
        mask: torch.BoolTensor
            Shape: [batch_size, num_orig_tokens].
        offsets: torch.LongTensor
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in token_ids. `token_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.
        wordpiece_mask: torch.BoolTensor
            Shape: [batch_size, num_wordpieces].
        type_ids: Optional[torch.LongTensor]
            Shape: [batch_size, num_wordpieces].
        segment_concat_mask: Optional[torch.BoolTensor]
            See `PretrainedTransformerEmbedder`.
        # Returns:
        Shape: [batch_size, num_orig_tokens, embedding_size].
        """
        # Shape: [batch_size, num_wordpieces, embedding_size].
        embeddings, attentions, unfold_map = self._matched_embedder(
            token_ids, wordpiece_mask, type_ids=type_ids, segment_concat_mask=segment_concat_mask
        )

        orig_embeddings = self._get_span_by_mean_pooling(embeddings, offsets)

        # # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # # span_mask: (batch_size, num_orig_tokens, max_span_length)
        # span_embeddings, span_mask = util.batched_span_select(embeddings.contiguous(), offsets)
        # span_mask = span_mask.unsqueeze(-1)
        # span_embeddings *= span_mask  # zero out paddings
        #
        # span_embeddings_sum = span_embeddings.sum(2)
        # span_embeddings_len = span_mask.sum(2)
        # # Shape: (batch_size, num_orig_tokens, embedding_size)
        # orig_embeddings = span_embeddings_sum / span_embeddings_len

        orig_block_attentions = self._get_match_attention_by_mean_pooling(attentions, offsets, unfold_map)


        # The following code block compresses self_attention
        # attentions: (batch_size, #attetnion, num_wordpieces1, num_wordpieces2)
        attentions = attentions.contiguous()
        batch_size = attentions.shape[0]
        num_attention = attentions.shape[1]
        num_wordpieces = attentions.shape[2]
        tmp_attentions = attentions.transpose(1, 2)
        # shape [batch, num_wordpieces1, num_attention, num_wordpieces2]
        tmp_attentions = tmp_attentions.reshape(batch_size, num_wordpieces, num_attention * num_wordpieces)
        partial_orig_attentions = self._get_span_by_mean_pooling(tmp_attentions, offsets)
        #Shape [batch, num_word1, #att * #pieces2]
        num_word = partial_orig_attentions.shape[1]
        partial_orig_attentions = partial_orig_attentions.view(batch_size, -1, num_wordpieces).transpose(1, 2)
        # Shape[batch, # piece2, #word1 * #attention]
        orig_attentions = self._get_span_by_mean_pooling(partial_orig_attentions, offsets)
        # Shape[batch, # word2, #word1 * #attention]
        orig_attentions = orig_attentions.view(batch_size, num_word, num_word, num_attention).transpose(1, 3)
        # Shape[batch, # attention, #word1, #word2]

        #return orig_embeddings, orig_attentions
        return orig_embeddings, orig_block_attentions
