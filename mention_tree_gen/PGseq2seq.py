from typing import Dict, List, Tuple

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.attention import LegacyAttention
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.util import masked_log_softmax, masked_softmax
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import BLEU
from allennlp.models.encoder_decoders import SimpleSeq2Seq


@Model.register("PG_seq2seq")
class PGSeq2Seq(SimpleSeq2Seq):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None,
                 beam_size: int = None,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 use_bleu: bool = True,
                 action_mask_getter = None) -> None:
        self._get_class_mask = action_mask_getter
        super().__init__(vocab, source_embedder, encoder, max_decoding_steps, attention, attention_function, beam_size, target_namespace, target_embedding_dim, scheduled_sampling_ratio, use_bleu)



    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                source_span: torch.LongTensor,
                target_tokens: Dict[str, torch.LongTensor] = None,
                loss_weights: torch.Tensor = None,
                sample: bool = False,
                get_prediction: bool=False) -> Dict[str, torch.Tensor]:

        """
        Make foward pass with decoder logic for producing the entire target sequence.
        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.
        loss_weights:
           weights used for per-token cross-entropy, can be used to implement reinforce
        Returns
        -------
        Dict[str, torch.Tensor]
        """
        state = self._encode(source_tokens, source_span)

        if target_tokens:
            state = self._init_decoder_state(state)
            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            assert(sample == False)
            output_dict = self._forward_loop(state, target_tokens, loss_weights, sample)
        else:
            output_dict = {}

        if not self.training or get_prediction:
            state = self._init_decoder_state(state)
            if sample:
                predictions = self._forward_loop(state, None, loss_weights, sample)
            else:
                predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if target_tokens and self._bleu:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                self._bleu(best_predictions, target_tokens["tokens"])

        return output_dict



    def _encode(self, source_tokens: Dict[str, torch.Tensor], source_span: torch.LongTensor) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        source_span = source_span.float()
        concat_input = torch.cat((embedded_input, torch.unsqueeze(source_span, -1)), -1)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(concat_input, source_mask)
        return {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
        }


    #def _get_class_mask(self, action_history):
    #    action_cfgs = self.decoder_vocab.get(action_history)
    #    for cfg in action_cfgs:
    #        pass


    @overrides
    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        if 'total_actions' not in state.keys():
            state['total_actions'] = torch.unsqueeze(last_predictions,0)
        else:
            state['total_actions'] = torch.cat((state['total_actions'], torch.unsqueeze(last_predictions,0)), -1)
        #for k in state.keys():
        #    print(state[k].shape)
        #print(state['total_actions'])
        output_projections, state = self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        #class_log_probabilities = F.log_softmax(output_projections, dim=-1)
        action_history = state['total_actions']
        class_mask = self._get_class_mask(action_history)
        class_log_probabilities = masked_log_softmax(output_projections, class_mask, dim=-1)

        return class_log_probabilities, state



    @overrides
    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor] = None,
                      loss_weights: torch.Tensor=None,
                      sample: bool=False) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.
        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]

        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        action_history = []
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._prepare_output_projections(input_choices, state)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            class_mask = self._get_class_mask(action_history)
            #output_projections = output_projections * class_mask

            # shape: (batch_size, num_classes)
            #class_probabilities = F.softmax(output_projections, dim=-1)
            #print(class_mask)
            #print(output_projections)
            class_probabilities = masked_softmax(output_projections, class_mask, dim=-1, memory_efficient=True)
            safe_class_probabilities = class_probabilities + 1e-45 * (1 - class_mask)
            plogps = -safe_class_probabilities * torch.log(safe_class_probabilities)
            #print(class_probabilities)
            #print(class_mask)
            #print(safe_class_probabilities)
            #print(plogps)
            #print(safe_class_probabilities)
            #print(plogps)
            #print(class_mask)
            #print(plogps*class_mask)
            entropy = torch.sum(plogps*class_mask)
            entropies.append(entropy.detach().cpu().numpy())
            #print(class_probabilities.shape)

            #print(class_mask.shape)
            #class_probabilities = class_probabilities * class_mask

            if not sample:
                # shape (predicted_classes): (batch_size,)
                _, predicted_classes = torch.max(class_probabilities, 1)
            else:
                predicted_classes = torch.multinomial(class_probabilities, 1)[:, 0]
            action_history.append(predicted_classes)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions, "entropies":entropies}

        if target_tokens:
            #print("CHECK1")
            #print(loss_weights)
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            #TODO implement code when loss_weights is not a scalar (e.g. batch)
            if loss_weights is not None:
                loss = self._get_loss(logits, targets, target_mask, average=None)
                loss = loss*loss_weights
            else:
                loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict


    @staticmethod
    @overrides
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor,
                  average: str='batch') -> torch.Tensor:
        """
        Compute loss.
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.
        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.
        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask, average=average)
