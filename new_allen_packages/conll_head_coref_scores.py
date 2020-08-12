from typing import Any, Dict, List, Tuple, Iterable
from collections import Counter

from overrides import overrides
import numpy as np
import torch

from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.conll_coref_scores import Scorer, ConllCorefScores
from .fullmetric_conll_coref_scores import FullMetricConllCorefScores

@Metric.register("conll_head_coref_scores")
class ConllHeadCorefScores(ConllCorefScores):
    def __init__(self) -> None:
        super().__init__()

    @overrides
    def __call__(self,  # type: ignore
                 predicted_antecedents: torch.Tensor,
                 metadata_list: List[Dict[str, Any]]):
        """
        Parameters
        ----------
        top_spans : ``torch.Tensor``
            (start, end) indices for all spans kept after span pruning in the model.
            Expected shape: (batch_size, num_spans, 2)
        antecedent_indices : ``torch.Tensor``
            For each span, the indices of all allowed antecedents for that span.  This is
            independent of the batch dimension, as it's just based on order in the document.
            Expected shape: (num_spans, num_antecedents)
        predicted_antecedents: ``torch.Tensor``
            For each span, this contains the index (into antecedent_indices) of the most likely
            antecedent for that span.
            Expected shape: (batch_size, num_spans)
        metadata_list : ``List[Dict[str, Any]]``
            A metadata dictionary for each instance in the batch.  We use the "clusters" key from
            this dictionary, which has the annotated gold coreference clusters for that instance.
        """
        predicted_antecedents = predicted_antecedents.detach().cpu()
        for i, metadata in enumerate(metadata_list):
            gold_clusters, mention_to_gold = self.get_gold_clusters(metadata["clusters"])
            predicted_clusters, mention_to_predicted = self.get_predicted_clusters(predicted_antecedents[i])
            for scorer in self.scorers:
                scorer.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

    @staticmethod
    @overrides
    def get_predicted_clusters(predicted_antecedents: torch.Tensor) -> Tuple[List[Tuple[Tuple[int, int], ...]],
                                                                             Dict[Tuple[int, int],
                                                                                  Tuple[Tuple[int, int], ...]]]:
        # Pytorch 0.4 introduced scalar tensors, so our calls to tuple() and such below don't
        # actually give ints unless we convert to numpy first.  So we do that here.
        # TODO pruning based on mask?
        #predicted_antecedents = predicted_antecedents.numpy()  # (num_spans,)
        predicted_antecedents = predicted_antecedents.numpy()  # (num_spans,)

        predicted_clusters_to_ids: Dict[Tuple[int, int], int] = {}
        clusters: List[List[Tuple[int, int]]] = []
        for i, predicted_antecedent in enumerate(predicted_antecedents):
            if predicted_antecedent < 0:
                continue

            # Find predicted index in the antecedent spans.
            predicted_index = predicted_antecedent
            # Must be a previous span.
            assert i > predicted_index
            antecedent_span: Tuple[int, int] = (predicted_index, predicted_index)  # type: ignore

            # Check if we've seen the span before.
            if antecedent_span in predicted_clusters_to_ids.keys():
                predicted_cluster_id: int = predicted_clusters_to_ids[antecedent_span]
            else:
                # We start a new cluster.
                predicted_cluster_id = len(clusters)
                clusters.append([antecedent_span])
                predicted_clusters_to_ids[antecedent_span] = predicted_cluster_id

            mention: Tuple[int, int] = (i, i)  # type: ignore
            clusters[predicted_cluster_id].append(mention)
            predicted_clusters_to_ids[mention] = predicted_cluster_id

        # finalise the spans and clusters.
        final_clusters = [tuple(cluster) for cluster in clusters]
        # Return a mapping of each mention to the cluster containing it.
        mention_to_cluster: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = {
                mention: final_clusters[cluster_id]
                for mention, cluster_id in predicted_clusters_to_ids.items()
                }
        #print(final_clusters)

        return final_clusters, mention_to_cluster


@Metric.register("conll_head_prune_coref_scores")
class ConllHeadPruneCorefScores(ConllCorefScores):
    def __init__(self) -> None:
        super().__init__()

    @overrides
    def __call__(self,  # type: ignore
                 predicted_antecedents: torch.Tensor,
                 top_head_indices: torch.Tensor,
                 metadata_list: List[Dict[str, Any]]):
        """
        Parameters
        ----------
        top_spans : ``torch.Tensor``
            (start, end) indices for all spans kept after span pruning in the model.
            Expected shape: (batch_size, num_spans, 2)
        antecedent_indices : ``torch.Tensor``
            For each span, the indices of all allowed antecedents for that span.  This is
            independent of the batch dimension, as it's just based on order in the document.
            Expected shape: (num_spans, num_antecedents)
        predicted_antecedents: ``torch.Tensor``
            For each span, this contains the index (into antecedent_indices) of the most likely
            antecedent for that span.
            Expected shape: (batch_size, num_spans)
        metadata_list : ``List[Dict[str, Any]]``
            A metadata dictionary for each instance in the batch.  We use the "clusters" key from
            this dictionary, which has the annotated gold coreference clusters for that instance.
        """
        predicted_antecedents = predicted_antecedents.detach().cpu()
        top_head_indices = top_head_indices.detach().cpu()
        for i, metadata in enumerate(metadata_list):
            gold_clusters, mention_to_gold = self.get_gold_clusters(metadata["clusters"])
            predicted_clusters, mention_to_predicted = self.get_predicted_clusters(predicted_antecedents[i], top_head_indices[i])
            for scorer in self.scorers:
                scorer.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

    @staticmethod
    @overrides
    def get_predicted_clusters(predicted_antecedents: torch.Tensor, top_head_indices: torch.Tensor) -> Tuple[List[Tuple[Tuple[int, int], ...]],
                                                                             Dict[Tuple[int, int],
                                                                                  Tuple[Tuple[int, int], ...]]]:
        # Pytorch 0.4 introduced scalar tensors, so our calls to tuple() and such below don't
        # actually give ints unless we convert to numpy first.  So we do that here.
        # TODO pruning based on mask?
        #predicted_antecedents = predicted_antecedents.numpy()  # (num_spans,)
        predicted_antecedents = predicted_antecedents.numpy()  # (num_spans,)

        predicted_clusters_to_ids: Dict[Tuple[int, int], int] = {}
        clusters: List[List[Tuple[int, int]]] = []
        for i, predicted_antecedent in enumerate(predicted_antecedents):
            if predicted_antecedent < 0:
                continue

            real_i = top_head_indices[i].item()
            # assert(real_i == i)
            # print("R")
            # print(real_i)
            # print("I")
            # print(i)
            # # real_i = i

            # Find predicted index in the antecedent spans.
            predicted_index = predicted_antecedent
            # Must be a previous span.
            try:
                assert real_i > predicted_index
            except:
                print(real_i)
                print(predicted_index)
                exit()
            antecedent_span: Tuple[int, int] = (predicted_index, predicted_index)  # type: ignore

            # Check if we've seen the span before.
            if antecedent_span in predicted_clusters_to_ids.keys():
                predicted_cluster_id: int = predicted_clusters_to_ids[antecedent_span]
            else:
                # We start a new cluster.
                predicted_cluster_id = len(clusters)
                clusters.append([antecedent_span])
                predicted_clusters_to_ids[antecedent_span] = predicted_cluster_id

            mention: Tuple[int, int] = (real_i, real_i)  # type: ignore
            clusters[predicted_cluster_id].append(mention)
            predicted_clusters_to_ids[mention] = predicted_cluster_id

        # finalise the spans and clusters.
        final_clusters = [tuple(cluster) for cluster in clusters]
        # Return a mapping of each mention to the cluster containing it.
        mention_to_cluster: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = {
                mention: final_clusters[cluster_id]
                for mention, cluster_id in predicted_clusters_to_ids.items()
                }
        #print(final_clusters)

        return final_clusters, mention_to_cluster








@Metric.register("fullmetric_conll_head_coref_scores")
class FullMetricConllHeadCorefScores(FullMetricConllCorefScores):
    def __init__(self) -> None:
        super().__init__()

    @overrides
    def __call__(self,  # type: ignore
                 predicted_antecedents: torch.Tensor,
                 metadata_list: List[Dict[str, Any]]):
        """
        Parameters
        ----------
        top_spans : ``torch.Tensor``
            (start, end) indices for all spans kept after span pruning in the model.
            Expected shape: (batch_size, num_spans, 2)
        antecedent_indices : ``torch.Tensor``
            For each span, the indices of all allowed antecedents for that span.  This is
            independent of the batch dimension, as it's just based on order in the document.
            Expected shape: (num_spans, num_antecedents)
        predicted_antecedents: ``torch.Tensor``
            For each span, this contains the index (into antecedent_indices) of the most likely
            antecedent for that span.
            Expected shape: (batch_size, num_spans)
        metadata_list : ``List[Dict[str, Any]]``
            A metadata dictionary for each instance in the batch.  We use the "clusters" key from
            this dictionary, which has the annotated gold coreference clusters for that instance.
        """
        predicted_antecedents = predicted_antecedents.detach().cpu()
        for i, metadata in enumerate(metadata_list):
            gold_clusters, mention_to_gold = self.get_gold_clusters(metadata["clusters"])
            predicted_clusters, mention_to_predicted = self.get_predicted_clusters(predicted_antecedents[i])
            for scorer in self.scorers:
                scorer.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

    @staticmethod
    @overrides
    def get_predicted_clusters(predicted_antecedents: torch.Tensor) -> Tuple[List[Tuple[Tuple[int, int], ...]],
                                                                             Dict[Tuple[int, int],
                                                                                  Tuple[Tuple[int, int], ...]]]:
        # Pytorch 0.4 introduced scalar tensors, so our calls to tuple() and such below don't
        # actually give ints unless we convert to numpy first.  So we do that here.
        # TODO pruning based on mask?
        #predicted_antecedents = predicted_antecedents.numpy()  # (num_spans,)
        predicted_antecedents = predicted_antecedents.numpy()  # (num_spans,)

        predicted_clusters_to_ids: Dict[Tuple[int, int], int] = {}
        clusters: List[List[Tuple[int, int]]] = []
        for i, predicted_antecedent in enumerate(predicted_antecedents):
            if predicted_antecedent < 0:
                continue

            # Find predicted index in the antecedent spans.
            predicted_index = predicted_antecedent
            # Must be a previous span.
            assert i > predicted_index
            antecedent_span: Tuple[int, int] = (predicted_index, predicted_index)  # type: ignore

            # Check if we've seen the span before.
            if antecedent_span in predicted_clusters_to_ids.keys():
                predicted_cluster_id: int = predicted_clusters_to_ids[antecedent_span]
            else:
                # We start a new cluster.
                predicted_cluster_id = len(clusters)
                clusters.append([antecedent_span])
                predicted_clusters_to_ids[antecedent_span] = predicted_cluster_id

            mention: Tuple[int, int] = (i, i)  # type: ignore
            clusters[predicted_cluster_id].append(mention)
            predicted_clusters_to_ids[mention] = predicted_cluster_id

        # finalise the spans and clusters.
        final_clusters = [tuple(cluster) for cluster in clusters]
        # Return a mapping of each mention to the cluster containing it.
        mention_to_cluster: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = {
                mention: final_clusters[cluster_id]
                for mention, cluster_id in predicted_clusters_to_ids.items()
                }
        #print(final_clusters)

        return final_clusters, mention_to_cluster


@Metric.register("fullmetric_conll_head_prune_coref_scores")
class FullMetricConllHeadPruneCorefScores(FullMetricConllCorefScores):
    def __init__(self) -> None:
        super().__init__()

    @overrides
    def __call__(self,  # type: ignore
                 predicted_antecedents: torch.Tensor,
                 top_head_indices: torch.Tensor,
                 metadata_list: List[Dict[str, Any]]):
        """
        Parameters
        ----------
        top_spans : ``torch.Tensor``
            (start, end) indices for all spans kept after span pruning in the model.
            Expected shape: (batch_size, num_spans, 2)
        antecedent_indices : ``torch.Tensor``
            For each span, the indices of all allowed antecedents for that span.  This is
            independent of the batch dimension, as it's just based on order in the document.
            Expected shape: (num_spans, num_antecedents)
        predicted_antecedents: ``torch.Tensor``
            For each span, this contains the index (into antecedent_indices) of the most likely
            antecedent for that span.
            Expected shape: (batch_size, num_spans)
        metadata_list : ``List[Dict[str, Any]]``
            A metadata dictionary for each instance in the batch.  We use the "clusters" key from
            this dictionary, which has the annotated gold coreference clusters for that instance.
        """
        predicted_antecedents = predicted_antecedents.detach().cpu()
        top_head_indices = top_head_indices.detach().cpu()
        for i, metadata in enumerate(metadata_list):
            gold_clusters, mention_to_gold = self.get_gold_clusters(metadata["clusters"])
            predicted_clusters, mention_to_predicted = self.get_predicted_clusters(predicted_antecedents[i], top_head_indices[i])
            for scorer in self.scorers:
                scorer.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

    @staticmethod
    @overrides
    def get_predicted_clusters(predicted_antecedents: torch.Tensor, top_head_indices: torch.Tensor) -> Tuple[List[Tuple[Tuple[int, int], ...]],
                                                                             Dict[Tuple[int, int],
                                                                                  Tuple[Tuple[int, int], ...]]]:
        # Pytorch 0.4 introduced scalar tensors, so our calls to tuple() and such below don't
        # actually give ints unless we convert to numpy first.  So we do that here.
        # TODO pruning based on mask?
        #predicted_antecedents = predicted_antecedents.numpy()  # (num_spans,)
        predicted_antecedents = predicted_antecedents.numpy()  # (num_spans,)

        predicted_clusters_to_ids: Dict[Tuple[int, int], int] = {}
        clusters: List[List[Tuple[int, int]]] = []
        for i, predicted_antecedent in enumerate(predicted_antecedents):
            if predicted_antecedent < 0:
                continue

            real_i = top_head_indices[i].item()
            # assert(real_i == i)
            # print("R")
            # print(real_i)
            # print("I")
            # print(i)
            # # real_i = i

            # Find predicted index in the antecedent spans.
            predicted_index = predicted_antecedent
            # Must be a previous span.
            try:
                assert real_i > predicted_index
            except:
                print(real_i)
                print(predicted_index)
                exit()
            antecedent_span: Tuple[int, int] = (predicted_index, predicted_index)  # type: ignore

            # Check if we've seen the span before.
            if antecedent_span in predicted_clusters_to_ids.keys():
                predicted_cluster_id: int = predicted_clusters_to_ids[antecedent_span]
            else:
                # We start a new cluster.
                predicted_cluster_id = len(clusters)
                clusters.append([antecedent_span])
                predicted_clusters_to_ids[antecedent_span] = predicted_cluster_id

            mention: Tuple[int, int] = (real_i, real_i)  # type: ignore
            clusters[predicted_cluster_id].append(mention)
            predicted_clusters_to_ids[mention] = predicted_cluster_id

        # finalise the spans and clusters.
        final_clusters = [tuple(cluster) for cluster in clusters]
        # Return a mapping of each mention to the cluster containing it.
        mention_to_cluster: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = {
                mention: final_clusters[cluster_id]
                for mention, cluster_id in predicted_clusters_to_ids.items()
                }
        #print(final_clusters)

        return final_clusters, mention_to_cluster



