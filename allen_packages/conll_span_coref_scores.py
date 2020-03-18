from typing import Any, Dict, List, Tuple, Iterable
from collections import Counter

from overrides import overrides
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np
import torch

from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.conll_coref_scores import Scorer

@Metric.register("conll_span_coref_scores")
class ConllSpanCorefScores(Metric):
    def __init__(self) -> None:
        self.scorers = [Scorer(m) for m in (Scorer.muc, Scorer.b_cubed, Scorer.ceafe)]

    @overrides
    def __call__(
        self,  # type: ignore
        top_spans: torch.Tensor,
        antecedent_indices: torch.Tensor,
        predicted_antecedents: torch.Tensor,
        metadata_list: List[Dict[str, Any]],
    ):
        """
        # Parameters
        top_spans : `torch.Tensor`
            (start, end) indices for all spans kept after span pruning in the model.
            Expected shape: (batch_size, num_spans, 2)
        antecedent_indices : `torch.Tensor`
            For each span, the indices of all allowed antecedents for that span.
            Expected shape: (batch_size, num_spans, num_antecedents)
        predicted_antecedents : `torch.Tensor`
            For each span, this contains the index (into antecedent_indices) of the most likely
            antecedent for that span.
            Expected shape: (batch_size, num_spans)
        metadata_list : `List[Dict[str, Any]]`
            A metadata dictionary for each instance in the batch.  We use the "clusters" key from
            this dictionary, which has the annotated gold coreference clusters for that instance.
        """
        top_spans, antecedent_indices, predicted_antecedents = self.detach_tensors(
            top_spans, antecedent_indices, predicted_antecedents
        )

        # They need to be in CPU because Scorer.ceafe uses a SciPy function.
        top_spans = top_spans.cpu()
        antecedent_indices = antecedent_indices.cpu()
        predicted_antecedents = predicted_antecedents.cpu()

        for i, metadata in enumerate(metadata_list):
            gold_clusters, mention_to_gold = self.get_gold_clusters(metadata["clusters"])
            predicted_clusters, mention_to_predicted = self.get_predicted_clusters(
                top_spans[i], antecedent_indices[i], predicted_antecedents[i]
            )
            for scorer in self.scorers:
                scorer.update(
                    predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold
                )

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float, float]:
        metrics = (lambda e: e.get_precision(), lambda e: e.get_recall(), lambda e: e.get_f1())
        precision, recall, f1_score = tuple(
            sum(metric(e) for e in self.scorers) / len(self.scorers) for metric in metrics
        )
        if reset:
            self.reset()
        return precision, recall, f1_score

    @overrides
    def reset(self):
        self.scorers = [Scorer(metric) for metric in (Scorer.muc, Scorer.b_cubed, Scorer.ceafe)]

    @staticmethod
    def get_gold_clusters(gold_clusters):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gold_cluster in gold_clusters:
            for mention in gold_cluster:
                mention_to_gold[mention] = gold_cluster
        return gold_clusters, mention_to_gold

    @staticmethod
    def get_predicted_clusters(
        top_spans: torch.Tensor,  # (num_spans, 2)
        antecedent_indices: torch.Tensor,  # (num_spans, num_antecedents)
        predicted_antecedents: torch.Tensor,  # (num_spans,)
    ) -> Tuple[
        List[Tuple[Tuple[int, int], ...]], Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]]
    ]:

        predicted_clusters_to_ids: Dict[Tuple[int, int], int] = {}
        clusters: List[List[Tuple[int, int]]] = []
        for i, predicted_antecedent in enumerate(predicted_antecedents):
            if predicted_antecedent < 0:
                continue

            # Find predicted index in the antecedent spans.
            predicted_index = antecedent_indices[i, predicted_antecedent]
            # Must be a previous span.
            assert i > predicted_index
            antecedent_span: Tuple[int, int] = tuple(  # type: ignore
                top_spans[predicted_index].tolist()
            )

            # Check if we've seen the span before.
            if antecedent_span in predicted_clusters_to_ids.keys():
                predicted_cluster_id: int = predicted_clusters_to_ids[antecedent_span]
            else:
                # We start a new cluster.
                predicted_cluster_id = len(clusters)
                clusters.append([antecedent_span])
                predicted_clusters_to_ids[antecedent_span] = predicted_cluster_id

            mention: Tuple[int, int] = tuple(top_spans[i].tolist())  # type: ignore
            clusters[predicted_cluster_id].append(mention)
            predicted_clusters_to_ids[mention] = predicted_cluster_id

        # finalise the spans and clusters.
        final_clusters = [tuple(cluster) for cluster in clusters]
        # Return a mapping of each mention to the cluster containing it.
        mention_to_cluster: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = {
            mention: final_clusters[cluster_id]
            for mention, cluster_id in predicted_clusters_to_ids.items()
        }

        return final_clusters, mention_to_cluster




    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)
