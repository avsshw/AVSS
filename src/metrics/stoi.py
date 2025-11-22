from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import BaseMetric


class STOIMetric(BaseMetric):
    """
    STOI Metric with PIT (Permutation Invariant Training)
    """

    def __init__(self, device="cpu", name=None):
        super().__init__(name=name)
        self.device = device
        stoi_base = ShortTimeObjectiveIntelligibility(fs=16000)
        self.metric = PermutationInvariantTraining(
            stoi_base, mode="speaker-wise", eval_func="max"
        ).to(device)

    def __call__(self, est_source, true_source, mixture, **kwargs):
        """
        Args:
            est_source: Estimated source [batch, num_sources, samples]
            true_source: True source [batch, num_sources, samples]
            mixture: Mixture signal [batch, samples]
        Returns:
            stoi: STOI score averaged over batch and sources (with PIT)
        """
        score = self.metric(est_source, true_source)
        return score.item()
