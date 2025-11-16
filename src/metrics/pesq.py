from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from src.metrics.base_metric import BaseMetric


class PESQMetric(BaseMetric):
    """
    PESQ Metric with PIT (Permutation Invariant Training)
    """

    def __init__(self, device="cpu", name=None):
        super().__init__(name=name)
        self.pit_pesq = PermutationInvariantTraining(
            PerceptualEvaluationSpeechQuality(fs=16000, mode="wb"),
            mode="speaker-wise",
            eval_func="max",
        )

    def __call__(self, est_source, true_source, mixture, **kwargs):
        """
        Args:
            est_source: Estimated source [batch, num_sources, samples]
            true_source: True source [batch, num_sources, samples]
            mixture: Mixture ignored here
        Returns:
            pesq: PESQ score (with PIT)
        """
        self.pit_pesq = self.pit_pesq.to(est_source.device)
        est_pesq = self.pit_pesq(est_source, true_source)
        return est_pesq.mean().item()
