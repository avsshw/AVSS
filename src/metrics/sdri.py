from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional.audio import signal_distortion_ratio

from src.metrics.base_metric import BaseMetric


class SDRiMetric(BaseMetric):
    """
    SDRi Metric with PIT (Permutation Invariant Training)
    """

    def __init__(self, device="cpu", name=None):
        super().__init__(name=name)
        self.pit_sdri = PermutationInvariantTraining(
            signal_distortion_ratio, mode="speaker-wise", eval_func="max"
        )

    def __call__(self, est_source, true_source, mixture, **kwargs):
        """
        Args:
            est_source: Estimated source [batch, num_sources, samples]
            true_source: True source [batch, num_sources, samples]
            mixture: Mixture signal [batch, samples]
        Returns:
            sdri: SDR improvement
        """
        batch_size, num_sources, _ = true_source.shape
        est_sdri = self.pit_sdri(est_source, true_source)
        mixture_expanded = mixture.unsqueeze(1).expand(-1, num_sources, -1)
        mix_flat = mixture_expanded.reshape(-1, mixture.size(-1))
        true_flat = true_source.reshape(-1, true_source.size(-1))
        mix_sdri_flat = signal_distortion_ratio(mix_flat, true_flat)
        mix_sdri = mix_sdri_flat.view(batch_size, num_sources).mean(dim=1)
        sdri = est_sdri - mix_sdri
        return sdri.mean().item()
