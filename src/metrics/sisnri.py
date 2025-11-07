import torch.nn.functional as F
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio

from src.metrics.base_metric import BaseMetric


class SISNRiMetric(BaseMetric):
    def __init__(self, device="cpu", name=None):
        super().__init__(name=name)
        self.pit_sisnr = PermutationInvariantTraining(
            scale_invariant_signal_noise_ratio, mode="speaker-wise", eval_func="max"
        )

    def __call__(self, est_source, true_source, mixture, **kwargs):
        """
        Args:
            est_source: [batch, num_sources, samples]
            true_source: [batch, num_sources, samples]
            mixture: [batch, samples]
        Returns:
            SI-SNRi
        """
        batch_size, num_sources, _ = true_source.shape
        est_sisnr = self.pit_sisnr(est_source, true_source)
        mixture_expanded = mixture.unsqueeze(1).expand(-1, num_sources, -1)
        mix_flat = mixture_expanded.reshape(-1, mixture.size(-1))
        true_flat = true_source.reshape(-1, true_source.size(-1))
        mix_sisnr_flat = scale_invariant_signal_noise_ratio(mix_flat, true_flat)
        mix_sisnr = mix_sisnr_flat.view(batch_size, num_sources).mean(dim=1)
        sisnri = est_sisnr - mix_sisnr
        return sisnri.mean().item()
