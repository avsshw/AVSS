import torch
import torch.nn as nn
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio


class SISNRLoss(nn.Module):
    """
    SISNRLoss with PTI
    """

    def __init__(self, num_sources=2, mode="speaker-wise", eval_func="max"):
        """
        Args:
            num_sources: Number of sources (kept for debug)
            mode
            eval_func
        """
        super().__init__()
        self.pit = PermutationInvariantTraining(
            scale_invariant_signal_noise_ratio,
            mode=mode,
            eval_func=eval_func,
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, **batch):
        """
        Compute PIT-SI-SNR loss

        Args:
            preds: Predicted sources [batch, num_sources, samples]
            targets: True sources [batch, num_sources, samples]
            **batch
        """
        best_metric = self.pit(preds, targets)
        loss = -best_metric
        return {"loss": loss}
