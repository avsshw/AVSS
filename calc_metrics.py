import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from src.metrics.pesq import PESQMetric
from src.metrics.sdri import SDRiMetric
from src.metrics.sisnri import SISNRiMetric
from src.metrics.stoi import STOIMetric


def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics for audio source separation predictions"
    )
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--ground_truth_dir", type=str, required=True)
    parser.add_argument("--mixture_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    gt_dir = Path(args.ground_truth_dir)
    mix_dir = Path(args.mixture_dir)
    device = args.device

    pred_files = sorted(pred_dir.glob("*_s1.wav"))

    sisnri_metric = SISNRiMetric(device=device, name="SI-SNRi")
    sdri_metric = SDRiMetric(device=device, name="SDRi")
    pesq_metric = PESQMetric(device=device, name="PESQ")
    stoi_metric = STOIMetric(device=device, name="STOI")

    metrics_sum = {
        "SI-SNRi": 0.0,
        "SDRi": 0.0,
        "PESQ": 0.0,
        "STOI": 0.0,
    }

    num_samples = 0
    for pred_s1_path in tqdm(pred_files):
        base_name = pred_s1_path.stem.replace("_s1", "")
        pred_s2_path = pred_dir / f"{base_name}_s2.wav"

        gt_s1_path = gt_dir / "s1" / f"{base_name}.wav"
        gt_s2_path = gt_dir / "s2" / f"{base_name}.wav"

        mixture_path = mix_dir / f"{base_name}.wav"

        pred_s1 = load_audio(pred_s1_path).to(device)
        pred_s2 = load_audio(pred_s2_path).to(device)
        gt_s1 = load_audio(gt_s1_path).to(device)
        gt_s2 = load_audio(gt_s2_path).to(device)
        mixture = load_audio(mixture_path).to(device)

        min_len = min(
            pred_s1.shape[0],
            pred_s2.shape[0],
            gt_s1.shape[0],
            gt_s2.shape[0],
            mixture.shape[0],
        )
        pred_s1 = pred_s1[:min_len]
        pred_s2 = pred_s2[:min_len]
        gt_s1 = gt_s1[:min_len]
        gt_s2 = gt_s2[:min_len]
        mixture = mixture[:min_len]

        est_source = torch.stack([pred_s1, pred_s2], dim=0)
        true_source = torch.stack([gt_s1, gt_s2], dim=0)

        est_source_batch = est_source.unsqueeze(0)
        true_source_batch = true_source.unsqueeze(0)
        mixture_batch = mixture.unsqueeze(0)

        sisnri = sisnri_metric(
            est_source=est_source_batch,
            true_source=true_source_batch,
            mixture=mixture_batch,
        )
        sdri = sdri_metric(
            est_source=est_source_batch,
            true_source=true_source_batch,
            mixture=mixture_batch,
        )
        pesq = pesq_metric(
            est_source=est_source_batch,
            true_source=true_source_batch,
            mixture=mixture_batch,
        )
        stoi = stoi_metric(
            est_source=est_source_batch,
            true_source=true_source_batch,
            mixture=mixture_batch,
        )

        metrics_sum["SI-SNRi"] += sisnri
        metrics_sum["SDRi"] += sdri
        metrics_sum["PESQ"] += pesq
        metrics_sum["STOI"] += stoi
        num_samples += 1

    for metric_name, total in metrics_sum.items():
        avg = total / num_samples
        print(f"{metric_name:15s}: {avg:.4f}")


if __name__ == "__main__":
    main()
