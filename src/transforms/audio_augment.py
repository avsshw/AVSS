import torch
import torch.nn as nn


class Normalize(nn.Module):
    "Normalize audio to [-1, 1] range"

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def __call__(self, audio):
        max_amp = audio.abs().max()
        if max_amp > self.eps:
            scaled = audio / (max_amp + self.eps)
            return scaled
        return audio


class InstanceTransforms(nn.Module):
    "Transforms for our dataset"

    def __init__(self, is_train=True, target_length=48000):
        super().__init__()
        self.train_mode = is_train
        self.target_length = target_length
        self._norm = Normalize()

    def __call__(self, batch):
        if "mix" in batch and batch["mix"] is not None:
            batch["mix"] = self._norm(batch["mix"])

        for label in ["label_1", "label_2"]:
            x = batch.get(label)
            if x is not None:
                batch[label] = self._norm(x)

        if batch.get("mix") is not None:
            cur_len = batch["mix"].shape[-1]
            if cur_len > self.target_length:
                rand_idx = torch.randint(
                    0, cur_len - self.target_length + 1, (1,)
                ).item()
                for name in ["mix", "label_1", "label_2"]:
                    seg = batch.get(name)
                    if seg is not None:
                        batch[name] = seg[..., rand_idx : rand_idx + self.target_length]
            else:
                if cur_len < self.target_length:
                    extra = self.target_length - cur_len
                    for name in ["mix", "label_1", "label_2"]:
                        seg = batch.get(name)
                        if seg is not None:
                            batch[name] = torch.nn.functional.pad(seg, (0, extra))
        return batch
