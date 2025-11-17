import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir: str, *args: Any, **kwargs: Any):
        self._data_dir = Path(data_dir)
        index = self._create_index()
        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind: int) -> dict[str, Any]:
        item = self._index[ind]

        def load_audio(path: str | None) -> torch.Tensor | None:
            if path is None:
                return None
            waveform, _ = torchaudio.load(path)
            return waveform.squeeze(0)

        def load_mouth(path: str | None) -> torch.Tensor | None:
            if path is None:
                return None
            with np.load(path) as data:
                return torch.from_numpy(data["data"]).float()

        result: dict[str, Any] = {
            "mix": load_audio(item["mix"]),
            "label_1": load_audio(item["label_1"]),
            "label_2": load_audio(item["label_2"]),
            "mouths_1": load_mouth(item["mouths_1"]),
            "mouths_2": load_mouth(item["mouths_2"]),
            "mix_path": item["mix"],
        }

        if self.instance_transforms is not None:
            result = self.instance_transforms(result)

        return result

    def _create_index(self) -> list[dict[str, str | None]]:
        index: list[dict[str, str | None]] = []

        audio_dir = self._data_dir / "audio"
        mix_dir = audio_dir / "mix"
        s1_dir = audio_dir / "s1"
        s2_dir = audio_dir / "s2"
        mouths_dir = self._data_dir / "mouths"

        if not mix_dir.is_dir():
            msg = f"Mix directory '{mix_dir}' does not exist"
            raise FileNotFoundError(msg)

        allowed_ext = {".wav", ".flac", ".mp3"}

        for mix_name in sorted(os.listdir(mix_dir)):
            mix_path = mix_dir / mix_name
            if not mix_path.is_file() or mix_path.suffix.lower() not in allowed_ext:
                continue

            mix_path_str = str(mix_path)

            s1_path: str | None = None
            s2_path: str | None = None
            mouths1_path: str | None = None
            mouths2_path: str | None = None

            if s1_dir.is_dir():
                candidate = s1_dir / mix_name
                if candidate.is_file():
                    s1_path = str(candidate)

            if s2_dir.is_dir():
                candidate = s2_dir / mix_name
                if candidate.is_file():
                    s2_path = str(candidate)

            stem = mix_path.stem
            if "_" in stem and mouths_dir.is_dir():
                spk1_id, spk2_id = stem.split("_", maxsplit=1)
                m1 = mouths_dir / f"{spk1_id}.npz"
                m2 = mouths_dir / f"{spk2_id}.npz"
                if m1.is_file():
                    mouths1_path = str(m1)
                if m2.is_file():
                    mouths2_path = str(m2)

            index.append(
                {
                    "path": mix_path_str,
                    "mix": mix_path_str,
                    "label_1": s1_path,
                    "label_2": s2_path,
                    "mouths_1": mouths1_path,
                    "mouths_2": mouths2_path,
                }
            )

        return index
