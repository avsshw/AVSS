import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchaudio

from src.datasets.base_dataset import BaseDataset


class Dataset(BaseDataset):
    def __init__(self, part: str, data_dir: str, *args, **kwargs):
        self._data_dir = Path(data_dir)
        index = self._create_index(part)
        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        item = self._index[ind]

        def load_audio(path):
            if path is None:
                return None
            waveform, _ = torchaudio.load(path)
            waveform = waveform.squeeze(0)
            return waveform

        def load_mouth(path):
            if path is None:
                return None
            with np.load(path) as data:
                return torch.from_numpy(data["data"]).float()

        result = {
            "mix": load_audio(item["mix"]),
            "label_1": load_audio(item["label_1"]),
            "label_2": load_audio(item["label_2"]),
            "mouths_1": load_mouth(item["mouths_1"]),
            "mouths_2": load_mouth(item["mouths_2"]),
            "mix_path": item["mix"],  # Add the mix file path
        }

        result = self.preprocess_data(result)
        return result

    def _create_index(self, part: str) -> List[Dict[str, str]]:
        index = []
        audio_dir = self._data_dir / "audio" / part
        mouths_dir = self._data_dir / "mouths"

        mix_dir = audio_dir / "mix"
        s1_dir = audio_dir / "s1"
        s2_dir = audio_dir / "s2"

        for mix_name in os.listdir(str(mix_dir)):
            if not mix_name.endswith(".wav"):
                continue

            mix_path = str(mix_dir / mix_name)
            if part == "test":
                index.append(
                    {
                        "path": mix_path,
                        "mix": mix_path,
                        "label_1": None,
                        "label_2": None,
                        "mouths_1": None,
                        "mouths_2": None,
                    }
                )
            else:
                stem = mix_name[:-4]
                spk1_id, spk2_id = stem.split("_")
                s1_path = str(s1_dir / mix_name)
                s2_path = str(s2_dir / mix_name)
                mouths1_path = str(mouths_dir / f"{spk1_id}.npz")
                mouths2_path = str(mouths_dir / f"{spk2_id}.npz")
                index.append(
                    {
                        "path": mix_path,
                        "mix": mix_path,
                        "label_1": s1_path,
                        "label_2": s2_path,
                        "mouths_1": mouths1_path,
                        "mouths_2": mouths2_path,
                    }
                )

        return index
