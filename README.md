# Audio-Visual Source Separation

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Inference](#inference)
- [Demo](#demo)
- [Useful Links](#useful)
- [Credits](#credits)
- [License](#license)


## About

"This repository contains a course project implementing **Audio-Visual Source Separation** using the [**RTFS-Net**](https://arxiv.org/pdf/2309.17189) architecture, developed as homework for the [**Deep Learning in Audio & Speech**](https://github.com/markovka17/dla/tree/2025/project_avss)


## Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/avsshw/AVSS.git
cd AVSS
pip install -r requirements.txt
```

The project uses uv for fast dependency management. If you prefer uv, run:
```bash
uv pip install -r requirements.txt
```

## Inference

To run inference on the example dataset:

```bash
# Download example data
sh scripts/download_inference_data.sh

# Download pretrained model
gdown https://drive.google.com/uc?id=1t7FFsG3hPcgUYuitekSMpggYLvzV6SXW
unzip rtfs.zip

# Run inference
sh scripts/inference.sh
```

You can also use your own dataset if it follows the format below and replace the first step with the:

```bash
!uv run python3 scripts/download_inference_data.py --link YOUR_LINK --download_location .
```

```bash
NameOfTheDirectoryWithUtterances
├── audio
│   ├── mix
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   ├── s1 # ground truth for the speaker s1, may not be given
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   └── s2 # ground truth for the speaker s2, may not be given
│       ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│       ├── FirstSpeakerID2_SecondSpeakerID2.wav
│       .
│       .
│       .
│       └── FirstSpeakerIDn_SecondSpeakerIDn.wav
└── mouths # contains video information for all speakers
    ├── FirstOrSecondSpeakerID1.npz # npz mouth-crop
    ├── FirstOrSecondSpeakerID2.npz
    .
    .
    .
    └── FirstOrSecondSpeakerIDn.npz
```

## Evaluation

If your dataset includes ground-truth clean sources (s1/, s2/), you can compute metrics directly (Si-SNRi, SI-SDR, PESQ, STOI) with the separate script:

```bash
uv run python3 calc_metrics.py \
    --predictions_dir data/saved/inference_custom_dir/test \
    --ground_truth_dir inference_dataset/audio \
    --mixture_dir inference_dataset/audio/mix
```


## Demo

A full working example is provided in Demo_AVSS.ipynb notebook included in this repository.
It walks through installation, inference, and evaluation more precisely.



## Useful links

1. [RTFS-Net](https://arxiv.org/pdf/2309.17189) - article with the architecture that was used in the project
2. [Hydra Documentation](https://hydra.cc/docs/intro/) — configuration framework used in this project
3. [PyTorch](https://pytorch.org/) — DL framework used in the project


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
