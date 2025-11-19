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

Download video encoder checkpoint [here](https://drive.google.com/file/d/1TGFG0dW5M3rBErgU8i0N7M1ys9YMIvgm/view) ??

## Inference

To run inference follow these stetps:

```bash
sh scripts/download_inference_data.sh
```

```bash
sh scripts/inference.sh
```

You can also use your own dataset if it follows the format below:

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
In order to do that: ???

If you already have ground truth and prediction files, you can evaluate them directly with the following metrics script: ???


## Demo

 The Installation and Inference steps described above are demonstrated in the ???.ipynb notebook included in this repository.



## Useful links

1. [RTFS-Net](https://arxiv.org/pdf/2309.17189) - article with the architecture that was used in the project
2. [Hydra Documentation](https://hydra.cc/docs/intro/) — configuration framework used in this project
3. [PyTorch](https://pytorch.org/) — DL framework used in the project


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
