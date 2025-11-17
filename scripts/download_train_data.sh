#!/bin/bash

set -e

echo "Downloading files..."

GOOGLE_DRIVE_MODEL_ID="1TGFG0dW5M3rBErgU8i0N7M1ys9YMIvgm"
MODEL_OUTPUT="lrw_resnet18_dctcn_video_boundary.pth"

if [ ! -f "${MODEL_OUTPUT}" ]; then
    echo "Downloading model weights from Google Drive: ${MODEL_OUTPUT}"
    uv run gdown ${GOOGLE_DRIVE_MODEL_ID} -O ${MODEL_OUTPUT}
else
    echo "${MODEL_OUTPUT} already exists, skipping..."
fi

GOOGLE_DRIVE_DATASET_ID="10b1yyPS8tm_zzVOe4M8thkGqDiT5bw7-"
DATASET_ZIP="dla_dataset.zip"
DATASET_DIR="dla_dataset"

if [ ! -d "${DATASET_DIR}" ]; then
    echo "Downloading dataset from Google Drive: ${DATASET_ZIP}"
    uv run gdown ${GOOGLE_DRIVE_DATASET_ID} -O ${DATASET_ZIP}
    
    echo "Extracting dataset..."
    unzip -q ${DATASET_ZIP}
    rm ${DATASET_ZIP}
    echo "Dataset extracted"
else
    echo "${DATASET_DIR} already exists, skipping..."
fi

echo ""
echo "Download complete!"

