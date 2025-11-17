#!/usr/bin/env python3

import os
import subprocess
import zipfile
from pathlib import Path


def download_google_drive(file_id: str, output_path: str):
    """Download file from Google Drive using gdown."""
    print(f"Downloading from Google Drive to {output_path}...")
    subprocess.run(["gdown", file_id, "-O", output_path], check=True)




def extract_zip(zip_path: str):
    """Extract zip file."""
    if zipfile.is_zipfile(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete!")
    else:
        print(f"{zip_path} is not a valid zip file")


def main():
    model_id = "1TGFG0dW5M3rBErgU8i0N7M1ys9YMIvgm"
    model_output = "lrw_resnet18_dctcn_video_boundary.pth"
    
    dataset_id = "10b1yyPS8tm_zzVOe4M8thkGqDiT5bw7-"
    dataset_zip = "dla_dataset.zip"
    dataset_dir = "dla_dataset"
    
    model_success = False
    dataset_success = False
    
    try:
        if not Path(model_output).exists():
            print("Downloading model weights from Google Drive...")
            download_google_drive(model_id, model_output)
            model_success = True
            print(f"{model_output} downloaded successfully\n")
        else:
            print(f"{model_output} already exists, skipping...\n")
            model_success = True
    except Exception as e:
        print(f"Model download failed: {e}\n")
    
    if not Path(dataset_dir).exists():
        try:
            print("Downloading dataset from Google Drive...")
            download_google_drive(dataset_id, dataset_zip)
            
            if Path(dataset_zip).exists():
                print("Extracting dataset...")
                extract_zip(dataset_zip)
                os.remove(dataset_zip)
                dataset_success = True
                print(f"Dataset extracted to {dataset_dir}\n")
        except Exception as e:
            print(f"Dataset download failed: {e}\n")
    else:
        print(f"{dataset_dir} already exists, skipping...\n")
        dataset_success = True
    
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    if model_success:
        print(f"✓ Model weights: {model_output}")
    else:
        print(f"✗ Model weights: Failed")
        print(f"  Manual download: https://drive.google.com/file/d/{model_id}/view")
    
    if dataset_success:
        print(f"✓ Dataset: {dataset_dir}/")
    else:
        print(f"✗ Dataset: Failed")
        print(f"  Manual download: https://drive.google.com/file/d/{dataset_id}/view")
    
    print("=" * 60)
    
    if model_success and dataset_success:
        print("\n✓ All downloads complete! Ready to train.")
    else:
        print("\n⚠ Some downloads failed. Please download manually.")


if __name__ == "__main__":
    main()

