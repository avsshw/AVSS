HYDRA_FULL_ERROR=1 uv run python3 inference.py \
    datasets.test.data_dir=inference_dataset \
    inferencer.from_pretrained=rtfs/model_best.pth \
    datasets=custom_dir
