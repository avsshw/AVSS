HYDRA_FULL_ERROR=1 uv run python3 inference.py \
    datasets.test.data_dir=inference_dataset \
    inferencer.from_pretrained=saved/testing/model_best.pth \
    model.use_video=True