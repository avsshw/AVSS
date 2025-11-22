import torch

# This collate_fn is written with the hope that it will be used both for simple without-video architectures like ConvTasNet and
# for more complicated like RTFS-Net and CTCNet


def collate_fn(dataset_items: list[dict], use_video=False, use_sources=True):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    mix = torch.stack([item["mix"] for item in dataset_items])
    batch = {"mix": mix}

    if "mix_path" in dataset_items[0]:
        batch["mix_path"] = [item["mix_path"] for item in dataset_items]

    if use_sources and dataset_items[0]["label_1"] is not None:
        sources = torch.zeros(
            (len(dataset_items), 2, dataset_items[0]["label_1"].shape[0]),
            dtype=mix.dtype,
        )
        for i, item in enumerate(dataset_items):
            sources[i, 0] = item["label_1"]
            sources[i, 1] = item["label_2"]
        batch["source"] = sources

    has_video = all(item.get("mouths_1") is not None and item.get("mouths_2") is not None for item in dataset_items)
    if has_video:
        sample_video = dataset_items[0]["mouths_1"]
        videos = torch.zeros((len(dataset_items), 2, *sample_video.shape), dtype=sample_video.dtype)
        for i, item in enumerate(dataset_items):
            videos[i, 0] = item["mouths_1"]
            videos[i, 1] = item["mouths_2"]
        batch["video"] = videos

    if "name" in dataset_items[0]:
        batch["name"] = [item["name"] for item in dataset_items]  # type: ignore[assignment]

    return batch
