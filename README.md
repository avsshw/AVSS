1. Download video encoder checkpoint [here](https://drive.google.com/file/d/1TGFG0dW5M3rBErgU8i0N7M1ys9YMIvgm/view)
2. Run `scripts/train.sh` to run one batch test.

### Inference data for custom directory

```sh
echo "deb http://repo.yandex.ru/yandex-disk/deb/ stable main" | sudo tee -a /etc/apt/sources.list.d/yandex-disk.list > /dev/null && wget http://repo.yandex.ru/yandex-disk/YANDEX-DISK-KEY.GPG -O- | sudo apt-key add - && sudo apt-get update && sudo apt-get install -y yandex-disk
```

```sh
yandex-disk setup
```

- **Download inference data from Yandex Disk** (if you want to reproduce the provided DLA setup):

```bash
uv run bash scripts/download_data_yandex_disk.sh
```

This script tries to download from `https://disk.yandex.ru/d/d06t0CmGLj0ijg`. If Yandex shows SmartCaptcha in the browser, download the archive manually, place it as `dla_dataset_inference.zip` in the project root, and rerun the script to extract into `dla_dataset/`.

### Running inference on the final model

- **Default behavior**: predictions are saved under `data/saved/inference_custom_dir/<partition>/`, where each separated source file is named as `<mix_name>_s1.wav` and `<mix_name>_s2.wav`, with `<mix_name>` matching the mixture filename.

- **Run inference on the final RTFS-Net model (two-speaker separation)**:

```bash
HYDRA_FULL_ERROR=1 uv run python3 inference.py
```

By default this uses:
- `src/configs/inference.yaml` (model `rtfsnet`, metrics `base`, transforms `audio_separation`)
- `src/configs/datasets/custom_dir.yaml` with `data_dir: "dla_dataset"` and a `CustomDirDataset` that parses `audio/mix`, optional `audio/s1`, `audio/s2`, and `mouths/`.

- **Override the custom directory path from CLI** (for your own mixed-speech directory):

```bash
HYDRA_FULL_ERROR=1 uv run python3 inference.py datasets.test.data_dir=/path/to/your/custom_dir
```

### Calculating separation metrics from saved predictions

After running inference, given directories with ground-truth and predicted sources, you can compute SI-SNRi, SDRi, PESQ, and STOI:

```bash
uv run python3 scripts/calc_metrics.py \
  --gt_s1_dir /path/to/dla_dataset/audio/val/s1 \
  --gt_s2_dir /path/to/dla_dataset/audio/val/s2 \
  --pred_s1_dir data/saved/inference_custom_dir/val \
  --pred_s2_dir data/saved/inference_custom_dir/val
```

Ground-truth paths can be changed to `train`/`test` or your own structure, as long as filenames match the mixture names and predictions follow the `<mix_name>_s1.wav` / `<mix_name>_s2.wav` convention produced by `inference.py`.