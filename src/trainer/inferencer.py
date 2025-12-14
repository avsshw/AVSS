from pathlib import Path

import torch
import torchaudio
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
        speaker_folder_names=None,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
            speaker_folder_names (list[str] | None): list of custom folder names
                for each speaker. If None, defaults to "speaker_1", "speaker_2", etc.
                The length should match the number of sources in the model.
        """
        assert skip_model_load or config.inferencer.get("from_pretrained") is not None, (
            "Provide checkpoint or set skip_model_load=True"
        )

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        num_sources = getattr(self.model, 'num_sources', 2)
        if speaker_folder_names is not None:
            self.speaker_folder_names = speaker_folder_names
        else:
            config_speaker_names = config.inferencer.get("speaker_folder_names", None)
            if config_speaker_names is not None:
                self.speaker_folder_names = config_speaker_names
            else:
                self.speaker_folder_names = [f"speaker_{i + 1}" for i in range(num_sources)]

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part, accumulation_steps=1, scaler=None):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        use_amp = scaler is not None
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch["mix"])
                batch["logits"] = outputs
        else:
            outputs = self.model(batch["mix"])
            batch["logits"] = outputs

        batch["est_source"] = batch["logits"]
        batch["mixture"] = batch["mix"]

        if "source" in batch:
            batch["true_source"] = batch["source"]

        if metrics is not None and "true_source" in batch:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        if self.save_path is not None:
            batch_size = batch["logits"].shape[0]
            num_sources = batch["logits"].shape[1]
            for i in range(batch_size):
                if "mix_path" in batch:
                    mix_path = Path(batch["mix_path"][i])
                    mix_name = mix_path.stem  #
                for src_idx in range(num_sources):
                    separated_audio = batch["logits"][i, src_idx].cpu()
                    speaker_name = self.speaker_folder_names[src_idx]
                    speaker_dir = self.save_path / part / speaker_name
                    speaker_dir.mkdir(exist_ok=True, parents=True)
                    output_filename = f"{mix_name}.wav"
                    output_path = speaker_dir / output_filename
                    torchaudio.save(output_path, separated_audio.unsqueeze(0))

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)
            for speaker_name in self.speaker_folder_names:
                speaker_dir = self.save_path / part / speaker_name
                speaker_dir.mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()
