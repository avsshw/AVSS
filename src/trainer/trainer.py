from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(
        self,
        batch,
        metrics: MetricTracker,
        accumulation_steps=1,
        scaler=None,
        batch_idx=None,
    ):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            if batch_idx is not None and batch_idx % accumulation_steps == 0:
                self.optimizer.zero_grad()

        has_source = "source" in batch
        if not has_source:
            metric_funcs = []

        use_amp = scaler is not None
        if use_amp:
            import torch.cuda.amp

            with torch.cuda.amp.autocast():
                video = batch.get("video", None)
                if video is not None:
                    outputs = self.model(batch["mix"], video=video)
                else:
                    outputs = self.model(batch["mix"])
                batch["output"] = outputs
                if has_source:
                    all_losses = self.criterion(
                        preds=outputs, targets=batch["source"], **batch
                    )
                    batch.update(all_losses)

            batch["est_source"] = outputs
            batch["mixture"] = batch["mix"]
            if has_source:
                batch["true_source"] = batch["source"]
                loss = batch["loss"] / accumulation_steps
                scaler.scale(loss).backward()
                if (
                    self.is_train
                    and batch_idx is not None
                    and (batch_idx + 1) % accumulation_steps == 0
                ):
                    grad_norm = self._get_grad_norm()
                    self._clip_grad_norm()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    if self.lr_scheduler is not None:
                        # for per batch schedulers (doesn't work for ReduceLROnPlateu)
                        import inspect
                        step_signature = inspect.signature(self.lr_scheduler.step)
                        if 'metrics' not in step_signature.parameters:
                            self.lr_scheduler.step()
                    batch["grad_norm"] = grad_norm
        else:
            # if for some reason AMP is not used
            video = batch.get("video", None)
            if video is not None:
                outputs = self.model(batch["mix"], video=video)
            else:
                outputs = self.model(batch["mix"])
            batch["output"] = outputs
            if has_source:
                all_losses = self.criterion(
                    preds=outputs, targets=batch["source"], **batch
                )
                batch.update(all_losses)

            batch["est_source"] = outputs
            batch["mixture"] = batch["mix"]
            if has_source:
                batch["true_source"] = batch["source"]
                loss = batch["loss"] / accumulation_steps
                if self.is_train:
                    loss.backward()
                    if batch_idx is not None and (batch_idx + 1) % accumulation_steps == 0:
                        grad_norm = self._get_grad_norm()
                        self._clip_grad_norm()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        if self.lr_scheduler is not None:
                            # for per batch schedulers (doesn't work for ReduceLROnPlateu)
                            import inspect
                            step_signature = inspect.signature(self.lr_scheduler.step)
                            if 'metrics' not in step_signature.parameters:
                                self.lr_scheduler.step()
                        batch["grad_norm"] = grad_norm

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        sample_rate = 16000
        self.writer.add_audio("mix", batch["mix"][0], sample_rate=sample_rate)
        self.writer.add_audio(
            "pred_source_1", batch["output"][0, 0], sample_rate=sample_rate
        )
        self.writer.add_audio(
            "pred_source_2", batch["output"][0, 1], sample_rate=sample_rate
        )
        if "source" in batch:
            self.writer.add_audio(
                "true_source_1", batch["source"][0, 0], sample_rate=sample_rate
            )
            self.writer.add_audio(
                "true_source_2", batch["source"][0, 1], sample_rate=sample_rate
            )
