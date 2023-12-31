import random
from random import shuffle
from typing import Sequence

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from tss_lib.postprocessing.base_postprocessor import BasePostprocessor
from tss_lib.trainer.base_trainer import BaseTrainer
from tss_lib.metric.base_metric import BaseMetric
from tss_lib.metric.pesq_metric import calc_pesq
from tss_lib.logger.utils import plot_spectrogram_to_buf
from tss_lib.loss.utils import calc_si_sdr
from tss_lib.utils import inf_loop, MetricTracker, get_lr


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics: Sequence[BaseMetric],
            optimizer,
            config,
            device,
            dataloaders,
            postprocessor: BasePostprocessor = None,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device, lr_scheduler)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.postprocessor = postprocessor
        self.lr_scheduler = lr_scheduler
        self.log_step = config["trainer"].get("log_step", 50)

        self.train_metrics = MetricTracker(
            "loss", "grad norm",
            *criterion.get_loss_parts_names(),
            *[m.name for m in self.metrics],
            writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss",
            *criterion.get_loss_parts_names(),
            *[m.name for m in self.metrics],
            writer=self.writer
        )
        self.accumulated_grad_steps = 0
        self.accumulate_grad_steps = config["trainer"].get("accumulate_grad_steps", 1)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["target_wave", "mixed_wave", "ref_wave", "target_speaker_id", "noise_speaker_id"]:
            if tensor_for_gpu in batch:
                batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch) -> dict:
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.criterion.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )

                self.writer.add_scalar(
                    "learning rate",
                    self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else get_lr(self.optimizer)
                )

                self._log_predictions(**batch)
                # self._log_spectrogram(batch["spectrogram"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        outputs = self.model(**batch)
        if is_train and self.accumulated_grad_steps == 0:
            self.optimizer.zero_grad()
        batch.update(outputs)

        batch["speakers_log_probs"] = F.log_softmax(batch["speakers_logits"], dim=-1)
        # criterion returns a dict, in which the final loss has a key 'loss'
        losses = self.criterion(**batch)
        batch.update(losses)
        if self.postprocessor is not None:
            batch = self.postprocessor(**batch)
        if is_train:
            (batch["loss"] / self.accumulate_grad_steps).backward()
            self.accumulated_grad_steps += 1
            if self.accumulated_grad_steps % self.accumulate_grad_steps == 0:
                self._clip_grad_norm()
                self.optimizer.step()
                self.accumulated_grad_steps = 0
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        for loss_part in losses:
            metrics.update(loss_part, batch[loss_part].item())
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.criterion.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch)
            # self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            mix_id,
            mixed_wave,
            ref_wave,
            ref_wave_length,
            speakers_log_probs,  # (batch_dim, num_classes)
            w1,
            w2,
            w3,
            target_wave=None,
            target_speaker_id=None,
            examples_to_log=10,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return
        pred_speaker_id = speakers_log_probs.argmax(-1)

        if target_wave is None:
            target_wave = [None] * mixed_wave.shape[0]
        if target_speaker_id is None:
            target_speaker_id = [None] * mixed_wave.shape[0]

        tuples = list(zip(mix_id, mixed_wave, ref_wave, ref_wave_length, w1, w2, w3,
                          target_wave, target_speaker_id, pred_speaker_id))
        shuffle(tuples)
        rows = {}

        for mix_id, mixed_wave, ref_wave, ref_wave_length, w1, w2, w3, target_wave, target_speaker_id, pred_speaker_id \
                in tuples[:examples_to_log]:
            rows[mix_id] = {
                "mixed_wave": self._create_audio_for_writer(mixed_wave),
                "ref_wave": self._create_audio_for_writer(ref_wave, ref_wave_length),
                "pred_w1": self._create_audio_for_writer(w1),
                "w1_SI-SDR": calc_si_sdr(target_wave, w1).item(),
                "w1_PESQ": calc_pesq(target_wave, w1).item(),
                "pred_w2": self._create_audio_for_writer(w2),
                "w2_SI-SDR": calc_si_sdr(target_wave, w2).item(),
                "w2_PESQ": calc_pesq(target_wave, w2).item(),
                "pred_w3": self._create_audio_for_writer(w3),
                "w3_SI-SDR": calc_si_sdr(target_wave, w3).item(),
                "w3_PESQ": calc_pesq(target_wave, w3).item(),
                "pred_speaker_id": pred_speaker_id,
            }
            if target_wave is not None:
                rows[mix_id]["target_wave"] = self._create_audio_for_writer(target_wave)
            if target_speaker_id is not None:
                rows[mix_id]["target_speaker_id"] = target_speaker_id
        table = pd.DataFrame.from_dict(rows, orient="index")\
                            .reset_index().rename(columns={'index': 'mix_id'})
        self.writer.add_table("predictions", table)

    def _create_audio_for_writer(self, audio: torch.Tensor, length=None):
        audio = audio.detach().cpu().squeeze()
        if length is not None:
            audio = audio[:length]
        return self.writer.create_audio(audio, sample_rate=self.config["preprocessing"]["sr"])

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
