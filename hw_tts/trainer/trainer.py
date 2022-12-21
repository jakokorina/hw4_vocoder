import typing as tp
import torch
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from hw_tts.base import BaseTrainer
from hw_tts.utils import inf_loop, MetricTracker
from hw_tts.loss import MelLoss, DiscriminatorLoss, GeneratorLoss, FeatureLoss
from hw_tts.audio import MelSpectrogramConfig, MelSpectrogramGenerator


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            mpd,
            msd,
            optimizer,
            optimizer_d,
            config,
            device,
            dataloaders,
            lr_scheduler_g=None,
            lr_scheduler_d=None,
            len_epoch=None,
            skip_oom=True,
            lambda_fm=2,
            lambda_mel=45
    ):
        super().__init__(model, mpd, msd, optimizer, optimizer_d, config, device)
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
        self.lr_scheduler_g = lr_scheduler_g
        self.lr_scheduler_d = lr_scheduler_d
        self.log_step = 50

        self.eval_loader = dataloaders["eval"]
        self.train_metrics = MetricTracker(
            "loss", "mel_loss", "generator_loss", "feature_map_loss", "grad norm", writer=self.writer
        )

        self.discr_loss = DiscriminatorLoss()
        self.feature_loss = FeatureLoss()
        self.generator_loss = GeneratorLoss()
        self.mel_loss = MelLoss()

        self.lambda_mel = lambda_mel
        self.lambda_fm = lambda_fm

        self.mel_spec_generator = MelSpectrogramGenerator().to(device)

    def _clip_grad_norm(self, model):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.mpd.train()
        self.msd.train()

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
                self.writer.set_step(batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["total_loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler_g.get_last_lr()[0]
                )

                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break

        log = last_train_metrics

        self.lr_scheduler_g.step()
        self.lr_scheduler_d.step()
        self._evaluation_epoch(epoch)

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        # performing 2 steps for each batch

        # Get Data
        spectrogram = batch["spectrogram"].float().to(self.device)
        wav_real = batch["audio"].float().to(self.device).unsqueeze(1)

        self.optimizer_d.zero_grad()

        # Forward
        wav_pred = self.model(spectrogram)
        spec_pred = self.mel_spec_generator(wav_pred.squeeze(1))
        if spectrogram.shape[-1] != spec_pred.shape[-1]:
            wav_real = F.pad(spectrogram, (0, wav_pred.shape[-1] - wav_real.shape[-1]))
            spectrogram = self.mel_spec_generator(wav_real)

        # Teaching discriminator
        y_mpd_pred, _ = self.mpd(wav_pred.detach())
        y_msd_pred, _ = self.msd(wav_pred.detach())

        y_mpd_real, _ = self.mpd(wav_real)
        y_msd_real, _ = self.msd(wav_real)

        # Calc loss
        loss_p = self.discr_loss(y_mpd_real, y_mpd_pred)
        loss_s = self.discr_loss(y_msd_real, y_msd_pred)

        loss_disc = loss_p + loss_s
        loss_disc.backward()
        self.optimizer_d.step()

        # Teaching generator
        self.optimizer.zero_grad()

        y_mpd_pred, fmp_mpd_pred = self.mpd(wav_pred.detach())
        y_msd_pred, fmp_msd_pred = self.msd(wav_pred.detach())

        y_mpd_real, fmp_mpd_real = self.mpd(wav_real)
        y_msd_real, fmp_msd_real = self.msd(wav_real)

        mel_loss = self.mel_loss(spectrogram, spec_pred)
        loss_fm_p = self.feature_loss(fmp_mpd_real, fmp_mpd_pred)
        loss_fm_s = self.feature_loss(fmp_msd_real, fmp_msd_pred)
        loss_g_p = self.generator_loss(y_mpd_pred)
        loss_g_s = self.generator_loss(y_msd_pred)

        total_loss = self.lambda_mel * mel_loss + self.lambda_fm * (loss_fm_p + loss_fm_s) + loss_g_p + loss_g_s
        total_loss.backward()
        self.optimizer.step()

        t_l = total_loss.detach().cpu().numpy()
        m_l = mel_loss.detach().cpu().numpy()
        fm_l = (loss_fm_s + loss_fm_p).detach().cpu().numpy()
        g_l = (loss_g_p + loss_g_s).detach().cpu().numpy()

        batch["total_loss"] = t_l
        batch["mel_loss"] = m_l
        batch["feature_map_loss"] = fm_l
        batch["generator_loss"] = g_l

        # Backward
        total_loss.backward()
        self._clip_grad_norm(self.model)
        self.optimizer.step()

        metrics.update("loss", t_l)
        metrics.update("mel_loss", m_l)
        metrics.update("feature_map_loss", fm_l)
        metrics.update("generator_loss", g_l)
        return batch

    def _evaluation_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.writer.set_step(epoch * self.len_epoch, "eval")
        for i, phn in tqdm(enumerate(self.eval_loader)):
            wav = self.model(phn).squeeze().detach()
            self._log_audio(f"result_{i}", wav)

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_audio(self, name: str, wav):
        self.writer.add_audio(name, wav, sample_rate=MelSpectrogramConfig.sr)

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
