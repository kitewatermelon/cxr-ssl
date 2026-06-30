import torch as th
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .resample import LossAwareSampler
from . import logger


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


class RCDMLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        diffusion,
        schedule_sampler,
        lr,
        weight_decay,
        lr_anneal_steps,
        microbatch,
        log_interval,
        ema_rates,
        ssl_model=None,
    ):
        super().__init__()
        self.model = model
        self.diffusion = diffusion
        self.schedule_sampler = schedule_sampler
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.microbatch = microbatch
        self.log_interval = log_interval
        self.ema_rates = (
            [ema_rates]
            if isinstance(ema_rates, float)
            else [float(x) for x in ema_rates.split(",")]
        )
        # ssl_model을 서브모듈로 등록 → Lightning이 올바른 GPU로 이동시킴
        self.ssl_model = ssl_model
        if self.ssl_model is not None:
            for p in self.ssl_model.parameters():
                p.requires_grad_(False)

    def training_step(self, batch, batch_idx):
        img_batch, batch_big, model_kwargs = batch

        # SSL feature 추출 (training_step 안에서 실행 → 올바른 GPU 보장)
        if self.ssl_model is not None:
            with th.no_grad():
                # ssl_model은 fp32로 실행 (bf16-mixed autocast 범위 밖)
                with th.amp.autocast("cuda", enabled=False):
                    model_kwargs["feat"] = self.ssl_model(
                        batch_big.to(self.device).float()
                    ).detach()

        total_loss = th.tensor(0.0, device=self.device)
        n_micro = 0

        for i in range(0, img_batch.shape[0], self.microbatch):
            micro = img_batch[i : i + self.microbatch]
            micro_cond = {k: v[i : i + self.microbatch] for k, v in model_kwargs.items()}
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)
            losses = self.diffusion.training_losses(
                self.model, micro, t, model_kwargs=micro_cond
            )
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            loss = (losses["loss"] * weights).mean()
            total_loss = total_loss + loss
            n_micro += 1
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

        mean_loss = total_loss / n_micro

        self.log("train/loss", mean_loss, on_step=True, on_epoch=False,
                 prog_bar=True, sync_dist=True)
        self.log("train/step", float(self.global_step), on_step=True,
                 on_epoch=False, sync_dist=False)

        if self.global_step % self.log_interval == 0:
            logger.logkv("step", self.global_step)
            logger.logkv(
                "samples",
                (self.global_step + 1) * self.trainer.world_size * img_batch.shape[0],
            )
            logger.dumpkvs()

        return mean_loss

    def configure_optimizers(self):
        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if not self.lr_anneal_steps:
            return opt

        def lr_lambda(current_step):
            frac_done = current_step / self.lr_anneal_steps
            return max(0.0, 1.0 - frac_done)

        scheduler = LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
