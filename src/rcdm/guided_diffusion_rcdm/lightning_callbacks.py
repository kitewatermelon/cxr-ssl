import os

import blobfile as bf
import torch as th
from pytorch_lightning.callbacks import Callback

from .nn import update_ema
from . import logger


class EMACallback(Callback):
    """매 step마다 EMA 파라미터를 업데이트한다."""

    def __init__(self, ema_rates):
        self.ema_rates = (
            [ema_rates]
            if isinstance(ema_rates, float)
            else [float(x) for x in ema_rates.split(",")]
        )
        self.ema_params = None

    def on_train_start(self, trainer, pl_module):
        self.ema_params = [
            [p.data.clone() for p in pl_module.model.parameters()]
            for _ in self.ema_rates
        ]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        model_params = list(pl_module.model.parameters())
        for rate, ema_param_list in zip(self.ema_rates, self.ema_params):
            update_ema(ema_param_list, model_params, rate=rate)


class RCDMCheckpointCallback(Callback):
    """기존 포맷(model000000.pt, ema_rate_000000.pt, opt000000.pt)으로 저장한다."""

    def __init__(self, save_interval, out_dir, ema_callback, resume_step=0):
        self.save_interval = save_interval
        self.out_dir = out_dir
        self.ema_callback = ema_callback
        self.resume_step = resume_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step + self.resume_step
        if step > 0 and step % self.save_interval == 0:
            self._save(trainer, pl_module, step)

    def on_train_end(self, trainer, pl_module):
        step = trainer.global_step + self.resume_step
        last_saved = (step // self.save_interval) * self.save_interval
        if step != last_saved:
            self._save(trainer, pl_module, step)

    def _save(self, trainer, pl_module, step):
        if trainer.is_global_zero:
            os.makedirs(self.out_dir, exist_ok=True)

            model_path = bf.join(self.out_dir, f"model{step:06d}.pt")
            logger.log(f"saving model to {model_path}...")
            th.save(pl_module.model.state_dict(), model_path)

            opt_path = bf.join(self.out_dir, f"opt{step:06d}.pt")
            th.save(pl_module.optimizers().optimizer.state_dict(), opt_path)

            for rate, ema_param_list in zip(
                self.ema_callback.ema_rates, self.ema_callback.ema_params
            ):
                state_dict = {
                    name: param
                    for (name, _), param in zip(
                        pl_module.model.named_parameters(), ema_param_list
                    )
                }
                ema_path = bf.join(self.out_dir, f"ema_{rate}_{step:06d}.pt")
                logger.log(f"saving EMA {rate} to {ema_path}...")
                th.save(state_dict, ema_path)

        # 모든 rank가 반드시 barrier를 호출해야 데드락 방지
        trainer.strategy.barrier()
