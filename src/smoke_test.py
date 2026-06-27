"""
1-epoch smoke test for LeJEPA, DINOv2, IJEPA.
Runs sequentially; on failure prints traceback and moves to next method.
"""

import traceback
import torch
import lightning as pl
from lightning.pytorch.strategies import DDPStrategy

METHODS = ["LeJEPA", "DINOv2", "IJEPA"]

COMMON_TRAINER_KWARGS = dict(
    max_epochs=1,
    accelerator="gpu",
    devices="auto",
    strategy="ddp",
    precision="16-mixed",
    log_every_n_steps=10,
    enable_checkpointing=False,
    logger=False,
    limit_train_batches=50,
    limit_val_batches=20,
)

DATAMODULE_KWARGS = dict(
    root="/mnt/nvme1/mimic-cxr-jpg",
    batch_size=64,
    num_workers=8,
    frontal_only=False,
)


def run_lejpa():
    from LeJEPA import MIMICCXRDataModule, LeJEPAModule
    from stable_pretraining.callbacks import TeacherStudentCallback
    from utils import get_common_callbacks

    pl.seed_everything(42)
    dm = MIMICCXRDataModule(**DATAMODULE_KWARGS)
    model = LeJEPAModule(arch="vit_small_patch16_224", lr=1.5e-4)
    trainer = pl.Trainer(
        **COMMON_TRAINER_KWARGS,
        callbacks=[
            TeacherStudentCallback(),
            *get_common_callbacks(model, num_classes=14, task="multilabel", queue_length=512),
        ],
    )
    trainer.fit(model, dm)


def run_dinov2():
    from dinov2 import MIMICCXRDataModule, DINOv2Module
    from stable_pretraining.callbacks import TeacherStudentCallback
    from utils import get_common_callbacks

    pl.seed_everything(42)
    dm = MIMICCXRDataModule(**DATAMODULE_KWARGS)
    model = DINOv2Module(arch="vit_small_patch16_224", lr=1.5e-4)
    trainer = pl.Trainer(
        **COMMON_TRAINER_KWARGS,
        callbacks=[
            TeacherStudentCallback(),
            *get_common_callbacks(model, num_classes=14, task="multilabel", queue_length=512),
        ],
    )
    trainer.fit(model, dm)


def run_ijepa():
    from ijepa import MIMICCXRDataModule, IJEPAModule
    from stable_pretraining.callbacks import TeacherStudentCallback
    from utils import get_common_callbacks

    pl.seed_everything(42)
    dm = MIMICCXRDataModule(**DATAMODULE_KWARGS)
    model = IJEPAModule(arch="vit_small_patch16_224", lr=1.5e-4)
    trainer = pl.Trainer(
        **{**COMMON_TRAINER_KWARGS, "strategy": DDPStrategy(find_unused_parameters=True)},
        callbacks=[
            TeacherStudentCallback(),
            *get_common_callbacks(model, num_classes=14, task="multilabel", queue_length=512),
        ],
    )
    trainer.fit(model, dm)


RUNNERS = {"LeJEPA": run_lejpa, "DINOv2": run_dinov2, "IJEPA": run_ijepa}

if __name__ == "__main__":
    results = {}
    for name in METHODS:
        print(f"\n{'='*60}")
        print(f"  Smoke test: {name}")
        print(f"{'='*60}")
        try:
            RUNNERS[name]()
            results[name] = "PASS"
            print(f"\n[{name}] PASS")
        except Exception:
            results[name] = "FAIL"
            print(f"\n[{name}] FAIL")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("  Results")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name:10s}  {status}")
