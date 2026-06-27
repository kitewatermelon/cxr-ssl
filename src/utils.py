import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
import stable_pretraining as spt


class _MultilabelBCE(nn.Module):
    """BCEWithLogitsLoss that accepts long targets (torchmetrics와 dtype 공유 가능)."""
    def forward(self, preds, target):
        return F.binary_cross_entropy_with_logits(preds, target.float())
from stable_pretraining.callbacks import (
    OnlineProbe, OnlineKNN, RankMe, LiDAR,
    EarlyStopping, LoggingCallback, ModuleSummary, TrainerInfo,
)


def get_common_callbacks(
    model,
    num_classes: int,
    task: str,                      # "multiclass" | "multilabel" | "binary"
    queue_length: int,
    embed_dim: int | None = None,   # None이면 model.embed_dim에서 자동 추출
    early_stopping: EarlyStopping | None = None,
):
    """
    SSL 공통 콜백 묶음을 반환합니다.

    Args:
        model:           spt.Module 인스턴스
        num_classes:     분류 클래스 수
        task:            torchmetrics task 문자열
        queue_length:    OnlineKNN / RankMe / LiDAR 공유 큐 크기
                         LiDAR 기본값(n_classes=100, samples_per_class=10) 기준
                         최소 1000 이상 권장
        embed_dim:       feature 차원. None이면 model.embed_dim 자동 사용
        early_stopping:  EarlyStopping 인스턴스. None이면 추가 안 함
    """
    dim = embed_dim if embed_dim is not None else model.embed_dim

    # torchmetrics: multilabel은 num_labels, 나머지는 num_classes
    if task == "multilabel":
        metric_kwargs = {"num_labels": num_classes}
        probe_loss = _MultilabelBCE()
    else:
        metric_kwargs = {"num_classes": num_classes}
        probe_loss = nn.CrossEntropyLoss()

    callbacks = [
        TrainerInfo(),
        ModuleSummary(),
        LoggingCallback(),

        # ── Online Linear Probe ──────────────────────────────────────────
        OnlineProbe(
            module=model,
            name="linear",
            input="cls_token",
            target="label",
            probe=nn.Linear(dim, num_classes),
            loss=probe_loss,
            metrics={
                "acc":   torchmetrics.Accuracy(task=task, **metric_kwargs),
                "auroc": torchmetrics.AUROC(task=task, **metric_kwargs),
            },
        ),

        # ── Online KNN (multiclass only: multilabel은 one-hot 미지원) ────────
        *([OnlineKNN(
            name="knn",
            input="cls_token",
            target="label",
            queue_length=queue_length,
            input_dim=dim,
            metrics={
                "acc":   torchmetrics.Accuracy(task=task, **metric_kwargs),
                "auroc": torchmetrics.AUROC(task=task, **metric_kwargs),
            },
            k=20,
            distance_metric="cosine",
        )] if task != "multilabel" else []),

        # ── Collapse 탐지 (큐 공유: target="cls_token" 동일) ─────────────
        RankMe(
            name="rankme",
            target="cls_token",
            queue_length=queue_length,
            target_shape=dim,
            verbose=True,
        ),

        LiDAR(
            name="lidar",
            target="cls_token",
            queue_length=queue_length,
            target_shape=dim,
        ),
    ]

    if early_stopping is not None:
        callbacks.append(early_stopping)

    return callbacks