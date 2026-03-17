"""
train.py
--------
Training entry point for retinal vessel segmentation with SegFormer.

Usage:
    python src/train.py

Tracks experiments with Weights & Biases. Set WANDB_API_KEY or run `wandb login`
before executing.
"""

import gc
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

import wandb
from livelossplot import PlotLosses
from livelossplot.outputs import ExtremaPrinter, MatplotlibPlot

from config import DatasetConfig, InferenceConfig, ModelConfig, TrainingConfig
from dataset import get_dataloader
from losses import dice_coef_loss
from metrics import mean_iou
from model import get_model
from utils import get_default_device, seed_everything


# ---------------------------------------------------------------------------
# One-epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, num_classes, device, epoch_idx, total_epochs):
    model.train()

    loss_rec   = MeanMetric()
    metric_rec = MeanMetric()
    acc_rec    = MulticlassAccuracy(num_classes=num_classes, average="micro")

    with tqdm(total=len(loader), ncols=120, ascii=True) as pbar:
        pbar.set_description(f"Train :: Epoch {epoch_idx}/{total_epochs}")

        for data, target in loader:
            pbar.update(1)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            with amp.autocast():
                outputs = model(pixel_values=data, labels=target, return_dict=True)
                logits  = outputs["logits"]
                upsampled = nn.functional.interpolate(
                    logits, size=target.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = dice_coef_loss(upsampled, target, num_classes=num_classes)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            logits   = upsampled.detach()
            pred_idx = logits.argmax(dim=1)
            metric   = mean_iou(pred_idx, target, num_classes=num_classes)

            acc_rec.update(pred_idx.cpu(), target.cpu())
            loss_rec.update(loss.detach().cpu(),   weight=data.shape[0])
            metric_rec.update(metric.cpu(),        weight=data.shape[0])

            pbar.set_postfix_str(
                f"Loss: {loss_rec.compute():.4f}  IoU: {metric_rec.compute():.4f}  "
                f"Acc: {acc_rec.compute():.4f}"
            )

    return loss_rec.compute(), metric_rec.compute(), acc_rec.compute()


@torch.no_grad()
def evaluate(model, loader, device, num_classes, epoch_idx, total_epochs, split="Valid"):
    model.eval()

    loss_rec   = MeanMetric()
    metric_rec = MeanMetric()
    acc_rec    = MulticlassAccuracy(num_classes=num_classes, average="micro")

    with tqdm(total=len(loader), ncols=120, ascii=True) as pbar:
        pbar.set_description(f"{split} :: Epoch {epoch_idx}/{total_epochs}")

        for data, target in loader:
            pbar.update(1)
            data, target = data.to(device), target.to(device)

            outputs = model(pixel_values=data, labels=target, return_dict=True)
            logits  = outputs["logits"]
            upsampled = nn.functional.interpolate(
                logits, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
            loss     = dice_coef_loss(upsampled, target, num_classes=num_classes)
            pred_idx = upsampled.argmax(dim=1)
            metric   = mean_iou(pred_idx, target, num_classes=num_classes)

            acc_rec.update(pred_idx.cpu(), target.cpu())
            loss_rec.update(loss.cpu(),    weight=data.shape[0])
            metric_rec.update(metric.cpu(), weight=data.shape[0])

        pbar.set_postfix_str(
            f"Loss: {loss_rec.compute():.4f}  IoU: {metric_rec.compute():.4f}  "
            f"Acc: {acc_rec.compute():.4f}"
        )

    return loss_rec.compute(), metric_rec.compute(), acc_rec.compute()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    # ---- WandB setup ----
    run = wandb.init(project="ICPR_BloodVessels_segmentation")
    cfg = run.config

    cfg.IMG_SIZE      = (DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH)
    cfg.MODEL_NAME    = ModelConfig.MODEL_NAME
    cfg.BATCH_SIZE    = TrainingConfig.BATCH_SIZE
    cfg.NUM_EPOCHS    = TrainingConfig.NUM_EPOCHS
    cfg.NUM_CLASSES   = DatasetConfig.NUM_CLASSES
    cfg.OPTIMIZER     = TrainingConfig.OPTIMIZER
    cfg.LEARNING_RATE = TrainingConfig.LEARNING_RATE
    cfg.WEIGHT_DECAY  = TrainingConfig.WEIGHT_DECAY
    cfg.LR_SCHEDULER  = TrainingConfig.LR_SCHEDULER

    # ---- Reproducibility & device ----
    seed_everything(41)
    device, gpu_available = get_default_device()

    # ---- Checkpoint path ----
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join(wandb.run.dir, "ckpt.tar")

    # ---- Data ----
    train_loader, valid_loader = get_dataloader(
        configs=cfg,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=gpu_available,
    )

    # ---- Model ----
    model = get_model(cfg).to(device)

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["LEARNING_RATE"],
        weight_decay=cfg["WEIGHT_DECAY"],
        amsgrad=True,
    )

    # ---- Scheduler ----
    milestones = [cfg["NUM_EPOCHS"] // 2]
    scheduler  = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    scaler        = amp.GradScaler()
    best_val_loss = float("inf")
    live_plot     = PlotLosses(outputs=[MatplotlibPlot(cell_size=(8, 3)), ExtremaPrinter()])

    for epoch in range(cfg["NUM_EPOCHS"]):
        torch.cuda.empty_cache()
        gc.collect()

        train_loss, train_iou, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler,
            cfg["NUM_CLASSES"], device, epoch + 1, cfg["NUM_EPOCHS"],
        )
        val_loss, val_iou, val_acc = evaluate(
            model, valid_loader, device,
            cfg["NUM_CLASSES"], epoch + 1, cfg["NUM_EPOCHS"],
        )
        scheduler.step()

        live_plot.update({
            "loss": train_loss, "val_loss": val_loss,
            "accuracy": train_acc, "val_accuracy": val_acc,
            "IoU": train_iou, "val_IoU": val_iou,
        })
        live_plot.send()

        wandb.log({
            "epoch": epoch, "loss": train_loss, "val_loss": val_loss,
            "accuracy": train_acc, "val_accuracy": val_acc,
            "IoU": train_iou, "val_IoU": val_iou,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wandb.run.summary["best_epoch"] = epoch
            print("✓ Model improved — saving checkpoint...", end=" ")
            torch.save(
                {"model": model.state_dict(), "opt": optimizer.state_dict(), "scaler": scaler.state_dict()},
                ckpt_path,
            )
            print("done.")

    run.finish()
    print(f"\nTraining complete. Best checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
