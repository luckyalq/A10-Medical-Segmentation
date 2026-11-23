from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.logger import CSVLogger
from utils.metrics import Confusion, segmentation_loss


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    log_path: str | Path,
    resume: bool = False,
) -> dict[str, float]:
    """使用AdamW、AMP、梯度裁剪、学习率调度、早停和断点续训训练模型。"""
    settings = config["training"]
    checkpoint_path = Path(checkpoint_path)
    log_path = Path(log_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_path.with_name(checkpoint_path.stem + "_last.pth")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=settings["scheduler_factor"], patience=settings["scheduler_patience"], min_lr=settings["min_learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=bool(settings["amp"] and device.type == "cuda"))
    logger = CSVLogger(log_path, ["epoch", "learning_rate", "train_loss", "val_loss", "dice", "iou", "precision", "recall", "fnr"])
    writer = SummaryWriter(log_dir=str(log_path.with_suffix("")))
    model.to(device)
    best_dice = -1.0
    best_metrics: dict[str, float] = {}
    stale = 0
    start_epoch = 1
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["epoch"]) + 1
        best_dice = float(state.get("best_dice", -1.0))
        stale = int(state.get("stale", 0))
        print(f"从第 {start_epoch} 轮恢复训练: {last_path}")
    freeze_epochs = int(config["model"].get("freeze_encoder_epochs", 0))
    for epoch in range(start_epoch, int(settings["epochs"]) + 1):
        model.train()
        if hasattr(model, "set_encoder_trainable"):
            model.set_encoder_trainable(epoch > freeze_epochs)
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train {epoch}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=bool(settings["amp"] and device.type == "cuda")):
                logits = model(images)
                loss = segmentation_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        validation = evaluate(model, val_loader, device, settings["threshold"])
        scheduler.step(validation["val_loss"])
        learning_rate = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "learning_rate": learning_rate, "train_loss": train_loss, **validation}
        logger.log(row)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/validation", validation["val_loss"], epoch)
        writer.add_scalar("metrics/dice", validation["dice"], epoch)
        writer.add_scalar("metrics/fnr", validation["fnr"], epoch)
        writer.add_scalar("optimization/learning_rate", learning_rate, epoch)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={validation['val_loss']:.4f} dice={validation['dice']:.4f}")
        if validation["dice"] > best_dice:
            best_dice = validation["dice"]
            best_metrics = validation
            stale = 0
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(), "epoch": epoch, "best_dice": best_dice, "stale": stale, "metrics": validation, "config": config}, checkpoint_path)
        else:
            stale += 1
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(), "epoch": epoch, "best_dice": best_dice, "stale": stale, "config": config}, last_path)
        if stale >= int(settings["patience"]):
            print(f"验证集连续 {stale} 轮未提升，在第 {epoch} 轮早停")
            break
    writer.close()
    return best_metrics


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, threshold: float) -> dict[str, float]:
    model.eval()
    confusion = Confusion()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        total_loss += segmentation_loss(logits, masks).item() * images.size(0)
        confusion.update(torch.sigmoid(logits), masks, threshold)
    return {"val_loss": total_loss / len(loader.dataset), **confusion.compute()}

