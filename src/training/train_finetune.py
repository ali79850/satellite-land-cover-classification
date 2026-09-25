"""
Fine-tunes ResNet18: unfreezes the backbone and continues training from the
frozen-head checkpoint (transfer_frozen_best.pth), using discriminative
learning rates — a low LR for pretrained backbone layers, a higher LR for
the already-partially-trained head.
"""

import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.dataset import EuroSATDataset
from src.data.transforms_pretrained import get_train_transforms_pretrained, get_eval_transforms_pretrained
from src.models.transfer_resnet18 import build_resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "models"
FROZEN_CHECKPOINT = CHECKPOINT_DIR / "transfer_frozen_best.pth"
LOG_PATH = PROJECT_ROOT / "reports" / "finetune_training_log.csv"

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 10
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train():
    torch.manual_seed(SEED)
    print(f"Using device: {DEVICE}")

    train_ds = EuroSATDataset(split="train", transform=get_train_transforms_pretrained())
    val_ds = EuroSATDataset(split="val", transform=get_eval_transforms_pretrained())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_resnet18(num_classes=10, freeze_backbone=True)
    model.load_state_dict(torch.load(FROZEN_CHECKPOINT, map_location="cpu"))
    for param in model.parameters():
        param.requires_grad = True
    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    criterion = nn.CrossEntropyLoss()

    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc")]
    head_params = [p for name, p in model.named_parameters() if name.startswith("fc")]

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ])

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_PATH.parent.mkdir(exist_ok=True)

    best_val_acc = 0.0
    log_rows = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_start = time.time()
        running_loss, correct, total_seen = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_seen += labels.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"  batch {batch_idx + 1}/{len(train_loader)}")

        train_loss = running_loss / total_seen
        train_acc = correct / total_seen
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"time={epoch_time:.1f}s")

        log_rows.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "epoch_time_sec": epoch_time,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / "finetune_best.pth")
            print(f"  -> New best val_acc={val_acc:.4f}, checkpoint saved.")

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining log saved to {LOG_PATH}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()