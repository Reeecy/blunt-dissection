# train_rgbd_unet.py

import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

# -------------------
# Config
# -------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_ROOT = "../final_dataset_split"
IMG_SIZE = 320
N_CLASSES = 4
BATCH_SIZE = 8
LR = 3e-4
EPOCHS = 100

# Depth scaling: feltételezzük, hogy 0–1000 mm között érdekes
DEPTH_MAX_MM = 1000.0


# -------------------
# Dataset
# -------------------

class RGBDMaskDataset(Dataset):
    def __init__(self, image_dir, depth_dir, mask_dir, augment=False):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.mask_dir  = mask_dir

        self.files = sorted(os.listdir(image_dir))

        if augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.2),
                    A.ShiftScaleRotate(
                        shift_limit=0.02,
                        scale_limit=0.05,
                        rotate_limit=5,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.5,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
                        p=0.5,
                    ),
                    A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
                    A.Resize(IMG_SIZE, IMG_SIZE),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(IMG_SIZE, IMG_SIZE),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # RGB
        img_path = os.path.join(self.image_dir, fname)
        rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # H,W,3

        # Depth (.npy)
        depth_name = fname.replace(".png", ".npy")
        depth_path = os.path.join(self.depth_dir, depth_name)
        depth = np.load(depth_path).astype(np.float32)  # H,W

        # Depth normalizálás 0–1 skálára
        depth = np.clip(depth, 0, DEPTH_MAX_MM) / DEPTH_MAX_MM
        depth = depth[..., None]  # H,W,1

        # Stack RGBD
        x = np.concatenate([rgb, depth], axis=-1)  # H,W,4

        # Mask (0..3)
        mask_path = os.path.join(self.mask_dir, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)

        # Albumentations expects dict with 'image' and 'mask'
        augmented = self.transform(image=x, mask=mask)
        x = augmented["image"]      # tensor C,H,W
        mask = augmented["mask"]    # tensor H,W

        # Ensure mask is torch.long (int64) for CrossEntropyLoss
        mask = mask.long()

        # x: float32, shape (4,H,W), mask: long, shape (H,W)
        return x, mask


# -------------------
# Losses & Metrics
# -------------------

class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        # logits: N,C,H,W
        # targets: N,H,W (long)
        probs = F.softmax(logits, dim=1)
        loss = 0.0

        for c in range(self.num_classes):
            # Skip background if akarod -> start from 1
            pred_c = probs[:, c]               # N,H,W
            targ_c = (targets == c).float()    # N,H,W

            num = 2.0 * (pred_c * targ_c).sum(dim=(1, 2))
            den = pred_c.sum(dim=(1, 2)) + targ_c.sum(dim=(1, 2)) + 1e-7

            dice_c = 1.0 - (num / den)
            loss += dice_c.mean()

        return loss / self.num_classes


def compute_iou_per_class(logits, targets, num_classes):
    # logits: N,C,H,W, targets: N,H,W
    preds = torch.argmax(logits, dim=1)  # N,H,W
    ious = []

    for c in range(num_classes):
        pred_c = (preds == c)
        targ_c = (targets == c)

        intersection = (pred_c & targ_c).sum().item()
        union = (pred_c | targ_c).sum().item()

        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)

    return ious  # list of IoUs per class


# -------------------
# Model
# -------------------

def build_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=4,          # RGBD
        classes=N_CLASSES,
    )
    return model


# -------------------
# Training
# -------------------

def main():
    train_ds = RGBDMaskDataset(
        os.path.join(DATASET_ROOT, "train", "images"),
        os.path.join(DATASET_ROOT, "train", "depth"),
        os.path.join(DATASET_ROOT, "train", "masks"),
        augment=True,
    )

    val_ds = RGBDMaskDataset(
        os.path.join(DATASET_ROOT, "val", "images"),
        os.path.join(DATASET_ROOT, "val", "depth"),
        os.path.join(DATASET_ROOT, "val", "masks"),
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model().to(DEVICE)

    ce_loss   = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes=N_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    patience = 15
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        # ---- TRAIN ----
        model.train()
        train_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for x, y in pbar:
            x = x.to(DEVICE, non_blocking=True)  # N,4,H,W
            y = y.to(DEVICE, non_blocking=True)  # N,H,W

            optimizer.zero_grad()
            logits = model(x)
            loss = ce_loss(logits, y) + dice_loss(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = train_loss_sum / len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        val_loss_sum = 0.0
        iou_sums = np.zeros(N_CLASSES, dtype=np.float64)
        iou_counts = np.zeros(N_CLASSES, dtype=np.int32)

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                logits = model(x)
                loss = ce_loss(logits, y) + dice_loss(logits, y)
                val_loss_sum += loss.item()

                ious = compute_iou_per_class(logits, y, N_CLASSES)
                for c, iou in enumerate(ious):
                    if not np.isnan(iou):
                        iou_sums[c] += iou
                        iou_counts[c] += 1

        val_loss = val_loss_sum / len(val_loader)
        class_ious = iou_sums / np.maximum(iou_counts, 1)
        mean_iou = np.nanmean(class_ious)

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  mIoU={mean_iou:.4f}")
        print(f"Class IoUs (0=bg,1=flesh,2=dissection,3=tool): {class_ious}")

        # LR scheduler
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), "best_unet_rgbd.pth")
            print("  --> Best model saved.")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
