import csv
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import math
import torch.nn as nn

import timm
from torch.serialization import safe_globals
from torch.torch_version import TorchVersion
from segmentation_vit.segmentation_dataloader import initialize_segmentation_dataloaders


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ViTUNet(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()

        self.backbone = backbone
        hidden_size = backbone.num_features
        self.decoder = nn.Sequential(
            ConvBlock(hidden_size, 256),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(256, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(128, 64),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, images):

        tokens = self.backbone.forward_features(images)
        num_prefix_tokens = self.backbone.num_prefix_tokens
        tokens = tokens[:, num_prefix_tokens:, :]

        batch_size, num_tokens, hidden_size = tokens.shape
        grid_size = math.isqrt(num_tokens)

        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Expected square patch grid, got {num_tokens} tokens")

        features = tokens.transpose(1, 2).reshape(batch_size, hidden_size, grid_size, grid_size)
        logits = self.decoder(features)
        return F.interpolate(
            logits,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


def initialize_vit_unet(num_classes=2, ckpt_path=None):

    backbone = timm.create_model("vit_base_patch16_224", pretrained=(ckpt_path is None), num_classes=0)

    if ckpt_path is not None:
        with safe_globals([TorchVersion]):
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True,)

        backbone.load_state_dict(checkpoint["ema_backbone"], strict=True,)
        print(f"Loaded ViT backbone from: {ckpt_path}")
    else:
        print("Loaded ImageNet pretrained ViT backbone")

    return ViTUNet(backbone, num_classes)

    

def compute_loss(logits, masks, edl=False):
    if edl:
        alpha = F.softplus(logits) + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        y = F.one_hot(masks.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        return (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1).mean()

    ce_loss = F.cross_entropy(logits, masks.long())
    probabilities = F.softmax(logits, dim=1)[:, 1]
    targets = masks.float()
    intersection = (probabilities * targets).sum(dim=(1, 2))
    denominator = probabilities.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    dice_loss = 1 - ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    return ce_loss + dice_loss


def get_predictions(logits, edl=False):
    if edl:
        alpha = F.softplus(logits) + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        probabilities = alpha / S
        uncertainty = (logits.shape[1] / S).squeeze(1)
    else:
        probabilities = F.softmax(logits, dim=1)
        uncertainty = None
    return probabilities.argmax(dim=1), uncertainty


def compute_metrics(preds, masks):
    preds, masks = preds.bool(), masks.bool()
    intersection = (preds & masks).sum(dim=(1, 2)).float()
    pred_sum = preds.sum(dim=(1, 2)).float()
    mask_sum = masks.sum(dim=(1, 2)).float()
    dice = (2 * intersection + 1e-6) / (pred_sum + mask_sum + 1e-6)
    iou = (intersection + 1e-6) / (pred_sum + mask_sum - intersection + 1e-6)
    return dice, iou


def train_segmentation(model, train_loader, val_loader, epochs, edl=False, save_dir="./results/segmentation"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_dice = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_count = 0.0, 0

        for images, masks in tqdm(train_loader, desc=f"Train Epoch {epoch}/{epochs}"):
            images, masks = images.to(device), masks.to(device).long()
            optimizer.zero_grad()
            logits = model(images)
            loss = compute_loss(logits, masks, edl)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_count += images.size(0)

        model.eval()
        val_loss, val_count = 0.0, 0
        dice_sum, iou_sum, uncertainty_sum, uncertainty_count = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device).long()
                logits = model(images)
                loss = compute_loss(logits, masks, edl)
                preds, uncertainty = get_predictions(logits, edl)
                dice, iou = compute_metrics(preds, masks)
                val_loss += loss.item() * images.size(0)
                val_count += images.size(0)
                dice_sum += dice.sum().item()
                iou_sum += iou.sum().item()
                if uncertainty is not None:
                    uncertainty_sum += uncertainty.sum().item()
                    uncertainty_count += uncertainty.numel()

        train_loss /= max(train_count, 1)
        val_loss /= max(val_count, 1)
        val_dice = dice_sum / max(val_count, 1)
        val_iou = iou_sum / max(val_count, 1)
        mean_uncertainty = uncertainty_sum / uncertainty_count if uncertainty_count else None

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_dice": val_dice, "val_iou": val_iou, "val_uncertainty": mean_uncertainty}
        history.append(row)

        with open(save_dir / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(history)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_dir / "best_model.pt")

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}" + (f" | Uncertainty: {mean_uncertainty:.4f}" if edl else ""))

    print(f"Training finished | Best Val Dice: {best_dice:.4f}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edl", action="store_true", help="Enable evidential deep learning",)
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to DINO or LeJEPA pretrained checkpoint",)
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save training results",)
    args = parser.parse_args()

    torch.manual_seed(42)

    train_loader, val_loader = initialize_segmentation_dataloaders(
        data_dir="/content/segmented-images",
        batch_size=256,
    )
    model = initialize_vit_unet(num_classes=2, ckpt_path=args.ckpt)

    save_dir = args.save_dir or "/content/drive/MyDrive/results/segmentation"

    train_segmentation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        edl=args.edl,
        save_dir=save_dir,
    )
