import csv
from pathlib import Path
import torch
import torch.nn.functional as F


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

        for images, masks in train_loader:
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