import csv
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from classification.classification_dataloader import initialize_classification_dataloaders
from utils import initialize_siglip2


class SigLIP2Classifier(nn.Module):
    def __init__(self, siglip2, num_classes):
        super().__init__()
        self.backbone = siglip2.vision_model
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, images):
        outputs = self.backbone(pixel_values=images)
        features = outputs.pooler_output
        return self.head(features)


def compute_loss(logits, targets, edl=False):
    if not edl:
        return F.cross_entropy(logits, targets)
    alpha = F.softplus(logits) + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    y = F.one_hot(targets, num_classes=logits.shape[1]).float()
    loss = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1).mean()
    return loss

def get_predictions(logits, edl=False):
    if edl:
        alpha = F.softplus(logits) + 1.0
        probabilities = alpha / alpha.sum(dim=1, keepdim=True)
        uncertainty = logits.shape[1] / alpha.sum(dim=1)
    else:
        probabilities = F.softmax(logits, dim=1)
        uncertainty = None
    return probabilities.argmax(dim=1), uncertainty


def train_classification(model, train_loader, val_loader, epochs, edl=False, save_dir="./results/classification"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_count = 0.0, 0

        for images, targets in tqdm(train_loader, desc=f"Train Epoch {epoch}/{epochs}"):
            images, targets = images.to(device), targets.to(device).long()
            optimizer.zero_grad()
            logits = model(images)
            loss = compute_loss(logits, targets, edl)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            train_count += targets.size(0)

        model.eval()
        val_loss, val_count = 0.0, 0
        all_preds, all_targets, all_uncertainties = [], [], []

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device).long()
                logits = model(images)
                loss = compute_loss(logits, targets, edl)
                preds, uncertainty = get_predictions(logits, edl)
                val_loss += loss.item() * targets.size(0)
                val_count += targets.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())
                if uncertainty is not None:
                    all_uncertainties.extend(uncertainty.cpu().tolist())

        train_loss /= max(train_count, 1)
        val_loss /= max(val_count, 1)
        labels = list(range(model.head.out_features)) 
        val_acc = accuracy_score(all_targets, all_preds)
        val_macro_precision = precision_score(all_targets, all_preds, labels=labels, average="macro", zero_division=0,)
        val_macro_recall = recall_score(all_targets, all_preds, labels=labels, average="macro", zero_division=0,)
        val_f1 = f1_score(all_targets, all_preds, labels=labels, average="macro", zero_division=0,)
        val_weighted_f1 = f1_score(all_targets, all_preds, labels=labels, average="weighted", zero_division=0,)
        mean_uncertainty = sum(all_uncertainties) / len(all_uncertainties) if all_uncertainties else None

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_precision": val_macro_precision,
            "val_macro_recall": val_macro_recall,
            "val_macro_f1": val_f1,
            "val_weighted_f1": val_weighted_f1,
            "val_uncertainty": mean_uncertainty,
        }
        history.append(row)

        with open(save_dir / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(history)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_dir / "best_model.pt")

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Macro-F1: {val_f1:.4f}" + (f" | Uncertainty: {mean_uncertainty:.4f}" if edl else ""))

    print(f"Training finished | Best Val Macro-F1: {best_f1:.4f}")
    return history


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchical", action="store_true")
    parser.add_argument(
        "--edl",
        action="store_true",
        help="Enable evidential deep learning",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to DINO or LeJEPA pretrained checkpoint",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save training results",
    )
    args = parser.parse_args()
    
    train_loader, val_loader, label_to_idx = initialize_classification_dataloaders(
        data_dir="/content/labeled-images",
        batch_size=512,
    )
    num_classes = len(label_to_idx)

    torch.manual_seed(42)

    siglip2 = initialize_siglip2(initialization="siglip2")
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        siglip2.vision_model.load_state_dict(checkpoint["model"], strict=True)
        print(f"Loaded pretrained backbone from: {args.ckpt}")
    model = SigLIP2Classifier(siglip2, num_classes)

    save_dir = args.save_dir or "/content/drive/MyDrive/results/classification"

    train_classification(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,
        edl=args.edl,
        save_dir=save_dir,
    )

    if args.hierarchical:
        from classification.hierarchical_independent import train_hierarchical_classifier
        train_hierarchical_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            label_to_idx=label_to_idx,
            ckpt_path=args.ckpt,
            single_stage_ckpt=str(Path(save_dir) / "best_model.pt"),
            edl=args.edl,
            save_dir=str(Path(save_dir) / "hierarchical_independent"),
            epochs=15,
        )
