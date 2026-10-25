import csv
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from classification_vit.train_classification import compute_loss, get_predictions


SWIN_NAME = "microsoft/swin-tiny-patch4-window7-224"


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def calculate_metrics(targets, preds, num_classes, uncertainties=None, class_names=None):
    labels = list(range(num_classes))

    report = classification_report(
        targets,
        preds,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "macro_precision": precision_score(
            targets, preds, labels=labels,
            average="macro", zero_division=0,
        ),
        "macro_recall": recall_score(
            targets, preds, labels=labels,
            average="macro", zero_division=0,
        ),
        "macro_f1": f1_score(
            targets, preds, labels=labels,
            average="macro", zero_division=0,
        ),
        "weighted_f1": f1_score(
            targets, preds, labels=labels,
            average="weighted", zero_division=0,
        ),
        "mean_uncertainty": (
            float(np.mean(uncertainties))
            if uncertainties is not None and len(uncertainties) > 0
            else None
        ),
        "classification_report": report,
    }

    cm = confusion_matrix(targets, preds, labels=labels)

    return metrics, cm


def evaluate_model(model, loader, device, edl=False, swin=False):
    model.eval()
    total_loss, total_count = 0.0, 0
    all_targets, all_preds, all_uncertainties = [], [], []

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device).long()
            logits = model(pixel_values=images).logits if swin else model(images)
            loss = compute_loss(logits, targets, edl)
            preds, uncertainty = get_predictions(logits, edl)

            total_loss += loss.item() * targets.size(0)
            total_count += targets.size(0)
            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            if uncertainty is not None:
                all_uncertainties.extend(uncertainty.cpu().tolist())

    metrics = {
        "loss": total_loss / max(total_count, 1),
        "accuracy": accuracy_score(all_targets, all_preds),
        "macro_f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "uncertainty": float(np.mean(all_uncertainties)) if all_uncertainties else None,
    }
    return metrics, all_targets, all_preds




def discover_groups(cm, threshold=15):
    num_classes = cm.shape[0]
    parent = list(range(num_classes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            if cm[i, j] + cm[j, i] >= threshold:
                union(i, j)

    components = defaultdict(list)
    for i in range(num_classes):
        components[find(i)].append(i)

    return sorted([sorted(group) for group in components.values() if len(group) > 1], key=lambda group: group[0])


def build_coarse_mapping(num_classes, merge_groups):
    merged = {cls for group in merge_groups for cls in group}
    leaf_to_coarse, coarse_to_leaf, group_to_coarse = {}, {}, {}
    coarse_idx = 0

    for cls in range(num_classes):
        if cls not in merged:
            leaf_to_coarse[cls] = coarse_idx
            coarse_to_leaf[coarse_idx] = cls
            coarse_idx += 1

    for group_id, group in enumerate(merge_groups):
        group_to_coarse[group_id] = coarse_idx
        for cls in group:
            leaf_to_coarse[cls] = coarse_idx
        coarse_idx += 1

    return leaf_to_coarse, coarse_to_leaf, group_to_coarse, coarse_idx


class SwinInferenceDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            pixels = self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].squeeze(0)
        return pixels, label


def evaluate_hierarchical(
    coarse_model,
    specialists,
    val_loader,
    val_samples,
    processor,
    coarse_to_leaf,
    group_to_coarse,
    merge_groups,
    num_classes,
    edl=False,
    batch_size=32,
):
    device = next(coarse_model.parameters()).device
    coarse_model.eval()

    for specialist in specialists:
        specialist.eval()

    coarse_preds = []
    coarse_uncertainties = []

    with torch.no_grad():
        for images, _ in val_loader:
            logits = coarse_model(images.to(device))
            preds, uncertainty = get_predictions(logits, edl)
            coarse_preds.extend(preds.cpu().tolist())
            if uncertainty is not None:
                coarse_uncertainties.extend(uncertainty.cpu().tolist())

    if len(coarse_preds) != len(val_samples):
        raise ValueError("Validation predictions and samples have different lengths.")

    leaf_to_coarse = {leaf: coarse_idx for coarse_idx, leaf in coarse_to_leaf.items()}

    for group_id, group in enumerate(merge_groups):
        coarse_idx = group_to_coarse[group_id]
        for leaf in group:
            leaf_to_coarse[leaf] = coarse_idx

    targets = [label for _, label in val_samples]
    coarse_targets = [leaf_to_coarse[label] for label in targets]
    num_coarse = len(coarse_to_leaf) + len(merge_groups)
    coarse_metrics, coarse_cm = calculate_metrics(
        coarse_targets,
        coarse_preds,
        num_classes=num_coarse,
        uncertainties=coarse_uncertainties if edl else None,
    )

    final_preds = [-1] * len(coarse_preds)
    routing = defaultdict(list)

    for index, coarse_idx in enumerate(coarse_preds):
        if coarse_idx in coarse_to_leaf:
            final_preds[index] = coarse_to_leaf[coarse_idx]
        else:
            routing[coarse_idx].append(index)

    for group_id, group in enumerate(merge_groups):
        coarse_idx = group_to_coarse[group_id]
        indices = routing.get(coarse_idx, [])
        if not indices:
            continue

        selected_samples = [val_samples[i] for i in indices]
        loader = DataLoader(
            SwinInferenceDataset(selected_samples, processor),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

        local_preds = []
        with torch.no_grad():
            for images, _ in loader:
                logits = specialists[group_id](pixel_values=images.to(device)).logits
                preds, _ = get_predictions(logits, edl)
                local_preds.extend(preds.cpu().tolist())

        for index, local_pred in zip(indices, local_preds):
            final_preds[index] = group[local_pred]

    if any(pred < 0 for pred in final_preds):
        raise RuntimeError("Some validation samples did not receive a final prediction.")

    final_metrics, final_cm = calculate_metrics(
        targets,
        final_preds,
        num_classes=num_classes,
    )

    return (
        final_metrics,
        final_cm,
        targets,
        final_preds,
        coarse_preds,
        coarse_metrics,
        coarse_cm,
    )


def evaluate_specialists(
    specialists,
    val_samples,
    processor,
    merge_groups,
    edl=False,
    batch_size=32,
):
    results = {}
    confusion_matrices = {}

    for group_id, group in enumerate(merge_groups):
        specialist = specialists[group_id]
        device = next(specialist.parameters()).device
        specialist.eval()
        selected_samples = [(path, label) for path, label in val_samples if label in group]

        if not selected_samples:
            continue

        loader = DataLoader(
            SwinInferenceDataset(selected_samples, processor),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

        local_targets = []
        local_preds = []
        local_uncertainties = []

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                logits = specialist(pixel_values=images).logits
                preds, uncertainty = get_predictions(logits, edl)
                local_preds.extend(preds.cpu().tolist())
                local_targets.extend(group.index(label) for label in targets.tolist())

                if uncertainty is not None:
                    local_uncertainties.extend(uncertainty.cpu().tolist())

        metrics, cm = calculate_metrics(
            local_targets,
            local_preds,
            num_classes=len(group),
            uncertainties=local_uncertainties if edl else None,
        )

        results[f"group_{group_id}"] = metrics
        confusion_matrices[f"group_{group_id}"] = cm

        print(f"Swin Group {group_id} | Acc: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")

    return results, confusion_matrices

