import csv
import json
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
from transformers import AutoImageProcessor, AutoModelForImageClassification

from utils import initialize_siglip2
from classification.train_classification import SigLIP2Classifier, compute_loss, get_predictions


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


def initialize_coarse_model(num_classes, ckpt_path=None):
    siglip2 = initialize_siglip2(initialization="siglip2")

    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        siglip2.vision_model.load_state_dict(checkpoint["model"], strict=True)
        print(f"Loaded pretrained SigLIP2 backbone: {ckpt_path}")

    return SigLIP2Classifier(siglip2, num_classes)

class HierarchicalDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        siglip_image, label = self.dataset[index]
        path, _ = self.dataset.samples[index]
        with Image.open(path) as image:
            swin_image = self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].squeeze(0)
        return siglip_image, swin_image, int(label)


def hierarchical_forward(coarse_model, specialists, siglip_images, swin_images, targets, leaf_to_coarse, group_to_coarse, merge_groups, num_classes, edl=False, epoch=1):
    coarse_logits = coarse_model(siglip_images)
    if edl:
        alpha = F.softplus(coarse_logits) + 1
        coarse_probs = alpha / alpha.sum(dim=1, keepdim=True)
    else:
        coarse_probs = F.softmax(coarse_logits, dim=1)

    final_probs = coarse_probs.new_zeros((targets.size(0), num_classes))
    merged = {cls for group in merge_groups for cls in group}

    for cls in range(num_classes):
        if cls not in merged:
            final_probs[:, cls] = coarse_probs[:, leaf_to_coarse[cls]]

    if edl:
        coarse_targets = torch.tensor(
            [leaf_to_coarse[t] for t in targets.tolist()],
            device=targets.device,
            dtype=torch.long,
        )
        total_loss = compute_loss(coarse_logits, coarse_targets, edl=True) * targets.size(0)


    for group_id, group in enumerate(merge_groups):
        mask = torch.isin(targets, torch.tensor(group, device=targets.device))
        if not mask.any():
            continue

        logits = specialists[group_id](pixel_values=swin_images[mask]).logits
        if edl:
            alpha = F.softplus(logits) + 1
            local_probs = alpha / alpha.sum(dim=1, keepdim=True)
        else:
            local_probs = F.softmax(logits, dim=1)

        group_probs = coarse_probs[mask, group_to_coarse[group_id]].unsqueeze(1)
        for local_idx, cls in enumerate(group):
            final_probs[mask, cls] = group_probs[:, 0] * local_probs[:, local_idx]

        if edl:
            local_targets = torch.tensor(
                [group.index(t) for t in targets[mask].tolist()],
                device=targets.device,
                dtype=torch.long,
            )
            total_loss = total_loss + compute_loss(logits, local_targets, edl=True) * mask.sum()
    if edl:
        loss = total_loss / targets.size(0)
    else:
        loss = F.nll_loss(torch.log(final_probs.clamp_min(1e-8)), targets)

    return final_probs, loss


def train_hierarchical_model(coarse_model, specialists, train_loader, val_loader, val_samples, processor, leaf_to_coarse, coarse_to_leaf, group_to_coarse, merge_groups, num_classes, epochs, save_dir, edl=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coarse_model = coarse_model.to(device)
    specialists = [model.to(device) for model in specialists]
    parameters = list(coarse_model.parameters())
    for specialist in specialists:
        parameters.extend(specialist.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)

    train_dataset = HierarchicalDataset(train_loader.dataset, processor)
    joint_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=train_loader.num_workers, pin_memory=True)
    best_f1, history = -1.0, []

    for epoch in range(1, epochs + 1):
        coarse_model.train()
        for specialist in specialists:
            specialist.train()

        train_loss, train_count = 0.0, 0

        for siglip_images, swin_images, targets in tqdm(joint_loader, desc=f"Hierarchical Epoch {epoch}/{epochs}"):
            siglip_images, swin_images, targets = siglip_images.to(device), swin_images.to(device), targets.to(device).long()
            optimizer.zero_grad()

            final_probs, loss = hierarchical_forward(coarse_model, specialists, siglip_images, swin_images, targets, leaf_to_coarse, group_to_coarse, merge_groups, num_classes, edl, epoch=epoch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            train_count += targets.size(0)

        metrics, _, _, _, _, _, _ = evaluate_hierarchical(
            coarse_model, specialists, val_loader, val_samples, processor,
            coarse_to_leaf, group_to_coarse, merge_groups, num_classes, edl=edl,
        )

        row = {"epoch": epoch, "train_loss": train_loss / max(train_count, 1), "val_accuracy": metrics["accuracy"], "val_macro_f1": metrics["macro_f1"]}
        history.append(row)
        save_csv(save_dir / "metrics.csv", history)

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(coarse_model.state_dict(), save_dir / "coarse_best_model.pt")
            for group_id, specialist in enumerate(specialists):
                torch.save(specialist.state_dict(), save_dir / f"swin_group_{group_id}_best_model.pt")

        print(f"Epoch {epoch}/{epochs} | Train Loss: {row['train_loss']:.4f} | Val Acc: {metrics['accuracy']:.4f} | Val Macro-F1: {metrics['macro_f1']:.4f}")

    coarse_model.load_state_dict(torch.load(save_dir / "coarse_best_model.pt", map_location=device, weights_only=True))
    for group_id, specialist in enumerate(specialists):
        specialist.load_state_dict(torch.load(save_dir / f"swin_group_{group_id}_best_model.pt", map_location=device, weights_only=True))

    print(f"Hierarchical training finished | Best Val Macro-F1: {best_f1:.4f}")
    return coarse_model, specialists, history


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


def train_hierarchical_classifier(model, train_loader, val_loader, label_to_idx, ckpt_path=None, single_stage_ckpt=None, edl=False, save_dir="./results/classification/hierarchical", epochs=20, threshold=15):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_to_idx)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    if single_stage_ckpt is None:
        raise ValueError("single_stage_ckpt must be provided.")

    model.load_state_dict(torch.load(single_stage_ckpt, map_location="cpu", weights_only=True))
    model = model.to(device)
    single_metrics, single_targets, single_preds = evaluate_model(model, val_loader, device, edl=edl)
    single_cm = confusion_matrix(single_targets, single_preds, labels=list(range(num_classes)))
    np.savetxt(save_dir / "single_stage_confusion_matrix.csv", single_cm, fmt="%d", delimiter=",")

    print(f"Single-stage | Acc: {single_metrics['accuracy']:.4f} | Macro-F1: {single_metrics['macro_f1']:.4f}")

    merge_groups = discover_groups(single_cm, threshold=threshold)
    print(f"Detected merge groups: {[[idx_to_label[i] for i in group] for group in merge_groups]}")

    if not merge_groups:
        print("No merge groups detected. Hierarchical training skipped.")
        with open(save_dir / "final_metrics.json", "w") as f:
            json.dump({"single_stage": single_metrics, "hierarchical": None, "reason": "No merge groups detected"}, f, indent=2)
        return

    leaf_to_coarse, coarse_to_leaf, group_to_coarse, num_coarse = build_coarse_mapping(num_classes, merge_groups)
    groups_info = {
        "threshold": threshold,
        "original_classes": {str(i): idx_to_label[i] for i in range(num_classes)},
        "merge_groups": [{"group_id": i, "class_indices": group, "class_names": [idx_to_label[j] for j in group]} for i, group in enumerate(merge_groups)],
        "leaf_to_coarse": {str(k): v for k, v in leaf_to_coarse.items()},
        "coarse_to_leaf": {str(k): v for k, v in coarse_to_leaf.items()},
        "group_to_coarse": {str(k): v for k, v in group_to_coarse.items()},
        "num_coarse_classes": num_coarse,
    }
    with open(save_dir / "groups.json", "w") as f:
        json.dump(groups_info, f, indent=2)

    train_samples = train_loader.dataset.samples
    val_samples = val_loader.dataset.samples

    if len(val_samples) != len(single_targets):
        raise ValueError("Validation dataset and predictions have different lengths.")

    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    processor = AutoImageProcessor.from_pretrained(SWIN_NAME)
    coarse_model = initialize_coarse_model(num_coarse, ckpt_path=ckpt_path)
    specialists = [
        AutoModelForImageClassification.from_pretrained(SWIN_NAME, num_labels=len(group), ignore_mismatched_sizes=True)
        for group in merge_groups
    ]

    coarse_model, specialists, history = train_hierarchical_model(
        coarse_model=coarse_model, specialists=specialists,
        train_loader=train_loader, val_loader=val_loader, val_samples=val_samples,
        processor=processor, leaf_to_coarse=leaf_to_coarse, coarse_to_leaf=coarse_to_leaf,
        group_to_coarse=group_to_coarse, merge_groups=merge_groups,
        num_classes=num_classes, epochs=epochs, save_dir=save_dir, edl=edl,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    coarse_model = coarse_model.to(device)
    hierarchical_metrics, hierarchical_cm, targets, final_preds, coarse_preds, coarse_metrics, coarse_cm = evaluate_hierarchical(
        coarse_model=coarse_model, specialists=specialists, val_loader=val_loader, val_samples=val_samples,
        processor=processor, coarse_to_leaf=coarse_to_leaf, group_to_coarse=group_to_coarse,
        merge_groups=merge_groups, num_classes=num_classes, edl=edl,
    )

    specialist_metrics, specialist_cms = evaluate_specialists(
        specialists=specialists, val_samples=val_samples, processor=processor,
        merge_groups=merge_groups, edl=edl,
    )

    np.savetxt(save_dir / "coarse_confusion_matrix.csv", coarse_cm, fmt="%d", delimiter=",",)
    np.savetxt(save_dir / "hierarchical_confusion_matrix.csv", hierarchical_cm, fmt="%d", delimiter=",")
    for group_name, cm in specialist_cms.items():
        np.savetxt(save_dir / f"swin_{group_name}_confusion_matrix.csv", cm, fmt="%d", delimiter=",",)

    
    prediction_rows = [
        {"image_path": str(val_samples[i][0]), "true_label": idx_to_label[targets[i]], "coarse_prediction": coarse_preds[i], "final_prediction": idx_to_label[final_preds[i]], "correct": int(targets[i] == final_preds[i])}
        for i in range(len(targets))
    ]
    save_csv(save_dir / "hierarchical_predictions.csv", prediction_rows)

    final_metrics = {
        "single_stage": single_metrics,
        "coarse": coarse_metrics,
        "specialists": specialist_metrics,
        "hierarchical": hierarchical_metrics,
        "training_history": history,
    }
    
    with open(save_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n========== Final Results ==========")
    print(f"Single-stage Macro-F1: {single_metrics['macro_f1']:.4f}")
    print(f"Hierarchical Macro-F1: {hierarchical_metrics['macro_f1']:.4f}")
    print(f"Hierarchical Accuracy: {hierarchical_metrics['accuracy']:.4f}")
    print(f"Results saved to: {save_dir}")
    return final_metrics