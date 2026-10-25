"""Independent ViT coarse classifier and Swin specialists with hard-routed inference."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from classification_vit.train_classification import compute_loss, get_predictions, initialize_vit_classifier
from classification_vit.hierarchical_utils import (
    SWIN_NAME, SwinInferenceDataset, build_coarse_mapping, calculate_metrics,
    discover_groups, evaluate_hierarchical, evaluate_model, evaluate_specialists,
    save_csv,
)


class CoarseDataset(Dataset):
    def __init__(self, dataset, leaf_to_coarse):
        self.dataset = dataset
        self.leaf_to_coarse = leaf_to_coarse

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, self.leaf_to_coarse[int(label)]


class SpecialistDataset(Dataset):
    def __init__(self, samples, processor, group):
        self.dataset = SwinInferenceDataset([(path, group.index(label)) for path, label in samples if label in group], processor)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def train_independent_model(model, train_loader, val_loader, epochs, checkpoint_path, edl=False, swin=False, name="Model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_f1, history = -1.0, []
    num_classes = model.config.num_labels if swin else model.head.out_features

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_count = 0.0, 0
        for images, targets in tqdm(train_loader, desc=f"{name} Epoch {epoch}/{epochs}"):
            images, targets = images.to(device), targets.to(device).long()
            optimizer.zero_grad()
            logits = model(pixel_values=images).logits if swin else model(images)
            loss = compute_loss(logits, targets, edl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_count += targets.size(0)

        model.eval()
        targets_all, preds_all, uncertainties = [], [], []
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device).long()
                logits = model(pixel_values=images).logits if swin else model(images)
                loss = compute_loss(logits, targets, edl)
                preds, uncertainty = get_predictions(logits, edl)
                val_loss += loss.item() * targets.size(0)
                val_count += targets.size(0)
                targets_all.extend(targets.cpu().tolist())
                preds_all.extend(preds.cpu().tolist())
                if uncertainty is not None:
                    uncertainties.extend(uncertainty.cpu().tolist())

        metrics, _ = calculate_metrics(targets_all, preds_all, num_classes, uncertainties if edl else None)
        row = {"epoch": epoch, "train_loss": total_loss / max(total_count, 1), "val_loss": val_loss / max(val_count, 1),
               "val_accuracy": metrics["accuracy"], "val_macro_precision": metrics["macro_precision"],
               "val_macro_recall": metrics["macro_recall"], "val_macro_f1": metrics["macro_f1"],
               "val_weighted_f1": metrics["weighted_f1"], "val_uncertainty": metrics["mean_uncertainty"]}
        history.append(row)
        save_csv(checkpoint_path.parent / f"{name.lower().replace(' ', '_')}_metrics.csv", history)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), checkpoint_path)
        print(f"{name} Epoch {epoch}/{epochs} | Train Loss: {row['train_loss']:.4f} | Val Acc: {metrics['accuracy']:.4f} | Val Macro-F1: {metrics['macro_f1']:.4f}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"{name} finished | Best Val Macro-F1: {best_f1:.4f}")
    return model, history


def train_hierarchical_classifier(model, train_loader, val_loader, label_to_idx, ckpt_path=None, single_stage_ckpt=None,
                                  edl=False, save_dir="./results/classification/hierarchical", epochs=20, threshold=15):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_to_idx)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    if single_stage_ckpt is None:
        raise ValueError("single_stage_ckpt must be provided.")
    if isinstance(val_loader.sampler, torch.utils.data.RandomSampler):
        raise ValueError("val_loader must have shuffle=False to align predictions with val_samples.")

    model.load_state_dict(torch.load(single_stage_ckpt, map_location="cpu", weights_only=True))
    model = model.to(device)
    single_metrics, single_targets, single_preds = evaluate_model(model, val_loader, device, edl=edl)
    single_metrics_full, single_cm = calculate_metrics(single_targets, single_preds, num_classes)
    single_metrics_full["loss"] = single_metrics["loss"]
    single_metrics_full["mean_uncertainty"] = single_metrics["uncertainty"]
    np.savetxt(save_dir / "single_stage_confusion_matrix.csv", single_cm, fmt="%d", delimiter=",")
    print(f"Single-stage | Acc: {single_metrics_full['accuracy']:.4f} | Macro-F1: {single_metrics_full['macro_f1']:.4f}")

    merge_groups = discover_groups(single_cm, threshold=threshold)
    print(f"Detected merge groups: {[[idx_to_label[i] for i in group] for group in merge_groups]}")
    if not merge_groups:
        print("No merge groups detected. Hierarchical training skipped.")
        with open(save_dir / "final_metrics.json", "w") as f:
            json.dump({"single_stage": single_metrics_full, "hierarchical": None, "reason": "No merge groups detected"}, f, indent=2)
        return

    leaf_to_coarse, coarse_to_leaf, group_to_coarse, num_coarse = build_coarse_mapping(num_classes, merge_groups)
    groups_info = {"threshold": threshold, "original_classes": {str(i): idx_to_label[i] for i in range(num_classes)},
                   "merge_groups": [{"group_id": i, "class_indices": group, "class_names": [idx_to_label[j] for j in group]} for i, group in enumerate(merge_groups)],
                   "leaf_to_coarse": {str(k): v for k, v in leaf_to_coarse.items()},
                   "coarse_to_leaf": {str(k): v for k, v in coarse_to_leaf.items()},
                   "group_to_coarse": {str(k): v for k, v in group_to_coarse.items()}, "num_coarse_classes": num_coarse}
    with open(save_dir / "groups.json", "w") as f:
        json.dump(groups_info, f, indent=2)

    train_samples, val_samples = train_loader.dataset.samples, val_loader.dataset.samples
    if len(val_samples) != len(single_targets):
        raise ValueError("Validation dataset and predictions have different lengths.")
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    coarse_model = initialize_vit_classifier(num_coarse, ckpt_path=ckpt_path)
    coarse_train_loader = DataLoader(CoarseDataset(train_loader.dataset, leaf_to_coarse), batch_size=train_loader.batch_size,
                                     shuffle=True, num_workers=train_loader.num_workers, pin_memory=True)
    coarse_val_loader = DataLoader(CoarseDataset(val_loader.dataset, leaf_to_coarse), batch_size=val_loader.batch_size,
                                   shuffle=False, num_workers=val_loader.num_workers, pin_memory=True)
    coarse_model, coarse_history = train_independent_model(coarse_model, coarse_train_loader, coarse_val_loader, epochs,
                                                           save_dir / "coarse_best_model.pt", edl=edl, name="Coarse")

    processor = AutoImageProcessor.from_pretrained(SWIN_NAME)
    specialists, specialist_histories = [], {}
    for group_id, group in enumerate(merge_groups):
        specialist = AutoModelForImageClassification.from_pretrained(SWIN_NAME, num_labels=len(group), ignore_mismatched_sizes=True)
        group_train_dataset = SpecialistDataset(train_samples, processor, group)
        group_val_dataset = SpecialistDataset(val_samples, processor, group)
        if not len(group_train_dataset) or not len(group_val_dataset):
            raise ValueError(f"Group {group_id} has no train or validation samples.")
        group_train_loader = DataLoader(group_train_dataset, batch_size=16, shuffle=True, num_workers=train_loader.num_workers, pin_memory=True)
        group_val_loader = DataLoader(group_val_dataset, batch_size=32, shuffle=False, num_workers=val_loader.num_workers, pin_memory=True)
        specialist, history = train_independent_model(specialist, group_train_loader, group_val_loader, epochs,
                                                      save_dir / f"swin_group_{group_id}_best_model.pt", edl=edl,
                                                      swin=True, name=f"Swin Group {group_id}")
        specialists.append(specialist.to("cpu"))
        if device.type == "cuda":
            torch.cuda.empty_cache()
        specialist_histories[f"group_{group_id}"] = history

    specialists = [specialist.to(device) for specialist in specialists]
    hierarchical_metrics, hierarchical_cm, targets, final_preds, coarse_preds, coarse_metrics, coarse_cm = evaluate_hierarchical(
        coarse_model, specialists, val_loader, val_samples, processor, coarse_to_leaf, group_to_coarse, merge_groups, num_classes, edl=edl)
    specialist_metrics, specialist_cms = evaluate_specialists(specialists, val_samples, processor, merge_groups, edl=edl)
    hierarchical_metrics.pop("mean_uncertainty", None)

    np.savetxt(save_dir / "coarse_confusion_matrix.csv", coarse_cm, fmt="%d", delimiter=",")
    np.savetxt(save_dir / "hierarchical_confusion_matrix.csv", hierarchical_cm, fmt="%d", delimiter=",")
    for group_name, cm in specialist_cms.items():
        np.savetxt(save_dir / f"swin_{group_name}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    prediction_rows = [{"image_path": str(val_samples[i][0]), "true_label": idx_to_label[targets[i]],
                        "coarse_prediction": coarse_preds[i], "final_prediction": idx_to_label[final_preds[i]],
                        "correct": int(targets[i] == final_preds[i])} for i in range(len(targets))]
    save_csv(save_dir / "hierarchical_predictions.csv", prediction_rows)
    final_metrics = {"single_stage": single_metrics_full, "coarse": coarse_metrics, "specialists": specialist_metrics,
                     "hierarchical": hierarchical_metrics, "training_history": {"coarse": coarse_history, "specialists": specialist_histories}}
    with open(save_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n========== Final Results ==========")
    print(f"Single-stage | Acc: {single_metrics_full['accuracy']:.4f} | Macro-F1: {single_metrics_full['macro_f1']:.4f}")
    print(f"Coarse | Acc: {coarse_metrics['accuracy']:.4f} | Macro-F1: {coarse_metrics['macro_f1']:.4f}")
    for group_name, metrics in specialist_metrics.items():
        print(f"Swin {group_name} | Acc: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Final Hierarchical | Acc: {hierarchical_metrics['accuracy']:.4f} | Macro-F1: {hierarchical_metrics['macro_f1']:.4f}")
    print(f"Results saved to: {save_dir}")
    return final_metrics
