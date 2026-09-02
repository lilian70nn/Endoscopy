"""
DINOv1 self-supervised pre-training.

Input:
    - Initialized vision model
    - Training DataLoader

Output:
    - Trained model

The training checkpoint is saved for later loading and evaluation.
"""



import copy
import math
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms
import lightning.pytorch as pl



# Default DINOv1 settings: based on the Meta repo provided

DINO_DEFAULTS = {
    "epochs": 100,
    "out_dim": 65536,
    "hidden_dim": 2048,
    "bottleneck_dim": 256,
    "local_crops_number": 8,
    "global_crops_scale": (0.4, 1.0),
    "local_crops_scale": (0.05, 0.4),
    "student_temp": 0.1,
    "warmup_teacher_temp": 0.04,
    "teacher_temp": 0.04,
    "warmup_teacher_temp_epochs": 0,
    "center_momentum": 0.9,
    "momentum_teacher": 0.996,
    "lr": 5e-4,
    "min_lr": 1e-6,
    "warmup_epochs": 10,
    "weight_decay": 0.04,
    "weight_decay_end": 0.4,
    "clip_grad": 3.0,
    "freeze_last_layer": 1,
    "norm_last_layer": True,
    "output_dir": "./dino_output"
}



# DINO augmentations

class GaussianBlur:
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p, self.radius_min, self.radius_max = p, radius_min, radius_max

    def __call__(self, image):
        if random.random() > self.p: return image
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))


class Solarization:
    def __init__(self, p): self.p = p

    def __call__(self, image):
        return ImageOps.solarize(image) if random.random() < self.p else image


class DataAugmentationDINO:
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4), local_crops_number=8):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ])
        normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.global_transfo1 = transforms.Compose([transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC), flip_and_color_jitter, GaussianBlur(1.0), normalize])
        self.global_transfo2 = transforms.Compose([transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC), flip_and_color_jitter, GaussianBlur(0.1), Solarization(0.2), normalize])
        self.local_transfo = transforms.Compose([transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC), flip_and_color_jitter, GaussianBlur(0.5), normalize])
        self.local_crops_number = local_crops_number

    def __call__(self, image):
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        crops.extend(self.local_transfo(image) for _ in range(self.local_crops_number))
        return crops


# DINO head: same structure as repo

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1: self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer: self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


# Convert arbitrary vision-model output -> [B, D]

def extract_features(output):
    if isinstance(output, torch.Tensor):
        if output.ndim == 2: return output
        if output.ndim == 3: return output[:, 0]
    if hasattr(output, "pooler_output") and output.pooler_output is not None: return output.pooler_output
    if hasattr(output, "image_embeds") and output.image_embeds is not None: return output.image_embeds
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None: return output.last_hidden_state[:, 0]
    if isinstance(output, (tuple, list)):
        output = output[0]
        return output if output.ndim == 2 else output[:, 0]
    raise ValueError("Cannot determine backbone feature. Provide a model whose output contains pooler_output, image_embeds, last_hidden_state, or a [B,D] tensor.")


def run_backbone(backbone, images):
    try: output = backbone(pixel_values=images)
    except (TypeError, ValueError): output = backbone(images)
    return extract_features(output)


# Student / Teacher wrapper

class DINOModel(nn.Module):
    def __init__(self, backbone, feature_dim, cfg, teacher=False):
        super().__init__()
        self.backbone = backbone
        self.head = DINOHead(feature_dim, cfg["out_dim"], norm_last_layer=False if teacher else cfg["norm_last_layer"], hidden_dim=cfg["hidden_dim"], bottleneck_dim=cfg["bottleneck_dim"])

    def forward(self, crops):
        if not isinstance(crops, list): crops = [crops]
        sizes = [x.shape[-1] for x in crops]
        outputs, start = [], 0
        while start < len(crops):
            end = start + 1
            while end < len(crops) and sizes[end] == sizes[start]: end += 1
            x = torch.cat(crops[start:end], dim=0)
            outputs.append(run_backbone(self.backbone, x))
            start = end
        return self.head(torch.cat(outputs, dim=0))


# DINO loss: centering + sharpening + cross-view CE

class DINOLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.student_temp = cfg["student_temp"]
        self.center_momentum = cfg["center_momentum"]
        self.ncrops = cfg["local_crops_number"] + 2
        self.register_buffer("center", torch.zeros(1, cfg["out_dim"]))
        warmup_epochs = cfg["warmup_teacher_temp_epochs"]
        warmup = np.linspace(cfg["warmup_teacher_temp"], cfg["teacher_temp"], warmup_epochs) if warmup_epochs > 0 else np.array([])
        remaining = np.ones(cfg["epochs"] - warmup_epochs) * cfg["teacher_temp"]
        self.teacher_temp_schedule = np.concatenate((warmup, remaining))

    def forward(self, student_output, teacher_output, epoch):
        student_out = (student_output / self.student_temp).chunk(self.ncrops)
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1).detach().chunk(2)
        total_loss, n_terms = 0.0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq: continue
                total_loss += torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1).mean()
                n_terms += 1
        total_loss /= n_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center /= len(teacher_output) * dist.get_world_size()
        else: batch_center /= len(teacher_output)
        self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))


# Official-style schedules / optimizer groups

def cosine_schedule(base_value, final_value, total_steps, warmup_steps=0, start_warmup_value=0):
    warmup = np.linspace(start_warmup_value, base_value, warmup_steps) if warmup_steps > 0 else np.array([])
    remaining = total_steps - warmup_steps
    iters = np.arange(remaining)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / max(1, remaining)))
    return np.concatenate((warmup, schedule))


def get_params_groups(model):
    regularized, not_regularized = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.endswith(".bias") or len(param.shape) == 1: not_regularized.append(param)
        else: regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def clip_gradients(model, clip):
    for _, p in model.named_parameters():
        if p.grad is None: continue
        norm = p.grad.data.norm(2)
        coef = clip / (norm + 1e-6)
        if coef < 1: p.grad.data.mul_(coef)


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer: return
    for name, p in model.named_parameters():
        if "last_layer" in name: p.grad = None


# Lightning DINO trainer

class DINOTrainer(pl.LightningModule):
    def __init__(self, backbone, feature_dim, steps_per_epoch, cfg):
        super().__init__()
        self.automatic_optimization = False
        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch
        self.student = DINOModel(copy.deepcopy(backbone), feature_dim, cfg, teacher=False)
        self.teacher = DINOModel(copy.deepcopy(backbone), feature_dim, cfg, teacher=True)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters(): p.requires_grad = False
        self.dino_loss = DINOLoss(cfg)
        self.augmentation = DataAugmentationDINO(cfg["global_crops_scale"], cfg["local_crops_scale"], cfg["local_crops_number"])
        total_steps = cfg["epochs"] * steps_per_epoch
        warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
        self.lr_schedule = cosine_schedule(cfg["lr"], cfg["min_lr"], total_steps, warmup_steps)
        self.wd_schedule = cosine_schedule(cfg["weight_decay"], cfg["weight_decay_end"], total_steps)
        self.momentum_schedule = cosine_schedule(cfg["momentum_teacher"], 1.0, total_steps)

    def configure_optimizers(self):
        return torch.optim.AdamW(get_params_groups(self.student), lr=self.cfg["lr"])

    def augment_batch(self, batch):
        if isinstance(batch, dict): batch = batch["image"]
        if isinstance(batch, tuple): batch = batch[0]
        per_image_crops = [self.augmentation(image.convert("RGB") if isinstance(image, Image.Image) else image) for image in batch]
        ncrops = self.cfg["local_crops_number"] + 2
        return [torch.stack([per_image_crops[b][crop_id] for b in range(len(per_image_crops))]).to(self.device, non_blocking=True) for crop_id in range(ncrops)]

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        step = self.current_epoch * self.steps_per_epoch + batch_idx
        step = min(step, len(self.lr_schedule) - 1)
        for group in optimizer.param_groups: group["lr"] = float(self.lr_schedule[step])
        optimizer.param_groups[0]["weight_decay"] = float(self.wd_schedule[step])
        images = self.augment_batch(batch)
        with torch.no_grad(): teacher_output = self.teacher(images[:2])
        student_output = self.student(images)
        loss = self.dino_loss(student_output, teacher_output, self.current_epoch)
        optimizer.zero_grad()
        self.manual_backward(loss)
        if self.cfg["clip_grad"] > 0: clip_gradients(self.student, self.cfg["clip_grad"])
        cancel_gradients_last_layer(self.current_epoch, self.student, self.cfg["freeze_last_layer"])
        optimizer.step()
        with torch.no_grad():
            m = float(self.momentum_schedule[step])
            for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
                teacher_param.data.mul_(m).add_(student_param.detach().data, alpha=1.0 - m)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("lr", float(self.lr_schedule[step]), prog_bar=False)
        self.log("teacher_momentum", float(self.momentum_schedule[step]), prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        if self.global_rank != 0: return
        Path(self.cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": self.current_epoch + 1,
            "student": self.student.state_dict(),
            "teacher": self.teacher.state_dict(),
            "student_backbone": self.student.backbone.state_dict(),
            "teacher_backbone": self.teacher.backbone.state_dict(),
            "dino_loss": self.dino_loss.state_dict(),
            "config": self.cfg
        }, Path(self.cfg["output_dir"]) / "checkpoint.pth")



# Public function: model + dataloader -> trained encoder

def train_dino(model, dataloader, config=None):
    cfg = DINO_DEFAULTS.copy()
    if config is not None: cfg.update(config)
    pl.seed_everything(0, workers=True)

    first_batch = next(iter(dataloader))
    if isinstance(first_batch, dict): sample = first_batch["image"][0]
    elif isinstance(first_batch, tuple): sample = first_batch[0][0]
    else: sample = first_batch[0]

    augmentation = DataAugmentationDINO(cfg["global_crops_scale"], cfg["local_crops_scale"], cfg["local_crops_number"])
    sample_tensor = augmentation(sample.convert("RGB") if isinstance(sample, Image.Image) else sample)[0].unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_model = copy.deepcopy(model).to(device).eval()
    with torch.no_grad(): feature_dim = run_backbone(probe_model, sample_tensor.to(device)).shape[-1]
    del probe_model

    module = DINOTrainer(model, feature_dim, len(dataloader), cfg)
    trainer = pl.Trainer(max_epochs=cfg["epochs"], accelerator="auto", devices="auto", precision="16-mixed" if torch.cuda.is_available() else "32-true", logger=False, enable_checkpointing=False, log_every_n_steps=10)
    trainer.fit(module, train_dataloaders=dataloader)

    trained_encoder = copy.deepcopy(module.teacher.backbone).cpu().eval()
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    torch.save(trained_encoder.state_dict(), Path(cfg["output_dir"]) / "trained_teacher_encoder.pth")
    torch.save(module.student.backbone.state_dict(), Path(cfg["output_dir"]) / "trained_student_encoder.pth")
    return trained_encoder