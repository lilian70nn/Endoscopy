"""
LeJEPA self-supervised pre-training.

Input:
    - Initialized vision model
    - Training DataLoader

The vision encoder checkpoint is saved after each epoch
for later downstream evaluation.
"""

import csv
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import lightning.pytorch as pl
from torchvision import transforms
from torch.distributed.nn.functional import all_reduce


# Default LeJEPA settings

LEJEPA_DEFAULTS = {
    "epochs": 5,
    "lambda": 0.1,
    "num_local_crops": 6,
    "num_slices": 256,
    "global_crops_scale": (0.4, 1.0),
    "local_crops_scale": (0.05, 0.4),
    "lr": 5e-4,
    "min_lr": 1e-6,
    "warmup_epochs": 1,
    "weight_decay": 0.01,
    "accumulate_grad_batches": 1,
    "devices": 4,
    "output_dir": "./lejepa_output"
}


# LeJEPA multi-crop augmentations

class DataAugmentationLeJEPA:
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4), local_crops_number=6):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ])
        normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.global_transfo1 = transforms.Compose([transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC), flip_and_color_jitter, normalize])
        self.global_transfo2 = transforms.Compose([transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC), flip_and_color_jitter, normalize])
        self.local_transfo = transforms.Compose([transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC), flip_and_color_jitter, normalize])
        self.local_crops_number = local_crops_number

    def __call__(self, image):
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        crops.extend(self.local_transfo(image) for _ in range(self.local_crops_number))
        return crops


# Convert arbitrary vision-model output -> [B, D]

def extract_features(output):
    if isinstance(output, torch.Tensor):
        if output.ndim == 2: return output
        if output.ndim == 3: return output.mean(dim=1)
    if hasattr(output, "pooler_output") and output.pooler_output is not None: return output.pooler_output
    if hasattr(output, "image_embeds") and output.image_embeds is not None: return output.image_embeds
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None: return output.last_hidden_state.mean(dim=1)
    if isinstance(output, (tuple, list)):
        output = output[0]
        return output if output.ndim == 2 else output.mean(dim=1)
    raise ValueError("Cannot determine backbone feature. Provide a model whose output contains pooler_output, image_embeds, last_hidden_state, or a [B,D] tensor.")


def run_backbone(backbone, images):
    try: output = backbone(pixel_values=images, interpolate_pos_encoding=True)
    except TypeError: output = backbone(images)
    return extract_features(output)


# SIGReg: Epps-Pulley statistic

def sigreg(x, global_step, num_slices=256):
    device = x.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(global_step))

    A = torch.randn(x.size(1), num_slices, generator=generator, device=device)
    A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)

    t = torch.linspace(-5, 5, 17, device=device)
    exp_f = torch.exp(-0.5 * t.square())

    x_t = (x @ A).unsqueeze(2) * t
    ecf = torch.exp(1j * x_t).mean(dim=0)

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        ecf = all_reduce(ecf, op=dist.ReduceOp.SUM) / world_size
    else:
        world_size = 1

    err = (ecf - exp_f).abs().square().mul(exp_f)
    n = x.size(0) * world_size

    return torch.trapz(err, t, dim=1) * n


# Official-style LR schedule

def cosine_schedule(base_value, final_value, total_steps, warmup_steps=0, start_warmup_value=0):
    warmup = np.linspace(start_warmup_value, base_value, warmup_steps) if warmup_steps > 0 else np.array([])
    remaining = total_steps - warmup_steps
    iters = np.arange(remaining)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / max(1, remaining)))
    return np.concatenate((warmup, schedule))


# Lightning LeJEPA trainer

class LeJEPATrainer(pl.LightningModule):
    def __init__(self, backbone, feature_dim, steps_per_epoch, cfg):
        super().__init__()
        self.automatic_optimization = False
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch
        self.augmentation = DataAugmentationLeJEPA(cfg["global_crops_scale"], cfg["local_crops_scale"], cfg["num_local_crops"])
        total_steps = cfg["epochs"] * steps_per_epoch
        warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
        self.lr_schedule = cosine_schedule(cfg["lr"], cfg["min_lr"], total_steps, warmup_steps)
        self.epoch_bar = None

    def configure_optimizers(self):
        return torch.optim.AdamW(self.backbone.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])

    def augment_batch(self, batch):
        if isinstance(batch, dict): batch = batch["image"]
        if isinstance(batch, tuple): batch = batch[0]
        per_image_crops = [self.augmentation(image.convert("RGB") if isinstance(image, Image.Image) else image) for image in batch]
        ncrops = self.cfg["num_local_crops"] + 2
        return [torch.stack([per_image_crops[b][crop_id] for b in range(len(per_image_crops))]).to(self.device, non_blocking=True) for crop_id in range(ncrops)]

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        step = self.current_epoch * self.steps_per_epoch + batch_idx
        step = min(step, len(self.lr_schedule) - 1)
        for group in optimizer.param_groups: group["lr"] = float(self.lr_schedule[step])

        views = self.augment_batch(batch)
        embeddings = [run_backbone(self.backbone, view) for view in views]

        global_embeddings = torch.stack(embeddings[:2], dim=0)
        centers = global_embeddings.mean(dim=0)
        pred_loss = torch.stack([(centers - emb).square().mean() for emb in embeddings]).mean()
        sigreg_loss = torch.stack([sigreg(emb, step, self.cfg["num_slices"]).mean() for emb in embeddings]).mean()
        loss = (1.0 - self.cfg["lambda"]) * pred_loss + self.cfg["lambda"] * sigreg_loss

        if self.epoch_bar is not None:
            self.epoch_bar.set_postfix(loss=f"{loss.detach().item():.4f}", pred=f"{pred_loss.detach().item():.4f}", sigreg=f"{sigreg_loss.detach().item():.4f}", lr=f"{float(self.lr_schedule[step]):.2e}")

        accum_steps = self.cfg["accumulate_grad_batches"]
        if batch_idx % accum_steps == 0: optimizer.zero_grad()
        self.manual_backward(loss / accum_steps)
        should_step = (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == self.steps_per_epoch
        if should_step: optimizer.step()

        batch_size = len(batch["image"]) if isinstance(batch, dict) else len(batch[0]) if isinstance(batch, tuple) else len(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("pred_loss", pred_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("sigreg_loss", sigreg_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("lr", float(self.lr_schedule[step]), prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        if self.global_rank != 0: return

        if self.epoch_bar is not None: self.epoch_bar.update(1)

        output_dir = Path(self.cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        epoch = self.current_epoch + 1

        torch.save({
            "epoch": epoch,
            "model": self.backbone.state_dict()
        }, output_dir / f"checkpoint_epoch_{epoch}.pth")

        metrics = self.trainer.callback_metrics
        train_loss = metrics.get("train_loss_epoch")
        pred_loss = metrics.get("pred_loss_epoch")
        sigreg_loss = metrics.get("sigreg_loss_epoch")
        log_file = output_dir / "training_log.csv"
        write_header = not log_file.exists()

        with log_file.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header: writer.writerow(["epoch", "train_loss", "prediction_loss", "sigreg_loss", "lr"])
            writer.writerow([
                epoch,
                float(train_loss.detach().cpu()) if train_loss is not None else "",
                float(pred_loss.detach().cpu()) if pred_loss is not None else "",
                float(sigreg_loss.detach().cpu()) if sigreg_loss is not None else "",
                float(self.lr_schedule[min(epoch * self.steps_per_epoch - 1, len(self.lr_schedule) - 1)])
            ])

    def on_train_start(self):
        if self.global_rank == 0:
            self.epoch_bar = tqdm(total=self.cfg["epochs"], desc="Epoch", position=0, leave=True, dynamic_ncols=True)

    def on_train_epoch_start(self):
        if self.epoch_bar is not None: self.epoch_bar.set_description(f"Epoch {self.current_epoch + 1}/{self.cfg['epochs']}")

    def on_train_end(self):
        if self.epoch_bar is not None: self.epoch_bar.close()


# Public function: LeJEPA pre-training

def train_lejepa(model, dataloader, config=None):
    cfg = LEJEPA_DEFAULTS.copy()
    if config is not None: cfg.update(config)
    pl.seed_everything(0, workers=True)

    feature_dim = model.config.hidden_size

    module = LeJEPATrainer(model, feature_dim, len(dataloader), cfg)
    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg["devices"],
        strategy="ddp" if cfg["devices"] > 1 else "auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=10,
    )
    trainer.fit(module, train_dataloaders=dataloader)