"""
LeJEPA self-supervised pre-training.

Implements:
    - Multi-crop augmentation
    - Prediction loss
    - SIGReg with Epps-Pulley statistic
    - SigLIP2 vision encoder training
"""

import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchvision import transforms
from torchvision.transforms import InterpolationMode


LEJEPA_DEFAULTS = {
    "epochs": 100,
    "lambda": 0.1,
    "num_local_crops": 6,
    "num_slices": 256,
    "lr": 5e-4,
    "min_lr": 1e-6,
    "weight_decay": 0.01,
    "warmup_epochs": 10,
    "output_dir": "./lejepa_output",
    "save_every": 5,
}


class LeJEPAAugmentation:
    def __init__(self, num_local_crops=6):
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.05, 0.4), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        self.num_local_crops = num_local_crops

    def __call__(self, image):
        global_views = [self.global_transform(image), self.global_transform(image)]
        local_views = [self.local_transform(image) for _ in range(self.num_local_crops)]
        return global_views, global_views + local_views


def extract_features(output):
    if isinstance(output, torch.Tensor):
        if output.ndim == 2: return output
        if output.ndim == 3: return output.mean(dim=1)

    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output

    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds

    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state.mean(dim=1)

    if isinstance(output, (tuple, list)):
        output = output[0]
        return output if output.ndim == 2 else output.mean(dim=1)

    raise ValueError(f"Unsupported backbone output type: {type(output)}")


def run_backbone(backbone, images):
    try:
        output = backbone(pixel_values=images, interpolate_pos_encoding=True)
    except TypeError:
        output = backbone(images)
    return extract_features(output)


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

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(ecf, op=torch.distributed.ReduceOp.SUM)
        ecf /= torch.distributed.get_world_size()
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    err = (ecf - exp_f).abs().square() * exp_f
    n = x.size(0) * world_size
    return torch.trapz(err, t, dim=1) * n


def cosine_schedule(base_value, final_value, epochs, steps_per_epoch, warmup_epochs=0):
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    schedule = []

    for step in range(total_steps):
        if step < warmup_steps:
            value = base_value * step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            value = final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress))
        schedule.append(value)

    return schedule


class LeJEPAModule(pl.LightningModule):
    def __init__(self, backbone, cfg, steps_per_epoch):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch
        self.augmentation = LeJEPAAugmentation(cfg["num_local_crops"])
        self.automatic_optimization = False

        self.lr_schedule = cosine_schedule(
            cfg["lr"],
            cfg["min_lr"],
            cfg["epochs"],
            steps_per_epoch,
            cfg["warmup_epochs"]
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.backbone.parameters(),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"]
        )

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        step = min(self.global_step, len(self.lr_schedule) - 1)

        for group in optimizer.param_groups:
            group["lr"] = self.lr_schedule[step]

        global_views = []
        all_views = [[] for _ in range(2 + self.cfg["num_local_crops"])]

        for image in batch:
            globals_i, all_i = self.augmentation(image)

            for i, view in enumerate(globals_i):
                global_views.append((i, view))

            for i, view in enumerate(all_i):
                all_views[i].append(view)

        global_batches = []
        for view_idx in range(2):
            views = [view for idx, view in global_views if idx == view_idx]
            global_batches.append(torch.stack(views).to(self.device))

        all_batches = [torch.stack(views).to(self.device) for views in all_views]

        global_embeddings = [run_backbone(self.backbone, x) for x in global_batches]
        all_embeddings = [run_backbone(self.backbone, x) for x in all_batches]

        centers = torch.stack(global_embeddings, dim=0).mean(dim=0)
        pred_loss = torch.stack([(centers - emb).square().mean() for emb in all_embeddings]).mean()

        sigreg_loss = torch.stack([
            sigreg(emb, self.global_step, self.cfg["num_slices"]).mean()
            for emb in all_embeddings
        ]).mean()

        loss = (1.0 - self.cfg["lambda"]) * pred_loss + self.cfg["lambda"] * sigreg_loss

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        batch_size = len(batch)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("pred_loss", pred_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("sigreg_loss", sigreg_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)

        return loss

    def on_train_epoch_end(self):
        if self.global_rank != 0:
            return

        output_dir = Path(self.cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        epoch = self.current_epoch + 1

        if epoch % self.cfg["save_every"] == 0 or epoch == self.cfg["epochs"]:
            torch.save({
                "epoch": epoch,
                "model": self.backbone.state_dict()
            }, output_dir / f"checkpoint_epoch_{epoch}.pth")

        metrics = self.trainer.callback_metrics

        train_loss = metrics.get("train_loss_epoch")
        pred_loss = metrics.get("pred_loss_epoch")
        sigreg_loss_value = metrics.get("sigreg_loss_epoch")

        log_file = output_dir / "training_log.csv"
        write_header = not log_file.exists()

        with log_file.open("a", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(["epoch", "train_loss", "prediction_loss", "sigreg_loss"])

            writer.writerow([
                epoch,
                float(train_loss.detach().cpu()) if train_loss is not None else "",
                float(pred_loss.detach().cpu()) if pred_loss is not None else "",
                float(sigreg_loss_value.detach().cpu()) if sigreg_loss_value is not None else "",
            ])


def train_lejepa(model, dataloader, config=None):
    cfg = LEJEPA_DEFAULTS.copy()

    if config is not None:
        cfg.update(config)

    module = LeJEPAModule(
        backbone=model,
        cfg=cfg,
        steps_per_epoch=len(dataloader)
    )

    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False
    )

    trainer.fit(module, train_dataloaders=dataloader)