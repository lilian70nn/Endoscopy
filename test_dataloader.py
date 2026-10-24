"""
Cortex DataLoader multi-GPU training test.

Keeps the original DINO training infrastructure:
    - Original Cortex DataLoader
    - Automatic GPU detection
    - Lightning DDP
    - Mixed precision
    - AdamW optimizer
    - LR / Weight decay schedules
    - Manual backward
    - Gradient clipping
    - Epoch checkpoint
    - Automatic checkpoint resume
    - Training CSV log

Replaces:
    - SigLIP2 -> TinyCNN
    - DINO loss -> Simple RGB prediction loss

Run:
    python test_cortex_training.py
"""

import csv
import re
from pathlib import Path
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms

from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint

from cortex_dataloader import initialize_dataloader
from utils import cosine_value, get_epoch_progress, get_lr


# ============================================================
# 1. Configuration
# ============================================================

DEFAULTS = {
    "epochs": 2,

    "image_size": 64,

    "lr": 5e-4,
    "min_lr": 1e-6,
    "warmup_epochs": 1,

    "weight_decay": 0.04,
    "weight_decay_end": 0.4,

    "clip_grad": 3.0,
    "accumulate_grad_batches": 1,

    "devices": (
        torch.cuda.device_count()
        if torch.cuda.is_available()
        else 1
    ),

    "output_dir": "./checkpoints/tiny_cortex_test",
}


# ============================================================
# 2. Tiny Model
# ============================================================

class TinyCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# 3. Optimizer parameter groups
# ============================================================

def get_params_groups(model):

    regularized = []
    not_regularized = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)

    return [
        {"params": regularized},
        {
            "params": not_regularized,
            "weight_decay": 0.0,
        },
    ]


# ============================================================
# 4. Gradient clipping
# ============================================================

def clip_gradients(model, clip):

    for _, p in model.named_parameters():

        if p.grad is None:
            continue

        norm = p.grad.data.norm(2)

        coef = clip / (norm + 1e-6)

        if coef < 1:
            p.grad.data.mul_(coef)


# ============================================================
# 5. Find latest checkpoint
# ============================================================

def find_latest_checkpoint(output_dir):

    output_dir = Path(output_dir)

    checkpoints = []

    for path in output_dir.glob("checkpoint_epoch_*.pth"):

        match = re.fullmatch(
            r"checkpoint_epoch_(\d+)\.pth",
            path.name,
        )

        if match:
            checkpoints.append(
                (int(match.group(1)), path)
            )

    if not checkpoints:
        return None

    return max(
        checkpoints,
        key=lambda x: x[0]
    )[1]


# ============================================================
# 6. Epoch checkpoint
# ============================================================

class EpochCheckpoint(ModelCheckpoint):

    FILE_EXTENSION = ".pth"

    def __init__(self, output_dir):

        super().__init__(
            dirpath=output_dir,
            filename="checkpoint_epoch_{epoch_plus_one}",
            save_top_k=-1,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_weights_only=False,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        )

    def on_train_epoch_end(self, trainer, pl_module):

        print(
            f"[DEBUG CKPT A] Rank {trainer.global_rank}: "
            f"checkpoint callback entered",
            flush=True,
        )

        trainer.callback_metrics["epoch_plus_one"] = (
            trainer.current_epoch + 1
        )

        super().on_train_epoch_end(
            trainer,
            pl_module,
        )

        print(
            f"[DEBUG CKPT B] Rank {trainer.global_rank}: "
            f"checkpoint callback returned",
            flush=True,
        )

        if trainer.is_global_zero:
            print(
                f"[DEBUG CKPT FILES] "
                f"{list(Path(self.dirpath).glob('checkpoint_epoch_*.pth'))}",
                flush=True,
            )


# ============================================================
# 7. Lightning Trainer
# ============================================================

class TinyTrainer(pl.LightningModule):

    def __init__(self, model, cfg):

        super().__init__()

        self.automatic_optimization = False

        self.cfg = cfg

        self.model = model

        self.transform = transforms.Compose([
            transforms.Resize(
                (cfg["image_size"], cfg["image_size"])
            ),
            transforms.ToTensor(),
        ])

        self.epoch_bar = None

        self.previous_steps = None
        self.current_epoch_steps = 0

        self.last_lr = cfg["lr"]
        self.last_weight_decay = cfg["weight_decay"]


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    def configure_optimizers(self):

        return torch.optim.AdamW(
            get_params_groups(self.model),
            lr=self.cfg["lr"],
        )


    # --------------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------------

    def prepare_batch(self, batch):

        if isinstance(batch, dict):
            batch = batch["image"]

        if isinstance(batch, tuple):
            batch = batch[0]

        images = torch.stack([
            self.transform(
                image.convert("RGB")
                if isinstance(image, Image.Image)
                else image
            )
            for image in batch
        ])

        return images.to(
            self.device,
            non_blocking=True,
        )


    # --------------------------------------------------------
    # Save checkpoint state
    # --------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):

        checkpoint["epoch_number"] = (
            self.current_epoch + 1
        )

        checkpoint["model"] = (
            self.model.state_dict()
        )

        checkpoint["previous_steps"] = (
            self.current_epoch_steps
            if self.current_epoch_steps > 0
            else self.previous_steps
        )

        checkpoint["current_epoch_steps"] = 0

        checkpoint["last_lr"] = self.last_lr

        checkpoint["last_weight_decay"] = (
            self.last_weight_decay
        )


    # --------------------------------------------------------
    # Load checkpoint state
    # --------------------------------------------------------

    def on_load_checkpoint(self, checkpoint):

        self.previous_steps = checkpoint.get(
            "previous_steps"
        )

        self.current_epoch_steps = checkpoint.get(
            "current_epoch_steps",
            0,
        )

        self.last_lr = checkpoint.get(
            "last_lr",
            self.cfg["lr"],
        )

        self.last_weight_decay = checkpoint.get(
            "last_weight_decay",
            self.cfg["weight_decay"],
        )


    # --------------------------------------------------------
    # Training step
    # --------------------------------------------------------

    def training_step(self, batch, batch_idx):

        print(
            f"[DEBUG TRAIN START] Rank {self.global_rank}: "
            f"step={batch_idx}",
            flush=True,
        )

        optimizer = self.optimizers()

        # Same epoch progress logic as DINO
        epoch_progress = get_epoch_progress(
            self.current_epoch,
            batch_idx,
            self.previous_steps,
        )

        # Learning rate schedule
        lr = get_lr(
            self.cfg,
            epoch_progress,
        )

        progress = epoch_progress / self.cfg["epochs"]

        # Weight decay schedule
        wd = cosine_value(
            self.cfg["weight_decay"],
            self.cfg["weight_decay_end"],
            progress,
        )

        self.last_lr = lr
        self.last_weight_decay = wd

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.param_groups[0]["weight_decay"] = wd

        self.current_epoch_steps = batch_idx + 1

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        images = self.prepare_batch(batch)

        # Self-supervised target: average RGB
        target = images.mean(dim=(2, 3))

        prediction = self.model(images)

        loss = F.mse_loss(
            prediction,
            target,
        )

        if self.epoch_bar is not None:
            self.epoch_bar.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                lr=f"{lr:.2e}",
            )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        accum_steps = self.cfg[
            "accumulate_grad_batches"
        ]

        if batch_idx % accum_steps == 0:
            optimizer.zero_grad()

        print(
            f"[DEBUG BACKWARD A] Rank {self.global_rank}: "
            f"step={batch_idx}",
            flush=True,
        )

        self.manual_backward(
            loss / accum_steps
        )

        print(
            f"[DEBUG BACKWARD B] Rank {self.global_rank}: "
            f"step={batch_idx}",
            flush=True,
        )

        # ----------------------------------------------------
        # Optimizer step
        # ----------------------------------------------------

        should_step = (
            (batch_idx + 1) % accum_steps == 0
        )

        if should_step:

            if self.cfg["clip_grad"] > 0:
                clip_gradients(
                    self.model,
                    self.cfg["clip_grad"],
                )

            optimizer.step()

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        batch_size = len(batch)

        print(
            f"[DEBUG TRAIN A] Rank {self.global_rank}: "
            f"step={batch_idx}, before sync_dist log",
            flush=True,
        )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        print(
            f"[DEBUG TRAIN B] Rank {self.global_rank}: "
            f"step={batch_idx}, after sync_dist log",
            flush=True,
        )

        self.log(
            "lr",
            lr,
            prog_bar=False,
        )

        self.log(
            "weight_decay",
            wd,
            prog_bar=False,
        )

        print(
            f"[DEBUG TRAIN END] Rank {self.global_rank}: "
            f"step={batch_idx}",
            flush=True,
        )

        return loss


    # --------------------------------------------------------
    # Epoch end
    # --------------------------------------------------------

    def on_train_epoch_end(self):

        print(
            f"[DEBUG 11] Rank {self.global_rank}: "
            f"Lightning on_train_epoch_end entered",
            flush=True,
        )

        self.previous_steps = (
            self.current_epoch_steps
        )

        self.current_epoch_steps = 0

        if self.global_rank != 0:
            return

        if self.epoch_bar is not None:
            self.epoch_bar.update(1)

        output_dir = Path(
            self.cfg["output_dir"]
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        epoch = self.current_epoch + 1

        train_loss = (
            self.trainer.callback_metrics.get(
                "train_loss_epoch"
            )
        )

        log_file = (
            output_dir / "training_log.csv"
        )

        write_header = (
            not log_file.exists()
        )

        with log_file.open(
            "a",
            newline="",
        ) as f:

            writer = csv.writer(f)

            if write_header:
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "lr",
                    "weight_decay",
                ])

            writer.writerow([
                epoch,
                (
                    float(
                        train_loss.detach().cpu()
                    )
                    if train_loss is not None
                    else ""
                ),
                float(self.last_lr),
                float(self.last_weight_decay),
            ])


    # --------------------------------------------------------
    # Training start
    # --------------------------------------------------------

    def on_train_start(self):

        if self.global_rank == 0:

            self.epoch_bar = tqdm(
                total=self.cfg["epochs"],
                initial=self.current_epoch,
                desc="Epoch",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )


    # --------------------------------------------------------
    # Epoch start
    # --------------------------------------------------------

    def on_train_epoch_start(self):

        if self.epoch_bar is not None:

            self.epoch_bar.set_description(
                f"Epoch {self.current_epoch + 1}/"
                f"{self.cfg['epochs']}"
            )


    # --------------------------------------------------------
    # Training end
    # --------------------------------------------------------

    def on_train_end(self):

        if self.epoch_bar is not None:
            self.epoch_bar.close()


# ============================================================
# 8. Public training function
# ============================================================

def train(model, dataloader, config=None):

    cfg = DEFAULTS.copy()

    if config is not None:
        cfg.update(config)

    pl.seed_everything(
        0,
        workers=True,
    )

    module = TinyTrainer(
        model,
        cfg,
    )

    # --------------------------------------------------------
    # Automatic checkpoint resume
    # --------------------------------------------------------

    checkpoint_path = find_latest_checkpoint(
        cfg["output_dir"]
    )

    if checkpoint_path is not None:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if (
            "state_dict" not in checkpoint
            or "loops" not in checkpoint
        ):
            raise RuntimeError(
                f"[Resume] {checkpoint_path} is not "
                "a full Lightning checkpoint."
            )

        del checkpoint

    if checkpoint_path is None:

        print(
            "[Resume] No checkpoint found. "
            "Starting from epoch 0.",
            flush=True,
        )

    else:

        print(
            f"[Resume] Found checkpoint: "
            f"{checkpoint_path}",
            flush=True,
        )

    # --------------------------------------------------------
    # Lightning Trainer
    # --------------------------------------------------------

    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],

        accelerator=(
            "gpu"
            if torch.cuda.is_available()
            else "cpu"
        ),

        devices=cfg.get(
            "devices",
            1,
        ),

        strategy=(
            DDPStrategy(
                timeout=timedelta(hours=6)
            )
            if cfg.get("devices", 1) > 1
            else "auto"
        ),

        precision=(
            "16-mixed"
            if torch.cuda.is_available()
            else "32-true"
        ),

        logger=False,

        enable_checkpointing=True,

        callbacks=[
            EpochCheckpoint(
                cfg["output_dir"]
            )
        ],

        enable_progress_bar=False,

        log_every_n_steps=10,
    )

    trainer.fit(
        module,
        train_dataloaders=dataloader,
        ckpt_path=(
            str(checkpoint_path)
            if checkpoint_path
            else None
        ),
    )


# ============================================================
# 9. Main
# ============================================================

def main():

    dataloader = initialize_dataloader(
        batch_size=256,
        cache_dir="./cortex_cache_tiny_test",
    )

    model = TinyCNN()

    train(
        model,
        dataloader,
        config={
            "epochs": 2,
            "output_dir": "./checkpoints/tiny_cortex_test",
        },
    )


if __name__ == "__main__":
    main()