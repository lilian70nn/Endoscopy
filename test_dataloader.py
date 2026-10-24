import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from PIL import Image
from torchvision import transforms
from datetime import timedelta
from lightning.pytorch.strategies import DDPStrategy

from cortex_dataloader import initialize_dataloader


# ============================================================
# 1. Tiny Model
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
# 2. Training
# ============================================================

class TinyTrainer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def training_step(self, batch, batch_idx):

        # Cortex DataLoader returns a list of PIL images
        images = torch.stack([
            self.transform(image.convert("RGB"))
            for image in batch
        ]).to(self.device)

        # Simple target: average RGB color
        target = images.mean(dim=(2, 3))

        # Forward
        prediction = self.model(images)

        # Loss
        loss = F.mse_loss(prediction, target)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch),
        )

        if batch_idx % 10 == 0:
            print(
                f"[Rank {self.global_rank}] "
                f"Epoch {self.current_epoch + 1} "
                f"Step {batch_idx} "
                f"Loss {loss.item():.6f}",
                flush=True,
            )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
        )


def train(model, dataloader):

    devices = (
        torch.cuda.device_count()
        if torch.cuda.is_available()
        else 1
    )

    print(f"Using {devices} GPU(s)" if torch.cuda.is_available()
          else "Using CPU", flush=True)

    trainer = pl.Trainer(
        max_epochs=2,

        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,

        strategy=(
            DDPStrategy(timeout=timedelta(hours=6))
            if devices > 1
            else "auto"
        ),

        precision="16-mixed" if torch.cuda.is_available() else "32-true",

        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(
        TinyTrainer(model),
        train_dataloaders=dataloader,
    )


# ============================================================
# 3. Main
# ============================================================

def main():

    dataloader = initialize_dataloader(
        batch_size=256,
        cache_dir="./cortex_cache_tiny_test",
    )

    model = TinyCNN()

    train(model, dataloader)


if __name__ == "__main__":
    main()