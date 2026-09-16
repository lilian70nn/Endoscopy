import torch
import torch.nn as nn
import lightning.pytorch as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dino import DINOTrainer, DINO_DEFAULTS
from lejepa import LeJEPATrainer, LEJEPA_DEFAULTS


class FakeDataset(Dataset):
    def __init__(self, n=32): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, idx): return Image.new("RGB", (224, 224), (idx % 255, 100, 100))


def collate_fn(batch): return batch


class FakeOutput:
    def __init__(self, x): self.pooler_output = x


class FakeBackbone(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(3, hidden_size)

    def forward(self, pixel_values=None, interpolate_pos_encoding=True):
        x = pixel_values.mean(dim=(2, 3))
        return FakeOutput(self.proj(x))


def test_dino():
    print("\n===== TESTING DINO LIGHTNING DDP =====", flush=True)

    loader = DataLoader(FakeDataset(32), batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    cfg = DINO_DEFAULTS.copy()
    cfg.update({
        "epochs": 1,
        "out_dim": 64,
        "hidden_dim": 64,
        "bottleneck_dim": 16,
        "local_crops_number": 2,
        "warmup_epochs": 0,
        "freeze_last_layer": 0,
        "devices": 4,
        "output_dir": "./test_output/dino",
    })

    module = DINOTrainer(FakeBackbone(32), 32, 2, cfg)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=4,
        strategy="ddp",
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_train_batches=2,
    )

    trainer.fit(module, train_dataloaders=loader)

    if trainer.global_rank == 0:
        print("[PASS] DINO Lightning 4-rank DDP", flush=True)


def test_lejepa():
    print("\n===== TESTING LEJEPA LIGHTNING DDP =====", flush=True)

    loader = DataLoader(FakeDataset(32), batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    cfg = LEJEPA_DEFAULTS.copy()
    cfg.update({
        "epochs": 1,
        "num_local_crops": 2,
        "num_slices": 16,
        "warmup_epochs": 0,
        "devices": 4,
        "output_dir": "./test_output/lejepa",
    })

    module = LeJEPATrainer(FakeBackbone(32), 32, 2, cfg)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=4,
        strategy="ddp",
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_train_batches=2,
    )

    trainer.fit(module, train_dataloaders=loader)

    if trainer.global_rank == 0:
        print("[PASS] LeJEPA Lightning 4-rank DDP", flush=True)


if __name__ == "__main__":
    pl.seed_everything(0)

    test_dino()
    test_lejepa()

    print("\nALL LIGHTNING DDP TESTS PASSED", flush=True)