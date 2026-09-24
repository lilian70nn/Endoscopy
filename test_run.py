"""
2-GPU DDP smoke test.
"""

from utils import initialize_siglip2
from cortex_dataloader import initialize_dataloader
from dino import train_dino


def main():
    # Small batch because V100 only has ~16 GB VRAM per GPU.
    # Cortex/DDP/shard logic remains unchanged.
    dataloader = initialize_dataloader(
        batch_size=2,
        devices=2,
    )

    model = initialize_siglip2(initialization="siglip2")

    train_dino(
        model.vision_model,
        dataloader,
        config={
            "devices": 2,
            "epochs": 1,
            "output_dir": "./checkpoints/ddp_smoke_test",
        },
    )


if __name__ == "__main__":
    main()