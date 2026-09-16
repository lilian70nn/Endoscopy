"""
Entry point for SSL pre-training experiments.

Runs DINOv1 and LeJEPA with:
    - SigLIP2 initialization
    - ImageNet initialization
"""


from utils import initialize_siglip2
from cortex_dataloader import initialize_dataloader

from dino import train_dino
from lejepa import train_lejepa


def main():
    dataloader = initialize_dataloader(batch_size=300)

    experiments = [
        # ("dino", "siglip2"),
        # ("dino", "imagenet"),
        ("lejepa", "siglip2"),
        # ("lejepa", "imagenet"),
    ]

    for ssl_method, initialization in experiments:
        model = initialize_siglip2(initialization=initialization)

        if ssl_method == "dino":
            train_dino(model.vision_model, dataloader, config={"output_dir": f"./checkpoints/dino_{initialization}"})
        elif ssl_method == "lejepa":
            train_lejepa(model.vision_model, dataloader, config={"output_dir": f"./checkpoints/lejepa_{initialization}"})

if __name__ == "__main__":
    main()
