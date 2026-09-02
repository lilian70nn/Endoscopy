"""
Entry point for SSL pre-training experiments.

Runs DINOv1 and LeJEPA with:
    - SigLIP2 initialization
    - ImageNet initialization
"""


from utils import initialize_siglip2, initialize_dataloader
from dino import train_dino
from lejepa import train_lejepa


def main():

    dataloader = initialize_dataloader(batch_size=64)

    experiments = [
        ("dino", "siglip2"),
        ("dino", "imagenet"),
        ("lejepa", "siglip2"),
        ("lejepa", "imagenet"),
    ]

    for ssl_method, initialization in experiments:

        model = initialize_siglip2(initialization=initialization)

        if ssl_method == "dino":
            trained_model = train_dino(model, dataloader)

        elif ssl_method == "lejepa":
            trained_model = train_lejepa(model, dataloader)


if __name__ == "__main__":
    main()