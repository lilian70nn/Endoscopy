"""
Shared utilities for:
    - SigLIP2 model initialization
    - Dataset and DataLoader initialization
    - Checkpoint loading
"""


import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

def initialize_siglip2(initialization="siglip2"):
    if initialization == "siglip2":
        # SigLIP2 pretrained ViT weights
        ...
    elif initialization == "imagenet":
        # ImageNet pretrained ViT weights
        ...
    else:
        raise ValueError(f"Unknown initialization: {initialization}")

    return model



def initialize_dataloader(batch_size=64):

    # Load raw images only.
    # SSL-specific augmentation is handled inside the training method.

    dataset = load_dataset(
        "BONS-AI-TUE-AMC/GastroNet5M",
        split="train"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: [item["image"] for item in batch]
    )

    return dataloader

def load_trained_model(initialization, checkpoint_path):
    model = initialize_siglip2(initialization=initialization)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])

    return model