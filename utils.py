"""
Shared utilities for:
    - SigLIP2 model initialization
    - Dataset and DataLoader initialization
    - Checkpoint loading
"""


import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoModel
from torchvision.models import vit_b_16, ViT_B_16_Weights




MODEL_NAME = "google/siglip2-base-patch16-224"

def initialize_siglip2(initialization="siglip2"):

    if initialization == "siglip2":
        # Architecture + SigLIP2 pretrained weights
        model = AutoModel.from_pretrained(MODEL_NAME)
        return model

    elif initialization == "imagenet":

        # Create SigLIP2 architecture with random weights
        config = AutoConfig.from_pretrained(MODEL_NAME)
        model = AutoModel.from_config(config)
        siglip_vit = model.vision_model
        # Load ImageNet-pretrained ViT-B/16
        imagenet_vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        src = imagenet_vit.state_dict()
        dst = siglip_vit.state_dict()
        # Patch emnedding
        dst["embeddings.patch_embedding.weight"] = src["conv_proj.weight"].clone()
        dst["embeddings.patch_embedding.bias"] = src["conv_proj.bias"].clone()

        # Position embedding
        dst["embeddings.position_embedding.weight"] = src["encoder.pos_embedding"][:, 1:, :].squeeze(0).clone()
        # Transformer blocks
        for i in range(12):

            tv = f"encoder.layers.encoder_layer_{i}"
            sl = f"encoder.layers.{i}"
            # LayerNorm 1
            dst[f"{sl}.layer_norm1.weight"] = src[f"{tv}.ln_1.weight"].clone()
            dst[f"{sl}.layer_norm1.bias"] = src[f"{tv}.ln_1.bias"].clone()

            # Attention Q/K/V
            q_w, k_w, v_w = src[f"{tv}.self_attention.in_proj_weight"].chunk(3, dim=0)
            q_b, k_b, v_b = src[f"{tv}.self_attention.in_proj_bias"].chunk(3, dim=0)
            dst[f"{sl}.self_attn.q_proj.weight"] = q_w.clone()
            dst[f"{sl}.self_attn.q_proj.bias"] = q_b.clone()
            dst[f"{sl}.self_attn.k_proj.weight"] = k_w.clone()
            dst[f"{sl}.self_attn.k_proj.bias"] = k_b.clone()
            dst[f"{sl}.self_attn.v_proj.weight"] = v_w.clone()
            dst[f"{sl}.self_attn.v_proj.bias"] = v_b.clone()

            # Attention output projection
            dst[f"{sl}.self_attn.out_proj.weight"] = src[f"{tv}.self_attention.out_proj.weight"].clone()
            dst[f"{sl}.self_attn.out_proj.bias"] = src[f"{tv}.self_attention.out_proj.bias"].clone()

            # LayerNorm 2
            dst[f"{sl}.layer_norm2.weight"] = src[f"{tv}.ln_2.weight"].clone()
            dst[f"{sl}.layer_norm2.bias"] = src[f"{tv}.ln_2.bias"].clone()

            # MLP

            dst[f"{sl}.mlp.fc1.weight"] = src[f"{tv}.mlp.0.weight"].clone()
            dst[f"{sl}.mlp.fc1.bias"] = src[f"{tv}.mlp.0.bias"].clone()
            dst[f"{sl}.mlp.fc2.weight"] = src[f"{tv}.mlp.3.weight"].clone()
            dst[f"{sl}.mlp.fc2.bias"] = src[f"{tv}.mlp.3.bias"].clone()

        # Final LayerNorm
        dst["post_layernorm.weight"] = src["encoder.ln.weight"].clone()
        dst["post_layernorm.bias"] = src["encoder.ln.bias"].clone()

        siglip_vit.load_state_dict(dst)

        return model

    else:
        raise ValueError(f"Unknown initialization: {initialization}")




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