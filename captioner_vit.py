import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import timm
from torch.serialization import safe_globals
from torch.torch_version import TorchVersion

from classification_vit.train_classification import ViTClassifier
from segmentation_vit.train_segmentation import ViTUNet


class VisualAdapter(nn.Module):
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.proj = nn.Linear(vision_dim, llm_dim)

    def forward(self, z):
        return self.proj(z)


class ClassificationAdapter(nn.Module):
    def __init__(self, num_classes, llm_dim):
        super().__init__()
        self.proj = nn.Linear(num_classes, llm_dim)

    def forward(self, probabilities):
        return self.proj(probabilities).unsqueeze(1)


class SegmentationAdapter(nn.Module):
    def __init__(self, num_classes, llm_dim, grid_size=8):
        super().__init__()
        self.grid_size = grid_size
        self.proj = nn.Linear(num_classes, llm_dim)

    def forward(self, probabilities):
        x = F.adaptive_avg_pool2d(probabilities, (self.grid_size, self.grid_size))
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)



class MultimodalAdapter(nn.Module):
    def __init__(self, vision_dim, llm_dim, cls_classes=None, seg_classes=None, seg_grid_size=8, use_evidence=False):
        super().__init__()
        if (cls_classes is None) != (seg_classes is None):
            raise ValueError("cls_classes and seg_classes must both be provided or both be None.")
        if use_evidence and cls_classes is None:
            raise ValueError("Evidence adapters require classification and segmentation adapters.")

        self.visual = VisualAdapter(vision_dim, llm_dim)
        self.cls = ClassificationAdapter(cls_classes, llm_dim) if cls_classes is not None else None
        self.seg = SegmentationAdapter(seg_classes, llm_dim, seg_grid_size) if seg_classes is not None else None
        self.cls_evidence = ClassificationAdapter(cls_classes, llm_dim) if use_evidence else None
        self.seg_evidence = SegmentationAdapter(seg_classes, llm_dim, seg_grid_size) if use_evidence else None

    def forward(self, z, cls_probs=None, seg_probs=None, cls_alpha=None, seg_alpha=None):
        tokens = [self.visual(z)]

        if (cls_probs is None) != (seg_probs is None):
            raise ValueError("Classification and segmentation predictions must be provided together.")
        if (cls_alpha is None) != (seg_alpha is None):
            raise ValueError("Classification and segmentation EDL parameters must be provided together.")

        if cls_probs is not None:
            if self.cls is None or self.seg is None:
                raise ValueError("Prediction adapters were not initialized.")
            tokens.extend([self.cls(cls_probs), self.seg(seg_probs)])

        if cls_alpha is not None:
            if cls_probs is None or self.cls_evidence is None or self.seg_evidence is None:
                raise ValueError("Evidence adapters are unavailable or prediction inputs are missing.")
            tokens.extend([self.cls_evidence(torch.log(cls_alpha)), self.seg_evidence(torch.log(seg_alpha))])

        return torch.cat(tokens, dim=1)


class CaptioningModel(nn.Module):
    def __init__(self, qwen_name, vision_encoder, cls_model=None, seg_model=None, cls_classes=None, seg_classes=None, seg_grid_size=8, use_evidence=False,):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.cls_model = cls_model
        self.seg_model = seg_model
        self.use_evidence = use_evidence

        self.tokenizer = AutoTokenizer.from_pretrained(qwen_name)
        self.decoder = AutoModelForCausalLM.from_pretrained(qwen_name)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.start_token_id = self.tokenizer.bos_token_id
        if self.start_token_id is None:
            self.start_token_id = self.tokenizer.eos_token_id
        if self.start_token_id is None:
            raise ValueError("Tokenizer has neither BOS nor EOS token.")

        vision_dim = self.vision_encoder.num_features
        llm_dim = self.decoder.config.hidden_size
        self.adapter = MultimodalAdapter(vision_dim, llm_dim, 
                                         cls_classes=cls_classes, 
                                         seg_classes = seg_classes,
                                         seg_grid_size=seg_grid_size,
                                         use_evidence=use_evidence)

        for model in (self.vision_encoder, self.cls_model, self.seg_model):
            if model is not None:
                model.requires_grad_(False)
                model.eval()

    def train(self, mode=True):
        super().train(mode)
        for model in (self.vision_encoder, self.cls_model, self.seg_model):
            if model is not None:
                model.eval()
        return self

    def extract_features(self, images):
        with torch.no_grad():
            z = self.vision_encoder.forward_features(images)
            cls_probs, seg_probs, cls_alpha, seg_alpha = None, None, None, None

            if self.cls_model is not None:

                cls_logits = self.cls_model(images)
                seg_logits = self.seg_model(images)

                if self.use_evidence:
                    cls_alpha = F.softplus(cls_logits) + 1.0
                    seg_alpha = F.softplus(seg_logits) + 1.0
                    cls_probs = cls_alpha / cls_alpha.sum(dim=-1, keepdim=True)
                    seg_probs = seg_alpha / seg_alpha.sum(dim=1, keepdim=True)
                else:
                    cls_alpha, seg_alpha = None, None
                    cls_probs = F.softmax(cls_logits, dim=-1)
                    seg_probs = F.softmax(seg_logits, dim=1)

        return self.adapter(z=z, cls_probs=cls_probs, seg_probs=seg_probs, cls_alpha=cls_alpha, seg_alpha=seg_alpha)

    def forward(self, images, captions):
        prefix = self.extract_features(images)
        batch_size = images.shape[0]

        caption_ids = [
            self.tokenizer.encode(
                caption, add_special_tokens=False, truncation=True, max_length=127,
            ) + [self.tokenizer.eos_token_id]
            for caption in captions
        ]

        max_len = max(len(ids) for ids in caption_ids)
        text_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long, device=images.device,)
        text_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=images.device)

        for i, ids in enumerate(caption_ids):
            length = len(ids)
            text_ids[i, :length] = torch.tensor(ids, dtype=torch.long, device=images.device,)
            text_mask[i, :length] = 1

        start_ids = torch.full((batch_size, 1), self.start_token_id, dtype=torch.long, device=images.device,)

        start_embeds = self.decoder.get_input_embeddings()(start_ids)
        text_embeds = self.decoder.get_input_embeddings()(text_ids)

        inputs_embeds = torch.cat([prefix, start_embeds, text_embeds], dim=1)
        inputs_embeds = inputs_embeds.to(self.decoder.dtype)

        prefix_mask = torch.ones(
            (batch_size, prefix.shape[1] + 1),
            dtype=text_mask.dtype,
            device=images.device,
        )
        attention_mask = torch.cat([prefix_mask, text_mask], dim=1)

        prefix_labels = torch.full(
            (batch_size, prefix.shape[1] + 1),
            -100,
            dtype=torch.long,
            device=images.device,
        )
        text_labels = text_ids.masked_fill(text_mask == 0, -100)
        labels = torch.cat([prefix_labels, text_labels], dim=1)

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, images, max_new_tokens=64):
        self.eval()

        prefix = self.extract_features(images)
        batch_size = images.shape[0]

        start_ids = torch.full(
            (batch_size, 1),
            self.start_token_id,
            dtype=torch.long,
            device=images.device,
        )

        start_embeds = self.decoder.get_input_embeddings()(start_ids)
        inputs_embeds = torch.cat([prefix, start_embeds], dim=1)
        inputs_embeds = inputs_embeds.to(self.decoder.dtype)

        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=images.device,
        )

        generated = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )


def load_checkpoint(path, key=None):
    with safe_globals([TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True,)
    if key is not None:
        checkpoint = checkpoint[key]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Invalid checkpoint: {path}")
    return checkpoint


def load_captioning_model(args):
    vision_encoder = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0,)
    vision_state = load_checkpoint(args.vit_ckpt, key="ema_backbone")

    vision_encoder.load_state_dict(vision_state, strict=True)

    cls_model, seg_model, cls_classes, seg_classes = None, None, None, None

    if args.baseline >= 2:
        if not args.cls_ckpt or not args.seg_ckpt:
            raise ValueError("Baseline 2/3 requires --cls_ckpt and --seg_ckpt.")

        cls_state = load_checkpoint(args.cls_ckpt)
        cls_classes = cls_state["head.weight"].shape[0]
        cls_backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0,)
        cls_model = ViTClassifier(cls_backbone, cls_classes)
        cls_model.load_state_dict(cls_state, strict=True)

        seg_state = load_checkpoint(args.seg_ckpt)
        seg_classes = seg_state["decoder.5.weight"].shape[0]
        seg_backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0,)
        seg_model = ViTUNet(seg_backbone, num_classes=seg_classes)
        seg_model.load_state_dict(seg_state, strict=True)


    model = CaptioningModel(
        qwen_name=args.qwen_name,
        vision_encoder=vision_encoder,
        cls_model=cls_model,
        seg_model=seg_model,
        cls_classes=cls_classes,
        seg_classes=seg_classes,
        seg_grid_size=args.seg_grid_size,
        use_evidence=args.baseline == 3,
    )

    if args.adapter_ckpt:
        model.adapter.load_state_dict(load_checkpoint(args.adapter_ckpt), strict=True)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--vit_ckpt", type=str, required=True)
    parser.add_argument("--cls_ckpt", type=str, default=None)
    parser.add_argument("--seg_ckpt", type=str, default=None)
    parser.add_argument("--adapter_ckpt", type=str, default=None)
    parser.add_argument("--qwen_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seg_grid_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_captioning_model(args).to(device).eval()

    images = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    with torch.no_grad():
        features = model.extract_features(images)
        print(f"Baseline: {args.baseline}")
        print(f"Adapter output: {tuple(features.shape)}")
        print(f"Generated text: {model.generate(images)}")


if __name__ == "__main__":
    main()