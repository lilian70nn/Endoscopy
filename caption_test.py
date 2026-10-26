import argparse

import torch
from PIL import Image
from torchvision import transforms

from captioner_vit import load_captioning_model


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
    parser.add_argument("--image_path", type=str, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_captioning_model(args).to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(args.image_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.extract_features(images)
        captions = model.generate(images)

    print(f"Baseline: {args.baseline}")
    print(f"Image: {args.image_path}")
    print(f"Adapter output: {tuple(features.shape)}")
    print(f"Generated caption: {captions[0]}")


if __name__ == "__main__":
    main()