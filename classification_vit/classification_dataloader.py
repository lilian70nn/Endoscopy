from pathlib import Path
from collections import Counter
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FindingDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return image, label

def initialize_classification_dataloaders(
    data_dir="./data/labeled-images",
    batch_size=32,
    min_images=100,
    image_size=224,
    num_workers=2,
    seed=42,
):
    data_dir = Path(data_dir)

    image_extensions = {".jpg", ".jpeg", ".png"}

    image_paths = [
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    ]

    counts = Counter(path.parent.name for path in image_paths)
    valid_findings = sorted([finding for finding, count in counts.items() if count > min_images])
    label_to_idx = {finding: index for index, finding in enumerate(valid_findings)}
    samples = [(path, label_to_idx[path.parent.name]) for path in image_paths if path.parent.name in label_to_idx]

    if not samples:
        raise ValueError("No finding classes contain more than 100 images.")

    labels = [label for _, label in samples]

    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.15,
        random_state=seed,
        stratify=labels,
    )

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_loader = DataLoader(
        FindingDataset(train_samples, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        FindingDataset(val_samples, val_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Number of classes: {len(valid_findings)}")
    print(f"Training images: {len(train_samples)}")
    print(f"Validation images: {len(val_samples)}")
    print("Finding classes:", label_to_idx)

    return train_loader, val_loader, label_to_idx
