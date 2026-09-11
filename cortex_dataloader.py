import os
import io
import math
import random
import zipfile
import subprocess
from pathlib import Path

import requests
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader


CORTEX_API = "https://cortex.thetavision.nl/api"
CORTEX_DATASET_ID = 2

GASTRONET_NUM_IMAGES = 5_000_000


class CortexGastroNetDataset(IterableDataset):
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

    def __init__(self, cookie, cache_dir="./cortex_cache", shuffle_shards=True, shuffle_images=True, seed=0):
        super().__init__()
        self.cookie = cookie
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shuffle_shards = shuffle_shards
        self.shuffle_images = shuffle_images
        self.seed = seed
        self.epoch = 0

        self.csrf_token = None
        for part in cookie.split(";"):
            part = part.strip()
            if part.startswith("csrftoken="):
                self.csrf_token = part.split("=", 1)[1]
                break

        if self.csrf_token is None:
            raise RuntimeError("Cannot find csrftoken in CORTEX_COOKIE.")

        self.session = requests.Session()
        self.session.headers.update({"Cookie": self.cookie, "X-CSRFToken": self.csrf_token, "Referer": "https://cortex.thetavision.nl/", "User-Agent": "Mozilla/5.0"})

        response = self.session.get(f"{CORTEX_API}/provided_file/", params={"limit": 10000, "provided_dataset": CORTEX_DATASET_ID}, timeout=60)
        response.raise_for_status()
        payload = response.json()

        self.files = payload["data"]
        self.files.sort(key=lambda x: x["file_name"])

        if len(self.files) == 0:
            raise RuntimeError("Cortex returned zero files.")

        print(f"[Cortex] Found {len(self.files)} ZIP shards.")

    def __len__(self):
        return GASTRONET_NUM_IMAGES

    def _get_download_url(self, file_id):
        response = self.session.post(f"{CORTEX_API}/provided_file/{file_id}/download_url/", timeout=60)
        response.raise_for_status()
        return response.json()["url"]

    def _download_zip(self, file_info):
        file_name = file_info["file_name"]
        expected_size = int(file_info.get("size", 0))
        destination = self.cache_dir / file_name

        if destination.exists() and expected_size > 0 and destination.stat().st_size == expected_size:
            print(f"[Cortex] Using existing {file_name}")
            return destination

        url = self._get_download_url(file_info["id"])
        print(f"[Cortex] Downloading {file_name}")

        command = ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "3", "--connect-timeout", "30", "-C", "-", "-o", str(destination), url]
        result = subprocess.run(command)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to download {file_name}, curl exit code={result.returncode}")

        if expected_size > 0 and destination.stat().st_size != expected_size:
            raise RuntimeError(f"Downloaded file has wrong size: {file_name}, got {destination.stat().st_size}, expected {expected_size}")

        return destination

    def _images_from_zip(self, zip_path, rng):
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [x for x in zf.namelist() if not x.endswith("/") and x.lower().endswith(self.IMAGE_EXTENSIONS)]

            if self.shuffle_images:
                rng.shuffle(names)

            print(f"[Cortex] {zip_path.name}: {len(names)} images")

            for name in names:
                try:
                    with zf.open(name, "r") as f:
                        data = f.read()

                    with Image.open(io.BytesIO(data)) as image:
                        image = image.convert("RGB")
                        image.load()
                        image = image.copy()

                    yield image

                except Exception as e:
                    print(f"[Cortex] Skipping bad image {name}: {e}")

    def __iter__(self):
        current_epoch = self.epoch
        self.epoch += 1

        rng = random.Random(self.seed + current_epoch)
        files = self.files.copy()

        if self.shuffle_shards:
            rng.shuffle(files)

        print(f"[Cortex] Starting dataset pass {current_epoch + 1}")

        for shard_index, file_info in enumerate(files):
            zip_path = None

            try:
                print(f"[Cortex] Shard {shard_index + 1}/{len(files)}: {file_info['file_name']}")
                zip_path = self._download_zip(file_info)

                for image in self._images_from_zip(zip_path, rng):
                    yield image

            finally:
                if zip_path is not None and zip_path.exists():
                    try:
                        zip_path.unlink()
                        print(f"[Cortex] Deleted {zip_path.name}")
                    except Exception as e:
                        print(f"[Cortex] Could not delete {zip_path}: {e}")


class CortexDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, **kwargs):
        self._cortex_batch_size = batch_size
        super().__init__(dataset, batch_size=batch_size, **kwargs)

    def __len__(self):
        return math.ceil(GASTRONET_NUM_IMAGES / self._cortex_batch_size)


def cortex_collate(batch):
    return batch


def initialize_dataloader(batch_size=64, cache_dir="./cortex_cache"):
    cookie = os.environ.get("CORTEX_COOKIE")

    if not cookie:
        raise RuntimeError(
            "CORTEX_COOKIE is not set.\n"
            "In the terminal run:\n"
            "export CORTEX_COOKIE='your complete Cortex Cookie'\n"
            "and then start training again."
        )

    dataset = CortexGastroNetDataset(cookie=cookie, cache_dir=cache_dir, shuffle_shards=True, shuffle_images=True, seed=0)

    dataloader = CortexDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=cortex_collate
    )

    print(f"[Cortex] DataLoader ready. Approximate steps per epoch: {len(dataloader)}")
    return dataloader