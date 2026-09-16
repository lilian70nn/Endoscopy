import os, io, random, zipfile, subprocess, requests
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import torch.distributed as dist

BASE_URL = "https://cortex.thetavision.nl"
DATASET_ID = 2
NUM_IMAGES = 4_820_653


class CortexSession:
    def __init__(self, access_url):
        self.access_url = access_url
        self.session = requests.Session()
        self.csrf = None
        self.login()

    def login(self):
        r = self.session.get(self.access_url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        login_page_url = r.url
        token = login_page_url.rstrip("/").split("/")[-1]

        b = self.session.get(
            f"{BASE_URL}/api/bootstrap/",
            headers={"Origin": BASE_URL, "Referer": login_page_url},
            timeout=60,
        )
        b.raise_for_status()
        self.csrf = b.json()["csrf_token"]

        r = self.session.post(
            f"{BASE_URL}/api/request/login/",
            json={"token": token},
            headers={"X-Csrftoken": self.csrf, "Origin": BASE_URL, "Referer": login_page_url},
            timeout=60,
        )
        r.raise_for_status()

        b = self.session.get(
            f"{BASE_URL}/api/bootstrap/",
            headers={"Origin": BASE_URL, "Referer": f"{BASE_URL}/dataset-provider/request/download/"},
            timeout=60,
        )
        b.raise_for_status()
        self.csrf = b.json()["csrf_token"]

    def get_download_url(self, file_id):
        endpoint = f"{BASE_URL}/api/provided_file/{file_id}/download_url/"

        for attempt in range(2):
            r = self.session.post(
                endpoint,
                headers={
                    "X-Csrftoken": self.csrf,
                    "Origin": BASE_URL,
                    "Referer": f"{BASE_URL}/dataset-provider/request/download/",
                },
                timeout=60,
            )

            if r.status_code == 200:
                return r.json()["url"]

            print(f"[Cortex] download_url HTTP {r.status_code}: {r.text[:300]}", flush=True)

            if r.status_code == 403 and attempt == 0:
                self.login()
                continue

            r.raise_for_status()

    def get_files(self):
        r = self.session.get(
            f"{BASE_URL}/api/provided_file/",
            params={"limit": 10000, "provided_dataset": DATASET_ID},
            timeout=60,
        )
        r.raise_for_status()
        files = r.json()["data"]
        files.sort(key=lambda x: x["file_name"])
        return files


class PrefetchDataLoader:
    def __init__(self, dataloader, prefetch_batches=2, devices=1):
        self.dataloader = dataloader
        self.prefetch_batches = prefetch_batches
        self.devices = devices

    def __len__(self):
        return NUM_IMAGES // (self.dataloader.batch_size * self.devices)

    @property
    def batch_size(self):
        return self.dataloader.batch_size

    def __iter__(self):
        q = queue.Queue(maxsize=self.prefetch_batches)
        sentinel = object()

        def producer():
            try:
                for batch in self.dataloader:
                    q.put(batch)
            except Exception as e:
                q.put(e)
            finally:
                q.put(sentinel)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item




class GastroNetCortexDataset(IterableDataset):
    def __init__(self, access_url, cache_dir="./cortex_cache", shuffle_shards=True, shuffle_images=True, seed=42, decode_workers=4, prefetch_size=1024):
        super().__init__()
        self.access_url = access_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shuffle_shards = shuffle_shards
        self.shuffle_images = shuffle_images
        self.seed = seed
        self.decode_workers = decode_workers
        self.prefetch_size = prefetch_size
        self.shard_bar = None
        self.image_bar = None

    def download_shard(self, cortex, info):
        path = self.cache_dir / info["file_name"]
        expected_size = int(info["size"])

        if path.exists() and path.stat().st_size == expected_size:
            return path

        for attempt in range(10):
            url = cortex.get_download_url(info["id"])

            subprocess.run([
                "curl",
                "-L",
                "--fail",
                "--show-error",
                "--connect-timeout", "30",
                "-C", "-",
                "-o", str(path),
                url,
            ])

            if path.exists() and path.stat().st_size == expected_size:
                return path

        raise RuntimeError(f"Download failed: {info['file_name']}")

    @staticmethod
    def decode_image(item):
        name, data = item
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
            return name, image, None
        except Exception as e:
            return name, None, e

    def __iter__(self):
        cortex = CortexSession(self.access_url)
        shards = cortex.get_files()
        rng = random.Random(self.seed)

        if self.shuffle_shards:
            rng.shuffle(shards)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        shards = shards[rank::world_size]

        if not shards:
            raise RuntimeError(f"No Cortex shards found for rank {rank}")

        if self.shard_bar is None:
            self.shard_bar = tqdm(total=len(shards), desc="Shards", leave=True)
        else:
            self.shard_bar.reset(total=len(shards))
            self.shard_bar.set_description("Shards")

        with ThreadPoolExecutor(max_workers=1) as download_pool, ThreadPoolExecutor(max_workers=self.decode_workers) as decode_pool:
            future = download_pool.submit(self.download_shard, cortex, shards[0])

            for i, info in enumerate(shards):
                path = future.result()

                if i + 1 < len(shards):
                    future = download_pool.submit(self.download_shard, cortex, shards[i + 1])

                try:
                    with zipfile.ZipFile(path, "r") as zf:
                        names = [
                            n for n in zf.namelist()
                            if n.lower().endswith((".png", ".jpg", ".jpeg")) and not n.endswith("/")
                        ]

                        if self.shuffle_images:
                            rng.shuffle(names)

                        if self.image_bar is None:
                            self.image_bar = tqdm(total=len(names), desc=info["file_name"], leave=True)
                        else:
                            self.image_bar.reset(total=len(names))
                            self.image_bar.set_description(info["file_name"])

                        for start in range(0, len(names), self.prefetch_size):
                            chunk_names = names[start:start + self.prefetch_size]
                            items = []

                            for name in chunk_names:
                                try:
                                    items.append((name, zf.read(name)))
                                except Exception as e:
                                    self.image_bar.update(1)
                                    print(f"[Cortex] Failed to read {name} from {info['file_name']}: {e}", flush=True)

                            for name, image, error in decode_pool.map(self.decode_image, items):
                                self.image_bar.update(1)

                                if error is not None:
                                    print(f"[Cortex] Failed to decode {name} from {info['file_name']}: {error}", flush=True)
                                    continue

                                yield image

                finally:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass

                self.shard_bar.update(1)


class CortexDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, **kwargs):
        self._batch_size_for_len = batch_size
        super().__init__(dataset, batch_size=batch_size, **kwargs)

    def __len__(self):
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        return NUM_IMAGES // (self._batch_size_for_len * world_size)


def initialize_dataloader(batch_size=512, cache_dir="./cortex_cache", devices=4):
    access_url = os.environ.get("CORTEX_ACCESS_URL")
    if not access_url:
        raise RuntimeError("CORTEX_ACCESS_URL is not set")

    dataset = GastroNetCortexDataset(
        access_url=access_url,
        cache_dir=cache_dir,
        shuffle_shards=False,
        shuffle_images=True,
        decode_workers=4,
        prefetch_size=1024,
    )

    dataloader = CortexDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
        collate_fn=lambda batch: batch,
    )

    return PrefetchDataLoader(dataloader, prefetch_batches=2, devices=devices)