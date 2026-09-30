import os, io, random, zipfile, subprocess, requests
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import torch.distributed as dist
import time
import torch
import shutil


BASE_URL = "https://cortex.thetavision.nl"
DATASET_ID = 2

def is_valid_archive(path):
    suffix = path.suffix.lower()

    if suffix == ".zip":
        return zipfile.is_zipfile(path)

    if suffix == ".7z":
        result = subprocess.run(
            ["7z", "t", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    return False




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
    def __init__(self, dataloader, prefetch_batches=2):
        self.dataloader = dataloader
        self.prefetch_batches = prefetch_batches

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

            if isinstance(item, Exception):
                raise item

            has_batch = 0 if item is sentinel else 1

            if dist.is_available() and dist.is_initialized():
                flag = torch.tensor(
                    has_batch,
                    device=torch.cuda.current_device(),
                    dtype=torch.int32,
                )
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                all_have_batch = flag.item() == 1
            else:
                all_have_batch = has_batch == 1

            if not all_have_batch:
                break

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
        attempt = 0

        while True:
            attempt += 1

            if path.exists():
                print(
                    f"[Cortex] Removing previous/incomplete "
                    f"{info['file_name']} ({path.stat().st_size} bytes)",
                    flush=True,
                )
                path.unlink()

            try:
                url = cortex.get_download_url(info["id"])

                print(
                    f"[Cortex] Downloading {info['file_name']} "
                    f"(attempt {attempt})",
                    flush=True,
                )

                result = subprocess.run([
                    "curl",
                    "-L",
                    "--fail",
                    "--show-error",
                    "--connect-timeout", "30",
                    "--retry", "3",
                    "--retry-delay", "5",
                    "-o", str(path),
                    url,
                ])

                actual_size = path.stat().st_size if path.exists() else 0

                if (
                    result.returncode == 0
                    and path.exists()
                    and actual_size == expected_size
                    and is_valid_archive(path)
                ):
                    print(
                        f"[Cortex] Download complete: {info['file_name']} "
                        f"({actual_size} bytes)",
                        flush=True,
                    )
                    return path

                print(
                    f"[Cortex] Download failed/incomplete: "
                    f"{info['file_name']}, "
                    f"curl={result.returncode}, "
                    f"size={actual_size}, "
                    f"expected={expected_size}",
                    flush=True,
                )

            except Exception as e:
                print(
                    f"[Cortex] Failed to get/download {info['file_name']}: {e}",
                    flush=True,
                )

            if path.exists():
                path.unlink()

            wait_seconds = min(10 * attempt, 120)
            time.sleep(wait_seconds)

    @staticmethod
    def decode_image(item):
        name, data = item
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
            return name, image, None
        except Exception as e:
            return name, None, e


    def iter_7z_images(self, path, rng, decode_pool):
        extract_dir = self.cache_dir / f"{path.stem}_extracted"

        if extract_dir.exists():
            shutil.rmtree(extract_dir)

        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run([
                "7z",
                "x",
                "-y",
                f"-o{extract_dir}",
                str(path),
            ])

            if result.returncode != 0:
                raise RuntimeError(f"Failed to extract {path.name}")

            image_paths = [
                p for p in extract_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower() in (".png", ".jpg", ".jpeg")
            ]

            if self.shuffle_images:
                rng.shuffle(image_paths)

            if self.image_bar is None:
                self.image_bar = tqdm(
                    total=len(image_paths),
                    desc=path.name,
                    leave=True,
                )
            else:
                self.image_bar.reset(total=len(image_paths))
                self.image_bar.set_description(path.name)

            for start in range(0, len(image_paths), self.prefetch_size):
                chunk_paths = image_paths[start:start + self.prefetch_size]

                items = []

                for image_path in chunk_paths:
                    try:
                        items.append(
                            (str(image_path), image_path.read_bytes())
                        )
                    except Exception as e:
                        self.image_bar.update(1)
                        print(
                            f"[Cortex] Failed to read {image_path} "
                            f"from {path.name}: {e}",
                            flush=True,
                        )

                for name, image, error in decode_pool.map(
                    self.decode_image,
                    items,
                ):
                    self.image_bar.update(1)

                    if error is not None:
                        print(
                            f"[Cortex] Failed to decode {name} "
                            f"from {path.name}: {error}",
                            flush=True,
                        )
                        continue

                    yield image

        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

    def __iter__(self):
        cortex = CortexSession(self.access_url)
        shards = cortex.get_files()
        rng = random.Random(self.seed)

        if self.shuffle_shards:
            rng.shuffle(shards)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        # Drop the final incomplete group so every rank gets the same number of shards.
        usable_shards = (len(shards) // world_size) * world_size
        shards = shards[:usable_shards]

        shards = shards[rank::world_size]

        # TEMP: only for testing end-of-epoch behavior
        # shards = shards[:5]

        if not shards:
            raise RuntimeError(f"No Cortex shards found for rank {rank}")

        if self.shard_bar is None:
            self.shard_bar = tqdm(total=len(shards), desc="Shards", leave=True)
        else:
            self.shard_bar.reset(total=len(shards))
            self.shard_bar.set_description("Shards")

        with ThreadPoolExecutor(max_workers=self.decode_workers) as decode_pool:
            for info in shards:
                path = self.download_shard(cortex, info)

                try:
                    suffix = path.suffix.lower()

                    if suffix == ".zip":
                        with zipfile.ZipFile(path, "r") as zf:
                            names = [
                                n for n in zf.namelist()
                                if n.lower().endswith((".png", ".jpg", ".jpeg"))
                                and not n.endswith("/")
                            ]

                            if self.shuffle_images:
                                rng.shuffle(names)

                            if self.image_bar is None:
                                self.image_bar = tqdm(
                                    total=len(names),
                                    desc=info["file_name"],
                                    leave=True,
                                )
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
                                        print(
                                            f"[Cortex] Failed to read {name} "
                                            f"from {info['file_name']}: {e}",
                                            flush=True,
                                        )

                                for name, image, error in decode_pool.map(
                                    self.decode_image,
                                    items,
                                ):
                                    self.image_bar.update(1)

                                    if error is not None:
                                        print(
                                            f"[Cortex] Failed to decode {name} "
                                            f"from {info['file_name']}: {error}",
                                            flush=True,
                                        )
                                        continue

                                    yield image

                    elif suffix == ".7z":
                        yield from self.iter_7z_images(
                            path,
                            rng,
                            decode_pool,
                        )

                    else:
                        raise RuntimeError(
                            f"Unsupported archive format: {path.name}"
                        )

                finally:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass

                self.shard_bar.update(1)




def initialize_dataloader(batch_size=512, cache_dir="./cortex_cache"):
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

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
        collate_fn=lambda batch: batch,
    )

    return PrefetchDataLoader(
        dataloader,
        prefetch_batches=2,
    )