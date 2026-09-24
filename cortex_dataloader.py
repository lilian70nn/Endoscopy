import os, io, random, zipfile, subprocess, requests, time
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import torch
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

        max_attempts = 10

        for attempt in range(max_attempts):
            try:
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

                print(
                    f"[Cortex] download_url HTTP {r.status_code} "
                    f"(attempt {attempt + 1}/{max_attempts}): "
                    f"{r.text[:300]}",
                    flush=True,
                )

                # Session / CSRF may have expired.
                if r.status_code == 403:
                    print(
                        "[Cortex] Session may have expired. Re-login...",
                        flush=True,
                    )
                    self.login()

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ) as e:
                print(
                    f"[Cortex] download_url request failed "
                    f"(attempt {attempt + 1}/{max_attempts}): "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

            # Do not immediately hammer Cortex again.
            if attempt < max_attempts - 1:
                wait_seconds = min(5 * (attempt + 1), 30)

                print(
                    f"[Cortex] Waiting {wait_seconds}s before retry...",
                    flush=True,
                )

                time.sleep(wait_seconds)

        raise RuntimeError(
            f"Failed to obtain download URL for file_id={file_id} "
            f"after {max_attempts} attempts"
        )

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

        for attempt in range(10):
            if path.exists():
                print(
                    f"[Cortex] Removing previous/incomplete "
                    f"{info['file_name']} ({path.stat().st_size} bytes)",
                    flush=True,
                )
                path.unlink()

            try:
                url = cortex.get_download_url(info["id"])
            except Exception as e:
                print(
                    f"[Cortex] Failed to obtain URL for "
                    f"{info['file_name']}: {e}",
                    flush=True,
                )
                continue

            print(
                f"[Cortex] Downloading {info['file_name']} "
                f"(attempt {attempt + 1}/10)",
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

        if path.exists():
            path.unlink()

        print(
            f"[Cortex] Giving up on {info['file_name']} after 10 attempts",
            flush=True,
        )

        return None

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

        usable_shards = (len(shards) // world_size) * world_size

        if usable_shards < len(shards) and rank == 0:
            print(
                f"[Cortex] Dropping final {len(shards) - usable_shards} shards "
                f"to keep complete groups of {world_size}",
                flush=True,
            )

        shards = shards[:usable_shards]
        shards = shards[rank::world_size]

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

                local_success = 1 if path is not None else 0

                if dist.is_available() and dist.is_initialized():
                    status = torch.tensor(
                        local_success,
                        device=f"cuda:{torch.cuda.current_device()}",
                        dtype=torch.int32,
                    )

                    dist.all_reduce(status, op=dist.ReduceOp.MIN)
                    group_success = status.item() == 1
                else:
                    group_success = local_success == 1

                if not group_success:
                    print(
                        f"[Cortex] Rank {rank}: skipping current shard group "
                        f"because at least one rank failed",
                        flush=True,
                    )

                    if path is not None:
                        try:
                            path.unlink()
                        except FileNotFoundError:
                            pass

                    self.shard_bar.update(1)
                    continue

                try:
                    with zipfile.ZipFile(path, "r") as zf:
                        names = [n for n in zf.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg")) and not n.endswith("/")]

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


def initialize_dataloader(batch_size=512, cache_dir="./cortex_cache", devices=8):
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