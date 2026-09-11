import os, io, math, random, zipfile, subprocess, requests
from pathlib import Path
from tqdm.notebook import tqdm
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader

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
            headers={
                "Origin": BASE_URL,
                "Referer": login_page_url,
            },
            timeout=60,
        )
        b.raise_for_status()
        self.csrf = b.json()["csrf_token"]

        r = self.session.post(
            f"{BASE_URL}/api/request/login/",
            json={"token": token},
            headers={
                "X-Csrftoken": self.csrf,
                "Origin": BASE_URL,
                "Referer": login_page_url,
            },
            timeout=60,
        )
        r.raise_for_status()

        b = self.session.get(
            f"{BASE_URL}/api/bootstrap/",
            headers={
                "Origin": BASE_URL,
                "Referer": f"{BASE_URL}/dataset-provider/request/download/",
            },
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

    def headers(self):
        return {"X-Csrftoken": self.csrf, "Origin": BASE_URL, "Referer": f"{BASE_URL}/dataset-provider/request/download/"}

    def get_files(self):
        r = self.session.get(f"{BASE_URL}/api/provided_file/", params={"limit": 10000, "provided_dataset": DATASET_ID}, timeout=60)
        r.raise_for_status()
        files = r.json()["data"]
        files.sort(key=lambda x: x["file_name"])
        return files

    # def get_download_url(self, file_id):
    #     r = self.session.post(f"{BASE_URL}/api/provided_file/{file_id}/download_url/", headers=self.headers(), timeout=60)
    #     if r.status_code == 403:
    #         self.login()
    #         r = self.session.post(f"{BASE_URL}/api/provided_file/{file_id}/download_url/", headers=self.headers(), timeout=60)
    #     r.raise_for_status()
    #     return r.json()["url"]


class GastroNetCortexDataset(IterableDataset):
    def __init__(self, access_url, cache_dir="./cortex_cache", shuffle_shards=True, shuffle_images=True, seed=42):
        super().__init__()
        self.access_url = access_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shuffle_shards = shuffle_shards
        self.shuffle_images = shuffle_images
        self.seed = seed

        self.shard_bar = None
        self.image_bar = None

    def download_shard(self, cortex, info):
        path = self.cache_dir / info["file_name"]
        expected_size = int(info["size"])

        if path.exists() and path.stat().st_size == expected_size:
            return path

        url = cortex.get_download_url(info["id"])

        cmd = [
            "curl", "-L", "--fail", "--silent", "--show-error",
            "--retry", "5", "--retry-delay", "5",
            "--connect-timeout", "30",
            "-C", "-", "-o", str(path), url
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            if path.exists() and path.stat().st_size >= expected_size:
                return path
            raise RuntimeError(f"Download failed: {info['file_name']}")

        if path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Wrong file size: {info['file_name']} "
                f"got {path.stat().st_size}, expected {expected_size}"
            )

        return path

    def __iter__(self):
        cortex = CortexSession(self.access_url)
        shards = cortex.get_files()
        rng = random.Random(self.seed)

        if self.shuffle_shards:
            rng.shuffle(shards)

        if self.shard_bar is None:
            self.shard_bar = tqdm(
                total=len(shards),
                desc="Shards",
                leave=True
            )
        else:
            self.shard_bar.reset(total=len(shards))
            self.shard_bar.set_description("Shards")

        for info in shards:
            path = self.download_shard(cortex, info)

            try:
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
                            leave=True
                        )
                    else:
                        self.image_bar.reset(total=len(names))
                        self.image_bar.set_description(info["file_name"])

                    for name in names:
                        try:
                            data = zf.read(name)
                            image = Image.open(io.BytesIO(data)).convert("RGB")
                            yield image
                            self.image_bar.update(1)
                        except Exception:
                            self.image_bar.update(1)

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
        return NUM_IMAGES // self._batch_size_for_len


def initialize_dataloader(batch_size=64, cache_dir="./cortex_cache"):
    access_url = os.environ.get("CORTEX_ACCESS_URL")
    if not access_url:
        raise RuntimeError("CORTEX_ACCESS_URL is not set")

    dataset = GastroNetCortexDataset(access_url=access_url, cache_dir=cache_dir, shuffle_shards=False, shuffle_images=True)

    return CortexDataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, collate_fn=lambda batch: batch)