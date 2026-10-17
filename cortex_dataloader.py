import os, io, random, zipfile, subprocess, requests, shutil, socket, time, threading, queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import IterableDataset
import torch
import torch.distributed as dist

BASE_URL = "https://cortex.thetavision.nl"
DATASET_ID = 2
ACCESS_URL_FILE = Path(__file__).resolve().parent / "cortex_access_url.txt"


def get_access_url():
    url = ACCESS_URL_FILE.read_text().strip()
    if not url:
        raise RuntimeError("Cortex access URL file is empty")
    return url


def is_valid_archive(path):
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return zipfile.is_zipfile(path)
    if suffix == ".7z":
        return subprocess.run(["7z", "t", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    return False


class CortexSession:
    def __init__(self, access_url, stop_event=None):
        self.access_url = access_url
        self.stop_event = stop_event
        self.session = requests.Session()
        self.csrf = None
        self.login()

    def _wait(self, seconds):
        if self.stop_event is None:
            time.sleep(seconds)
            return False
        return self.stop_event.wait(seconds)

    def login(self):
        attempt = 0
        while self.stop_event is None or not self.stop_event.is_set():
            attempt += 1
            try:
                self.access_url = get_access_url()
                r = self.session.get(self.access_url, allow_redirects=True, timeout=60)
                r.raise_for_status()
                login_page_url = r.url
                token = login_page_url.rstrip("/").split("/")[-1]
                headers = {"Origin": BASE_URL, "Referer": login_page_url}
                b = self.session.get(f"{BASE_URL}/api/bootstrap/", headers=headers, timeout=60)
                b.raise_for_status()
                self.csrf = b.json()["csrf_token"]
                r = self.session.post(f"{BASE_URL}/api/request/login/", json={"token": token}, headers={"X-Csrftoken": self.csrf, **headers}, timeout=60)
                r.raise_for_status()
                b = self.session.get(f"{BASE_URL}/api/bootstrap/", headers={"Origin": BASE_URL, "Referer": f"{BASE_URL}/dataset-provider/request/download/"}, timeout=60)
                b.raise_for_status()
                self.csrf = b.json()["csrf_token"]
                return
            except Exception as e:
                wait = min(10 * attempt, 120)
                print(f"[Cortex] Login failed (attempt {attempt}): {e}. Retrying in {wait}s...", flush=True)
                if self._wait(wait):
                    break
        raise RuntimeError("Cortex login stopped")

    def get_download_url(self, file_id):
        endpoint = f"{BASE_URL}/api/provided_file/{file_id}/download_url/"
        for attempt in range(2):
            r = self.session.post(endpoint, headers={"X-Csrftoken": self.csrf, "Origin": BASE_URL, "Referer": f"{BASE_URL}/dataset-provider/request/download/"}, timeout=60)
            if r.status_code == 200:
                return r.json()["url"]
            print(f"[Cortex] download_url HTTP {r.status_code}: {r.text[:300]}", flush=True)
            if r.status_code == 403 and attempt == 0:
                self.login()
                continue
            r.raise_for_status()
        raise RuntimeError(f"Unable to obtain download URL for file {file_id}")

    def get_files(self):
        r = self.session.get(f"{BASE_URL}/api/provided_file/", params={"limit": 10000, "provided_dataset": DATASET_ID}, timeout=60)
        r.raise_for_status()
        files = r.json()["data"]
        files.sort(key=lambda x: x["file_name"])
        return files


class GastroNetCortexDataset(IterableDataset):
    def __init__(self, access_url, cache_dir="./cortex_cache", shuffle_shards=False, shuffle_images=True, seed=42, decode_workers=4, prefetch_size=1024, prefetch_shards=5, download_workers=5, max_download_attempts=10):
        super().__init__()
        self.access_url = access_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shuffle_shards = shuffle_shards
        self.shuffle_images = shuffle_images
        self.seed = seed
        self.decode_workers = decode_workers
        self.prefetch_size = prefetch_size
        self.prefetch_shards = prefetch_shards
        self.download_workers = download_workers
        self.max_download_attempts = max_download_attempts

    @staticmethod
    def decode_image(item):
        name, data = item
        try:
            with Image.open(io.BytesIO(data)) as source:
                image = source.convert("RGB")
                image.load()
            return name, image, None
        except Exception as e:
            return name, None, e

    def download_shard(self, info, stop_event):
        path = self.cache_dir / info["file_name"]
        expected_size = int(info["size"])
        cortex = None

        try:
            for attempt in range(1, self.max_download_attempts + 1):
                if stop_event.is_set():
                    return None

                path.unlink(missing_ok=True)

                try:
                    if cortex is None:
                        cortex = CortexSession(self.access_url, stop_event)
                    url = cortex.get_download_url(info["id"])
                    print(f"[Cortex] Downloading {info['file_name']} (attempt {attempt}/{self.max_download_attempts})", flush=True)

                    log_path = self.cache_dir / f".curl_{os.getpid()}_{threading.get_ident()}.log"
                    process = None

                    try:
                        with open(log_path, "wb") as log_file:
                            process = subprocess.Popen(
                                ["curl", "-L", "--fail", "--show-error", "--connect-timeout", "30", "--max-time", "3600", "-o", str(path), url],
                                stdout=subprocess.DEVNULL, stderr=log_file
                            )

                            while process.poll() is None:
                                if stop_event.wait(0.5):
                                    process.terminate()
                                    try:
                                        process.wait(timeout=5)
                                    except subprocess.TimeoutExpired:
                                        process.kill()
                                        process.wait()
                                    break

                            returncode = process.wait()

                        stderr = log_path.read_text(errors="replace").strip()

                    finally:
                        if process is not None and process.poll() is None:
                            process.kill()
                            process.wait()
                        log_path.unlink(missing_ok=True)

                    if stop_event.is_set():
                        return None

                    actual_size = path.stat().st_size if path.exists() else 0
                    if returncode == 0 and actual_size == expected_size and is_valid_archive(path):
                        print(f"[Cortex] Download complete: {info['file_name']} ({actual_size} bytes)", flush=True)
                        return path

                    print(
                        f"[Cortex] Download failed: file={info['file_name']}, attempt={attempt}/{self.max_download_attempts}, "
                        f"host={socket.gethostname()}, curl_returncode={returncode}, "
                        f"downloaded_bytes={actual_size}, expected_bytes={expected_size}, stderr={stderr}",
                        flush=True
                    )

                    if returncode == 23:
                        try:
                            st = os.statvfs(self.cache_dir)
                            print(f"[Cortex][Storage] filesystem_available_bytes={st.f_bavail * st.f_frsize}", flush=True)
                        except OSError as e:
                            print(f"[Cortex][Storage] statvfs failed: errno={e.errno}, {e}", flush=True)

                        probe = self.cache_dir / f".write_probe_{os.getpid()}_{threading.get_ident()}"
                        try:
                            with open(probe, "wb") as f:
                                f.write(b"\0" * 1024 * 1024)
                                f.flush()
                                os.fsync(f.fileno())
                            print("[Cortex][Storage] 1 MiB write probe: OK", flush=True)
                        except OSError as e:
                            print(f"[Cortex][Storage] write probe failed: errno={e.errno}, {e}", flush=True)
                        finally:
                            probe.unlink(missing_ok=True)

                except Exception as e:
                    if stop_event.is_set():
                        return None
                    print(f"[Cortex] Download error: file={info['file_name']}, attempt={attempt}/{self.max_download_attempts}, error={type(e).__name__}: {e}", flush=True)

                path.unlink(missing_ok=True)
                if attempt < self.max_download_attempts and stop_event.wait(min(10 * attempt, 120)):
                    return None

            print(f"[Cortex][SKIPPED] {info['file_name']} failed after {self.max_download_attempts} attempts.", flush=True)
            return None

        finally:
            if cortex is not None:
                cortex.session.close()
            if stop_event.is_set():
                path.unlink(missing_ok=True)

    def iter_zip_images(self, path, rng, decode_pool):
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg")) and not n.endswith("/")]
            if self.shuffle_images:
                rng.shuffle(names)

            with tqdm(total=len(names), desc=path.name, leave=True) as image_bar:
                for start in range(0, len(names), self.prefetch_size):
                    items = []
                    for name in names[start:start + self.prefetch_size]:
                        try:
                            items.append((name, zf.read(name)))
                        except Exception as e:
                            image_bar.update(1)
                            print(f"[Cortex] Failed to read {name} from {path.name}: {e}", flush=True)

                    for name, image, error in decode_pool.map(self.decode_image, items):
                        image_bar.update(1)
                        if error is not None:
                            print(f"[Cortex] Failed to decode {name} from {path.name}: {error}", flush=True)
                            continue
                        yield image

    def iter_7z_images(self, path, rng, decode_pool):
        extract_dir = self.cache_dir / f"{path.stem}_extracted"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(["7z", "x", "-y", f"-o{extract_dir}", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to extract {path.name}: {result.stderr.strip()}")

            image_paths = [p for p in extract_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")]
            if self.shuffle_images:
                rng.shuffle(image_paths)

            with tqdm(total=len(image_paths), desc=path.name, leave=True) as image_bar:
                for start in range(0, len(image_paths), self.prefetch_size):
                    items = []
                    for image_path in image_paths[start:start + self.prefetch_size]:
                        try:
                            items.append((str(image_path), image_path.read_bytes()))
                        except Exception as e:
                            image_bar.update(1)
                            print(f"[Cortex] Failed to read {image_path}: {e}", flush=True)

                    for name, image, error in decode_pool.map(self.decode_image, items):
                        image_bar.update(1)
                        if error is not None:
                            print(f"[Cortex] Failed to decode {name}: {error}", flush=True)
                            continue
                        yield image
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

    def iter_archive_images(self, path, rng, decode_pool):
        if path.suffix.lower() == ".zip":
            yield from self.iter_zip_images(path, rng, decode_pool)
        elif path.suffix.lower() == ".7z":
            yield from self.iter_7z_images(path, rng, decode_pool)
        else:
            raise RuntimeError(f"Unsupported archive format: {path.name}")

    def __iter__(self):
        stop_event = threading.Event()
        yield from self.iter_images(stop_event)

    def iter_images(self, stop_event):
        print("[Cortex] Cleaning cache before epoch...", flush=True)
        for path in self.cache_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        print("[Cortex] Cache cleaned.", flush=True)

        cortex = CortexSession(self.access_url, stop_event)
        try:
            shards = cortex.get_files()
        finally:
            cortex.session.close()

        rng = random.Random(self.seed)
        if self.shuffle_shards:
            rng.shuffle(shards)
        if not shards:
            raise RuntimeError("No Cortex shards found")

        print(f"[Cortex] Found {len(shards)} archives. Download lookahead={self.prefetch_shards}.", flush=True)

        futures = {}
        next_submit = 0
        skipped = 0
        completed = 0

        with ThreadPoolExecutor(max_workers=self.download_workers) as download_pool, ThreadPoolExecutor(max_workers=self.decode_workers) as decode_pool:
            def fill_window(current_index):
                nonlocal next_submit
                limit = min(len(shards), current_index + self.prefetch_shards + 1)
                while next_submit < limit and not stop_event.is_set():
                    futures[next_submit] = download_pool.submit(self.download_shard, shards[next_submit], stop_event)
                    next_submit += 1

            try:
                with tqdm(total=len(shards), desc="Shards", leave=True) as shard_bar:
                    for index, info in enumerate(shards):
                        if stop_event.is_set():
                            break

                        fill_window(index)
                        path = futures.pop(index).result()

                        if stop_event.is_set():
                            break

                        if path is None:
                            skipped += 1
                            print(f"[Cortex][SKIPPED] Moving past {info['file_name']}.", flush=True)
                            shard_bar.update(1)
                            continue

                        try:
                            yield from self.iter_archive_images(path, rng, decode_pool)
                        finally:
                            path.unlink(missing_ok=True)
                            print(f"[Cortex] Deleted used archive: {path.name}", flush=True)

                        completed += 1
                        shard_bar.update(1)

                if not stop_event.is_set():
                    print(f"[Cortex] Archive stream complete: processed={completed}, skipped={skipped}, total={len(shards)}.", flush=True)

            finally:
                stop_event.set()
                for future in futures.values():
                    future.cancel()


class PrefetchDataLoader:
    def __init__(self, dataloader, prefetch_batches=2, batch_size=None):
        self.dataloader = dataloader
        self.prefetch_batches = prefetch_batches
        self._batch_size = batch_size if batch_size is not None else dataloader.batch_size

    @property
    def batch_size(self):
        return self._batch_size

    def __iter__(self):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        global_batch_size = self.batch_size * world_size
        stop_event = threading.Event()
        batch_queue = queue.Queue(maxsize=max(1, self.prefetch_batches))
        sentinel = object()
        producer_error = []
        producer_thread = None

        def producer():
            iterator = None
            remaining = []
            steps = 0

            try:
                iterator = self.dataloader.iter_images(stop_event)

                for image in iterator:
                    if stop_event.is_set():
                        break

                    remaining.append(image)
                    if len(remaining) < global_batch_size:
                        continue

                    while not stop_event.is_set():
                        try:
                            batch_queue.put(remaining, timeout=0.5)
                            steps += 1
                            break
                        except queue.Full:
                            continue

                    remaining = []

                if not stop_event.is_set():
                    print(f"[Cortex] Epoch data complete: steps={steps}, global_batch_size={global_batch_size}, dropped_images={len(remaining)}.", flush=True)

            except Exception as e:
                producer_error.append(e)
                print(f"[Cortex][ERROR] Producer failed: {type(e).__name__}: {e}", flush=True)

            finally:
                if iterator is not None:
                    try:
                        iterator.close()
                    except Exception as e:
                        producer_error.append(e)
                        print(f"[Cortex][ERROR] Producer cleanup failed: {e}", flush=True)

                while not stop_event.is_set():
                    try:
                        batch_queue.put(sentinel, timeout=0.5)
                        break
                    except queue.Full:
                        continue

        if rank == 0:
            producer_thread = threading.Thread(target=producer, name="cortex-producer", daemon=False)
            producer_thread.start()

        try:
            while True:
                global_batch = None
                status = 1

                if rank == 0:
                    item = batch_queue.get()
                    if item is sentinel:
                        status = -1 if producer_error else 0
                    else:
                        global_batch = item

                if world_size > 1:
                    backend = dist.get_backend()
                    device = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")
                    status_tensor = torch.tensor([status], dtype=torch.int32, device=device)
                    dist.broadcast(status_tensor, src=0)
                    status = int(status_tensor.item())

                if status == 0:
                    break

                if status == -1:
                    if rank == 0:
                        raise RuntimeError("Cortex producer failed") from producer_error[0]
                    raise RuntimeError("Cortex producer failed on Rank 0; check Rank 0 logs")

                if world_size == 1:
                    batch = global_batch
                else:
                    scatter_list = [global_batch[i * self.batch_size:(i + 1) * self.batch_size] for i in range(world_size)] if rank == 0 else None
                    received = [None]
                    dist.scatter_object_list(received, scatter_list, src=0)
                    batch = received[0]

                yield batch

        finally:
            if rank == 0:
                stop_event.set()
                if producer_thread is not None:
                    producer_thread.join()


def initialize_dataloader(batch_size=512, cache_dir="./cortex_cache"):
    dataset = GastroNetCortexDataset(
        access_url=get_access_url(),
        cache_dir=cache_dir,
        shuffle_shards=False,
        shuffle_images=True,
        decode_workers=4,
        prefetch_size=1024,
        prefetch_shards=5,
        download_workers=5,
        max_download_attempts=10
    )
    return PrefetchDataLoader(dataset, prefetch_batches=2, batch_size=batch_size)