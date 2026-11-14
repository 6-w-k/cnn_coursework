"""Download pinned upstream sources and the official CIFAR-10 Python archive."""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATA_MD5 = "c58f30108f718f92721af3b95e74349a"


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def download(url, path, expected_md5=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not expected_md5 or digest(path, "md5") == expected_md5):
        print(f"Already downloaded: {path.name}", flush=True)
        return
    partial = path.with_name(path.name + ".part")
    for attempt in range(1, 4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "CNN-Coursework-Preparation/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=45) as response:
                append = offset > 0 and response.status == 206
                if not append:
                    offset = 0
                total = offset + int(response.headers.get("Content-Length", 0))
                last_print = time.monotonic()
                print(f"Downloading {url} (attempt {attempt}, offset {offset})", flush=True)
                with partial.open("ab" if append else "wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        offset += len(block)
                        if time.monotonic() - last_print >= 10:
                            print(f"  {offset / 1048576:.1f} / {total / 1048576:.1f} MiB", flush=True)
                            last_print = time.monotonic()
                if total and offset != total:
                    raise IOError(f"Incomplete response: {offset}/{total} bytes")
            if expected_md5 and digest(partial, "md5") != expected_md5:
                partial.unlink()
                raise ValueError("Downloaded archive MD5 does not match official value")
            partial.replace(path)
            print(f"Saved {path.name}: {path.stat().st_size} bytes", flush=True)
            return
        except (OSError, ValueError, http.client.HTTPException) as exc:
            print(f"Attempt {attempt} failed: {exc}", flush=True)
            if attempt == 3:
                raise
            time.sleep(2)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def download_ranges(url, path, workers):
    """Speed up slow per-connection servers; still verify the full official MD5."""
    if path.exists() and digest(path, "md5") == DATA_MD5:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith("bytes 0-0/"):
            raise RuntimeError("Server does not support byte ranges; use --workers 1")
        total = int(content_range.split("/")[-1])
    chunk_size = 4 * 1024 * 1024
    chunks = path.parent / "download_chunks"
    chunks.mkdir(exist_ok=True)
    partial = path.with_name(path.name + ".part")
    # Reuse complete chunks of any prior sequential download.
    if partial.exists():
        with partial.open("rb") as stream:
            for index in range(partial.stat().st_size // chunk_size):
                target = chunks / f"{index:04d}.chunk"
                block = stream.read(chunk_size)
                if not target.exists():
                    target.write_bytes(block)

    def fetch(index):
        start = index * chunk_size
        end = min(start + chunk_size, total) - 1
        target = chunks / f"{index:04d}.chunk"
        if target.exists() and target.stat().st_size == end - start + 1:
            return index, "cached"
        for attempt in range(1, 4):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                        raise RuntimeError("Unexpected byte range response")
                    block = response.read()
                if len(block) != end - start + 1:
                    raise IOError("Incomplete chunk")
                target.write_bytes(block)
                return index, "downloaded"
            except (OSError, RuntimeError, http.client.HTTPException) as exc:
                print(f"Chunk {index} attempt {attempt} failed: {exc}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(attempt * 2)
        raise RuntimeError("Unreachable")

    count = (total + chunk_size - 1) // chunk_size
    print(f"Range download: {total} bytes, {count} chunks, {workers} workers; {url}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, index) for index in range(count)]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            index, status = future.result()
            print(f"  {completed}/{count} chunks complete (chunk {index}: {status})", flush=True)
    with partial.open("wb") as stream:
        for index in range(count):
            stream.write((chunks / f"{index:04d}.chunk").read_bytes())
    if digest(partial, "md5") != DATA_MD5:
        raise ValueError("Assembled archive MD5 mismatch")
    partial.replace(path)


def sources():
    manifest_path = ROOT / "references" / "sources.json"
    # Repeated runs use exactly the recorded revisions.
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    revisions = previous.get("revisions", {})
    for repo, branch in [("kuangliu/pytorch-cifar", "master"), ("pytorch/vision", "main")]:
        if repo not in revisions:
            request = urllib.request.Request(
                f"https://api.github.com/repos/{repo}/commits/{branch}",
                headers={"User-Agent": "CNN-Coursework-Preparation/1.0"},
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                revisions[repo] = json.load(response)["sha"]
    # Save the pinned revisions before downloading files, so retries stay pinned.
    write_json(manifest_path, {"revisions": revisions})
    sha = revisions["kuangliu/pytorch-cifar"]
    archive = ROOT / "downloads" / f"pytorch-cifar-{sha}.zip"
    download(f"https://codeload.github.com/kuangliu/pytorch-cifar/zip/{sha}", archive)
    target = ROOT / "references" / "pytorch-cifar"
    if not target.exists():
        with zipfile.ZipFile(archive) as package:
            destination = ROOT / "references"
            for member in package.infolist():
                resolved = (destination / member.filename).resolve()
                if not resolved.is_relative_to(destination.resolve()):
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            package.extractall(destination)
        unpacked = ROOT / "references" / f"pytorch-cifar-{sha}"
        unpacked.rename(target)
    else:
        print("Preserving existing reference source directory", flush=True)
    official_sha = revisions["pytorch/vision"]
    for upstream_name, local_name in [
        ("torchvision/models/resnet.py", "resnet.py"),
        ("LICENSE", "LICENSE"),
    ]:
        download(
            f"https://raw.githubusercontent.com/pytorch/vision/{official_sha}/{upstream_name}",
            ROOT / "references" / "torchvision" / local_name,
        )
    files = {}
    for path in sorted((ROOT / "references").rglob("*")):
        if path.is_file() and path != manifest_path and "__pycache__" not in path.parts:
            files[str(path.relative_to(ROOT)).replace("\\", "/")] = digest(path)
    write_json(manifest_path, {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "revisions": revisions,
        "repositories": ["https://github.com/kuangliu/pytorch-cifar", "https://github.com/pytorch/vision"],
        "official_snapshot_note": "resnet.py and LICENSE only, not the whole torchvision repository; relative imports require installed torchvision",
        "sha256": files,
    })
    print(f"Source provenance: {manifest_path}", flush=True)


def data(url, workers):
    archive = ROOT / "data" / "cifar-10-python.tar.gz"
    manifest_path = ROOT / "data" / "download_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    reused = archive.exists() and digest(archive, "md5") == DATA_MD5
    if workers > 1:
        download_ranges(url, archive, workers)
    else:
        download(url, archive, DATA_MD5)
    print("Official archive MD5 verified", flush=True)
    with tarfile.open(archive, "r:gz") as package:
        package.extractall(ROOT / "data", filter="data")
    preserve_source = reused and previous.get("md5") == DATA_MD5
    write_json(manifest_path, {
        "recorded_at_utc": previous.get("recorded_at_utc") if preserve_source else datetime.now(timezone.utc).isoformat(),
        "last_extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "url": previous["url"] if preserve_source else url,
        "existing_verified_archive_reused": reused,
        "dataset_page": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "size_bytes": archive.stat().st_size,
        "md5": digest(archive, "md5"),
        "sha256": digest(archive),
        "official_md5_matches": True,
    })
    print(f"CIFAR-10 extracted into {ROOT / 'data'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["all", "sources", "data"], default="all")
    parser.add_argument("--data-url", default=DATA_URL)
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 17))
    args = parser.parse_args()
    if args.part in ("all", "sources"):
        sources()
    if args.part in ("all", "data"):
        data(args.data_url, args.workers)


if __name__ == "__main__":
    main()
