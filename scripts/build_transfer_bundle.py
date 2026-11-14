"""Package source and original dataset once, excluding credentials and caches."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = [p for p in ROOT.iterdir() if p.is_file()]
    for name in ["scripts", "references", "logs"]:
        files.extend(p for p in (ROOT / name).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    files.extend([ROOT / "data/cifar-10-python.tar.gz", ROOT / "data/download_manifest.json"])
    samples = ROOT / "data/sample_png"
    if samples.exists():
        files.extend(p for p in samples.iterdir() if p.is_file())
    output = ROOT / "downloads/server_bundle.tar"
    output.parent.mkdir(exist_ok=True)
    with tarfile.open(output, "w") as archive:
        for path in sorted(set(files)):
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    with output.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {"bundle": str(output), "size_bytes": output.stat().st_size,
                "sha256": sha256, "file_count": len(set(files)),
                "excluded": [".local (SSH helper and known-host records)", "downloads", "data/download_chunks", "extracted CIFAR batches", "Python caches", "checkpoints"]}
    (ROOT / "logs/server_transfer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
