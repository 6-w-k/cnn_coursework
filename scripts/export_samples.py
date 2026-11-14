"""Export two untouched CIFAR-10 training images per class for visual inspection."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive = ROOT / "data" / "cifar-10-python.tar.gz"
    with archive.open("rb") as stream:
        archive_md5 = hashlib.file_digest(stream, "md5").hexdigest()
    if archive_md5 != "c58f30108f718f92721af3b95e74349a":
        raise RuntimeError("Archive does not match official MD5")
    batches = ROOT / "data" / "cifar-10-batches-py"
    with (batches / "batches.meta").open("rb") as stream:
        names = pickle.load(stream, encoding="bytes")[b"label_names"]
    with (batches / "data_batch_1").open("rb") as stream:
        batch = pickle.load(stream, encoding="bytes")
    pixels = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(batch[b"labels"])
    output = ROOT / "data" / "sample_png"
    output.mkdir(exist_ok=True)
    entries = []
    for label, raw_name in enumerate(names):
        name = raw_name.decode("ascii")
        for index in np.flatnonzero(labels == label)[:2]:
            filename = f"train_{int(index):05d}_{name}.png"
            path = output / filename
            Image.fromarray(pixels[index]).save(path)
            with Image.open(path) as image:
                if not np.array_equal(np.asarray(image), pixels[index]):
                    raise RuntimeError(f"PNG pixel mismatch: {filename}")
            entries.append({"file": filename, "class": name, "label": label,
                            "source_batch": "data_batch_1", "index_in_batch": int(index)})
    manifest = {"archive_md5": archive_md5, "source_batch_shape": list(batch[b"data"].shape),
                "image_shape": [32, 32, 3], "exported_count": len(entries),
                "png_pixels_verified_equal_to_source": True, "samples": entries}
    (output / "samples.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Archive MD5 verified: {archive_md5}")
    print(f"Decoded data_batch_1: {pixels.shape}, labels: {len(labels)}")
    print(f"Exported {len(entries)} PNGs (two per class), original pixels verified: {output}")


if __name__ == "__main__":
    main()
