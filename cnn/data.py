import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def prepare_split(root, seed=42):
    root = Path(root)
    data = datasets.CIFAR10(str(root / "data"), train=True, download=False)
    path = root / "splits" / f"seed{seed}.json"
    if not path.exists():
        rng = np.random.default_rng(seed)
        targets = np.asarray(data.targets)
        train, val = [], []
        for label in range(10):
            indices = rng.permutation(np.flatnonzero(targets == label))
            val.extend(indices[:500].tolist())
            train.extend(indices[500:].tolist())
        split = {"seed": seed, "classes": data.classes,
                 "train": rng.permutation(train).tolist(), "validation": rng.permutation(val).tolist(),
                 "train_per_class": 4500, "validation_per_class": 500}
        write_json(path, split)
    split = json.loads(path.read_text(encoding="utf-8"))
    train, val = split["train"], split["validation"]
    if len(train) != 45000 or len(val) != 5000 or set(train) & set(val) or set(train + val) != set(range(50000)):
        raise RuntimeError("Invalid/disjointness failure in fixed training/validation split")
    targets = np.asarray(data.targets)
    if not np.array_equal(np.bincount(targets[train], minlength=10), [4500] * 10):
        raise RuntimeError("Training split is not stratified")
    if not np.array_equal(np.bincount(targets[val], minlength=10), [500] * 10):
        raise RuntimeError("Validation split is not stratified")
    return split, hashlib.sha256(path.read_bytes()).hexdigest()


def loaders(root, seed, batch_size, workers):
    split, digest = prepare_split(root, seed)
    normalize = transforms.Normalize(MEAN, STD)
    training = datasets.CIFAR10(str(Path(root) / "data"), train=True, download=False,
        transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))
    validation = datasets.CIFAR10(str(Path(root) / "data"), train=True, download=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    generator = torch.Generator().manual_seed(seed)
    common = {"num_workers": workers, "pin_memory": True, "worker_init_fn": seed_worker}
    train = torch.utils.data.DataLoader(torch.utils.data.Subset(training, split["train"]),
        batch_size=batch_size, shuffle=True, generator=generator, **common)
    val = torch.utils.data.DataLoader(torch.utils.data.Subset(validation, split["validation"]),
        batch_size=256, shuffle=False, **common)
    return train, val, generator, digest, split["classes"]
