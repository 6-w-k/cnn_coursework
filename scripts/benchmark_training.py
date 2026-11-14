"""Short real-data timing run, not a full training or accuracy experiment."""
import argparse
import gc
import importlib.util
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 300:
        parser.error("--steps must be between 1 and 300")
    if not torch.cuda.is_available():
        raise RuntimeError("This timing configuration requires CUDA")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    spec = importlib.util.spec_from_file_location(
        "reference_resnet", ROOT / "references/pytorch-cifar/models/resnet.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    normalizer = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalizer])
    val_transform = transforms.Compose([transforms.ToTensor(), normalizer])
    full_train = datasets.CIFAR10(str(ROOT / "data"), train=True, download=False, transform=train_transform)
    val_view = datasets.CIFAR10(str(ROOT / "data"), train=True, download=False, transform=val_transform)
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(42)).tolist()
    train = torch.utils.data.Subset(full_train, indices[:45000])
    val = torch.utils.data.Subset(val_view, indices[45000:])
    train_loader = torch.utils.data.DataLoader(train, batch_size=128, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=256, num_workers=0, pin_memory=True)
    measurements = []
    for use_amp in [False, True]:
        torch.manual_seed(42)
        model = source.ResNet18().cuda().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        iterator = iter(train_loader)

        def training_step():
            images, labels = next(iterator)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            return loss

        print(f"Warming up: {'AMP FP16' if use_amp else 'FP32'}", flush=True)
        for _ in range(5):
            training_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(args.steps):
            loss = training_step()
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - started
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss during timing check")
        peak_allocated = torch.cuda.max_memory_allocated()
        model.eval()
        val_iterator = iter(val_loader)

        def validation_step():
            images, labels = next(val_iterator)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                return nn.functional.cross_entropy(logits, labels)

        for _ in range(3):
            validation_step()
        val_steps = 10
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(val_steps):
            validation_step()
        torch.cuda.synchronize()
        validation_seconds = time.perf_counter() - started
        train_epoch = training_seconds / args.steps * len(train_loader)
        val_epoch = validation_seconds / val_steps * len(val_loader)
        epoch_seconds = train_epoch + val_epoch
        row = {
            "precision": "amp_fp16" if use_amp else "fp32",
            "timed_training_steps": args.steps, "training_seconds": training_seconds,
            "training_seconds_per_batch": training_seconds / args.steps,
            "timed_validation_steps": val_steps, "validation_seconds": validation_seconds,
            "validation_seconds_per_batch": validation_seconds / val_steps,
            "peak_pytorch_allocated_memory_bytes": peak_allocated,
            "estimated_train_epoch_seconds": train_epoch,
            "estimated_validation_epoch_seconds": val_epoch,
            "estimated_epoch_seconds": epoch_seconds,
            "estimated_single_model_hours": {str(e): epoch_seconds * e / 3600 for e in [50, 100, 200]},
            "five_baseline_equivalent_runs_hours": {str(e): epoch_seconds * e * 5 / 3600 for e in [50, 100, 200]},
        }
        measurements.append(row)
        print(json.dumps(row, indent=2), flush=True)
        del model, optimizer, scaler, iterator, val_iterator, loss
        gc.collect()
        torch.cuda.empty_cache()
    report = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Short baseline timing only; estimates extrapolate measured batches, not complete epochs or actual variant runtimes. No accuracy benchmark.",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "input_image_size": [3, 32, 32],
        "training_samples": len(train), "validation_samples": len(val),
        "training_batch_size": 128, "validation_batch_size": 256,
        "training_batches_per_epoch": len(train_loader),
        "validation_batches_per_epoch": len(val_loader),
        "dataloader_workers": 0, "cpu_threads": 4,
        "training_augmentation": ["RandomCrop(32,padding=4)", "RandomHorizontalFlip"],
        "excluded_overhead": ["initial imports", "dataset decoding at startup", "worker startup", "checkpoint saving", "plotting", "final test evaluation"],
        "limitations": ["Laptop power settings, temperature and competing workloads can change sustained speed.",
                        "SE, depthwise convolution and other activations require separate measurement; five baseline equivalents are only a scheduling reference.",
                        "Timing does not determine how many epochs are needed to reach the target accuracy."],
        "measurements": measurements,
    }
    target = ROOT / "logs" / "training_timing.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved timing report: {target}", flush=True)


if __name__ == "__main__":
    main()
