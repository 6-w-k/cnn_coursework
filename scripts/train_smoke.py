"""Short baseline training, validation, checkpoint-save and reload check."""
import argparse
import hashlib
import importlib.util
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tag", default="training_smoke")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2:
        parser.error("Positive step count and batch size >= 2 required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU training check")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location(
        "reference_resnet", ROOT / "references/pytorch-cifar/models/resnet.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_data = datasets.CIFAR10(str(ROOT / "data"), train=True, download=False,
        transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm]))
    val_data = datasets.CIFAR10(str(ROOT / "data"), train=True, download=False,
        transform=transforms.Compose([transforms.ToTensor(), norm]))
    indices = torch.randperm(50000, generator=torch.Generator().manual_seed(42)).tolist()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_data, indices[:45000]), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(42), num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_data, indices[45000:]), batch_size=args.batch_size, num_workers=0)
    if args.steps > len(train_loader):
        parser.error("Step count exceeds one training epoch")
    model = source.ResNet18().cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    losses = []
    samples = 0
    initial_weight = model.conv1.weight.detach().clone()
    torch.cuda.synchronize()
    started = time.perf_counter()
    iterator = iter(train_loader)
    for step in range(args.steps):
        images, labels = next(iterator)
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(images), labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step + 1}")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        samples += len(labels)
        if step == 0 or (step + 1) % 8 == 0 or step + 1 == args.steps:
            print(f"Training step {step + 1}/{args.steps}: loss={loss.item():.6f}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    updated = not torch.equal(initial_weight, model.conv1.weight.detach())
    finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    if not updated or not finite:
        raise RuntimeError("Parameters did not update or became non-finite")
    model.eval()
    validation_losses = []
    validation_samples = 0
    with torch.no_grad():
        for index, (images, labels) in enumerate(val_loader):
            images, labels = images.cuda(), labels.cuda()
            logits = model(images)
            validation_loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(validation_loss):
                raise RuntimeError("Non-finite validation loss")
            validation_losses.append(validation_loss.item())
            validation_samples += len(labels)
            if index == 0:
                reload_images, expected_logits = images, logits.clone()
            if index == 3:
                break
    checkpoint = ROOT / "checkpoints" / f"{args.tag}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "completed_steps": len(losses), "seed": 42}, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = source.ResNet18().cuda().eval()
    restored.load_state_dict(payload["model"])
    with torch.no_grad():
        torch.testing.assert_close(restored(reload_images), expected_logits)
    with checkpoint.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "Short training plumbing check only; no full epoch, no benchmark accuracy, no test set evaluation.",
        "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "model": "CIFAR-10 adapted ResNet-18, original ReLU baseline",
        "batch_size": args.batch_size, "training_steps": len(losses), "training_samples_processed": samples,
        "training_step_losses": losses, "training_mean_loss": sum(losses) / len(losses),
        "training_seconds_under_current_server_load": elapsed,
        "weights_updated": updated, "parameters_finite": finite,
        "validation_batches": len(validation_losses), "validation_samples_processed": validation_samples,
        "validation_losses": validation_losses,
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_saved_and_reload_outputs_match": True,
        "test_set_used": False,
    }
    output = ROOT / "logs" / f"{args.tag}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Training/validation/checkpoint check passed: {output}", flush=True)


if __name__ == "__main__":
    main()
