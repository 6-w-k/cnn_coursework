"""Full reproducible training. All reported metrics originate from saved outputs."""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

from cnn.data import MEAN, STD, loaders, write_json
from cnn.models import ResNet18, VARIANTS, model_statistics


def save_checkpoint(path, state):
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def capture_rng(generator):
    state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
            "loader": generator.get_state(), "python": random.getstate(),
            "numpy": [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])]}


def restore_rng(state, generator):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])
    generator.set_state(state["loader"])
    random.setstate(state["python"])
    s = state["numpy"]
    np.random.set_state((s[0], np.array(s[1], dtype=np.uint32), s[2], s[3], s[4]))


@torch.no_grad()
def evaluate(model, loader, limit=0, collect=False):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    targets, predictions, probabilities = [], [], []
    for index, (images, labels) in enumerate(loader):
        if limit and index >= limit:
            break
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        predicted = logits.argmax(1)
        total += len(labels)
        correct += (predicted == labels).sum().item()
        if collect:
            targets.extend(labels.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
            probabilities.append(logits.softmax(1).cpu().numpy())
    if not total or not np.isfinite(loss_sum):
        raise RuntimeError("Invalid evaluation output")
    result = {"loss": loss_sum / total, "accuracy_pct": 100.0 * correct / total,
              "correct": correct, "samples": total}
    if collect:
        result.update(targets=targets, predictions=predictions,
                      probabilities=np.concatenate(probabilities))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "experiments/plan.json"))
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--stop-after-epoch", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.limit_train_batches or args.limit_val_batches) and not args.smoke:
        parser.error("Partial data runs require --smoke and cannot produce formal test results")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for key, value in [("epochs", args.epochs), ("batch_size", args.batch_size), ("workers", args.workers)]:
        if value is not None:
            config[key] = value
    config.update(variant=args.variant, smoke=args.smoke,
                  limit_train_batches=args.limit_train_batches, limit_validation_batches=args.limit_val_batches)
    if config["epochs"] < 1 or config["batch_size"] < 2 or config["workers"] < 0:
        parser.error("Invalid training configuration")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these scheduled experiments")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    last_path, best_path = output / "last.pt", output / "best.pt"
    if (output / "history.csv").exists() and not args.resume:
        raise RuntimeError("Refusing to overwrite an existing training run; use --resume")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    train, val, generator, split_digest, classes = loaders(ROOT, seed, config["batch_size"], config["workers"])
    model = ResNet18(args.variant)
    statistics = model_statistics(model)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
        momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    start, best_acc, best_loss, best_epoch = 1, -1.0, float("inf"), 0
    total_train_seconds, total_validation_seconds = 0.0, 0.0
    total_training_samples = 0
    if args.resume:
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state["config"] != config or state["split_sha256"] != split_digest:
            raise RuntimeError("Resume configuration/split does not match checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"], generator)
        start = state["epoch"] + 1
        best_acc, best_loss, best_epoch = state["best_accuracy_pct"], state["best_loss"], state["best_epoch"]
        total_train_seconds = state["total_train_seconds"]
        total_validation_seconds = state["total_validation_seconds"]
        total_training_samples = state["total_training_samples"]
        # A crash between writing history and checkpoint may leave an uncommitted row.
        history_path = output / "history.csv"
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames
                committed = [row for row in reader if int(row["epoch"]) <= state["epoch"]]
            if not any(int(row["epoch"]) == state["epoch"] for row in committed):
                committed.append(state["history_row"])
            with history_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(committed)
    write_json(output / "config.json", config)
    write_json(output / "model_statistics.json", statistics)
    environment = {"python": sys.version, "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "split_sha256": split_digest, "seed": seed,
        "deterministic_algorithms": True, "precision": "float32", "classes": classes}
    write_json(output / "environment.json", environment)
    status = {"variant": args.variant, "state": "running", "pid": os.getpid(),
              "epochs": config["epochs"], "smoke": args.smoke, "gpu": environment["cuda_visible_devices"]}
    print(json.dumps({"started": status, "statistics": statistics}), flush=True)
    fields = ["epoch", "learning_rate", "train_loss", "train_accuracy_pct", "train_samples",
              "validation_loss", "validation_accuracy_pct", "validation_samples", "train_seconds", "validation_seconds"]
    with (output / "history.csv").open("a", newline="", encoding="utf-8") as history:
        writer = csv.DictWriter(history, fieldnames=fields)
        if history.tell() == 0:
            writer.writeheader()
        for epoch in range(start, config["epochs"] + 1):
            model.train()
            seen, correct, loss_sum = 0, 0, 0.0
            lr = optimizer.param_groups[0]["lr"]
            torch.cuda.synchronize()
            started = time.perf_counter()
            for index, (images, labels) in enumerate(train):
                if args.limit_train_batches and index >= args.limit_train_batches:
                    break
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss: epoch {epoch}, batch {index}")
                loss.backward()
                optimizer.step()
                seen += len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
                loss_sum += loss.item() * len(labels)
                if index == 0 or (index + 1) % 100 == 0:
                    status.update(epoch=epoch, batch=index + 1, train_batches=len(train), updated_at_utc=datetime.now(timezone.utc).isoformat())
                    write_json(output / "status.json", status)
                    print(f"{args.variant}: epoch {epoch}/{config['epochs']} batch {index + 1}/{len(train)} loss={loss.item():.5f}", flush=True)
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - started
            started = time.perf_counter()
            validation = evaluate(model, val, args.limit_val_batches)
            torch.cuda.synchronize()
            validation_seconds = time.perf_counter() - started
            total_train_seconds += train_seconds
            total_validation_seconds += validation_seconds
            total_training_samples += seen
            improved = (validation["accuracy_pct"] > best_acc or
                        (validation["accuracy_pct"] == best_acc and validation["loss"] < best_loss))
            if improved:
                best_acc, best_loss, best_epoch = validation["accuracy_pct"], validation["loss"], epoch
            scheduler.step()
            row = {"epoch": epoch, "learning_rate": lr, "train_loss": loss_sum / seen,
                "train_accuracy_pct": 100.0 * correct / seen, "train_samples": seen,
                "validation_loss": validation["loss"], "validation_accuracy_pct": validation["accuracy_pct"],
                "validation_samples": validation["samples"], "train_seconds": train_seconds,
                "validation_seconds": validation_seconds}
            state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "rng": capture_rng(generator), "config": config,
                "split_sha256": split_digest, "best_accuracy_pct": best_acc, "best_loss": best_loss,
                "best_epoch": best_epoch, "classes": classes, "total_train_seconds": total_train_seconds,
                "total_validation_seconds": total_validation_seconds, "total_training_samples": total_training_samples,
                "history_row": row}
            if improved:
                save_checkpoint(best_path, state)
            save_checkpoint(last_path, state)
            writer.writerow(row)
            history.flush()
            os.fsync(history.fileno())
            status.update(epoch=epoch, epoch_complete=True, validation_accuracy_pct=validation["accuracy_pct"],
                          best_validation_accuracy_pct=best_acc, updated_at_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / "status.json", status)
            print(json.dumps(row), flush=True)
            if args.stop_after_epoch and epoch >= args.stop_after_epoch:
                status.update(state="paused_for_resume")
                write_json(output / "status.json", status)
                return
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    metrics = {"variant": args.variant, "seed": seed, "epochs": config["epochs"], "smoke": args.smoke,
        "best_epoch": best_epoch, "best_validation_accuracy_pct": best_acc, "best_validation_loss": best_loss,
        "training_split_size": 45000, "validation_split_size": 5000, "split_sha256": split_digest,
        "total_training_samples_processed": total_training_samples,
        "total_training_seconds": total_train_seconds, "total_validation_seconds": total_validation_seconds,
        "model_statistics": statistics, "environment": environment, "test_evaluated": False}
    if not args.smoke:
        test_data = datasets.CIFAR10(str(ROOT / "data"), train=False, download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False,
            num_workers=config["workers"], pin_memory=True)
        test = evaluate(model, test_loader, collect=True)
        targets, predicted = np.array(test.pop("targets")), np.array(test.pop("predictions"))
        probabilities = test.pop("probabilities")
        confusion = np.bincount(targets * 10 + predicted, minlength=100).reshape(10, 10)
        diagonal = np.diag(confusion)
        precision = np.divide(diagonal, confusion.sum(0), out=np.zeros(10, dtype=float), where=confusion.sum(0) != 0)
        recall = diagonal / confusion.sum(1)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10), where=(precision + recall) != 0)
        test.update(confusion_matrix=confusion.tolist(), macro_f1=float(f1.mean()),
            per_class=[{"class": name, "precision": float(precision[i]), "recall": float(recall[i]),
                        "f1": float(f1[i]), "samples": int(confusion[i].sum())} for i, name in enumerate(classes)])
        np.savez_compressed(output / "test_predictions.npz", targets=targets, predictions=predicted, probabilities=probabilities)
        metrics.update(test_evaluated=True, test=test)
    with best_path.open("rb") as f:
        metrics["best_checkpoint_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    write_json(output / "metrics.json", metrics)
    status.update(state="completed", best_epoch=best_epoch, test_evaluated=metrics["test_evaluated"],
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "status.json", status)
    print(json.dumps({"completed": args.variant, "metrics": str(output / 'metrics.json'),
                      "test": metrics.get("test")}), flush=True)


if __name__ == "__main__":
    main()
