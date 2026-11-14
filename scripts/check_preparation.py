"""Verify real downloaded samples and one GPU optimization step; not a benchmark."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]


def with_activation(baseline, activation):
    # The downloaded reference uses functional F.relu, not nn.ReLU modules.
    # FX rewrites the actual calls without modifying the archived source.
    import copy
    graph_model = torch.fx.symbolic_trace(copy.deepcopy(baseline))
    replaced = 0
    for node in graph_model.graph.nodes:
        if node.op == "call_function" and node.target is nn.functional.relu:
            node.target = activation
            node.kwargs = {}
            replaced += 1
    if replaced != 17:
        raise RuntimeError(f"Expected 17 activation calls, found {replaced}")
    graph_model.graph.lint()
    graph_model.recompile()
    return graph_model, replaced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="preparation_check")
    parser.add_argument("--requirements-name", default="requirements.txt")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = ROOT / "references" / "pytorch-cifar" / "models" / "resnet.py"
    spec = importlib.util.spec_from_file_location("reference_cifar_resnet", reference)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10(str(ROOT / "data"), train=True, download=False, transform=transform)
    test = datasets.CIFAR10(str(ROOT / "data"), train=False, download=False, transform=transform)
    if not train._check_integrity() or not test._check_integrity():
        raise RuntimeError("CIFAR-10 member integrity check failed")
    if len(train) != 50000 or len(test) != 10000:
        raise RuntimeError("Unexpected dataset split sizes")
    train_counts = np.bincount(train.targets, minlength=10).tolist()
    test_counts = np.bincount(test.targets, minlength=10).tolist()
    if train_counts != [5000] * 10 or test_counts != [1000] * 10:
        raise RuntimeError("Unexpected per-class counts")
    loader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    baseline = source.ResNet18().to(device).eval()
    with torch.no_grad():
        reference_logits = baseline(images)
    results = {}
    for name, activation in [("relu", nn.functional.relu), ("gelu", nn.functional.gelu), ("mish", nn.functional.mish)]:
        # All three start from the same convolution and normalization weights.
        model, activation_calls = with_activation(baseline, activation)
        model = model.to(device).eval()
        with torch.no_grad():
            initial_logits = model(images)
        difference = (initial_logits - reference_logits).abs().max().item()
        if name == "relu":
            torch.testing.assert_close(initial_logits, reference_logits)
        elif difference == 0:
            raise RuntimeError(f"Activation {name} had no effect on the output")
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.conv1.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if list(logits.shape) != [8, 10] or not torch.isfinite(logits).all():
            raise RuntimeError(f"Invalid model output for {name}")
        loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss for {name}")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f"Non-finite gradients for {name}")
        optimizer.step()
        updated = not torch.equal(before, model.conv1.weight.detach())
        if not updated:
            raise RuntimeError(f"Weights did not update for {name}")
        model.eval()
        with torch.no_grad():
            prediction = model(images)
        if not torch.isfinite(prediction).all():
            raise RuntimeError(f"Non-finite evaluation output for {name}")
        results[name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "activation_function_calls": activation_calls,
            "activation_function": activation.__name__,
            "initial_max_abs_difference_from_original_relu": difference,
            "input_shape": list(images.shape),
            "output_shape": list(logits.shape),
            "single_batch_training_loss": loss.item(),
            "finite_gradients": True,
            "weights_updated": updated,
            "eval_forward_passed": True,
        }
        print(f"PASS: {name}, parameters={results[name]['parameters']}, loss={loss.item():.6f}", flush=True)
        del optimizer, model
    metadata_versions = {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "numpy", "pillow"]}
    versions = {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                "numpy": str(np.__version__), "pillow": str(PIL.__version__)}
    upstream_dir = ROOT / "references" / "pytorch-cifar"
    compatibility = subprocess.run(
        [sys.executable, "-c", "import utils"], cwd=upstream_dir,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "Preparation smoke check only: real training samples, one optimization step per activation. No accuracy benchmark or full training performed.",
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": versions,
            "package_metadata_versions": metadata_versions,
            "package_metadata_mismatches": {name: {"imported": version, "metadata": metadata_versions[name]}
                                            for name, version in versions.items() if version != metadata_versions[name]},
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else None,
        },
        "dataset": {
            "train_size": len(train), "test_size": len(test),
            "classes": train.classes,
            "train_class_counts": train_counts, "test_class_counts": test_counts,
            "member_md5_verified": True,
        },
        "activation_smoke_checks": results,
        "upstream_original_entrypoint": {
            "utils_import_returncode": compatibility.returncode,
            "utils_import_stdout": compatibility.stdout,
            "utils_import_stderr": compatibility.stderr,
            "note": "Original utilities assume an interactive Unix terminal (stty); original main.py defaults to SimpleDLA and has no Windows multiprocessing main guard. Use our preparation check instead; full training entrypoint still needs adaptation.",
        },
    }
    output = ROOT / "logs" / f"{args.tag}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Pin the versions actually used, including CUDA wheel suffixes.
    wheel_index = "cu" + torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"
    (ROOT / args.requirements_name).write_text(
        f"--extra-index-url https://download.pytorch.org/whl/{wheel_index}\n"
        + "\n".join(f"{key}=={value}" for key, value in versions.items()) + "\n", encoding="utf-8")
    print(f"Preparation check passed on {device}. Report: {output}", flush=True)


if __name__ == "__main__":
    main()
