"""Verify architectural invariants, actual training and exact checkpoint resume."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from cnn.data import write_json
from cnn.models import ResNet18, VARIANTS, SE, DepthwiseSeparable, model_statistics


def source_hashes():
    import hashlib
    paths = list((ROOT / "cnn").glob("*.py")) + [ROOT / "scripts/train.py", ROOT / "experiments/plan.json"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory = Path(args.output) if args.output else ROOT / "runs" / ("verification_" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("reference_resnet", ROOT / "references/pytorch-cifar/models/resnet.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    original, baseline = reference.ResNet18().eval(), ResNet18().eval()
    baseline.load_state_dict(original.state_dict(), strict=True)
    images = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(original(images), baseline(images))
    stats = {}
    for variant in VARIANTS:
        model = ResNet18(variant)
        stats[variant] = model_statistics(model)
        if variant in ("gelu", "mish"):
            activation = nn.GELU if variant == "gelu" else nn.Mish
            assert sum(isinstance(m, activation) for m in model.modules()) == 17
        if variant == "depthwise":
            assert sum(isinstance(m, DepthwiseSeparable) for m in model.modules()) == 16
            assert model.conv1.groups == 1
        if variant == "se":
            assert sum(isinstance(m, SE) for m in model.modules()) == 8
    assert stats["depthwise"]["parameters"] < stats["baseline"]["parameters"] < stats["se"]["parameters"]
    assert stats["depthwise"]["macs"] < stats["baseline"]["macs"] < stats["se"]["macs"]
    for v in ("gelu", "mish"):
        assert stats[v]["parameters"] == stats["baseline"]["parameters"]
        assert stats[v]["macs"] == stats["baseline"]["macs"]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    common = [sys.executable, str(ROOT / "scripts/train.py"), "--smoke", "--batch-size", "8",
              "--workers", "0", "--limit-train-batches", "2", "--limit-val-batches", "2"]
    for variant in VARIANTS:
        cmd = common + ["--variant", variant, "--epochs", "1", "--output", str(directory / variant)]
        subprocess.run(cmd, env=env, check=True)
        metrics = json.loads((directory / variant / "metrics.json").read_text())
        assert metrics["smoke"] and not metrics["test_evaluated"]
        assert metrics["model_statistics"] == stats[variant]
    resume_dir, uninterrupted = directory / "resume", directory / "uninterrupted"
    resume_common = common.copy()
    resume_common[resume_common.index("--workers") + 1] = "2"
    base = resume_common + ["--variant", "baseline", "--epochs", "2"]
    subprocess.run(base + ["--output", str(resume_dir), "--stop-after-epoch", "1"], env=env, check=True)
    subprocess.run(base + ["--output", str(resume_dir), "--resume"], env=env, check=True)
    subprocess.run(base + ["--output", str(uninterrupted)], env=env, check=True)
    a = torch.load(resume_dir / "last.pt", map_location="cpu", weights_only=True)
    b = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=True)
    assert a["epoch"] == b["epoch"] == 2
    for name in a["model"]:
        if not torch.equal(a["model"][name], b["model"][name]):
            raise RuntimeError("Resume does not reproduce uninterrupted weights: " + name)
    sample = ROOT / "data/sample_png/train_00000_frog.png"
    subprocess.run([sys.executable, str(ROOT / "scripts/predict.py"),
                    str(directory / "baseline/best.pt"), str(sample)], env=env, check=True)
    result = {"status": "passed", "scope": "Architecture and real-data short-run verification, not accuracy results",
        "baseline_matches_upstream": True, "all_five_variants_trained_and_checkpoints_loaded": True,
        "resume_matches_uninterrupted_weights_exactly": True, "multiprocess_data_loading_passed": True,
        "image_prediction_cli_passed": True, "test_set_used": False,
        "statistics": stats, "source_sha256": source_hashes(), "artifacts": str(directory)}
    write_json(ROOT / "logs/five_experiment_verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
