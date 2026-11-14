"""Audit saved predictions and export final tables from actual training outputs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_directory")
    args = parser.parse_args()
    root = Path(args.results_directory).resolve()
    names = {"baseline": "ResNet-18 基线", "depthwise": "深度可分离卷积", "se": "SE 注意力",
             "gelu": "GELU", "mish": "Mish"}
    metrics = {}
    for variant in names:
        folder = root / variant
        m = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        assert m["epochs"] == 200 and not m["smoke"] and m["test_evaluated"]
        with (folder / "history.csv").open(newline="", encoding="utf-8") as f:
            history = list(csv.DictReader(f))
        assert [int(row["epoch"]) for row in history] == list(range(1, 201))
        best = max(history, key=lambda x: (float(x["validation_accuracy_pct"]), -float(x["validation_loss"])))
        assert int(best["epoch"]) == m["best_epoch"]
        assert float(best["validation_accuracy_pct"]) == m["best_validation_accuracy_pct"]
        with np.load(folder / "test_predictions.npz") as predictions:
            actual, predicted, probability = (predictions[key] for key in ("targets", "predictions", "probabilities"))
            assert actual.shape == predicted.shape == (10000,) and probability.shape == (10000, 10)
            assert np.array_equal(probability.argmax(1), predicted)
            assert np.allclose(probability.sum(1), 1.0, atol=2e-6)
            correct = int((actual == predicted).sum())
            matrix = np.bincount(actual * 10 + predicted, minlength=100).reshape(10, 10)
        assert correct == m["test"]["correct"]
        assert m["test"]["accuracy_pct"] == 100.0 * correct / 10000
        assert matrix.tolist() == m["test"]["confusion_matrix"]
        metrics[variant] = m
    assert len({m["split_sha256"] for m in metrics.values()}) == 1
    baseline = metrics["baseline"]
    derived = {}
    lines = ["# 五组最终实验结果", "", "每组训练 200 轮，种子为 42；使用验证集选择最佳权重，随后在 10000 张测试图像上评估。",
        "训练/验证划分为 45000/5000 张，五组使用相同划分 SHA-256。此表直接从 metrics.json 生成，并用保存的逐样本预测重新核对准确率和混淆矩阵。", "",
        "| 模型 | 最佳验证准确率 | 测试准确率 | 相对基线（百分点） | 参数量 | Conv/Linear MACs |",
        "|---|---:|---:|---:|---:|---:|"]
    for variant, label in names.items():
        m = metrics[variant]
        delta = m["test"]["accuracy_pct"] - baseline["test"]["accuracy_pct"]
        derived[variant] = {"test_accuracy_pct": m["test"]["accuracy_pct"], "accuracy_change_percentage_points": delta,
            "parameter_reduction_pct": 100 * (1 - m["model_statistics"]["parameters"] / baseline["model_statistics"]["parameters"]),
            "mac_reduction_pct": 100 * (1 - m["model_statistics"]["macs"] / baseline["model_statistics"]["macs"])}
        lines.append(f"| {label} | {m['best_validation_accuracy_pct']:.2f}% | {m['test']['accuracy_pct']:.2f}% | {delta:+.2f} | {m['model_statistics']['parameters']:,} | {m['model_statistics']['macs']:,} |")
    lines += ["", "计算量口径：只统计卷积和全连接层 MACs，FLOPs=2×MACs，不含 BN、激活、池化及逐元素操作；理论运算量不等于实测推理速度。", "",
        f"深度可分离版本参数量减少 {derived['depthwise']['parameter_reduction_pct']:.2f}%，MACs 减少 {derived['depthwise']['mac_reduction_pct']:.2f}%，测试准确率下降 {-derived['depthwise']['accuracy_change_percentage_points']:.2f} 个百分点。",
        f"SE 本次测试准确率最高，相对基线提升 {derived['se']['accuracy_change_percentage_points']:.2f} 个百分点。GELU 和 Mish 在本次统一设置下没有超过 ReLU 基线。", "",
        "这是一个随机种子的结果，不能将小幅提升解释为统计上稳定显著的优势。所有精度均来自实际预测，未引用上游结果替代本项目输出。"]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "derived_comparison.json").write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
    (root / "prediction_audit.json").write_text(json.dumps({"status": "passed", "variants": list(names),
        "epochs_each": 200, "predictions_checked_each": 10000, "same_split_verified": True,
        "all_prediction_accuracies_and_confusion_matrices_match": True}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(derived, indent=2), flush=True)


if __name__ == "__main__":
    main()
