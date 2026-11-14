"""Four-GPU work queue. Never interrupts pre-existing GPU jobs."""
import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def gpu_available(index, allowed_background):
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    mapping = {int(row.split(",")[0]): row.split(",")[1].strip() for row in rows.splitlines()}
    utilization = {int(row.split(",")[0]): int(row.split(",")[2].strip()) for row in rows.splitlines()}
    if index not in mapping or utilization[index] > 10:
        return False
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    for line in processes.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or parts[0] != mapping[index]:
            continue
        try:
            permitted = int(parts[1]) in allowed_background and float(parts[2]) <= 64
        except ValueError:
            permitted = False
        if not permitted:
            return False
    return True


def summarize(directory, variants):
    rows = []
    for variant in variants:
        path = directory / variant / "metrics.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        if m["smoke"] or not m["test_evaluated"]:
            continue
        rows.append({"variant": variant, "parameters": m["model_statistics"]["parameters"],
            "macs": m["model_statistics"]["macs"], "flops": m["model_statistics"]["flops"],
            "epochs": m["epochs"], "best_epoch": m["best_epoch"],
            "validation_accuracy_pct": m["best_validation_accuracy_pct"],
            "test_accuracy_pct": m["test"]["accuracy_pct"], "test_macro_f1": m["test"]["macro_f1"],
            "training_seconds": m["total_training_seconds"]})
    write_json(directory / "comparison.json", {"completed_experiments": len(rows), "results": rows,
        "limitations": "One seed. Training times were measured under concurrent four-GPU execution and are not isolated inference speed comparisons."})
    if rows:
        temp = directory / "comparison.csv.tmp"
        with temp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(directory / "comparison.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=str(ROOT / "experiments/plan.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-background-pid", type=int, action="append", default=[])
    args = parser.parse_args()
    verification = json.loads((ROOT / "logs/five_experiment_verification.json").read_text())
    if verification["status"] != "passed":
        raise RuntimeError("Five-model verification has not passed")
    for name, expected in verification["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Source changed after verification: " + name)
    config = json.loads(Path(args.config).read_text())
    variants, gpus = config["variants"], config["gpus"]
    directory = Path(args.run_dir).resolve()
    if directory.exists() and not args.resume:
        raise RuntimeError("Run directory already exists; choose a new directory or --resume")
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "coordinator.lock"
    if lock.exists():
        old_pid = int(lock.read_text())
        try:
            command_line = Path(f"/proc/{old_pid}/cmdline").read_bytes()
        except FileNotFoundError:
            command_line = b""
        if b"launch_experiments.py" in command_line:
            raise RuntimeError(f"Coordinator {old_pid} is already active")
    lock.write_text(str(os.getpid()))
    write_json(directory / "plan.json", config)
    write_json(directory / "source_manifest.json", verification["source_sha256"])
    if not (directory / "source_snapshot.tar.gz").exists():
        with tarfile.open(directory / "source_snapshot.tar.gz", "w:gz") as archive:
            for folder in ("cnn", "scripts", "experiments"):
                for path in (ROOT / folder).rglob("*"):
                    if path.is_file() and "__pycache__" not in path.parts:
                        archive.add(path, arcname=path.relative_to(ROOT).as_posix())
    pending, jobs, active = [], {}, {}
    for variant in variants:
        result = directory / variant / "metrics.json"
        completed = False
        if args.resume and result.exists():
            metrics = json.loads(result.read_text())
            completed = metrics.get("test_evaluated") and not metrics.get("smoke") and metrics["epochs"] == config["epochs"]
        jobs[variant] = {"state": "completed" if completed else "pending", "gpu": None, "pid": None}
        if not completed:
            pending.append(variant)
    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    state = {"coordinator_pid": os.getpid(), "run_directory": str(directory), "state": "running",
             "jobs": jobs, "gpu_indices": gpus, "allowed_small_background_pids": args.allow_background_pid,
             "started_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        while pending or active:
            for gpu, (process, variant, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                jobs[variant].update(state="completed" if code == 0 else "failed", exit_code=code,
                                     finished_at_utc=datetime.now(timezone.utc).isoformat())
                del active[gpu]
                summarize(directory, variants)
                print(json.dumps({"finished": variant, "gpu": gpu, "exit_code": code}), flush=True)
            if stopping:
                for process, _, _ in active.values():
                    process.terminate()
                for process, variant, log in active.values():
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    log.close()
                    jobs[variant]["state"] = "stopped"
                for variant in pending:
                    jobs[variant]["state"] = "not_started"
                active.clear()
                pending.clear()
                state["state"] = "stopped"
                break
            for gpu in gpus:
                if not pending or gpu in active or not gpu_available(gpu, set(args.allow_background_pid)):
                    continue
                variant = pending.pop(0)
                target = directory / variant
                target.mkdir(exist_ok=True)
                cmd = [sys.executable, "-u", str(ROOT / "scripts/train.py"), "--config", str(directory / "plan.json"),
                       "--variant", variant, "--output", str(target)]
                if (target / "last.pt").exists():
                    cmd.append("--resume")
                elif (target / "history.csv").exists():
                    lines = (target / "history.csv").read_text().splitlines()
                    if len(lines) <= 1:
                        (target / "history.csv").unlink()
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                    CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS=str(config["cpu_threads"]),
                    MKL_NUM_THREADS=str(config["cpu_threads"]), OPENBLAS_NUM_THREADS="1")
                log = (target / "stdout.log").open("ab")
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (process, variant, log)
                jobs[variant].update(state="running", gpu=gpu, pid=process.pid,
                    started_at_utc=datetime.now(timezone.utc).isoformat(), command=cmd)
                print(json.dumps({"started": variant, "gpu": gpu, "pid": process.pid}), flush=True)
            state.update(updated_at_utc=datetime.now(timezone.utc).isoformat(), pending=list(pending))
            write_json(directory / "queue.json", state)
            time.sleep(5)
        if not stopping:
            state["state"] = "completed" if all(j["state"] == "completed" for j in jobs.values()) else "failed"
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "queue.json", state)
        summarize(directory, variants)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
