import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run_directory")
args = parser.parse_args()
root = Path(args.run_directory)
queue = json.loads((root / "queue.json").read_text())
summary = {"state": queue["state"], "coordinator_pid": queue["coordinator_pid"], "experiments": {}}
for name, job in queue["jobs"].items():
    status_path = root / name / "status.json"
    progress = json.loads(status_path.read_text()) if status_path.exists() else {}
    summary["experiments"][name] = {"state": job["state"], "gpu": job["gpu"], "pid": job["pid"],
        "epoch": progress.get("epoch"), "epochs": progress.get("epochs"), "batch": progress.get("batch"),
        "validation_accuracy_pct": progress.get("validation_accuracy_pct")}
print(json.dumps(summary, indent=2))
