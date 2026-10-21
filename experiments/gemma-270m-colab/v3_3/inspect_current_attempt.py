# Paste into one new cell in the existing Colab notebook. No training or TPU imports.
import json
from pathlib import Path

folder = OUTPUT
print("RUN FOLDER:", folder)
print("CURRENT WORK:", WORK)
for name in ("training.started", "attempt_status.json", "timed_training.started.json",
             "kernel_verification.json", "dispatch_preparation.json", "training.jsonl",
             "checkpoint_manifest.json", "result.json", "failure.json"):
    path = folder / name
    print(f"{name}: {'EXISTS' if path.exists() else 'absent'}")
    if path.is_file() and path.suffix in (".json", ".jsonl"):
        print(path.read_text()[-12000:])
active = []
for directory in Path("/proc").iterdir():
    if not directory.name.isdigit():
        continue
    try:
        command = (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, PermissionError):
        continue
    if "pilot_V3_3.py" in command and "--phase train" in command and str(folder) in command:
        active.append({"pid": directory.name, "command": command})
print("ACTIVE TRAIN PROCESSES:", json.dumps(active, indent=2))
print("No training, downloads, marker deletion or TPU initialization was performed by this cell.")
