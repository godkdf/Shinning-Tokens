#!/usr/bin/env python3
"""Package prepared inputs and launch all workers of an existing v5e-16 slice."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import uuid
import zipfile

from pilot_common import read_json, sha256_file, split_gcs_uri, write_json


def resource_is_v5e16(resource):
    if resource.get("acceleratorType") == "v5litepod-16":
        return True
    accelerator = resource.get("acceleratorConfig", {})
    return (accelerator.get("type", "").upper() == "V5LITEPOD"
            and accelerator.get("topology") == "4x4")


def package_bundle(root, data_dir, destination):
    source = {}
    paths = [*root.glob("*.py"), *root.glob("requirements-*"), root / "config.json"]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            source[path.name] = sha256_file(path)
            archive.write(path, path.name)
        for path in sorted(data_dir.rglob("*")):
            if path.is_file() and ".cache" not in path.relative_to(data_dir).parts:
                archive.write(path, "prepared/" + str(path.relative_to(data_dir)))
        archive.writestr("source_manifest.json", json.dumps(source, indent=2))
    return source


def launch(args):
    if not shutil.which("gcloud"):
        raise RuntimeError("Google Cloud CLI is required; run this notebook in Cloud Shell or an authenticated workstation.")
    split_gcs_uri(args.output_uri)
    if args.hourly_usd <= 0:
        raise ValueError("Supply the actual hourly rate for the complete 16-chip slice.")
    root = Path(__file__).resolve().parent
    manifest = read_json(args.data_dir / "manifest.json")
    config = read_json(args.data_dir / "resolved_config.json")
    if manifest.get("cpu_fixture") or config["expected_devices"] != 16:
        raise ValueError("Cloud launch requires the real 16-chip pilot data.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    destination = args.output_uri.rstrip("/") + "/" + run_id
    local_dir = root / "runs" / run_id
    local_dir.mkdir(parents=True)
    common = ["--project", args.project, "--zone", args.zone]
    observed_start = time.monotonic()
    description = subprocess.check_output(["gcloud", "compute", "tpus", "tpu-vm", "describe", args.tpu_name,
        *common, "--format=json"], text=True)
    resource = json.loads(description)
    if not resource_is_v5e16(resource):
        raise RuntimeError("Selected resource is not a 16-chip TPU v5e training slice.")
    if resource.get("state") != "READY":
        raise RuntimeError(f"TPU is not READY: {resource.get('state')}")
    if len(resource.get("networkEndpoints", [])) < 2:
        raise RuntimeError("Expected a multi-worker TPU slice; inspect resource endpoints.")
    write_json(local_dir / "resource.json", resource)
    bundle = local_dir / "bundle.zip"
    source_manifest = package_bundle(root, args.data_dir, bundle)
    bundle_uri = destination + "/bundle.zip"
    subprocess.run(["gcloud", "storage", "cp", str(bundle), bundle_uri, "--project", args.project], check=True)
    bootstrap = base64.b64encode((root / "bootstrap_worker.py").read_bytes()).decode()
    loader = "import base64; exec(compile(base64.b64decode(" + repr(bootstrap) + "), 'bootstrap_worker.py', 'exec'))"
    remote_command = shlex.join(["python3", "-c", loader, "--bundle-uri", bundle_uri,
                                "--output-uri", destination, "--run-id", run_id])
    record = {"run_id": run_id, "output_uri": destination, "resource": resource,
        "source_manifest": source_manifest, "bundle_sha256": sha256_file(bundle),
        "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "slice_hourly_usd": args.hourly_usd,
        "cost_scope": "Observed job interval only; excludes allocation time before launch and after return. Total bill unknown."}
    write_json(local_dir / "launch.json", record)
    print(f"Launching all workers. Results: {destination}", flush=True)
    command = ["gcloud", "compute", "tpus", "tpu-vm", "ssh", args.tpu_name, *common,
               "--worker=all", "--quiet", "--command", remote_command]
    returncode = 1
    try:
        with (local_dir / "launcher.log").open("w") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
    finally:
        record["launcher_returncode"] = returncode
        record["observed_job_seconds"] = time.monotonic() - observed_start
        record["observed_job_compute_estimate_usd"] = args.hourly_usd * record["observed_job_seconds"] / 3600
        write_json(local_dir / "launch.json", record)
        for filename in ("launch.json", "resource.json", "launcher.log"):
            if (local_dir / filename).exists():
                subprocess.run(["gcloud", "storage", "cp", str(local_dir / filename),
                                destination + "/" + filename, "--project", args.project], check=True)
        for filename in ("result.json", "verification.json", "checkpoint_meta.json"):
            subprocess.run(["gcloud", "storage", "cp", destination + "/" + filename,
                            str(local_dir / filename), "--project", args.project], check=False)
        print(f"Local reports: {local_dir}", flush=True)
        print("The existing TPU allocation is still running. Its bill continues until it is stopped or deleted.", flush=True)
    if returncode:
        raise RuntimeError(f"Pilot failed with exit code {returncode}; inspect {local_dir / 'launcher.log'}")
    return local_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--tpu-name", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--hourly-usd", type=float, required=True, help="Total rate for all 16 chips.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).with_name("prepared"))
    args = parser.parse_args()
    launch(args)


if __name__ == "__main__":
    main()
