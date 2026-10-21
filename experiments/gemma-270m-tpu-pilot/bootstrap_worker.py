#!/usr/bin/env python3
"""Standard-library bootstrap executed on every worker of an existing TPU slice."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile


def metadata_token():
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["access_token"]


def fetch_bundle(uri, destination):
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError("Expected a gs:// bundle URI.")
    name = urllib.parse.quote(parsed.path.lstrip("/"), safe="")
    url = f"https://storage.googleapis.com/storage/v1/b/{parsed.netloc}/o/{name}?alt=media"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {metadata_token()}"})
    with urllib.request.urlopen(request, timeout=300) as source, destination.open("wb") as output:
        import shutil
        shutil.copyfileobj(source, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not all(character.isalnum() or character == "-" for character in args.run_id):
        raise ValueError("Invalid run ID.")
    work = Path.home() / "gemma-270m-tpu-pilot" / args.run_id
    work.mkdir(parents=True, exist_ok=False)
    archive = work / "bundle.zip"
    fetch_bundle(args.bundle_uri, archive)
    with zipfile.ZipFile(archive) as bundle:
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in bundle.namelist()):
            raise ValueError("Unsafe archive path.")
        bundle.extractall(work)
    tooling = work / "bootstrap-packages"
    subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(tooling), "uv==0.11.15"], check=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tooling)
    uv = [sys.executable, "-m", "uv"]
    venv = work / ".venv"
    subprocess.run([*uv, "venv", "--python", "3.12", "--managed-python", str(venv)], env=environment, check=True)
    python = venv / "bin" / "python"
    subprocess.run([*uv, "pip", "install", "--python", str(python), "-r", str(work / "requirements-tpu.lock")],
                   env=environment, check=True)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["JAX_PLATFORMS"] = "tpu"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    subprocess.run([str(python), str(work / "run_pilot.py"), "--data-dir", str(work / "prepared"),
                    "--run-dir", str(work / "runs"), "--output-uri", args.output_uri],
                   env=environment, cwd=work, timeout=1800, check=True)


if __name__ == "__main__":
    main()
