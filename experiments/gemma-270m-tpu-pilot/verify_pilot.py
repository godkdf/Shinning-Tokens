#!/usr/bin/env python3
"""Distributed checkpoint reload test plus standalone evidence-to-report verifier."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from pilot_common import assess_report, download_file, load_checkpoint, read_json, sha256_file, write_json


def verify_reload(state, checkpoint_dir, output_uri, executable, validation,
                  local_indices, data, global_batch, manifest_hash):
    import jax
    from flax import serialization
    from pilot_core import evaluate_all, host_value

    started = time.monotonic()
    destination = Path(checkpoint_dir) / "reload"
    destination.mkdir()
    for filename in ("checkpoint.npz", "checkpoint.tree.json", "checkpoint_meta.json"):
        if output_uri:
            download_file(f"{output_uri}/{filename}", destination / filename)
        else:
            import shutil
            shutil.copy2(Path(checkpoint_dir) / filename, destination / filename)
    metadata = read_json(destination / "checkpoint_meta.json")
    if sha256_file(destination / "checkpoint.npz") != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch after download.")
    if sha256_file(destination / "checkpoint.tree.json") != metadata["tree_sha256"]:
        raise ValueError("Checkpoint tree SHA-256 mismatch after download.")
    saved = load_checkpoint(destination / "checkpoint.npz")
    original = serialization.to_state_dict(jax.tree.map(host_value, state))

    def equal(a, b):
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
        return np.array_equal(a, b)

    exact = equal(original, saved)
    restored = serialization.from_state_dict(state, saved)
    restored = jax.device_put(restored, jax.tree.leaves(state)[0].sharding)
    jax.block_until_ready(restored)
    evaluation = evaluate_all(restored.params, validation, executable, local_indices,
                              data, global_batch, manifest_hash)
    return {"state_exact": exact, "persistent_verified": bool(output_uri),
            "evaluation": evaluation, "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = read_json(args.run_dir / "result.json")
    metadata = read_json(args.run_dir / "checkpoint_meta.json")
    for filename, key in (("checkpoint.npz", "checkpoint_sha256"), ("checkpoint.tree.json", "tree_sha256")):
        path = args.run_dir / filename
        if path.exists() and sha256_file(path) != metadata[key]:
            raise ValueError(f"Local file changed after verification: {filename}")
    report = assess_report(result)
    write_json(args.run_dir / "verification.json", report)
    import json
    print(json.dumps(report, indent=2))
    if result.get("cpu_fixture"):
        raise SystemExit(0 if result["evidence"]["checkpoint_state_exact"] else 1)
    raise SystemExit(0 if report["engineering_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
