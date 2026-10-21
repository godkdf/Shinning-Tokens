"""Portable data, checkpoint, and report helpers. No accelerator imports here."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_gcs_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError("Persistent output must be a gs://bucket/prefix URI.")
    return parsed.netloc, parsed.path.strip("/")


def upload_file(path, uri):
    from google.cloud import storage
    bucket, name = split_gcs_uri(uri)
    storage.Client().bucket(bucket).blob(name).upload_from_filename(str(path), timeout=1200)


def download_file(uri, path):
    from google.cloud import storage
    bucket, name = split_gcs_uri(uri)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket).blob(name).download_to_filename(str(path), timeout=1200)


def chunk_document(ids, sequence_length, bos_id, eos_id, pad_id=0):
    """Every target appears once; consecutive chunks overlap by one input token."""
    if pad_id in ids:
        raise ValueError("Document encodes a padding ID; cannot silently mask real text.")
    sequence = [bos_id, *ids, eos_id]
    for start in range(0, len(sequence) - 1, sequence_length):
        window = sequence[start:start + sequence_length + 1]
        count = len(window) - 1
        tokens = np.full(sequence_length, pad_id, np.int32)
        targets = np.full(sequence_length, pad_id, np.int32)
        mask = np.zeros(sequence_length, np.bool_)
        tokens[:count] = window[:-1]
        targets[:count] = window[1:]
        mask[:count] = True
        yield tokens, targets, mask


def save_checkpoint(path, state_dict):
    """Store arrays without pickle, with a JSON description of the state tree."""
    arrays = {}

    def encode(node):
        if isinstance(node, dict):
            return {key: encode(value) for key, value in node.items()}
        name = f"leaf_{len(arrays):05d}"
        array = np.asarray(node)
        if array.dtype.hasobject:
            raise TypeError("Checkpoint contains a non-array value.")
        arrays[name] = array
        return {"__array__": name}

    tree = encode(state_dict)
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)
    write_json(path.with_suffix(".tree.json"), tree)


def load_checkpoint(path):
    path = Path(path)
    tree = read_json(path.with_suffix(".tree.json"))
    with np.load(path, allow_pickle=False) as archive:
        def decode(node):
            if set(node) == {"__array__"}:
                return np.array(archive[node["__array__"]])
            return {key: decode(value) for key, value in node.items()}
        return decode(tree)


def assess_report(result):
    """Turn measured evidence into requirement dispositions, never assumed passes."""
    evidence = result["evidence"]
    checks = {
        "TPU-001": result["device_count"] == 16 and result["platform"] == "tpu",
        "TPU-002": evidence["distributed_initialized_before_devices"],
        "TPU-003": evidence["native_architecture"] and result["initialization"] == "random",
        "TPU-004": evidence["tied_embedding"],
        "TPU-005": evidence["distinct_batch_indices"],
        "TPU-006": evidence["global_loss_reference_pass"],
        "TPU-007": evidence["gradient_collective_present"],
        "TPU-008": result["compilation_finished_before_training"],
        "TPU-009": result["untimed_updates"] == 0,
        "TPU-010": evidence["workers_agree_on_updates"],
        "TPU-011": evidence["synchronized_timing"],
        "TPU-012": result["initial_eval"]["manifest_hash"] == result["final_eval"]["manifest_hash"],
        "TPU-013": evidence["checkpoint_state_exact"] and evidence["persistent_checkpoint_verified"],
        "TPU-014": result["reload_loss_error"] <= result["config"]["reload_loss_tolerance"],
        "NUMERICS": result["nonfinite_updates"] == 0,
        "UPDATES": result["updates"] >= result["config"]["minimum_updates"],
        "WEIGHTS": evidence["transformer_parameters_changed"],
        "TIME": (result["training_seconds"] >= result["config"]["training_seconds"]
                 and result["last_update_start_seconds"] < result["config"]["training_seconds"]),
        "DATA": evidence["no_shared_document_hashes"],
    }
    engineering = all(checks.values())
    return {
        "engineering_status": "PASS" if engineering else "FAIL",
        "learning_status": "PASS" if result["final_eval"]["loss"] < result["initial_eval"]["loss"] else "FAIL",
        "requirements": {key: "PASS" if passed else "FAIL" for key, passed in checks.items()},
        "cpu_fixture": result.get("cpu_fixture", False),
    }
