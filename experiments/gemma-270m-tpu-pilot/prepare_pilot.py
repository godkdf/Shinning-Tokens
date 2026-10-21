#!/usr/bin/env python3
"""Freeze FineWeb documents and native tokenizer artifacts before TPU training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from pilot_common import chunk_document, read_json, sha256_file, write_json


def prepare(config, destination):
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from tokenizers import Tokenizer

    started = time.monotonic()
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace prepared data: {destination}")
    destination.mkdir(parents=True)
    api = HfApi()
    model_sha = api.model_info(config["model_id"], revision=config["model_revision"]).sha
    data_sha = api.dataset_info(config["dataset_id"], revision=config["dataset_revision"]).sha
    tokenizer_dir = destination / "tokenizer"
    snapshot_download(config["model_id"], revision=model_sha, local_dir=tokenizer_dir,
                      allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                                      "special_tokens_map.json", "tokenizer.model"])
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    model_config = read_json(tokenizer_dir / "config.json")
    text_config = model_config.get("text_config", model_config)
    expected = {"hidden_size": 640, "intermediate_size": 2048, "num_hidden_layers": 18,
                "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 256,
                "vocab_size": 262144}
    for key, value in expected.items():
        if text_config.get(key) != value:
            raise ValueError(f"Unexpected native model setting {key}: {text_config.get(key)}")
    bos = text_config["bos_token_id"]
    eos = text_config["eos_token_id"]
    if isinstance(eos, list):
        eos = eos[0]
    pad = text_config.get("pad_token_id", 0)
    if pad != 0:
        raise ValueError("Gemma library expects padding ID 0.")
    stream = load_dataset(config["dataset_id"], config["dataset_config"], split="train",
                          revision=data_sha, streaming=True)
    selected = {"train": [], "validation": []}
    seen = set()
    desired = config["train_documents"] + config["validation_documents"]
    scanned = 0
    for document in stream:
        scanned += 1
        text = document["text"]
        if not text.strip():
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        split = "validation" if len(selected["validation"]) < config["validation_documents"] else "train"
        selected[split].append({"id": str(document.get("id", digest)), "sha256": digest, "text": text})
        if len(seen) == desired:
            break
    if len(seen) != desired:
        raise RuntimeError(f"Only found {len(seen)} unique nonempty documents; need {desired}.")
    manifest = {"model_id": config["model_id"], "model_revision": model_sha,
                "dataset_id": config["dataset_id"], "dataset_config": config["dataset_config"],
                "dataset_revision": data_sha, "documents_scanned": scanned,
                "sequence_length": config["sequence_length"], "special_ids": {"bos": bos, "eos": eos, "pad": pad},
                "native_config": expected, "splits": {}, "files": {}}
    for split, documents in selected.items():
        rows, metadata, roundtrips = [], [], 0
        with (destination / f"{split}.jsonl").open("w", encoding="utf-8") as output:
            for document_index, document in enumerate(documents):
                output.write(json.dumps(document, ensure_ascii=False) + "\n")
                ids = tokenizer.encode(document["text"], add_special_tokens=False).ids
                roundtrips += tokenizer.decode(ids, skip_special_tokens=False) == document["text"]
                if not ids or min(ids) < 0 or max(ids) >= expected["vocab_size"]:
                    raise ValueError(f"Invalid token IDs for {split} document {document_index}.")
                rows.extend((*chunk, document_index) for chunk in chunk_document(
                    ids, config["sequence_length"], bos, eos, pad))
                metadata.append({key: document[key] for key in ("id", "sha256")})
        tokens, targets, masks, doc_indices = zip(*rows)
        arrays = {"tokens": np.stack(tokens), "targets": np.stack(targets),
                  "mask": np.stack(masks), "document_index": np.array(doc_indices, np.int32)}
        np.savez_compressed(destination / f"{split}.npz", **arrays)
        manifest["splits"][split] = {"documents": metadata, "blocks": len(rows),
            "scored_tokens": int(arrays["mask"].sum()), "exact_roundtrip_documents": roundtrips,
            "original_utf8_bytes": sum(len(doc["text"].encode("utf-8")) for doc in documents)}
        print(f"{split}: {len(documents)} documents, {len(rows)} blocks", flush=True)
    if manifest["splits"]["train"]["blocks"] < config["expected_devices"]:
        raise ValueError("Need at least one distinct training block per chip.")
    for path in sorted(destination.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            manifest["files"][str(path.relative_to(destination))] = sha256_file(path)
    manifest["preparation_seconds"] = time.monotonic() - started
    write_json(destination / "manifest.json", manifest)
    write_json(destination / "resolved_config.json", {**config, "model_revision": model_sha, "dataset_revision": data_sha})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("prepared"))
    parser.add_argument("--reuse", action="store_true", help="Reuse a prepared sample only after checking its hashes and configuration.")
    args = parser.parse_args()
    requested = read_json(args.config)
    if args.reuse and args.output.exists():
        manifest = read_json(args.output / "manifest.json")
        resolved = read_json(args.output / "resolved_config.json")
        for key, value in requested.items():
            if key not in {"model_revision", "dataset_revision"} and resolved.get(key) != value:
                raise ValueError(f"Cached configuration changed: {key}")
        for key in ("model_revision", "dataset_revision"):
            if requested[key] != "main" and requested[key] != resolved[key]:
                raise ValueError(f"Cached pinned revision differs: {key}")
        for filename, digest in manifest["files"].items():
            if sha256_file(args.output / filename) != digest:
                raise ValueError(f"Cached file changed: {filename}")
        print("Reusing verified frozen sample; revisions remain pinned to its manifest.")
    else:
        prepare(requested, args.output)


if __name__ == "__main__":
    main()
