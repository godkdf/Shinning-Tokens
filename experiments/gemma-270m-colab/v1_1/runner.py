
"""Self-contained one-chip Colab Gemma pilot, embedded in its notebook."""
from __future__ import annotations
import argparse
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile
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


MODEL_HELPERS = '@struct.dataclass\nclass PilotState:\n    step: jax.Array\n    params: object\n    opt_state: object\n    rng: jax.Array\n    data_position: jax.Array\n\n\ndef make_model(cpu_fixture=False):\n    # Kauldron\'s dtype registry assumes Linux\'s extended-precision NumPy alias.\n    # This compatibility alias is used only by the explicitly marked Mac CPU fixture.\n    if cpu_fixture and not hasattr(np, "float128"):\n        np.float128 = np.longdouble\n    from gemma import gm\n    model = gm.nn.Gemma3_270M(dtype=jnp.float32)\n    if cpu_fixture:\n        config = dataclasses.replace(model.config, num_embed=128, embed_dim=32,\n            hidden_dim=64, num_heads=2, head_dim=16, num_kv_heads=1,\n            attention_types=model.config.attention_types[:2], sliding_window_size=8)\n        model = model.clone(config=config)\n    return model\n\n\ndef make_optimizer(config, params):\n    schedule = optax.join_schedules([\n        optax.linear_schedule(0.0, config["learning_rate"], config["warmup_updates"]),\n        optax.constant_schedule(config["learning_rate"]),\n    ], [config["warmup_updates"]])\n    mask = jax.tree.map(lambda value: value.ndim > 1, params)\n    optimizer = optax.chain(optax.clip_by_global_norm(config["gradient_clip"]),\n        optax.adamw(schedule, b1=config["adam_b1"], b2=config["adam_b2"],\n                    eps=config["adam_eps"], weight_decay=config["weight_decay"], mask=mask))\n    return optimizer, schedule\n\n\ndef loss_parts(model, params, batch):\n    # Master weights remain FP32. Only the forward computation sees BF16 copies.\n    compute_params = jax.tree.map(lambda value: value.astype(jnp.bfloat16), params)\n    logits = model.apply({"params": compute_params}, tokens=batch["tokens"]).logits\n    logits = logits.astype(jnp.float32)\n    target_logits = jnp.take_along_axis(logits, batch["targets"][..., None], axis=-1)[..., 0]\n    losses = jax.scipy.special.logsumexp(logits, axis=-1) - target_logits\n    weights = batch["mask"].astype(jnp.float32)\n    return jnp.sum(jnp.where(weights > 0, losses, 0.0)), jnp.sum(weights)\n'

def load_model_helpers(cpu_test=False):
    accelerator_imports(cpu_test)
    exec(MODEL_HELPERS, globals())



def accelerator_imports(cpu_test=False):
    global jax, jnp, optax, struct, serialization, traverse_util
    os.environ["JAX_PLATFORMS"] = "cpu" if cpu_test else "tpu"
    import jax
    import jax.numpy as jnp
    import optax
    from flax import struct, serialization, traverse_util
    return jax


def preflight(cpu_test=False):
    accelerator_imports(cpu_test)
    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError(f"This notebook requires exactly ONE v5e chip; found {len(devices)} devices: {devices}. Select the v5e-1 TPU runtime in Colab.")
    device = devices[0]
    if not cpu_test:
        kind = device.device_kind.lower().replace(" ", "")
        if device.platform != "tpu" or not any(label in kind for label in ("v5lite", "v5e")):
            raise RuntimeError(f"Expected a v5e-1 TPU, found {device}. Change the Colab runtime hardware accelerator.")
    return {"platform": device.platform, "device_kind": device.device_kind,
            "device_count": len(devices), "cpu_test": cpu_test,
            "python": sys.version, "jax": jax.__version__,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}


def numpy_leaf(value):
    if isinstance(value, jax.Array):
        return np.asarray(jax.device_get(value))
    return np.asarray(value)


def archive_tree(state_dict):
    leaves = []
    def encode(node):
        if isinstance(node, dict):
            return {key: encode(value) for key, value in node.items()}
        name = f"arrays/{len(leaves):05d}.npy"
        leaves.append((name, node))
        return {"__array__": name}
    return encode(state_dict), leaves


def save_archive(path, state, metadata):
    """Write one array at a time; never materialize a full host checkpoint copy."""
    tree, leaves = archive_tree(serialization.to_state_dict(state))
    temporary = Path(str(path) + ".partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, leaf in leaves:
            array = numpy_leaf(leaf)
            with archive.open(name, "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, array, allow_pickle=False)
            del array
        archive.writestr("checkpoint.json", json.dumps({"tree": tree, "metadata": metadata}, allow_nan=False))
    temporary.replace(path)
    return len(leaves)


def verify_archive(path, state):
    """Compare the persistent checkpoint with current state, one array at a time."""
    tree, leaves = archive_tree(serialization.to_state_dict(state))
    with zipfile.ZipFile(path) as archive:
        description = json.loads(archive.read("checkpoint.json"))
        if tree != description["tree"]:
            raise ValueError("Checkpoint tree differs from the original state.")
        for name, leaf in leaves:
            with archive.open(name) as stream:
                stored = np.lib.format.read_array(stream, allow_pickle=False)
            actual = numpy_leaf(leaf)
            if stored.dtype != actual.dtype or not np.array_equal(stored, actual):
                raise ValueError(f"Checkpoint state mismatch: {name}")
            del stored, actual
    return len(leaves)


def restore_archive(path, abstract_state):
    """Restore arrays directly to the TPU before reconstructing the state class."""
    with zipfile.ZipFile(path) as archive:
        description = json.loads(archive.read("checkpoint.json"))
        def decode(node):
            if set(node) == {"__array__"}:
                with archive.open(node["__array__"]) as stream:
                    array = np.lib.format.read_array(stream, allow_pickle=False)
                value = jax.device_put(array)
                jax.block_until_ready(value)
                return value
            return {key: decode(value) for key, value in node.items()}
        restored_dict = decode(description["tree"])
    return serialization.from_state_dict(abstract_state, restored_dict)


def create_fixture(directory, config):
    directory.mkdir(parents=True, exist_ok=False)
    splits = {}
    for split, count in (("train",64),("validation",32)):
        rows = []
        docs = []
        for index in range(count):
            ids = [3 + index % 7] * 23
            rows.extend((*row,index) for row in chunk_document(ids,16,2,1))
            docs.append({"id":f"{split}-{index}","sha256":hashlib.sha256(f"{split}-{index}".encode()).hexdigest()})
        x,y,m,indices = zip(*rows)
        np.savez_compressed(directory/f"{split}.npz", tokens=np.stack(x),targets=np.stack(y),
                            mask=np.stack(m),document_index=np.array(indices,np.int32))
        splits[split] = {"documents":docs,"blocks":len(rows)}
    manifest = {"splits":splits,"cpu_test":True,"files":{}}
    for name in ("train.npz","validation.npz"):
        manifest["files"][name] = sha256_file(directory/name)
    write_json(directory/"manifest.json",manifest)
    write_json(directory/"resolved_config.json",{**config,"sequence_length":16,"training_seconds":0.5,"minimum_updates":5})


VARIANTS = (("B", False, 1), ("C", True, 1), ("D", True, 2), ("E", True, 4))


def pack_chunks(arrays):
    """Best-fit decreasing; keep each original chunk intact and reset its positions."""
    length = arrays["tokens"].shape[1]
    counts = arrays["mask"].sum(axis=1).astype(np.int32)
    expected_mask = np.arange(length)[None, :] < counts[:, None]
    if not np.array_equal(arrays["mask"], expected_mask) or np.any(counts <= 0):
        raise ValueError("Source chunks must have a nonempty contiguous target mask.")
    buckets = [[] for _ in range(length + 1)]
    bins, placements = [], []
    for source in np.argsort(-counts, kind="stable"):
        size = int(counts[source])
        available = next((r for r in range(size, length + 1) if buckets[r]), None)
        if available is None:
            row, offset = len(bins), 0
            bins.append([])
            remaining = length - size
        else:
            row = buckets[available].pop()
            offset = length - available
            remaining = available - size
        segment = len(bins[row]) + 1
        bins[row].append(int(source))
        placements.append((int(source), row, offset, size, segment))
        if remaining:
            buckets[remaining].append(row)
    shape = (len(bins), length)
    packed = {"tokens": np.zeros(shape, np.int32), "targets": np.zeros(shape, np.int32),
              "mask": np.zeros(shape, np.bool_), "segments": np.zeros(shape, np.int32),
              "positions": np.zeros(shape, np.int32)}
    for source, row, offset, size, segment in placements:
        end = offset + size
        for key in ("tokens", "targets", "mask"):
            packed[key][row, offset:end] = arrays[key][source, :size]
        packed["segments"][row, offset:end] = segment
        packed["positions"][row, offset:end] = np.arange(size, dtype=np.int32)
    verify_packing(arrays, packed, placements)
    return packed, placements


def verify_packing(original, packed, placements):
    seen = set()
    for source, row, offset, size, segment in placements:
        if source in seen:
            raise ValueError("An original chunk was packed twice.")
        seen.add(source)
        for key in ("tokens", "targets", "mask"):
            if not np.array_equal(original[key][source, :size], packed[key][row, offset:offset+size]):
                raise ValueError(f"Packing changed {key} in original chunk {source}.")
        if not np.all(packed["segments"][row, offset:offset+size] == segment):
            raise ValueError("Packing segment IDs are inconsistent.")
        if not np.array_equal(packed["positions"][row, offset:offset+size], np.arange(size)):
            raise ValueError("Packing changed a chunk's original positions.")
    if seen != set(range(len(original["tokens"]))):
        raise ValueError("Packing lost original chunks.")
    if int(original["mask"].sum()) != int(packed["mask"].sum()):
        raise ValueError("Packing changed the number of training targets.")


def packed_attention(batch):
    segments = batch["segments"]
    length = segments.shape[-1]
    indices = jnp.arange(length)
    causal = indices[:, None] >= indices[None, :]
    same_segment = segments[..., :, None] == segments[..., None, :]
    real = (segments[..., :, None] > 0) & (segments[..., None, :] > 0)
    # Padded queries attend to themselves, keeping softmax finite; their losses are masked.
    padded_diagonal = (segments[..., :, None] == 0) & jnp.eye(length, dtype=jnp.bool_)
    return (same_segment & causal & real) | padded_diagonal


def suite_loss(model, params, batch):
    compute_params = jax.tree.map(lambda value: value.astype(jnp.bfloat16), params)
    kwargs = {}
    if "segments" in batch:
        kwargs = {"positions": batch["positions"], "attention_mask": packed_attention(batch)}
    logits = model.apply({"params": compute_params}, tokens=batch["tokens"], **kwargs).logits.astype(jnp.float32)
    selected = jnp.take_along_axis(logits, batch["targets"][..., None], axis=-1)[..., 0]
    loss = jax.scipy.special.logsumexp(logits, axis=-1) - selected
    weights = batch["mask"].astype(jnp.float32)
    return jnp.where(batch["mask"], loss, 0.0).sum(), weights.sum()


def batch_indices(order, cursor, batch_size):
    return order[(cursor + np.arange(batch_size)) % len(order)]


def host_batch(arrays, indices):
    return {key: value[indices] for key, value in arrays.items()}


def check_baseline(manifest, baseline):
    for key in ("model_revision", "dataset_revision"):
        if manifest[key] != baseline["result"]["config"][key]:
            raise ValueError(f"Data/model revision differs from the measured baseline: {key}")
    for filename, digest in baseline["expected_files"].items():
        if manifest["files"].get(filename) != digest:
            raise ValueError(f"Prepared source/tokenizer differs from v1.0: {filename}")
    for split, expected in baseline["splits"].items():
        actual = manifest["splits"][split]
        digest = hashlib.sha256(json.dumps([d["sha256"] for d in actual["documents"]], separators=(",", ":")).encode()).hexdigest()
        if digest != expected["ordered_document_hash"]:
            raise ValueError(f"Ordered {split} documents differ from the measured baseline.")
        if any(actual[key] != expected[key] for key in ("blocks", "scored_tokens")):
            raise ValueError(f"Token/chunk counts differ from the measured baseline: {split}")


def prepare_cached(config, work, output):
    fields = ("model_id", "model_revision", "dataset_id", "dataset_config", "dataset_revision",
              "train_documents", "validation_documents", "sequence_length")
    identity = {key: config[key] for key in fields}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    cache = work.parent / "data-cache" / key
    cache.parent.mkdir(parents=True, exist_ok=True)
    hit = (cache / "manifest.json").is_file()
    if not hit:
        import tempfile
        with tempfile.TemporaryDirectory(dir=cache.parent) as temporary:
            staged = Path(temporary) / "prepared"
            prepare(config, staged)
            staged.rename(cache)
    manifest = read_json(cache / "manifest.json")
    for filename, digest in manifest["files"].items():
        if sha256_file(cache / filename) != digest:
            raise ValueError(f"Cached input checksum mismatch: {filename}")
    check_baseline(manifest, read_json(work / "baseline_v1_0.json"))
    prepared = work / "prepared"
    prepared.mkdir(exist_ok=True)
    for path in cache.iterdir():
        if path.name != "resolved_config.json":
            (prepared / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    write_json(prepared / "resolved_config.json", {**config, "model_revision": manifest["model_revision"],
                                                  "dataset_revision": manifest["dataset_revision"]})
    write_json(output / "preparation.json", {"cache_hit": hit, "cache_key": key,
                                            "original_preparation_seconds": manifest["preparation_seconds"]})


def is_oom(error):
    message = str(error).lower()
    return any(word in message for word in ("resource_exhausted", "out of memory", "failed to allocate"))


def run_suite(work, output, cpu_test=False):
    from collections import deque
    suite_started = time.monotonic()
    environment = preflight(cpu_test)
    load_model_helpers(cpu_test)
    work, output = Path(work), Path(output)
    prepared = work / "prepared"
    config = read_json(prepared / "resolved_config.json")
    manifest = read_json(prepared / "manifest.json")
    baseline = read_json(work / "baseline_v1_0.json")
    manifest_hash = sha256_file(prepared / "manifest.json")
    for filename, digest in manifest["files"].items():
        if sha256_file(prepared / filename) != digest:
            raise ValueError(f"Prepared input changed: {filename}")
    train_hashes = {d["sha256"] for d in manifest["splits"]["train"]["documents"]}
    validation_hashes = {d["sha256"] for d in manifest["splits"]["validation"]["documents"]}
    if train_hashes & validation_hashes:
        raise ValueError("Training and validation document hashes overlap.")
    if not cpu_test:
        check_baseline(manifest, baseline)
        if shutil.disk_usage(work).free < 8 * 1024**3:
            raise RuntimeError("The full winner checkpoint needs at least 8 GiB free local disk.")
    write_json(output / "environment.json", environment)
    write_json(output / "config.json", config)
    shutil.copyfile(prepared / "manifest.json", output / "data_manifest.json")
    def load_arrays(split):
        with np.load(prepared / f"{split}.npz", allow_pickle=False) as archive:
            return {key: archive[key] for key in ("tokens", "targets", "mask")}
    unpacked, validation = load_arrays("train"), load_arrays("validation")
    phase = time.monotonic()
    packed, placements = pack_chunks(unpacked)
    packing_seconds = time.monotonic() - phase
    np.savez_compressed(output / "packing_map.npz", placements=np.asarray(placements, np.int32))
    packing = {"algorithm": "best-fit decreasing, intact chunks, stable length ties",
               "original_chunks": len(unpacked["tokens"]), "packed_sequences": len(packed["tokens"]),
               "scored_targets_before": int(unpacked["mask"].sum()),
               "scored_targets_after": int(packed["mask"].sum()),
               "unpacked_utilization": float(unpacked["mask"].mean()),
               "packed_utilization": float(packed["mask"].mean()), "packing_seconds": packing_seconds,
               "every_original_chunk_verified": True, "positions_reset_per_original_chunk": True,
               "cross_chunk_attention": False}
    write_json(output / "packing.json", packing)
    model = make_model(cpu_test)
    initializer = jax.jit(lambda key: model.init(key, tokens=jnp.ones((1, 1), jnp.int32))["params"])
    evaluation_jit = jax.jit(lambda params, batch: suite_loss(model, params, batch))
    eval_example = jax.device_put(host_batch(validation, np.array([0])))
    eval_executable = None
    reference_initial_loss = None
    reference_probe = None
    results = []
    candidate = work / "winner_candidate.npz"
    winner = None
    candidate_write_seconds = 0.0

    def evaluate(params):
        total, count = 0.0, 0
        for index in range(len(validation["tokens"])):
            batch = jax.device_put(host_batch(validation, np.array([index])))
            nll, scored = jax.block_until_ready(eval_executable(params, batch))
            total += float(nll)
            count += int(scored)
        if count == 0:
            raise ValueError("Validation has zero scored tokens.")
        return {"loss": total / count, "nll_sum": total, "scored_tokens": count,
                "manifest_sha256": manifest_hash}

    for label, use_packing, batch_size in VARIANTS:
        print(f"\nRun {label}: packed={use_packing}, batch={batch_size}, {config['training_seconds']} seconds", flush=True)
        arrays = packed if use_packing else unpacked
        order = np.random.default_rng(config["seed"]).permutation(len(arrays["tokens"]))
        run_dir = output / label
        run_dir.mkdir(exist_ok=True)
        run_config = {**config, "variant": label, "packing": use_packing, "local_batch_per_chip": batch_size}
        state = train_executable = example = None
        pending = []
        prefetch = deque()
        try:
            phase = time.monotonic()
            params = initializer(jax.random.PRNGKey(config["seed"]))
            jax.block_until_ready(params)
            flat = traverse_util.flatten_dict(params)
            parameter_count = sum(leaf.size for leaf in flat.values())
            if not cpu_test and parameter_count != 268_098_176:
                raise ValueError(f"Incorrect native model parameter count: {parameter_count}")
            if not all(leaf.dtype == jnp.float32 for leaf in flat.values()):
                raise TypeError("Master parameters must be FP32.")
            embedding_shape = (model.config.num_embed, model.config.embed_dim)
            tied = sum(leaf.shape == embedding_shape for leaf in flat.values()) == 1
            probe_path = next(path for path, leaf in sorted(flat.items()) if path[0].startswith("layer_") and leaf.ndim > 1)
            def probe(p):
                return np.array(jax.device_get(traverse_util.flatten_dict(p)[probe_path].reshape(-1)[:64]))
            initial_probe = probe(params)
            if reference_probe is None:
                reference_probe = initial_probe.copy()
            if not np.array_equal(reference_probe, initial_probe):
                raise ValueError("Initial transformer weights changed between variants.")
            optimizer, schedule = make_optimizer(config, params)
            state = PilotState(jnp.array(0, jnp.int32), params, optimizer.init(params),
                               jax.random.PRNGKey(config["seed"]), jnp.array(0, jnp.int32))
            del flat, params
            jax.block_until_ready(state)
            initialization_seconds = time.monotonic() - phase
            def training_step(current, batch):
                def objective(p):
                    total, count = suite_loss(model, p, batch)
                    return total / jnp.maximum(count, 1.0), count
                (loss, count), grads = jax.value_and_grad(objective, has_aux=True)(current.params)
                norm = optax.tree.norm(grads)
                finite = jnp.isfinite(loss) & jnp.isfinite(norm) & (count > 0)
                def apply(_):
                    updates, opt_state = optimizer.update(grads, current.opt_state, current.params)
                    return current.replace(step=current.step + 1, params=optax.apply_updates(current.params, updates),
                                           opt_state=opt_state, data_position=current.data_position + batch_size)
                updated = jax.lax.cond(finite, apply, lambda _: current, operand=None)
                return updated, {"loss": loss, "tokens": count, "gradient_norm": norm,
                                 "finite": finite, "learning_rate": schedule(current.step)}
            example = jax.device_put(host_batch(arrays, batch_indices(order, 0, batch_size)))
            phase = time.monotonic()
            train_executable = jax.jit(training_step, donate_argnums=(0,)).lower(state, example).compile()
            if eval_executable is None:
                eval_executable = evaluation_jit.lower(state.params, eval_example).compile()
            compilation_seconds = time.monotonic() - phase
            del example
            phase = time.monotonic()
            initial_eval = evaluate(state.params)
            initial_eval_seconds = time.monotonic() - phase
            if reference_initial_loss is None:
                reference_initial_loss = initial_eval["loss"]
            if abs(initial_eval["loss"] - reference_initial_loss) > 1e-6:
                raise ValueError("Initial validation loss changed between variants.")
            if not cpu_test and abs(initial_eval["loss"] - baseline["result"]["initial_eval"]["loss"]) > 0.001:
                raise ValueError("Initial validation differs from v1.0 by more than 0.001 nats/token.")
            write_json(run_dir / "config.json", run_config)
            metrics_log = work / f"training_{label}.jsonl"
            updates = tokens_processed = fetch_cursor = 0
            last_start = 0.0
            def enqueue():
                nonlocal fetch_cursor
                indices = batch_indices(order, fetch_cursor, batch_size)
                host = host_batch(arrays, indices)
                prefetch.append((jax.device_put(host), int(host["mask"].sum())))
                fetch_cursor += batch_size
            with metrics_log.open("w", buffering=1024*1024) as log:
                def flush_metrics():
                    if not pending:
                        return
                    host_metrics = jax.device_get([item[0] for item in pending])
                    for metric, (_, update, start, end, expected_tokens) in zip(host_metrics, pending):
                        row = {key: np.asarray(value).item() for key, value in metric.items()}
                        if not row["finite"] or int(row["tokens"]) != expected_tokens:
                            raise FloatingPointError("Nonfinite update or inconsistent scored-token count; run rejected.")
                        row.update(update=update, start_seconds=start, end_seconds=end)
                        log.write(json.dumps(row, allow_nan=False) + "\n")
                    pending.clear()
                jax.block_until_ready(state)
                started = time.monotonic()
                # Prefetch startup and all input transfers are charged to the training clock.
                enqueue()
                enqueue()
                while True:
                    decision = time.monotonic() - started
                    if decision >= config["training_seconds"]:
                        break
                    batch, scored_tokens = prefetch.popleft()
                    last_start = decision
                    state, metrics = train_executable(state, batch)
                    # Prepare the next input while this update executes; no future update is queued.
                    enqueue()
                    jax.block_until_ready((state, metrics))
                    updates += 1
                    tokens_processed += scored_tokens
                    end = time.monotonic() - started
                    pending.append((metrics, updates, last_start, end, scored_tokens))
                    if len(pending) >= 32:
                        flush_metrics()
                    if updates % 256 == 0:
                        print(f"{label}: {updates} updates, {end:.2f} seconds", flush=True)
                training_seconds = time.monotonic() - started
                phase = time.monotonic()
                flush_metrics()
                metrics_drain_seconds = time.monotonic() - phase
            prefetch.clear()
            changed = not np.array_equal(initial_probe, probe(state.params))
            phase = time.monotonic()
            final_eval = evaluate(state.params)
            final_eval_seconds = time.monotonic() - phase
            completed_step, completed_position = int(state.step), int(state.data_position)
            checks = {"V11-001_SINGLE_V5E": not cpu_test and environment["device_count"] == 1,
                      "V11-002_NATIVE_MODEL": not cpu_test and parameter_count == 268_098_176,
                      "V11-003_TIED_EMBEDDING": tied,
                      "V11-004_DISJOINT_DATA": not bool(train_hashes & validation_hashes),
                      "V11-005_TIMED_UPDATES": training_seconds >= config["training_seconds"] and last_start < config["training_seconds"],
                      "V11-006_FINITE_APPLIED_UPDATES": completed_step == updates and updates >= config["minimum_updates"],
                      "V11-007_TRANSFORMER_CHANGED": changed,
                      "V11-008_IDENTICAL_INITIALIZATION": np.array_equal(reference_probe, initial_probe),
                      "V11-009_SAME_VALIDATION": initial_eval["manifest_sha256"] == final_eval["manifest_sha256"],
                      "V11-010_PACKING_PRESERVED": packing["every_original_chunk_verified"],
                      "V11-011_DATA_CURSOR": completed_position == updates * batch_size}
            logical_positions = updates * batch_size * config["sequence_length"]
            result = {"variant": label, "status": "COMPLETE", "cpu_test": cpu_test, "packing": use_packing,
                      "batch_size": batch_size, "model_parameters": parameter_count,
                      "training_seconds": training_seconds, "training_overrun_seconds": training_seconds-config["training_seconds"],
                      "last_update_start_seconds": last_start, "updates": updates, "scored_tokens": tokens_processed,
                      "tokens_per_second": tokens_processed/training_seconds,
                      "input_positions": logical_positions, "useful_position_fraction": tokens_processed/logical_positions,
                      "initial_eval": initial_eval, "final_eval": final_eval,
                      "initialization_seconds": initialization_seconds, "compilation_seconds": compilation_seconds,
                      "initial_eval_seconds": initial_eval_seconds, "final_eval_seconds": final_eval_seconds,
                      "metrics_drain_seconds": metrics_drain_seconds, "memory_stats": jax.devices()[0].memory_stats(),
                      "memory_scope": "Device lifetime through this variant; peaks are cumulative across B-E.",
                      "training_compute_units_estimate": training_seconds*config["compute_units_per_hour"]/3600,
                      "speedup_vs_historical_A": None if cpu_test else tokens_processed/training_seconds/baseline["result"]["tokens_per_second"],
                      "validation_loss_difference_vs_A": None if cpu_test else final_eval["loss"]-baseline["result"]["final_eval"]["loss"],
                      "requirements": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
                      "engineering_status": "PASS" if all(checks.values()) else "FAIL",
                      "learning_status": "PASS" if final_eval["loss"] < initial_eval["loss"] else "FAIL",
                      "checkpoint_scope": "Only the selected winner receives persistent checkpoint verification."}
            eligible = all(value for key, value in checks.items() if not(cpu_test and key in ("V11-001_SINGLE_V5E", "V11-002_NATIVE_MODEL")))
            if not eligible:
                raise RuntimeError(f"Run {label} failed an engineering requirement: {result['requirements']}")
            if winner is None or final_eval["loss"] < winner["final_eval"]["loss"]:
                phase = time.monotonic()
                save_archive(candidate, state, {"version": "1.1", "variant": label, "config": run_config,
                                               "parameter_count": parameter_count, "data_manifest_sha256": manifest_hash,
                                               "model_initialization": "random", "cpu_test": cpu_test})
                candidate_write_seconds += time.monotonic() - phase
                winner = result
            results.append(result)
            write_json(run_dir / "result.json", result)
            shutil.copyfile(metrics_log, run_dir / "training.jsonl")
            print(f"{label}: {result['tokens_per_second']:,.0f} tokens/s; validation loss {final_eval['loss']:.6f}", flush=True)
        except Exception as error:
            if not is_oom(error):
                raise
            failed = {"variant": label, "status": "OOM", "batch_size": batch_size,
                      "packing": use_packing, "cpu_test": cpu_test, "error": str(error),
                      "engineering_status": "FAIL"}
            results.append(failed)
            write_json(run_dir / "result.json", failed)
            print(f"{label}: ran out of memory; recorded as OOM, continuing.", flush=True)
        finally:
            state = train_executable = example = None
            pending.clear()
            prefetch.clear()
            gc.collect()
        write_json(output / "partial_results.json", {"version": "1.1", "runs": results})
    if winner is None:
        raise RuntimeError("No configuration completed successfully.")
    print(f"Selected {winner['variant']} by lowest final validation loss. Verifying full checkpoint...", flush=True)
    phase = time.monotonic()
    # Shape-only initializer preserves the selected variant's complete AdamW state structure.
    def shape_template(key):
        p = model.init(key, tokens=jnp.ones((1, 1), jnp.int32))["params"]
        optimizer, _ = make_optimizer(config, p)
        return PilotState(jnp.array(0, jnp.int32), p, optimizer.init(p), key, jnp.array(0, jnp.int32))
    template = jax.eval_shape(shape_template, jax.random.PRNGKey(config["seed"]))
    state = restore_archive(candidate, template)
    jax.block_until_ready(state)
    checkpoint = output / "checkpoint.npz"
    shutil.copyfile(candidate, checkpoint)
    verified_leaves = verify_archive(checkpoint, state)
    checkpoint_hash = sha256_file(checkpoint)
    del state
    gc.collect()
    state = restore_archive(checkpoint, template)
    jax.block_until_ready(state)
    reloaded = evaluate(state.params)
    reload_error = abs(reloaded["loss"] - winner["final_eval"]["loss"])
    cursor_ok = int(state.step) == winner["updates"] and int(state.data_position) == winner["updates"]*winner["batch_size"]
    if reload_error > config["reload_loss_tolerance"] or not cursor_ok:
        raise ValueError("Selected persistent checkpoint failed reload verification.")
    checkpoint_result = {"variant": winner["variant"], "status": "PASS", "checkpoint_bytes": checkpoint.stat().st_size,
                         "checkpoint_sha256": checkpoint_hash, "array_leaves_verified": verified_leaves,
                         "reload_loss_error": reload_error, "reloaded_eval": reloaded,
                         "persistent_save_verify_reload_seconds": time.monotonic()-phase,
                         "all_local_candidate_writes_seconds": candidate_write_seconds}
    write_json(output / "checkpoint_verification.json", checkpoint_result)
    if (prepared / "train.jsonl").exists():
        import gzip
        with gzip.open(output / "source_documents.jsonl.gz", "wb") as destination:
            for filename in ("validation.jsonl", "train.jsonl"):
                with (prepared / filename).open("rb") as source:
                    shutil.copyfileobj(source, destination)
    completed = [r for r in results if r["status"] == "COMPLETE"]
    elapsed = time.monotonic() - config["notebook_start_monotonic"]
    suite = {"version": "1.1", "cpu_test": cpu_test, "baseline_rerun": False,
             "baseline": baseline["result"], "runs": results, "packing": packing,
             "selected_variant": winner["variant"], "selection_rule": "Lowest final validation CE among completed valid runs; ties keep earlier run.",
             "fastest_variant": max(completed, key=lambda r: r["tokens_per_second"])["variant"],
             "checkpoint": checkpoint_result, "active_training_seconds": sum(r["training_seconds"] for r in completed),
             "suite_phase_seconds": time.monotonic()-suite_started, "notebook_elapsed_seconds": elapsed,
             "compute_units_per_hour_assumed": config["compute_units_per_hour"],
             "notebook_compute_units_estimate": elapsed*config["compute_units_per_hour"]/3600,
             "cost_note": "Elapsed-time estimate at the user-supplied rate; Colab's meter is authoritative. Includes recorded setup/waiting.",
             "engineering_status": "PASS" if len(completed)==4 and not cpu_test and all(r["engineering_status"]=="PASS" for r in completed) else "INCOMPLETE_OR_CPU_TEST",
             "comparison_note": "A is a historical single run. B-E each use one seed and 32 validation documents; rankings are screening results, not statistical proof."}
    write_json(output / "result.json", suite)
    print(json.dumps({key: suite[key] for key in ("engineering_status", "selected_variant", "fastest_variant", "active_training_seconds", "notebook_compute_units_estimate")}, indent=2), flush=True)
    return suite


def suite_fixture(work, output):
    config = read_json(work / "config.json")
    create_fixture(work / "prepared", config)
    config = read_json(work / "prepared" / "resolved_config.json")
    # Uneven synthetic lengths exercise packing, padding, and cyclic batch boundaries.
    for split, count in (("train", 64), ("validation", 32)):
        rows = []
        for index in range(count):
            rows.extend(chunk_document([3 + index % 7] * (2 + index % 31), 16, 2, 1))
        x, y, mask = zip(*rows)
        np.savez_compressed(work / "prepared" / f"{split}.npz", tokens=np.stack(x), targets=np.stack(y), mask=np.stack(mask))
    manifest = read_json(work / "prepared" / "manifest.json")
    for filename in ("train.npz", "validation.npz"):
        manifest["files"][filename] = sha256_file(work / "prepared" / filename)
    write_json(work / "prepared" / "manifest.json", manifest)
    return run_suite(work, output, cpu_test=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "prepare", "suite", "cpu-test"), required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "preflight":
            info = preflight()
            write_json(args.output / "preflight.json", info)
            print(json.dumps({key: info[key] for key in ("platform", "device_kind", "device_count")}))
        elif args.phase == "prepare":
            prepare_cached(read_json(args.work / "config.json"), args.work, args.output)
        elif args.phase == "cpu-test":
            suite_fixture(args.work, args.output)
        else:
            run_suite(args.work, args.output)
    except Exception:
        write_json(args.output / "failure.json", {"phase": args.phase, "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()


