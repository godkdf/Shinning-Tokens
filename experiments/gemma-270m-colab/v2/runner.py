"""Gemma Colab V2: one fixed 60-second scratch-training run on one v5e TPU.
Embedded verbatim in the self-contained notebook. CPU fixtures are development only.
"""
from __future__ import annotations
import argparse
import dataclasses
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


MODEL_HELPERS = '@struct.dataclass\nclass PilotState:\n    step: jax.Array\n    params: object\n    opt_state: object\n    rng: jax.Array\n    data_position: jax.Array\n\n\ndef make_model(cpu_fixture=False):\n    # Kauldron\'s dtype registry assumes Linux\'s extended-precision NumPy alias.\n    # This compatibility alias is used only by the explicitly marked Mac CPU fixture.\n    if cpu_fixture and not hasattr(np, "float128"):\n        np.float128 = np.longdouble\n    from gemma import gm\n    model = gm.nn.Gemma3_270M(dtype=jnp.float32)\n    if cpu_fixture:\n        config = dataclasses.replace(model.config, num_embed=128, embed_dim=32,\n            hidden_dim=64, num_heads=2, head_dim=16, num_kv_heads=1,\n            attention_types=model.config.attention_types[:2], sliding_window_size=8)\n        model = model.clone(config=config)\n    return model\n\n\ndef make_optimizer(config, params):\n    schedule = optax.join_schedules([\n        optax.linear_schedule(0.0, config["learning_rate"], config["warmup_updates"]),\n        optax.constant_schedule(config["learning_rate"]),\n    ], [config["warmup_updates"]])\n    mask = jax.tree.map(lambda value: value.ndim > 1, params)\n    optimizer = optax.chain(optax.clip_by_global_norm(config["gradient_clip"]),\n        optax.adamw(schedule, b1=config["adam_b1"], b2=config["adam_b2"],\n                    eps=config["adam_eps"], weight_decay=config["weight_decay"], mask=mask))\n    return optimizer, schedule\n'

def load_model_helpers(cpu_test=False):
    accelerator_imports(cpu_test)
    exec(MODEL_HELPERS, globals())


PACKING_VERSION = "V2-best-fit-intact-chunks-1"
INPUT_KEYS = ("tokens", "targets", "mask", "positions", "segment_ids")


def split_identity(documents):
    return hashlib.sha256(("\n".join(d["sha256"] for d in documents) + "\n").encode()).hexdigest()


def pack_chunks(arrays, seed):
    """Best-fit decreasing, preserving each original chunk's complete context."""
    tokens, targets, masks = (arrays[key] for key in ("tokens", "targets", "mask"))
    count, length = tokens.shape
    lengths = masks.sum(axis=1).astype(int)
    if np.any(lengths <= 0) or not np.array_equal(masks, np.arange(length)[None, :] < lengths[:, None]):
        raise ValueError("Input chunks must have a nonempty contiguous prefix of scored targets.")
    shuffled = np.random.default_rng(seed).permutation(count)
    ordered = shuffled[np.argsort(-lengths[shuffled], kind="stable")]
    buckets = [[] for _ in range(length + 1)]
    bins, assignments = [], []
    for source in ordered:
        size = int(lengths[source])
        free = next((room for room in range(size, length + 1) if buckets[room]), None)
        if free is None:
            index = len(bins)
            bins.append(0)
            free = length
        else:
            index = buckets[free].pop()
        start = bins[index]
        bins[index] += size
        buckets[free - size].append(index)
        assignments.append([int(source), index, start, size])
    shape = (len(bins), length)
    packed = {"tokens": np.zeros(shape, np.int32), "targets": np.zeros(shape, np.int32),
              "mask": np.zeros(shape, np.bool_), "positions": np.zeros(shape, np.int32),
              "segment_ids": np.full(shape, -1, np.int32)}
    seen = np.zeros(count, np.int32)
    for source, index, start, size in assignments:
        sl = slice(start, start + size)
        packed["tokens"][index, sl] = tokens[source, :size]
        packed["targets"][index, sl] = targets[source, :size]
        packed["mask"][index, sl] = True
        packed["positions"][index, sl] = np.arange(size)
        packed["segment_ids"][index, sl] = source
        seen[source] += 1
    # Verify content and positions before discarding the unpacked training arrays.
    for source, index, start, size in assignments:
        sl = slice(start, start + size)
        if not (np.array_equal(packed["tokens"][index, sl], tokens[source, :size])
                and np.array_equal(packed["targets"][index, sl], targets[source, :size])
                and np.array_equal(packed["positions"][index, sl], np.arange(size))):
            raise ValueError("Packing changed a source chunk.")
    if not np.all(seen == 1) or int(packed["mask"].sum()) != int(lengths.sum()):
        raise ValueError("Packing duplicated or dropped source targets.")
    report = {"version": PACKING_VERSION, "source_chunks": count, "packed_sequences": len(bins),
              "scored_tokens": int(lengths.sum()), "positions": int(np.prod(shape)),
              "occupancy": float(lengths.sum() / np.prod(shape)),
              "targets_preserved": True, "context_preserved": True,
              "assignments_source_bin_start_length": assignments}
    return packed, report


def verify_prepared(directory, baseline, cpu_test=False):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    for name, digest in manifest["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Cached data failed its checksum: {name}")
    split_hashes = {name: split_identity(split["documents"]) for name, split in manifest["splits"].items()}
    if not cpu_test and split_hashes != baseline["split_hashes"]:
        raise ValueError("Selected documents differ from the measured V1 data.")
    train_hashes = {d["sha256"] for d in manifest["splits"]["train"]["documents"]}
    valid_hashes = {d["sha256"] for d in manifest["splits"]["validation"]["documents"]}
    if train_hashes & valid_hashes:
        raise ValueError("Training and validation documents overlap.")
    packing = read_json(directory / "packing.json")
    if packing["version"] != PACKING_VERSION or not all(packing[k] for k in ("targets_preserved", "context_preserved")):
        raise ValueError("Invalid packing record.")
    return manifest, packing, split_hashes


def finalize_packing(directory, config):
    directory = Path(directory)
    with np.load(directory / "train.npz", allow_pickle=False) as archive:
        original = {key: archive[key] for key in ("tokens", "targets", "mask")}
    packed, report = pack_chunks(original, config["seed"])
    np.savez_compressed(directory / "train.npz", **packed)
    write_json(directory / "packing.json", report)
    manifest = read_json(directory / "manifest.json")
    manifest["packing_version"] = PACKING_VERSION
    for name in ("train.npz", "validation.npz", "packing.json", "train.jsonl", "validation.jsonl"):
        if (directory / name).exists():
            manifest["files"][name] = sha256_file(directory / name)
    write_json(directory / "manifest.json", manifest)
    return report


def prepare_cached(config, work, output):
    started = time.monotonic()
    baseline = read_json(Path(work) / "baseline_v1.json")
    identity = {key: config[key] for key in ("model_id", "model_revision", "dataset_id", "dataset_config",
                "dataset_revision", "train_documents", "validation_documents", "sequence_length", "seed")}
    identity.update(packing=PACKING_VERSION, document_hashes=baseline["split_hashes"])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    local_cache = Path(work).parent / "data-cache" / key
    persistent_cache = Path(output).parent.parent / "V2-cache" / "data" / key
    prepared = Path(work) / "prepared"
    if prepared.exists():
        raise FileExistsError("This run already has prepared data. Continue to training.")
    reused = False
    if local_cache.exists():
        verify_prepared(local_cache, baseline)
        reused = True
    elif persistent_cache.exists():
        verify_prepared(persistent_cache, baseline)
        local_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(persistent_cache, local_cache)
        reused = True
    else:
        local_cache.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(str(local_cache) + ".partial")
        prepare(config, staging)
        finalize_packing(staging, config)
        verify_prepared(staging, baseline)
        staging.rename(local_cache)
        persistent_cache.parent.mkdir(parents=True, exist_ok=True)
        persistent_staging = Path(str(persistent_cache) + ".partial")
        shutil.copytree(local_cache, persistent_staging)
        persistent_staging.rename(persistent_cache)
    shutil.copytree(local_cache, prepared)
    # Runtime timings belong to this run, never the run which populated the cache.
    write_json(prepared / "resolved_config.json", config)
    manifest, packing, _ = verify_prepared(prepared, baseline)
    write_json(output / "preparation.json", {"cache_key": key, "cache_reused": reused,
               "preparation_seconds": time.monotonic() - started,
               "packed_sequences": packing["packed_sequences"], "occupancy": packing["occupancy"]})
    print(f"Prepared {packing['packed_sequences']:,} packed sequences; occupancy={packing['occupancy']:.2%}; cache reused={reused}", flush=True)


def packed_attention_mask(segment_ids):
    length = segment_ids.shape[-1]
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    causal = jnp.arange(length)[:, None] >= jnp.arange(length)[None, :]
    valid = segment_ids >= 0
    allowed = same_segment & causal[None, :, :] & valid[:, :, None] & valid[:, None, :]
    # Padding queries attend to themselves so every softmax has at least one key.
    return allowed | ((~valid)[:, :, None] & jnp.eye(length, dtype=jnp.bool_)[None, :, :])


def loss_parts(model, params, batch):
    compute_params = jax.tree.map(lambda x: x.astype(jnp.bfloat16), params)
    arguments = {"tokens": batch["tokens"]}
    if "segment_ids" in batch:
        arguments.update(positions=batch["positions"], attention_mask=packed_attention_mask(batch["segment_ids"]))
    logits = model.apply({"params": compute_params}, **arguments).logits.astype(jnp.float32)
    selected = jnp.take_along_axis(logits, batch["targets"][..., None], axis=-1)[..., 0]
    losses = jax.scipy.special.logsumexp(logits, axis=-1) - selected
    weights = batch["mask"].astype(jnp.float32)
    return jnp.sum(jnp.where(batch["mask"], losses, 0.0)), jnp.sum(weights)


def batch_training_arrays(arrays, batch_size, seed):
    order = np.random.default_rng(seed).permutation(len(arrays["tokens"]))
    padded = (-len(order)) % batch_size
    result = {}
    for key, values in arrays.items():
        values = values[order]
        if padded:
            values = np.pad(values, ((0, padded), (0, 0)), constant_values=-1 if key == "segment_ids" else 0)
        result[key] = values.reshape((-1, batch_size, values.shape[-1]))
    return result, hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()


def make_training_functions(model, optimizer):
    def one_step(state, batch):
        def objective(params):
            total, count = loss_parts(model, params, batch)
            return total / jnp.maximum(count, 1), count
        (loss, count), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        norm = optax.global_norm(grads)
        finite = jnp.isfinite(loss) & jnp.isfinite(norm) & (count > 0)
        def apply(_):
            updates, new_opt = optimizer.update(grads, state.opt_state, state.params)
            return state.replace(step=state.step + 1, params=optax.apply_updates(state.params, updates),
                                 opt_state=new_opt, data_position=state.data_position + 1)
        state = jax.lax.cond(finite, apply, lambda _: state, operand=None)
        return state, (loss, count, norm, finite)

    def run_group(state, training, requested_steps):
        initial = (state, jnp.array(0, jnp.int32), jnp.array(0., jnp.float32),
                   jnp.array(0, jnp.int32), jnp.array(0., jnp.float32), jnp.array(True))
        def condition(carry):
            return (carry[1] < requested_steps) & carry[5]
        def body(carry):
            state, steps, nll, tokens, max_norm, finite = carry
            index = state.data_position % training["tokens"].shape[0]
            batch = jax.tree.map(lambda values: jax.lax.dynamic_index_in_dim(values, index, keepdims=False), training)
            state, (loss, count, norm, good) = one_step(state, batch)
            return (state, steps + good.astype(jnp.int32), nll + loss * count,
                    tokens + jnp.where(good, count, 0).astype(jnp.int32), jnp.maximum(max_norm, norm), finite & good)
        state, steps, nll, tokens, norm, finite = jax.lax.while_loop(condition, body, initial)
        return state, {"updates": steps, "nll_sum": nll, "tokens": tokens,
                       "max_gradient_norm": norm, "finite": finite}
    return one_step, jax.jit(run_group, donate_argnums=(0,))


def train_pilot(work, output):
    work, output = Path(work), Path(output)
    # An exclusive marker prevents a notebook-cell rerun from paying for another run.
    with (work / "training.started").open("x") as marker:
        marker.write("V2: one training attempt; automatic retry disabled.\n")
    environment = preflight()
    load_model_helpers()
    config = read_json(work / "prepared" / "resolved_config.json")
    baseline = read_json(work / "baseline_v1.json")
    data_dir = work / "prepared"
    manifest, packing, split_hashes = verify_prepared(data_dir, baseline)
    manifest_hash = sha256_file(data_dir / "manifest.json")
    if shutil.disk_usage(work).free < 4 * 1024**3:
        raise RuntimeError("V2 requires at least 4 GiB of free local disk for the full checkpoint.")
    with np.load(data_dir / "train.npz", allow_pickle=False) as archive:
        training_host = {key: archive[key] for key in INPUT_KEYS}
    training_host, order_hash = batch_training_arrays(training_host, config["local_batch_per_chip"], config["seed"])
    with np.load(data_dir / "validation.npz", allow_pickle=False) as archive:
        validation_host = {key: archive[key] for key in ("tokens", "targets", "mask")}
    if int(validation_host["mask"].sum()) != baseline["final_eval"]["scored_tokens"]:
        raise ValueError("Validation target count differs from V1.")
    if int(training_host["mask"].sum()) != packing["scored_tokens"]:
        raise ValueError("Batch construction changed the target count.")
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment)
    transfer_start = time.monotonic()
    training = jax.device_put(training_host)
    validation = jax.device_put(validation_host)
    jax.block_until_ready((training, validation))
    resident_inputs = all(jax.devices()[0] in leaf.devices() for leaf in jax.tree.leaves((training, validation)))
    input_transfer_seconds = time.monotonic() - transfer_start
    input_bytes = sum(x.nbytes for x in jax.tree.leaves(training_host))
    del training_host, validation_host
    print(f"Inputs on TPU: {input_bytes / 1024**2:.1f} MiB of packed training data.", flush=True)
    probe_segments = jnp.array([[0, 0, 1, 1, -1]], jnp.int32)
    probe_mask = np.asarray(jax.jit(packed_attention_mask)(probe_segments))[0]
    expected_mask = np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                              [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]], np.bool_)
    isolated_segments = np.array_equal(probe_mask, expected_mask)
    if not isolated_segments:
        raise ValueError("Packed attention mask failed causal/segment/padding verification.")

    model = make_model()
    initialization_start = time.monotonic()
    params = jax.jit(lambda key: model.init(key, tokens=jnp.ones((1, 1), jnp.int32))["params"])(jax.random.PRNGKey(config["seed"]))
    jax.block_until_ready(params)
    flat = traverse_util.flatten_dict(params)
    parameter_count = sum(x.size for x in flat.values())
    if parameter_count != 268_098_176 or not all(x.dtype == jnp.float32 for x in flat.values()):
        raise ValueError("The native parameter count or FP32 master-weight requirement failed.")
    embedding_shape = (262144, 640)
    tied = (flat[("embedder", "input_embedding")].shape == embedding_shape
            and sum(x.shape == embedding_shape for x in flat.values()) == 1)
    probe_path = next(path for path, value in sorted(flat.items()) if path[0].startswith("layer_") and value.ndim > 1)
    def probe(parameters):
        return np.array(jax.device_get(traverse_util.flatten_dict(parameters)[probe_path].reshape(-1)[:64]))
    initial_probe = probe(params)
    optimizer, schedule = make_optimizer(config, params)
    state = PilotState(jnp.array(0, jnp.int32), params, optimizer.init(params),
                       jax.random.PRNGKey(config["seed"]), jnp.array(0, jnp.int32))
    del flat, params
    jax.block_until_ready(state)
    initialization_seconds = time.monotonic() - initialization_start
    _, group_jit = make_training_functions(model, optimizer)

    def evaluation_all(parameters, arrays):
        def body(index, totals):
            batch = jax.tree.map(lambda value: jax.lax.dynamic_slice_in_dim(value, index, 1), arrays)
            nll, count = loss_parts(model, parameters, batch)
            # Float64 is intentionally unnecessary: report FP32 accumulation explicitly.
            return totals[0] + nll, totals[1] + count
        return jax.lax.fori_loop(0, arrays["tokens"].shape[0], body,
                                (jnp.array(0., jnp.float32), jnp.array(0., jnp.float32)))
    evaluation_jit = jax.jit(evaluation_all)
    print("Compiling one dynamic-count training loop and the validation loop; zero weight updates...", flush=True)
    compilation_start = time.monotonic()
    executable = group_jit.lower(state, training, jnp.array(1, jnp.int32)).compile()
    evaluator = evaluation_jit.lower(state.params, validation).compile()
    compilation_seconds = time.monotonic() - compilation_start
    if int(state.step) != 0:
        raise ValueError("An optimizer update occurred before timing.")
    try:
        analysis = executable.memory_analysis()
        compiled_memory = {key: int(getattr(analysis, key)) for key in
            ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")}
    except (AttributeError, TypeError, NotImplementedError):
        compiled_memory = None

    def evaluate(parameters):
        nll, tokens = jax.block_until_ready(evaluator(parameters, validation))
        nll, tokens = float(nll), int(tokens)
        if tokens != baseline["final_eval"]["scored_tokens"] or not math.isfinite(nll):
            raise ValueError("Invalid validation loss or target count.")
        return {"loss": nll / tokens, "nll_sum": nll, "scored_tokens": tokens,
                "document_hash": split_hashes["validation"], "accumulation_dtype": "float32"}

    print("Initial validation...", flush=True)
    phase = time.monotonic()
    before = evaluate(state.params)
    initial_eval_seconds = time.monotonic() - phase
    jax.block_until_ready(state)
    print("Starting the ONE 60-second V2 training run...", flush=True)
    log_rows = []
    updates = tokens_processed = 0
    max_seconds_per_update = 0.0
    last_progress = last_start = 0.0
    started = time.monotonic()
    while True:
        elapsed = time.monotonic() - started
        remaining = config["training_seconds"] - elapsed
        if remaining <= 0:
            break
        # First dispatch is one timed update. Near the deadline, return to one update.
        group_size = config["updates_per_dispatch"]
        threshold = max(2.0, 2 * group_size * max_seconds_per_update)
        requested = group_size if updates > 0 and remaining > threshold else 1
        last_start = time.monotonic() - started
        if last_start >= config["training_seconds"]:
            break
        call_start = time.monotonic()
        state, metrics = jax.block_until_ready(executable(state, training, jnp.array(requested, jnp.int32)))
        metrics = {key: np.asarray(value).item() for key, value in metrics.items()}
        if not metrics["finite"] or metrics["updates"] != requested:
            raise FloatingPointError("Nonfinite training value; affected update rejected; no retry.")
        completed = int(metrics["updates"])
        updates += completed
        tokens_processed += int(metrics["tokens"])
        duration = time.monotonic() - call_start
        max_seconds_per_update = max(max_seconds_per_update, duration / completed)
        end_seconds = time.monotonic() - started
        log_rows.append({**metrics, "update": updates, "requested_updates": requested,
                         "start_seconds": last_start, "end_seconds": end_seconds,
                         "loss": metrics["nll_sum"] / metrics["tokens"]})
        if len(log_rows) == 1 or end_seconds - last_progress >= 2:
            print(f"updates={updates:,} tokens={tokens_processed:,} loss={log_rows[-1]['loss']:.5f} elapsed={end_seconds:.2f}s", flush=True)
            last_progress = end_seconds
    jax.block_until_ready(state)
    training_seconds = time.monotonic() - started
    # Disk writes happen after timing; no per-update transfers or log flushes.
    with (output / "training.jsonl").open("w") as log:
        for row in log_rows:
            log.write(json.dumps(row, allow_nan=False) + "\n")
    changed = not np.array_equal(initial_probe, probe(state.params))
    if int(state.step) != updates or int(state.data_position) != updates:
        raise ValueError("Optimizer count and data cursor differ from completed updates.")
    phase = time.monotonic()
    after = evaluate(state.params)
    final_eval_seconds = time.monotonic() - phase
    memory_stats = jax.devices()[0].memory_stats()
    checkpoint_start = time.monotonic()
    metadata = {"version": "V2", "config": config, "parameter_count": parameter_count,
                "model_initialization": "random", "model_implementation": "gemma==4.0.1; gm.nn.Gemma3_270M",
                "data_manifest_sha256": manifest_hash, "packing_version": PACKING_VERSION,
                "data_order_sha256": order_hash, "data_batches": training["tokens"].shape[0],
                "data_position_unit": "completed batches; fixed seeded order repeats after a complete pass",
                "updates": updates, "checkpoint_fields": ["step", "params", "opt_state", "rng", "data_position"]}
    print("Saving one full resumable checkpoint (approximately 3.2 GB)...", flush=True)
    local_checkpoint = work / "checkpoint.npz"
    leaves = save_archive(local_checkpoint, state, metadata)
    digest = sha256_file(local_checkpoint)
    checkpoint = output / "checkpoint.npz"
    partial = output / "checkpoint.npz.partial"
    shutil.copyfile(local_checkpoint, partial)
    if sha256_file(partial) != digest:
        raise ValueError("Persistent checkpoint checksum differs from the local file.")
    partial.rename(checkpoint)
    with zipfile.ZipFile(checkpoint) as archive:
        saved = json.loads(archive.read("checkpoint.json"))
        tree = saved["tree"]
        if set(tree) != {"step", "params", "opt_state", "rng", "data_position"} or saved["metadata"] != metadata:
            raise ValueError("Checkpoint fields or metadata differ from the final state.")
        if sum(name.startswith("arrays/") for name in archive.namelist()) != leaves:
            raise ValueError("Checkpoint array count differs.")
    write_json(output / "checkpoint_manifest.json", {"sha256": digest, "bytes": checkpoint.stat().st_size,
                "array_leaves": leaves, "verification": "local/persistent SHA256 equality and state-schema inspection"})
    checkpoint_seconds = time.monotonic() - checkpoint_start
    for name in ("manifest.json", "packing.json"):
        shutil.copyfile(data_dir / name, output / ("data_manifest.json" if name == "manifest.json" else name))
    import gzip
    with gzip.open(output / "source_documents.jsonl.gz", "wb") as stream:
        for name in ("validation.jsonl", "train.jsonl"):
            with (data_dir / name).open("rb") as source:
                shutil.copyfileobj(source, stream)
    runtime_elapsed = time.monotonic() - config["notebook_start_monotonic"]
    rate = config["compute_units_per_hour"]
    throughput = tokens_processed / training_seconds
    occupancy = tokens_processed / (updates * config["local_batch_per_chip"] * config["sequence_length"])
    checks = {
        "V2-001_ONE_RUN": True,
        "V2-002_ONE_V5E": environment["platform"] == "tpu" and environment["device_count"] == 1,
        "V2-003_TARGETS_PRESERVED": packing["targets_preserved"],
        "V2-004_ISOLATED_SEGMENTS": isolated_segments,
        "V2-005_CONTEXT_PRESERVED": packing["context_preserved"],
        "V2-006_RESIDENT_INPUTS": resident_inputs,
        "V2-007_GROUP_SIZE": all(1 <= row["requested_updates"] <= 16 for row in log_rows),
        "V2-008_TIMING": training_seconds >= config["training_seconds"] and last_start < config["training_seconds"],
        "V2-009_FINITE": all(row["finite"] for row in log_rows),
        "V2-010_VALIDATION": before["document_hash"] == after["document_hash"] and before["scored_tokens"] == after["scored_tokens"],
        "V2-011_CHECKPOINT": checkpoint.exists(),
        "V2-012_TIMINGS": all(value >= 0 for value in (initialization_seconds, compilation_seconds,
                                  input_transfer_seconds, initial_eval_seconds, final_eval_seconds, checkpoint_seconds)),
        "V2-013_NATIVE_MODEL": parameter_count == 268_098_176 and tied,
        "V2-014_ZERO_UNTIMED_UPDATES": int(state.step) == updates,
        "V2-015_TRANSFORMER_CHANGED": changed,
        "V2-016_MINIMUM_UPDATES": updates >= config["minimum_updates"],
    }
    result = {"version": "V2", "cpu_test": False, "run_count": 1, "config": config,
        "device": environment["device_kind"], "model_parameters": parameter_count,
        "input_transfer_seconds": input_transfer_seconds, "resident_training_bytes": input_bytes,
        "initialization_seconds": initialization_seconds, "compilation_seconds": compilation_seconds,
        "initial_eval_seconds": initial_eval_seconds, "final_eval_seconds": final_eval_seconds,
        "training_seconds": training_seconds, "training_overrun_seconds": training_seconds - config["training_seconds"],
        "last_dispatch_start_seconds": last_start, "host_dispatches": len(log_rows),
        "updates": updates, "scored_tokens": tokens_processed, "tokens_per_second": throughput,
        "training_occupancy": occupancy, "prepared_occupancy": packing["occupancy"],
        "data_passes": updates // training["tokens"].shape[0], "data_manifest_sha256": manifest_hash,
        "initial_eval": before, "final_eval": after, "final_learning_rate": float(schedule(state.step)),
        "checkpoint_save_verify_seconds": checkpoint_seconds, "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": digest, "checkpoint_array_leaves": leaves,
        "memory_stats": memory_stats, "compiled_memory_estimate": compiled_memory,
        "memory_scope": "JAX allocator counters sampled after final validation; compiler estimate reported separately.",
        "notebook_elapsed_seconds": runtime_elapsed,
        "compute_units_estimate": {"rate_per_hour": rate, "training": training_seconds * rate / 3600,
             "through_report": runtime_elapsed * rate / 3600, "overhead_through_report": (runtime_elapsed - training_seconds) * rate / 3600},
        "cost_note": "Estimates at the user-supplied 2.92 CU/hour; Colab billing is authoritative. Runtime after report creation is additional.",
        "v1_comparison": {"tokens_per_second": baseline["tokens_per_second"],
             "speed_ratio": throughput / baseline["tokens_per_second"], "final_loss": baseline["final_eval"]["loss"],
             "final_loss_difference": after["loss"] - baseline["final_eval"]["loss"], "source": baseline["source"]},
        "performance_goals": {"occupancy_at_least_95_percent": occupancy >= .95,
             "tokens_per_second_at_least_18000": throughput >= 18000,
             "final_loss_at_most_v1": after["loss"] <= baseline["final_eval"]["loss"]},
        "requirements": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "engineering_status": "PASS" if all(checks.values()) else "FAIL",
        "learning_status": "PASS" if after["loss"] < before["loss"] else "FAIL"}
    write_json(output / "result.json", result)
    print(json.dumps({key: result[key] for key in ("engineering_status", "learning_status", "updates", "training_seconds", "tokens_per_second", "v1_comparison")}, indent=2), flush=True)
    if not all(checks.values()):
        raise RuntimeError("V2 engineering requirement failed. Saved result.json contains the details.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["preflight", "prepare", "train"], required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "preflight":
            info = preflight()
            write_json(args.output / "preflight.json", info)
            print(json.dumps({key: info[key] for key in ("platform", "device_kind", "device_count", "jax")}, indent=2))
        elif args.phase == "prepare":
            prepare_cached(read_json(args.work / "config.json"), args.work, args.output)
        else:
            cache = Path(args.work).parent / "jax-cache-V2"
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache)
            os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "1"
            try:
                train_pilot(args.work, args.output)
            finally:
                # Persist cache once after execution; all live compilation uses local storage.
                persistent = args.output.parent.parent / "V2-cache" / "jax"
                try:
                    persistent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(cache, persistent, dirs_exist_ok=True)
                except OSError as error:
                    write_json(args.output / "cache_warning.json", {"warning": str(error), "training_retried": False})
    except Exception:
        write_json(args.output / "failure.json", {"phase": args.phase, "error": traceback.format_exc(), "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()
