"""Gemma Colab V3: one fixed multilingual tokenizer-screening pilot on one v5e TPU.
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


def packed_attention_mask(segment_ids):
    length = segment_ids.shape[-1]
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    causal = jnp.arange(length)[:, None] >= jnp.arange(length)[None, :]
    valid = segment_ids >= 0
    allowed = same_segment & causal[None, :, :] & valid[:, :, None] & valid[:, None, :]
    # Padding queries attend to themselves so every softmax has at least one key.
    return allowed | ((~valid)[:, :, None] & jnp.eye(length, dtype=jnp.bool_)[None, :, :])


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



"""V3 extensions, concatenated with base.py by build_notebook.py; no local execution."""
import functools
from collections import Counter, defaultdict
from urllib.parse import urlsplit, urlunsplit

PROTOCOL = "V3-static-byte-mix-1"
PACKING_VERSION = "V3-intact-chunks-1"
LANGUAGES = ["eng_Latn", "fra_Latn", "spa_Latn", "por_Latn", "cmn_Hani", "arb_Arab",
             "hin_Deva", "swh_Latn", "zul_Latn", "xho_Latn", "tsn_Latn", "sot_Latn"]
WIKI_CODES = ["en", "fr", "es", "pt", "zh", "ar", "hi", "sw", "zu", "xh", "tn", "st"]
DOMAINS = ["pdf_edu", "dclm", "web_edu", "multilingual_web", "math", "code", "translations", "wiki_holdout"]
SMOL = "HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled"
SOURCE_REPOS = [SMOL, "HuggingFaceFW/fineweb-2", "HuggingFaceTB/finemath",
                "codeparrot/codeparrot-clean", "HuggingFaceFW/finetranslations", "HuggingFaceFW/finewiki"]
INPUT_KEYS = ("tokens", "targets", "mask", "positions", "segment_ids", "text_mask", "byte_weights", "language_ids", "domain_ids")
EXTRA_KEYS = ("text_mask", "byte_weights", "language_ids", "domain_ids")


def recipe():
    # Basis points of original UTF-8 bytes, not tokenizer-dependent token counts.
    multilingual = [10, 10, 8, 15, 15, 15, 8, 5, 5, 5, 4]
    buckets = [dict(key="pdf_edu", repo=SMOL, subset=None, language="eng_Latn", domain="pdf_edu", bp=2750),
               dict(key="dclm", repo=SMOL, subset=None, language="eng_Latn", domain="dclm", bp=1650),
               dict(key="web_edu", repo=SMOL, subset=None, language="eng_Latn", domain="web_edu", bp=1100)]
    buckets += [dict(key="fw2_" + lang, repo=SOURCE_REPOS[1], subset=lang, language=lang,
                     domain="multilingual_web", bp=30 * share) for lang, share in zip(LANGUAGES[1:], multilingual)]
    buckets += [dict(key="math", repo=SOURCE_REPOS[2], subset="finemath-3plus", language="eng_Latn", domain="math", bp=500),
                dict(key="code", repo=SOURCE_REPOS[3], subset=None, language="eng_Latn", domain="code", bp=500)]
    buckets += [dict(key="translations_" + lang, repo=SOURCE_REPOS[4], subset=lang, language=lang,
                     domain="translations", bp=125) for lang in ("fra_Latn", "arb_Arab", "hin_Deva", "swh_Latn")]
    return {"protocol": PROTOCOL, "training_bytes_target": 64 * 1024**2,
            "in_mix_eval_bytes_target": 256 * 1024, "wiki_eval_bytes_per_language": 64 * 1024,
            "max_document_span_bytes": 32 * 1024, "byte_quota_tolerance": 3,
            "source_weight_unit": "original UTF-8 bytes", "buckets": buckets,
            "split_rule": "SHA256(source group) modulo 10: 0 held out, 1..9 train",
            "sampling_scope": "bounded deterministic shuffled stream prefixes; not a uniform full-corpus sample"}


def protocol_fingerprint():
    return hashlib.sha256(json.dumps(recipe(), sort_keys=True).encode()).hexdigest()


def canonical_group(row, text):
    metadata = row.get("metadata")
    url = row.get("url") or (metadata.get("url") if isinstance(metadata, dict) else None)
    if url:
        parsed = urlsplit(str(url))
        return "url:" + urlunsplit(("", parsed.netloc.lower().removeprefix("www."), parsed.path.rstrip("/"), parsed.query, ""))
    if row.get("repo_name"):
        return "code-repo:" + str(row["repo_name"])
    return "id:" + str(row.get("id") or row.get("blob_id") or hashlib.sha256(text.encode()).hexdigest())


def utf8_prefix(text, limit):
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def immutable_sources(shared):
    from huggingface_hub import HfApi
    path = shared / "source-lock.json"
    if path.exists():
        return read_json(path)
    api = HfApi()
    lock = {repo: api.dataset_info(repo, revision="main").sha for repo in SOURCE_REPOS}
    write_json(path, lock)
    return lock


def verify_raw(directory):
    manifest = read_json(directory / "manifest.json")
    if manifest["recipe_sha256"] != protocol_fingerprint():
        raise ValueError("The shared raw corpus belongs to another V3 protocol.")
    for name, digest in manifest["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Raw corpus checksum differs: {name}")
    return manifest


def build_raw_corpus(destination, lock):
    from datasets import load_dataset, get_dataset_config_names
    destination.mkdir(parents=True, exist_ok=False)
    rules = recipe()
    seen_hashes = set()
    forbidden_groups = set()
    docs = {"train": [], "in_mix": [], "wiki": []}
    counts, rejected = Counter(), Counter()
    summaries = {}

    def add(split, row, text, language, domain, bucket, remaining, forced_group=None):
        if not isinstance(text, str) or not text.strip() or remaining <= rules["byte_quota_tolerance"]:
            return 0
        group = forced_group or canonical_group(row, text)
        if split == "train" and group in forbidden_groups:
            rejected["reserved_evaluation_group"] += 1
            return 0
        if split != "wiki" and "wikipedia.org" in group:
            rejected["wikipedia_url_in_training_sources"] += 1
            return 0
        full_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # UTF-8 clipping preserves the selected prefix exactly; no normalization/cleanup.
        span = utf8_prefix(text, min(remaining, rules["max_document_span_bytes"]))
        if not span:
            return 0
        digest = hashlib.sha256(span.encode("utf-8")).hexdigest()
        if full_digest in seen_hashes or digest in seen_hashes:
            rejected["exact_duplicate"] += 1
            return 0
        seen_hashes.update((full_digest, digest))
        if split != "train":
            forbidden_groups.add(group)
        source = {key: row.get(key) for key in ("id", "url", "repo_name", "path", "license", "dataset") if row.get(key) is not None}
        docs[split].append({"text": span, "sha256": digest, "original_document_sha256": full_digest,
            "source_group": group, "source": source, "language": language, "domain": domain,
            "bucket": bucket, "utf8_bytes": len(span.encode("utf-8")), "source_byte_start": 0,
            "source_byte_end": len(span.encode("utf-8")), "clipped": span != text})
        return len(span.encode("utf-8"))

    # Reserve FineWiki first; it is never a training source.
    wiki_repo = SOURCE_REPOS[-1]
    names = set(get_dataset_config_names(wiki_repo, revision=lock[wiki_repo]))
    for language, code in zip(LANGUAGES, WIKI_CODES):
        subset = next((candidate for candidate in (code, code + "wiki") if candidate in names), None)
        if subset is None:
            raise ValueError(f"Pinned FineWiki has no configuration for {language}.")
        quota = rules["wiki_eval_bytes_per_language"]
        collected = 0
        for index, row in enumerate(load_dataset(wiki_repo, name=subset, split="train", streaming=True, revision=lock[wiki_repo])):
            if index >= 100000:
                break
            collected += add("wiki", row, row.get("text"), language, "wiki_holdout", "wiki_" + language, quota - collected)
            if quota - collected <= rules["byte_quota_tolerance"]:
                break
        if quota - collected > rules["byte_quota_tolerance"]:
            raise RuntimeError(f"FineWiki quota shortfall for {language}: {collected}/{quota} bytes.")
        summaries["wiki_" + language] = {"bytes": collected, "subset": subset}
        print(f"Reserved FineWiki {language}: {collected:,} bytes", flush=True)

    # One iterator for all three parts of the pre-shuffled English source.
    groups = defaultdict(list)
    for bucket in rules["buckets"]:
        groups[(bucket["repo"], bucket["subset"])].append(bucket)
    for (repo, subset), buckets in groups.items():
        targets = {b["key"]: {"train": round(rules["training_bytes_target"] * b["bp"] / 10000),
                                  "in_mix": round(rules["in_mix_eval_bytes_target"] * b["bp"] / 10000)} for b in buckets}
        actual = {key: {"train": 0, "in_mix": 0} for key in targets}
        stream = load_dataset(repo, name=subset, split="train", streaming=True, revision=lock[repo])
        for index, row in enumerate(stream):
            if index >= 300000:
                break
            if repo == SMOL:
                origin = str(row.get("dataset", ""))
                key = "pdf_edu" if "finepdfs" in origin else "web_edu" if "fineweb-edu" in origin else "dclm" if "dclm" in origin else None
                if key is None:
                    raise ValueError(f"Unknown upstream Smol-Data source label: {origin}")
                bucket = next(b for b in buckets if b["key"] == key)
            else:
                bucket = buckets[0]
                key = bucket["key"]
            if row.get("early_stop") or row.get("is_truncated"):
                rejected["source_early_stop_or_truncation"] += 1
                continue
            if repo == SOURCE_REPOS[4]:
                if int(row.get("edu_score", -1)) < 2:
                    rejected["translation_edu_below_2"] += 1
                    continue
                texts = [(row.get("og_full_text"), bucket["language"]), (row.get("translated_text"), "eng_Latn")]
            else:
                texts = [(row.get("content") if bucket["domain"] == "code" else row.get("text"), bucket["language"])]
            group = canonical_group(row, texts[0][0] or "")
            split = "in_mix" if int(hashlib.sha256(group.encode()).hexdigest()[:16], 16) % 10 == 0 else "train"
            for text, language in texts:
                remaining = targets[key][split] - actual[key][split]
                actual[key][split] += add(split, row, text, language, bucket["domain"], key, remaining, group)
            if all(targets[k][s] - actual[k][s] <= rules["byte_quota_tolerance"] for k in targets for s in ("train", "in_mix")):
                break
        for key in targets:
            for split in ("train", "in_mix"):
                if targets[key][split] - actual[key][split] > rules["byte_quota_tolerance"]:
                    raise RuntimeError(f"Data quota shortfall {key}/{split}: {actual[key][split]}/{targets[key][split]}. No source substitution.")
            summaries[key] = {"targets": targets[key], "actual": actual[key], "repository": repo, "subset": subset}
        print(f"Collected source {repo} / {subset}: {actual}", flush=True)

    # Deficit scheduling interleaves the whole spans in a fixed byte-weighted order.
    by_bucket = defaultdict(list)
    for doc in docs["train"]:
        by_bucket[doc["bucket"]].append(doc)
    rng = np.random.default_rng(42)
    weights = {b["key"]: b["bp"] / 10000 for b in rules["buckets"]}
    for rows in by_bucket.values():
        rng.shuffle(rows)
    used = Counter()
    interleaved = []
    while any(by_bucket.values()):
        key = min((k for k, rows in by_bucket.items() if rows), key=lambda k: used[k] / weights[k])
        doc = by_bucket[key].pop()
        interleaved.append(doc)
        used[key] += doc["utf8_bytes"]
    docs["train"] = interleaved
    for split, rows in docs.items():
        with (destination / (split + ".jsonl")).open("w", encoding="utf-8") as output:
            for doc in rows:
                output.write(json.dumps(doc, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
    train_groups = {d["source_group"] for d in docs["train"]}
    eval_groups = {d["source_group"] for s in ("in_mix", "wiki") for d in docs[s]}
    if train_groups & eval_groups:
        raise ValueError("Source group overlaps training and evaluation.")
    manifest = {"protocol": PROTOCOL, "recipe_sha256": protocol_fingerprint(), "source_revisions": lock,
        "recipe": rules, "buckets": summaries, "rejected": dict(rejected), "document_counts": dict(counts),
        "original_bytes": {s: sum(d["utf8_bytes"] for d in rows) for s, rows in docs.items()},
        "language_bytes": {s: dict(Counter({lang: sum(d["utf8_bytes"] for d in rows if d["language"] == lang) for lang in LANGUAGES})) for s, rows in docs.items()},
        "cross_split_source_groups": 0, "deduplication_scope": "global exact source/span hashes and source groups; near duplicates are not exhaustively removed",
        "files": {s + ".jsonl": sha256_file(destination / (s + ".jsonl")) for s in docs}}
    write_json(destination / "manifest.json", manifest)
    return manifest


def prepare_tokenizer(config, destination):
    from tokenizers import Tokenizer
    from huggingface_hub import HfApi, snapshot_download
    destination.mkdir(parents=True, exist_ok=True)
    reference = config["tokenizer_reference"]
    if Path(reference).is_dir():
        for name in ("tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json"):
            if (Path(reference) / name).exists():
                shutil.copyfile(Path(reference) / name, destination / name)
        revision = None
    else:
        revision = HfApi().model_info(reference, revision=config["tokenizer_revision"]).sha
        snapshot_download(reference, revision=revision, local_dir=destination,
                          allow_patterns=["tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json"])
    tokenizer = Tokenizer.from_file(str(destination / "tokenizer.json"))
    ids = sorted(set(tokenizer.get_vocab(with_added_tokens=True).values()))
    # Every tokenizer ID is mapped to a positive dense model ID; model padding is always 0.
    mapping = {old: index + 1 for index, old in enumerate(ids)}
    rows = len(ids) + 3
    total = 100326016 + rows * 640
    if total >= 300000000:
        raise ValueError(f"Tokenizer requires {rows} rows and {total:,} parameters, exceeding the strict <300M class.")
    info = {"reference": reference, "revision": revision, "tokenizer_sha256": sha256_file(destination / "tokenizer.json"),
        "vocabulary_ids": len(ids), "original_id_space": max(ids) + 1, "model_rows": rows,
        "pad_id": 0, "bos_id": len(ids) + 1, "eos_id": len(ids) + 2,
        "id_mapping": "sorted native IDs map to 1..N; PAD=0, BOS=N+1, EOS=N+2",
        "expected_parameters": total, "model_type": json.loads((destination / "tokenizer.json").read_text())["model"]["type"]}
    write_json(destination / "identity.json", info)
    write_json(destination / "native_ids_in_dense_order.json", ids)
    return tokenizer, mapping, info


def tokenize_split(documents, tokenizer, mapping, identity, sequence_length):
    arrays = defaultdict(list)
    text_token_count = 0
    for doc in documents:
        encoded = tokenizer.encode(doc["text"], add_special_tokens=False)
        if tokenizer.decode(encoded.ids, skip_special_tokens=False).encode("utf-8") != doc["text"].encode("utf-8"):
            raise ValueError(f"Tokenizer is not byte-lossless on fixed input {doc['sha256']} ({doc['language']}); corpus was not modified.")
        ids = [mapping[index] for index in encoded.ids]
        if not ids:
            raise ValueError("Nonempty source text produced zero tokens.")
        # Attribute each original byte once using the monotonic original-character offset frontier.
        char_bytes = np.concatenate(([0], np.cumsum([len(c.encode("utf-8")) for c in doc["text"]])))
        frontier = np.maximum.accumulate(np.array([0] + [end for _, end in encoded.offsets], np.int64))
        frontier = np.clip(frontier, 0, len(doc["text"]))
        frontier[-1] = len(doc["text"])
        byte_weights = np.diff(char_bytes[frontier]).astype(np.int32)
        sequence = [identity["bos_id"], *ids, identity["eos_id"]]
        weights = np.concatenate((byte_weights, [0])).astype(np.int32)
        for start in range(0, len(sequence) - 1, sequence_length):
            window = sequence[start:start + sequence_length + 1]
            size = len(window) - 1
            for key, values, dtype, padding in (
                ("tokens", window[:-1], np.int32, 0), ("targets", window[1:], np.int32, 0),
                ("mask", [True] * size, np.bool_, False),
                ("text_mask", [start + k < len(ids) for k in range(size)], np.bool_, False),
                ("byte_weights", weights[start:start + size], np.int32, 0),
                ("language_ids", [LANGUAGES.index(doc["language"])] * size, np.int32, 0),
                ("domain_ids", [DOMAINS.index(doc["domain"])] * size, np.int32, 0)):
                row = np.full(sequence_length, padding, dtype)
                row[:size] = values
                arrays[key].append(row)
        text_token_count += len(ids)
    result = {key: np.stack(value) for key, value in arrays.items()}
    if int(result["byte_weights"].sum()) != sum(doc["utf8_bytes"] for doc in documents):
        raise ValueError("Original-byte attribution failed.")
    packed, packing = pack_chunks(result, 42)
    for key in EXTRA_KEYS:
        packed[key] = np.zeros_like(packed["tokens"], dtype=result[key].dtype)
        for source, target, start, length in packing["assignments_source_bin_start_length"]:
            packed[key][target, start:start + length] = result[key][source, :length]
    packing["text_tokens"] = text_token_count
    packing["original_bytes"] = int(packed["byte_weights"].sum())
    return packed, packing


def prepare_cached(config, work, output):
    started = time.monotonic()
    work, output = Path(work), Path(output)
    shared = output.parent.parent / "V3-shared" / protocol_fingerprint()[:16]
    shared.mkdir(parents=True, exist_ok=True)
    lock = immutable_sources(shared)
    raw = shared / "raw"
    if not raw.exists():
        staging = shared / "raw.partial"
        build_raw_corpus(staging, lock)
        staging.rename(raw)
        raw_reused = False
    else:
        raw_reused = True
    raw_manifest = verify_raw(raw)
    prepared = work / "prepared"
    prepared.mkdir(exist_ok=False)
    tokenizer, mapping, identity = prepare_tokenizer(config, prepared / "tokenizer")
    token_key = hashlib.sha256((identity["tokenizer_sha256"] + sha256_file(raw / "manifest.json") + PACKING_VERSION).encode()).hexdigest()[:24]
    token_cache = shared / "tokenized" / token_key
    if token_cache.exists():
        cached = read_json(token_cache / "tokenization.json")
        for name, digest in cached["files"].items():
            if sha256_file(token_cache / name) != digest:
                raise ValueError("Cached tokenization checksum mismatch.")
            shutil.copyfile(token_cache / name, prepared / name)
        shutil.copyfile(token_cache / "tokenization.json", prepared / "tokenization.json")
        token_reused = True
    else:
        details = {"tokenizer": identity, "raw_manifest_sha256": sha256_file(raw / "manifest.json"), "splits": {}, "files": {}}
        for split in ("train", "in_mix", "wiki"):
            arrays, packing = tokenize_split(read_jsonl(raw / (split + ".jsonl")), tokenizer, mapping, identity, config["sequence_length"])
            name = split + ".npz"
            np.savez_compressed(prepared / name, **arrays)
            details["splits"][split] = {key: value for key, value in packing.items() if not key.startswith("assignments_")}
            details["files"][name] = sha256_file(prepared / name)
            print(f"Tokenized {split}: {packing['text_tokens']:,} text targets; occupancy={packing['occupancy']:.2%}", flush=True)
        write_json(prepared / "tokenization.json", details)
        token_cache.parent.mkdir(parents=True, exist_ok=True)
        token_cache.mkdir(exist_ok=False)
        for name in [*details["files"], "tokenization.json"]:
            shutil.copyfile(prepared / name, token_cache / name)
        token_reused = False
    config = {**config, "tokenizer_identity": identity, "shared_directory": str(shared)}
    write_json(prepared / "resolved_config.json", config)
    shutil.copyfile(raw / "manifest.json", prepared / "manifest.json")
    for source, destination in ((prepared / "manifest.json", output / "data_manifest.json"),
                                (prepared / "tokenization.json", output / "tokenization.json"),
                                (prepared / "tokenizer/identity.json", output / "tokenizer_identity.json"),
                                (shared / "source-lock.json", output / "source-lock.json")):
        shutil.copyfile(source, destination)
    shutil.copytree(prepared / "tokenizer", output / "tokenizer")
    write_json(output / "preparation.json", {"raw_cache_reused": raw_reused, "token_cache_reused": token_reused,
        "preparation_seconds": time.monotonic() - started, "raw_manifest_sha256": sha256_file(raw / "manifest.json"),
        "tokenizer_sha256": identity["tokenizer_sha256"], "shared_directory": str(shared)})


def load_model_helpers(cpu_test=False):
    accelerator_imports(cpu_test)
    # Retain the serializable state type; V3 defines its own model and optimizer.
    exec(MODEL_HELPERS[:MODEL_HELPERS.index("\n\ndef make_model")], globals())


def make_model(config):
    from gemma import gm
    model = gm.nn.Gemma3_270M(dtype=jnp.float32)
    return model.clone(config=dataclasses.replace(model.config, num_embed=config["tokenizer_identity"]["model_rows"]))


def make_optimizer(config, params):
    def warmup(rate):
        return optax.join_schedules([optax.linear_schedule(0., rate, 5), optax.constant_schedule(rate)], [5])
    specs = {}
    muon_count = adam_count = 0
    for path, value in traverse_util.flatten_dict(params).items():
        name = "/".join(path)
        spec = None
        if path[0].startswith("layer_") and value.ndim >= 2:
            # Matrix axes follow the native Gemma einsum equations; remaining axes are batches.
            if "q_einsum" in name:
                spec = optax.contrib.MuonDimensionNumbers(reduction_axis=1, output_axis=(0, 2))
            elif "kv_einsum" in name:
                spec = optax.contrib.MuonDimensionNumbers(reduction_axis=2, output_axis=(1, 3))
            elif "attn_vec_einsum" in name:
                spec = optax.contrib.MuonDimensionNumbers(reduction_axis=(0, 1), output_axis=2)
            elif path[-1] == "gating_einsum":
                spec = optax.contrib.MuonDimensionNumbers(reduction_axis=2, output_axis=1)
            elif value.ndim == 2:
                spec = optax.contrib.MuonDimensionNumbers(reduction_axis=0, output_axis=1)
            else:
                raise ValueError(f"Unclassified transformer projection for Muon: {name} {value.shape}")
        specs[path] = spec
        if spec is None:
            adam_count += value.size
        else:
            muon_count += value.size
    if muon_count != 100270080:
        raise ValueError(f"Muon partition has {muon_count:,} parameters; expected 100,270,080.")
    decay = jax.tree.map(lambda value: value.ndim > 1, params)
    transform = optax.contrib.muon(warmup(config["muon_learning_rate"]), beta=.95, ns_steps=5,
        nesterov=True, mu_dtype=jnp.float32, weight_decay=.01, weight_decay_mask=decay,
        adam_learning_rate=warmup(config["adam_learning_rate"]), adam_b1=.9, adam_b2=.999,
        adam_weight_decay=.01, muon_weight_dimension_numbers=traverse_util.unflatten_dict(specs),
        consistent_rms=None)
    return optax.chain(optax.clip_by_global_norm(1.), transform), {"muon_parameters": muon_count,
        "adamw_parameters": adam_count, "muon_lr": config["muon_learning_rate"],
        "adamw_lr": config["adam_learning_rate"], "shape_scaling": "Optax width scaling; consistent_rms=None"}


def install_splash_attention():
    from gemma.gm.nn import _modules
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm
    native_attention = _modules.Attention

    @functools.lru_cache(maxsize=8)
    def attention_kernel(length, heads, local):
        mask = sm.CausalMask((length, length))
        if local:
            mask = mask & sm.LocalMask((length, length), window_size=(511, 0), offset=0)
        blocks = sk.BlockSizes(block_q=128, block_kv=128, block_kv_compute=128,
                              block_q_dq=128, block_kv_dq=128, block_q_dkv=128,
                              block_kv_dkv=128, block_kv_dkv_compute=128,
                              use_fused_bwd_kernel=False)
        return sk.make_splash_mqa(sm.MultiHeadMask([mask] * heads), head_shards=1, q_seq_shards=1,
                                 block_sizes=blocks)

    class SplashAttention(native_attention):
        def __call__(self, x, segment_pos, cache, attn_mask):
            if cache is not None or self.num_kv_heads != 1 or self.attn_logits_soft_cap is not None:
                raise ValueError("V3 Splash path requires cache-free native Gemma 270M MQA.")
            query = self.q_einsum("BTD,NDH->BTNH", x)
            key, value = self.kv_einsum("BSD,CKDH->CBSKH", x)
            if self.use_qk_norm:
                query, key = self._query_norm(query), self._key_norm(key)
            query = _modules._positional_embeddings.apply_rope(query, segment_pos,
                base_frequency=self.rope_base_frequency, scale_factor=self.rope_scale_factor)
            key = _modules._positional_embeddings.apply_rope(key, segment_pos,
                base_frequency=self.rope_base_frequency, scale_factor=self.rope_scale_factor)
            query = (query * self.query_pre_attn_scalar).transpose(0, 2, 1, 3)
            # First permitted causal key identifies each intact packed segment.
            ids = jnp.argmax(attn_mask, axis=-1).astype(jnp.int32)
            kernel = attention_kernel(x.shape[1], self.num_heads, self.attn_type == _modules.AttentionType.LOCAL_SLIDING)
            encoded = jax.vmap(lambda q, k, v, s: kernel(q, k, v, sk.SegmentIds(s, s)))(
                query, key[:, :, 0, :], value[:, :, 0, :], ids)
            output = self.attn_vec_einsum("BTNH,NHD->BTD", encoded.transpose(0, 2, 1, 3))
            return None, output
    _modules.Attention = SplashAttention
    return attention_kernel


def hidden_only(model, tokens, positions, attention_mask):
    # Same native blocks and parameter paths; the huge decode tensor is never requested.
    x = model.embedder.encode(tokens)
    for block in model.blocks:
        _, x = block(x, positions, None, attention_mask)
    return model.final_norm(x)


def token_losses(model, params, batch):
    import tokamax
    compute = jax.tree.map(lambda value: value.astype(jnp.bfloat16), params)
    hidden = model.apply({"params": compute}, batch["tokens"], batch["positions"],
                         packed_attention_mask(batch["segment_ids"]), method=hidden_only)
    weights = compute["embedder"]["input_embedding"].T
    return tokamax.linear_softmax_cross_entropy_loss(hidden.reshape((-1, 640)),
        batch["targets"].reshape(-1), weights, reduction="none", implementation="mosaic_tpu").reshape(batch["targets"].shape).astype(jnp.float32)


def loss_parts(model, params, batch):
    losses = token_losses(model, params, batch)
    return jnp.sum(jnp.where(batch["mask"], losses, 0.)), jnp.sum(batch["mask"], dtype=jnp.float32)


def make_training_functions(model, optimizer):
    def one_step(state, batch):
        def objective(params):
            total, count = loss_parts(model, params, batch)
            return total / jnp.maximum(count, 1), count
        (loss, count), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        norm = optax.global_norm(grads)
        good = jnp.isfinite(loss) & jnp.isfinite(norm) & (count > 0)
        def apply(_):
            updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
            return state.replace(step=state.step + 1, params=optax.apply_updates(state.params, updates),
                                 opt_state=opt_state, data_position=state.data_position + 1)
        updated = jax.lax.cond(good, apply, lambda _: state, None)
        return updated, (loss, count, norm, good)

    def group(state, training, requested):
        initial = (state, jnp.array(0, jnp.int32), jnp.array(0., jnp.float32), jnp.array(0, jnp.int32),
                   jnp.array(0., jnp.float32), jnp.array(True), jnp.zeros(len(DOMAINS), jnp.int32),
                   jnp.zeros(len(LANGUAGES), jnp.int32))
        def condition(carry):
            return (carry[1] < requested) & carry[5]
        def body(carry):
            state, steps, nll, count, max_norm, finite, domain_bytes, language_bytes = carry
            index = state.data_position % training["tokens"].shape[0]
            batch = jax.tree.map(lambda values: jax.lax.dynamic_index_in_dim(values, index, keepdims=False), training)
            state, (loss, tokens, norm, good) = one_step(state, batch)
            weights = jnp.where(good, batch["byte_weights"], 0).reshape(-1)
            db = jnp.bincount(batch["domain_ids"].reshape(-1), weights=weights, length=len(DOMAINS)).astype(jnp.int32)
            lb = jnp.bincount(batch["language_ids"].reshape(-1), weights=weights, length=len(LANGUAGES)).astype(jnp.int32)
            return (state, steps + good.astype(jnp.int32), nll + loss * tokens,
                    count + jnp.where(good, tokens, 0).astype(jnp.int32), jnp.maximum(max_norm, norm),
                    finite & good, domain_bytes + db, language_bytes + lb)
        state, steps, nll, count, norm, finite, db, lb = jax.lax.while_loop(condition, body, initial)
        return state, {"updates": steps, "nll_sum": nll, "tokens": count, "max_gradient_norm": norm,
                       "finite": finite, "domain_bytes": db, "language_bytes": lb}
    return one_step, jax.jit(group, donate_argnums=(0,))


def make_evaluator(model):
    def evaluate(params, arrays):
        initial = (jnp.zeros(len(LANGUAGES), jnp.float32), jnp.zeros(len(LANGUAGES), jnp.float32),
                   jnp.zeros(len(DOMAINS), jnp.float32), jnp.zeros(len(DOMAINS), jnp.float32), jnp.array(0., jnp.float32))
        def body(index, totals):
            batch = jax.tree.map(lambda values: jax.lax.dynamic_index_in_dim(values, index, keepdims=False), arrays)
            losses = token_losses(model, params, batch)
            nll = jnp.where(batch["text_mask"], losses, 0.).reshape(-1)
            byte_weights = batch["byte_weights"].reshape(-1).astype(jnp.float32)
            language = batch["language_ids"].reshape(-1)
            domain = batch["domain_ids"].reshape(-1)
            ln = jnp.bincount(language, weights=nll, length=len(LANGUAGES))
            lb = jnp.bincount(language, weights=byte_weights, length=len(LANGUAGES))
            dn = jnp.bincount(domain, weights=nll, length=len(DOMAINS))
            db = jnp.bincount(domain, weights=byte_weights, length=len(DOMAINS))
            eos = jnp.sum(jnp.where(batch["mask"] & ~batch["text_mask"], losses, 0.))
            return totals[0] + ln, totals[1] + lb, totals[2] + dn, totals[3] + db, totals[4] + eos
        return jax.lax.fori_loop(0, arrays["tokens"].shape[0], body, initial)
    return jax.jit(evaluate)


def readable_evaluation(values, raw_bytes):
    ln, lb, dn, db, eos = [np.asarray(value) for value in values]
    if int(round(float(lb.sum()))) != raw_bytes or not np.isfinite(ln).all():
        raise ValueError("Evaluation original-byte denominator or finite-loss check failed.")
    languages = {language: {"bpb": float(ln[i] / (lb[i] * math.log(2))), "nll_nats": float(ln[i]),
                            "original_utf8_bytes": int(lb[i])} for i, language in enumerate(LANGUAGES) if lb[i] > 0}
    domains = {domain: {"bpb": float(dn[i] / (db[i] * math.log(2))), "original_utf8_bytes": int(db[i])}
               for i, domain in enumerate(DOMAINS) if db[i] > 0}
    return {"pooled_bpb": float(ln.sum() / (lb.sum() * math.log(2))),
        "language_macro_bpb": float(np.mean([value["bpb"] for value in languages.values()])),
        "languages": languages, "domains": domains, "original_utf8_bytes": raw_bytes,
        "eos_nll_nats_separate": float(eos), "scoring": "all original text targets once; virtual EOS excluded from BPB; fixed 1024-token chunk context"}


def initial_backbone_digest(params):
    digest = hashlib.sha256()
    for path, value in sorted(traverse_util.flatten_dict(params).items()):
        if path[0] == "embedder":
            continue
        digest.update("/".join(path).encode())
        digest.update(np.asarray(jax.device_get(value)).tobytes())
    return digest.hexdigest()


def verify_tpu_kernels(attention_kernel):
    """Colab-only synthetic kernel verification; never updates the pilot model."""
    import tokamax
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    started = time.monotonic()
    keys = jax.random.split(jax.random.PRNGKey(314159), 5)
    x = jax.random.normal(keys[0], (1024, 640), dtype=jnp.bfloat16) * .1
    w = jax.random.normal(keys[1], (640, 2049), dtype=jnp.bfloat16) * .1
    labels = jnp.arange(1024, dtype=jnp.int32) % w.shape[1]
    def fused(x, w):
        return tokamax.linear_softmax_cross_entropy_loss(x, labels, w, reduction="mean", implementation="mosaic_tpu")
    def reference(x, w):
        logits = jnp.matmul(x.astype(jnp.float32), w.astype(jnp.float32))
        return jnp.mean(jax.scipy.special.logsumexp(logits, axis=-1) - logits[jnp.arange(1024), labels])
    actual, actual_grad = jax.block_until_ready(jax.jit(jax.value_and_grad(fused, argnums=(0, 1)))(x, w))
    expected, expected_grad = jax.block_until_ready(jax.jit(jax.value_and_grad(reference, argnums=(0, 1)))(x, w))
    def relative(a, b):
        a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
        return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-12))
    report = {"loss_relative_error": relative(actual, expected),
              "loss_gradient_relative_errors": [relative(a, b) for a, b in zip(actual_grad, expected_grad)],
              "attention": {}, "loss_tolerance": .005, "gradient_tolerance": .05,
              "attention_tolerance": .03, "optimizer_updates": 0}
    q = jax.random.normal(keys[2], (4, 1024, 256), dtype=jnp.bfloat16) / 16
    k = jax.random.normal(keys[3], (1024, 256), dtype=jnp.bfloat16)
    v = jax.random.normal(keys[4], (1024, 256), dtype=jnp.bfloat16)
    # Two documents plus isolated padding rows; local case crosses the 512-position boundary.
    ids = jnp.concatenate((jnp.zeros(600, jnp.int32), jnp.full(300, 600, jnp.int32), jnp.arange(900, 1024)))
    positions = jnp.arange(1024)
    for local in (False, True):
        mask = (ids[:, None] == ids[None, :]) & (positions[:, None] >= positions[None, :])
        if local:
            mask &= positions[None, :] > positions[:, None] - 512
        def splash(q, k, v):
            return attention_kernel(1024, 4, local)(q, k, v, sk.SegmentIds(ids, ids))
        def dense(q, k, v):
            logits = jnp.einsum("hqd,kd->hqk", q.astype(jnp.float32), k.astype(jnp.float32))
            probabilities = jax.nn.softmax(jnp.where(mask[None], logits, -1e30), axis=-1)
            return jnp.einsum("hqk,kd->hqd", probabilities, v.astype(jnp.float32))
        def value_and_output(function, q, k, v):
            output = function(q, k, v)
            return jnp.mean(output.astype(jnp.float32) ** 2), output
        (a, ao), ag = jax.block_until_ready(jax.jit(jax.value_and_grad(
            functools.partial(value_and_output, splash), argnums=(0, 1, 2), has_aux=True))(q, k, v))
        (b, bo), bg = jax.block_until_ready(jax.jit(jax.value_and_grad(
            functools.partial(value_and_output, dense), argnums=(0, 1, 2), has_aux=True))(q, k, v))
        report["attention"]["local" if local else "global"] = {
            "output_relative_error": relative(ao, bo), "gradient_relative_errors": [relative(aa, bb) for aa, bb in zip(ag, bg)]}
    report["passed"] = (report["loss_relative_error"] <= .005
        and max(report["loss_gradient_relative_errors"]) <= .05
        and all(row["output_relative_error"] <= .03 and max(row["gradient_relative_errors"]) <= .05
                for row in report["attention"].values()))
    report["seconds"] = time.monotonic() - started
    if not report["passed"]:
        raise ValueError("TPU kernel verification failed: " + json.dumps(report))
    return report


def train_pilot(work, output):
    work, output = Path(work), Path(output)
    # This persists in Drive, so rerunning the cell or reconnecting cannot repeat the attempt.
    with (output / "training.started").open("x") as marker:
        marker.write("V3: one training attempt; automatic retry disabled.\n")
    environment = preflight()
    load_model_helpers()
    data_dir = work / "prepared"
    config = read_json(data_dir / "resolved_config.json")
    manifest = verify_raw(Path(config["shared_directory"]) / "raw")
    tokenization = read_json(data_dir / "tokenization.json")
    identity = config["tokenizer_identity"]
    if tokenization["raw_manifest_sha256"] != sha256_file(data_dir / "manifest.json"):
        raise ValueError("Tokenized and raw corpus manifests differ.")
    if shutil.disk_usage(work).free < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free local disk is required for the resumable checkpoint.")
    host = {}
    orders = {}
    for split in ("train", "in_mix", "wiki"):
        name = split + ".npz"
        if sha256_file(data_dir / name) != tokenization["files"][name]:
            raise ValueError("Prepared input checksum failed: " + name)
        with np.load(data_dir / name, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in INPUT_KEYS}
        if int(arrays["byte_weights"].sum()) != manifest["original_bytes"][split]:
            raise ValueError("Prepared original-byte count differs: " + split)
        host[split], orders[split] = batch_training_arrays(arrays, config["local_batch_per_chip"], config["seed"])
    input_bytes = sum(value.nbytes for value in host["train"].values())
    phase = time.monotonic()
    device_data = jax.device_put(host)
    jax.block_until_ready(device_data)
    input_transfer_seconds = time.monotonic() - phase
    resident = all(jax.devices()[0] in leaf.devices() for leaf in jax.tree.leaves(device_data))
    del host, arrays
    training = device_data["train"]
    print(f"Training inputs on TPU: {input_bytes / 1024**2:.1f} MiB.", flush=True)
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment)

    phase = time.monotonic()
    model = make_model(config)
    if model.config.final_logit_softcap is not None:
        raise ValueError("The fused linear head requires the native uncapped Gemma 270M logits.")
    # Native one-position initialization precedes installation of the 128-block Splash kernel.
    params = jax.jit(lambda key: model.init(key, tokens=jnp.ones((1, 1), jnp.int32))["params"])(jax.random.PRNGKey(config["seed"]))
    jax.block_until_ready(params)
    flat = traverse_util.flatten_dict(params)
    parameter_count = sum(value.size for value in flat.values())
    embedding_shape = (identity["model_rows"], 640)
    tied = flat[("embedder", "input_embedding")].shape == embedding_shape and sum(value.shape == embedding_shape for value in flat.values()) == 1
    if parameter_count != identity["expected_parameters"] or not tied or not all(value.dtype == jnp.float32 for value in flat.values()):
        raise ValueError("Parameter count, tied table, or FP32 master weights differ from the V3 specification.")
    backbone_digest = initial_backbone_digest(params)
    backbone_key = hashlib.sha256(json.dumps({"gemma": environment["packages"].get("gemma"),
        "jax": environment["jax"], "seed": config["seed"], "backbone": "Gemma3_270M"}, sort_keys=True).encode()).hexdigest()[:16]
    fingerprint_path = Path(config["shared_directory"]) / ("backbone-" + backbone_key + ".json")
    fingerprint = {"sha256": backbone_digest, "transformer_and_norm_parameters": 100326016, "seed": config["seed"]}
    if fingerprint_path.exists() and read_json(fingerprint_path) != fingerprint:
        raise ValueError("Transformer initialization differs from the shared tokenizer-comparison baseline.")
    write_json(fingerprint_path, fingerprint)
    write_json(output / "backbone_initialization.json", fingerprint)
    probe_path = next(path for path, value in sorted(flat.items()) if path[0].startswith("layer_") and value.ndim > 1)
    def probe(parameters):
        return np.asarray(jax.device_get(traverse_util.flatten_dict(parameters)[probe_path].reshape(-1)[:64]))
    initial_probe = probe(params)
    optimizer, partition = make_optimizer(config, params)
    write_json(output / "optimizer_partition.json", partition)
    state = PilotState(jnp.array(0, jnp.int32), params, optimizer.init(params), jax.random.PRNGKey(config["seed"]), jnp.array(0, jnp.int32))
    del params, flat
    jax.block_until_ready(state)
    initialization_seconds = time.monotonic() - phase
    attention_kernel = install_splash_attention()
    print("Verifying fused loss and segmented local/global attention on this TPU; zero model updates...", flush=True)
    kernel_report = verify_tpu_kernels(attention_kernel)
    write_json(output / "kernel_verification.json", kernel_report)
    _, group_jit = make_training_functions(model, optimizer)
    evaluation_jit = make_evaluator(model)
    print("Compiling training and held-out evaluation; zero model updates...", flush=True)
    phase = time.monotonic()
    executable = group_jit.lower(state, training, jnp.array(1, jnp.int32)).compile()
    evaluators = {split: evaluation_jit.lower(state.params, device_data[split]).compile() for split in ("in_mix", "wiki")}
    compilation_seconds = time.monotonic() - phase
    if int(state.step) != 0:
        raise ValueError("Optimizer update occurred before timing.")
    try:
        analysis = executable.memory_analysis()
        compiled_memory = {name: int(getattr(analysis, name)) for name in
            ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")}
    except (AttributeError, TypeError, NotImplementedError):
        compiled_memory = None
    def evaluate(parameters):
        return {split: readable_evaluation(jax.block_until_ready(evaluators[split](parameters, device_data[split])),
            manifest["original_bytes"][split]) for split in ("in_mix", "wiki")}
    phase = time.monotonic()
    before = evaluate(state.params)
    initial_eval_seconds = time.monotonic() - phase
    print(f"Initial FineWiki macro BPB={before['wiki']['language_macro_bpb']:.5f}. Starting ONE 60-second V3 run...", flush=True)
    log_rows = []
    updates = tokens_processed = 0
    domain_bytes = np.zeros(len(DOMAINS), np.int64)
    language_bytes = np.zeros(len(LANGUAGES), np.int64)
    max_seconds_per_update = 0.
    last_progress = last_start = 0.
    jax.block_until_ready(state)
    started = time.monotonic()
    while True:
        remaining = config["training_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            break
        group_size = config["updates_per_dispatch"]
        requested = group_size if updates > 0 and remaining > max(2., 2 * group_size * max_seconds_per_update) else 1
        last_start = time.monotonic() - started
        if last_start >= config["training_seconds"]:
            break
        call_start = time.monotonic()
        state, metrics = jax.block_until_ready(executable(state, training, jnp.array(requested, jnp.int32)))
        metrics = {key: np.asarray(value).tolist() for key, value in metrics.items()}
        if not metrics["finite"] or metrics["updates"] != requested:
            raise FloatingPointError("Nonfinite loss/gradient; affected update rejected. No retry.")
        completed = int(metrics["updates"])
        updates += completed
        tokens_processed += int(metrics["tokens"])
        domain_bytes += np.array(metrics["domain_bytes"], np.int64)
        language_bytes += np.array(metrics["language_bytes"], np.int64)
        max_seconds_per_update = max(max_seconds_per_update, (time.monotonic() - call_start) / completed)
        end = time.monotonic() - started
        log_rows.append({**metrics, "update": updates, "requested_updates": requested, "start_seconds": last_start,
                         "end_seconds": end, "loss": metrics["nll_sum"] / metrics["tokens"]})
        if len(log_rows) == 1 or end - last_progress >= 2:
            print(f"updates={updates:,} tokens={tokens_processed:,} loss={log_rows[-1]['loss']:.5f} elapsed={end:.2f}s", flush=True)
            last_progress = end
    jax.block_until_ready(state)
    training_seconds = time.monotonic() - started
    with (output / "training.jsonl").open("w") as stream:
        for row in log_rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    if not log_rows:
        raise RuntimeError("No optimizer updates completed.")
    changed = not np.array_equal(initial_probe, probe(state.params))
    final_finite = bool(jax.device_get(jax.jit(lambda p: jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(p)])))(state.params)))
    phase = time.monotonic()
    after = evaluate(state.params)
    final_eval_seconds = time.monotonic() - phase
    memory_stats = jax.devices()[0].memory_stats()
    phase = time.monotonic()
    metadata = {"version": "V3", "protocol": PROTOCOL, "config": config, "parameter_count": parameter_count,
        "initialization": "random; zero pretrained weights", "backbone_initialization_sha256": backbone_digest,
        "optimizer": partition, "raw_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "tokenization_manifest_sha256": sha256_file(data_dir / "tokenization.json"), "packing_version": PACKING_VERSION,
        "data_order_sha256": orders, "training_batches": training["tokens"].shape[0],
        "data_position_unit": "completed batches in fixed seeded order; repeats after a full pass", "updates": updates,
        "checkpoint_fields": ["step", "params", "opt_state", "rng", "data_position"]}
    print("Saving ONE full checkpoint to Drive and verifying its checksum...", flush=True)
    local_checkpoint = work / "checkpoint.npz"
    leaves = save_archive(local_checkpoint, state, metadata)
    digest = sha256_file(local_checkpoint)
    partial = output / "checkpoint.npz.partial"
    shutil.copyfile(local_checkpoint, partial)
    if sha256_file(partial) != digest:
        raise ValueError("Drive checkpoint checksum differs from the local checkpoint.")
    checkpoint = output / "checkpoint.npz"
    partial.rename(checkpoint)
    with zipfile.ZipFile(checkpoint) as archive:
        saved = json.loads(archive.read("checkpoint.json"))
        schema_verified = (set(saved["tree"]) == {"step", "params", "opt_state", "rng", "data_position"}
            and saved["metadata"] == metadata and sum(name.startswith("arrays/") for name in archive.namelist()) == leaves)
    if not schema_verified:
        raise ValueError("Saved checkpoint schema or array count differs.")
    write_json(output / "checkpoint_manifest.json", {"sha256": digest, "bytes": checkpoint.stat().st_size,
        "array_leaves": leaves, "verification": "local/Drive SHA256 equality plus state-schema and leaf-count inspection",
        "restore_executed": False})
    checkpoint_seconds = time.monotonic() - phase
    bytes_processed = int(domain_bytes.sum())
    checks = {
        "V3-001_ONE_ATTEMPT": (output / "training.started").exists(),
        "V3-002_ONE_V5E": environment["platform"] == "tpu" and environment["device_count"] == 1,
        "V3-003_SOURCE_PINS": set(manifest["source_revisions"]) == set(SOURCE_REPOS),
        "V3-004_FROZEN_CORPUS": tokenization["raw_manifest_sha256"] == sha256_file(data_dir / "manifest.json"),
        "V3-005_SPLIT_SEPARATION": manifest["cross_split_source_groups"] == 0,
        "V3-006_LOSSLESS_TOKENIZATION": all(tokenization["splits"][s]["original_bytes"] == manifest["original_bytes"][s] for s in ("train", "in_mix", "wiki")),
        "V3-007_INTACT_PACKING": all(tokenization["splits"][s]["targets_preserved"] and tokenization["splits"][s]["context_preserved"] for s in ("train", "in_mix", "wiki")),
        "V3-008_KERNEL_EQUIVALENCE": kernel_report["passed"],
        "V3-009_RESIDENT_INPUTS": resident,
        "V3-010_GEMMA_BACKBONE": parameter_count == identity["expected_parameters"] and parameter_count < 300000000 and tied,
        "V3-011_MUON_PARTITION": partition["muon_parameters"] == 100270080,
        "V3-012_TIMED_DISPATCH": all(row["start_seconds"] < config["training_seconds"] for row in log_rows),
        "V3-013_DISPATCH_BOUND": all(1 <= row["updates"] <= 16 for row in log_rows),
        "V3-014_FINITE_UPDATES": all(row["finite"] for row in log_rows) and final_finite,
        "V3-015_ZERO_EXTRA_UPDATES": int(state.step) == updates and int(state.data_position) == updates,
        "V3-016_FIXED_EVALUATION": len(after["wiki"]["languages"]) == 12 and all(after[s]["original_utf8_bytes"] == before[s]["original_utf8_bytes"] for s in ("in_mix", "wiki")),
        "V3-017_CHECKPOINT": schema_verified,
        "V3-018_CHANGED_WEIGHTS": changed,
        "V3-019_MINIMUM_UPDATES": updates >= 5,
        "V3-020_BYTE_ACCOUNTING": bytes_processed == int(language_bytes.sum()) and bytes_processed > 0,
    }
    runtime_elapsed = time.monotonic() - config["notebook_start_monotonic"]
    rate = config["compute_units_per_hour"]
    result = {"version": "V3", "protocol": PROTOCOL, "run_count": 1, "config": config,
        "device": environment["device_kind"], "model_parameters": parameter_count, "optimizer": partition,
        "initial_backbone_sha256": backbone_digest, "input_transfer_seconds": input_transfer_seconds,
        "resident_training_bytes": input_bytes, "initialization_seconds": initialization_seconds,
        "kernel_verification_seconds": kernel_report["seconds"], "compilation_seconds": compilation_seconds,
        "initial_eval_seconds": initial_eval_seconds, "final_eval_seconds": final_eval_seconds,
        "training_seconds": training_seconds, "training_overrun_seconds": max(0., training_seconds - config["training_seconds"]),
        "last_dispatch_start_seconds": last_start, "host_dispatches": len(log_rows), "updates": updates,
        "scored_tokens": tokens_processed, "tokens_per_second": tokens_processed / training_seconds,
        "processed_original_bytes_attributed": bytes_processed, "original_bytes_per_second_attributed": bytes_processed / training_seconds,
        "byte_accounting_note": "Monotonic tokenizer offset attribution; includes repeated passes; not unique bytes or a claim of per-token byte boundaries.",
        "training_domain_byte_shares": {name: int(domain_bytes[i]) / bytes_processed for i, name in enumerate(DOMAINS)},
        "training_language_byte_shares": {name: int(language_bytes[i]) / bytes_processed for i, name in enumerate(LANGUAGES)},
        "training_occupancy": tokens_processed / (updates * config["local_batch_per_chip"] * config["sequence_length"]),
        "prepared_occupancy": tokenization["splits"]["train"]["occupancy"],
        "completed_data_passes": updates // training["tokens"].shape[0], "training_batches": training["tokens"].shape[0],
        "data_manifest_sha256": sha256_file(data_dir / "manifest.json"), "initial_eval": before, "final_eval": after,
        "primary_metric": "FineWiki mean of 12 per-language BPB values; lower is better",
        "checkpoint_save_verify_seconds": checkpoint_seconds, "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": digest, "checkpoint_array_leaves": leaves, "memory_stats": memory_stats,
        "compiled_memory_estimate": compiled_memory, "memory_scope": "Allocator snapshot after validation; compiler estimate separately.",
        "notebook_elapsed_seconds": runtime_elapsed,
        "compute_units_estimate": {"rate_per_hour": rate, "training": training_seconds * rate / 3600,
            "through_report": runtime_elapsed * rate / 3600, "overhead_through_report": (runtime_elapsed - training_seconds) * rate / 3600},
        "cost_note": "User-supplied 2.92 CU/hour; connected runtime after report adds cost; Colab billing is authoritative.",
        "historical_v2_tokens_per_second": 45559.448, "historical_comparison_note": "V2 used different data, batch, optimizer and vocabulary handling; not a controlled V2/V3 speed or loss comparison.",
        "performance_goals": {"occupancy_at_least_95_percent": tokens_processed / (updates * 4 * 1024) >= .95},
        "requirements": {key: "PASS" if passed else "FAIL" for key, passed in checks.items()},
        "engineering_status": "PASS" if all(checks.values()) else "FAIL",
        "learning_status": "PASS" if after["wiki"]["language_macro_bpb"] < before["wiki"]["language_macro_bpb"] else "FAIL"}
    write_json(output / "result.json", result)
    print(json.dumps({name: result[name] for name in ("engineering_status", "learning_status", "updates", "training_seconds", "tokens_per_second", "original_bytes_per_second_attributed")}, indent=2), flush=True)
    print(f"FineWiki macro BPB: {before['wiki']['language_macro_bpb']:.6f} -> {after['wiki']['language_macro_bpb']:.6f}", flush=True)
    if not all(checks.values()):
        raise RuntimeError("V3 engineering requirement failed; result.json was saved.")


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
            print(json.dumps({name: info[name] for name in ("platform", "device_kind", "device_count", "jax")}, indent=2))
        elif args.phase == "prepare":
            prepare_cached(read_json(args.work / "config.json"), args.work, args.output)
        else:
            cache = args.work.parent / "jax-cache-V3"
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache)
            os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "1"
            try:
                train_pilot(args.work, args.output)
            finally:
                persistent = args.output.parent.parent / "V3-cache" / "jax"
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
