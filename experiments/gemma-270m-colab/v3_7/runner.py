"""Gemma Colab V3.7: one fixed multilingual tokenizer-screening pilot on one v5e TPU.
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


MODEL_HELPERS = '@struct.dataclass\nclass PilotState:\n    step: jax.Array\n    params: object\n    opt_state: object\n    rng: jax.Array\n    data_position: jax.Array\n'

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


def verify_raw(directory):
    manifest = read_json(directory / "manifest.json")
    if manifest["recipe_sha256"] != protocol_fingerprint():
        raise ValueError("The shared raw corpus belongs to another V3 protocol.")
    for name, digest in manifest["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Raw corpus checksum differs: {name}")
    return manifest



"""V3.7 parallel source preparation. Embedded in the notebook; not run locally."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading

DOWNLOAD_WORKERS = 6
PREFIX_CACHE_VERSION = "v31-source-prefix-1"
PREFIX_FACTOR = 1.25
PREFIX_PART_BYTES = 4 * 1024**2
PREFIX_JOB_LIMIT_BYTES = 256 * 1024**2


def immutable_sources(shared):
    from huggingface_hub import HfApi
    path = shared / "source-lock.json"
    if path.exists():
        return read_json(path)
    def resolve(repo):
        return HfApi().dataset_info(repo, revision="main").sha
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        revisions = list(pool.map(resolve, SOURCE_REPOS))
    lock = dict(zip(SOURCE_REPOS, revisions))
    write_json(path, lock)
    return lock


def source_jobs(lock):
    from datasets import get_dataset_config_names
    rules = recipe()
    wiki_repo = SOURCE_REPOS[-1]
    names = set(get_dataset_config_names(wiki_repo, revision=lock[wiki_repo]))
    wiki = []
    for language, code in zip(LANGUAGES, WIKI_CODES):
        subset = next((name for name in (code, code + "wiki") if name in names), None)
        if subset is None:
            raise ValueError(f"Pinned FineWiki has no configuration for {language}.")
        wiki.append({"key": "wiki_" + language, "kind": "wiki", "repo": wiki_repo,
            "subset": subset, "revision": lock[wiki_repo], "language": language,
            "targets": {"wiki:wiki_" + language: rules["wiki_eval_bytes_per_language"]}, "row_limit": 100000})
    grouped = defaultdict(list)
    for bucket in rules["buckets"]:
        grouped[(bucket["repo"], bucket["subset"])].append(bucket)
    training = []
    for (repo, subset), buckets in grouped.items():
        targets = {}
        for bucket in buckets:
            targets["train:" + bucket["key"]] = round(rules["training_bytes_target"] * bucket["bp"] / 10000)
            targets["in_mix:" + bucket["key"]] = round(rules["in_mix_eval_bytes_target"] * bucket["bp"] / 10000)
        training.append({"key": "smol_english" if repo == SMOL else buckets[0]["key"], "kind": "training",
            "repo": repo, "subset": subset, "revision": lock[repo], "buckets": buckets,
            "targets": targets, "row_limit": 300000})
    # Canonical merge order is V3 order, independent of thread completion order.
    return wiki + training


def candidate_records(job, row):
    rules = recipe()
    if job["kind"] == "wiki":
        texts = [(row.get("text"), job["language"])]
        bucket_key, domain = job["key"], "wiki_holdout"
    else:
        if row.get("early_stop") or row.get("is_truncated"):
            return []
        if job["repo"] == SMOL:
            origin = str(row.get("dataset", ""))
            bucket_key = "pdf_edu" if "finepdfs" in origin else "web_edu" if "fineweb-edu" in origin else "dclm" if "dclm" in origin else None
            if bucket_key is None:
                raise ValueError("Unknown upstream Smol-Data source label: " + origin)
            bucket = next(bucket for bucket in job["buckets"] if bucket["key"] == bucket_key)
        else:
            bucket = job["buckets"][0]
            bucket_key = bucket["key"]
        domain = bucket["domain"]
        if job["repo"] == SOURCE_REPOS[4]:
            score = row.get("edu_score")
            if score is None or int(score) < 2:
                return []
            texts = [(row.get("og_full_text"), bucket["language"]), (row.get("translated_text"), "eng_Latn")]
        else:
            texts = [(row.get("content") if domain == "code" else row.get("text"), bucket["language"])]
    group = canonical_group(row, texts[0][0] or "")
    if job["kind"] != "wiki" and "wikipedia.org" in group:
        return []
    split = "wiki" if job["kind"] == "wiki" else "in_mix" if int(hashlib.sha256(group.encode()).hexdigest()[:16], 16) % 10 == 0 else "train"
    provenance = {key: row.get(key) for key in ("id", "url", "repo_name", "path", "license", "dataset") if row.get(key) is not None}
    candidates = []
    for text, language in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        encoded = text.encode("utf-8")
        span = utf8_prefix(text, rules["max_document_span_bytes"])
        candidates.append({"text": span, "original_document_sha256": hashlib.sha256(encoded).hexdigest(),
            "source_was_clipped": span != text, "source_group": group, "source": provenance,
            "language": language, "domain": domain, "bucket": bucket_key, "split": split,
            "utf8_bytes": len(span.encode("utf-8"))})
    return candidates


def materialize_prefix(job, root, factor, report_status, stop):
    """Cache deterministic source prefixes in independently committed 4 MiB target batches."""
    from datasets import load_dataset, DownloadConfig
    started = time.monotonic()
    job_hash = hashlib.sha256(json.dumps({"version": PREFIX_CACHE_VERSION, "job": job}, sort_keys=True).encode()).hexdigest()
    directory = root / job_hash[:24]
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "state.json"
    state = read_json(state_path) if state_path.exists() else {"job_sha256": job_hash, "job": job,
        "rows_read": 0, "candidate_bytes": {}, "candidate_documents": {}, "parts": [], "exhausted": False}
    if state["job_sha256"] != job_hash:
        raise ValueError("Source-prefix cache identity mismatch: " + job["key"])
    for part in state["parts"]:
        if sha256_file(directory / part["name"]) != part["sha256"]:
            raise ValueError("Source-prefix cache checksum mismatch: " + job["key"])
    original_rows = state["rows_read"]
    original_parts = len(state["parts"])
    desired = {key: math.ceil(value * factor) for key, value in job["targets"].items()}
    def complete():
        return all(state["candidate_bytes"].get(key, 0) >= quota and state["candidate_documents"].get(key, 0) >= 4
                   for key, quota in desired.items())
    def status(stage):
        report_status(job["key"], {"stage": stage, "rows": state["rows_read"],
            "candidate_bytes": sum(state["candidate_bytes"].values()), "seconds": time.monotonic() - started})
    def result():
        return {"directory": str(directory), "rows_read": state["rows_read"], "candidate_parts": len(state["parts"]),
            "candidate_bytes": sum(state["candidate_bytes"].values()), "prefix_cache_reused": original_parts > 0,
            "new_rows_read": state["rows_read"] - original_rows, "seconds": time.monotonic() - started}
    if complete() or state["exhausted"]:
        status("cached")
        return result()
    buffer = bytearray()
    committed_bytes = sum(part["bytes"] for part in state["parts"])
    last_flush = time.monotonic()
    def commit():
        nonlocal buffer, last_flush, committed_bytes
        if buffer:
            name = f"part-{len(state['parts']):05d}.jsonl"
            temporary = directory / (name + ".partial")
            temporary.write_bytes(buffer)
            temporary.replace(directory / name)
            state["parts"].append({"name": name, "bytes": len(buffer), "sha256": hashlib.sha256(buffer).hexdigest()})
            committed_bytes += len(buffer)
            buffer = bytearray()
        # A cursor is committed only after all candidates from its row are committed.
        write_json(state_path, state)
        last_flush = time.monotonic()
    status("opening stream")
    kwargs = {"name": job["subset"], "split": "train", "streaming": True, "revision": job["revision"],
              "download_config": DownloadConfig(max_retries=1)}
    if job["repo"] == SOURCE_REPOS[4]:
        # Avoid downloading duplicate original/translated chunk-list columns.
        kwargs["columns"] = ["id", "url", "og_full_text", "translated_text", "og_language", "edu_score", "early_stop"]
    try:
        stream = load_dataset(job["repo"], **kwargs)
        if state["rows_read"]:
            stream = stream.skip(state["rows_read"])
        status("reading")
        for row in stream:
            if stop.is_set():
                commit()
                status("stopped; cache saved")
                return result()
            if state["rows_read"] >= job["row_limit"]:
                state["exhausted"] = True
                break
            records = candidate_records(job, row)
            encoded_records = [(json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8") for record in records]
            if committed_bytes + len(buffer) + sum(map(len, encoded_records)) > PREFIX_JOB_LIMIT_BYTES:
                raise RuntimeError("Source-prefix cache would exceed its 256 MiB bound: " + job["key"])
            for record, encoded_record in zip(records, encoded_records):
                buffer.extend(encoded_record)
                stratum = record["split"] + ":" + record["bucket"]
                state["candidate_bytes"][stratum] = state["candidate_bytes"].get(stratum, 0) + record["utf8_bytes"]
                state["candidate_documents"][stratum] = state["candidate_documents"].get(stratum, 0) + 1
            state["rows_read"] += 1
            if len(buffer) >= PREFIX_PART_BYTES or time.monotonic() - last_flush >= 5:
                commit()
                status("reading")
            if complete():
                break
        else:
            state["exhausted"] = True
        commit()
        status("ready")
        return result()
    except Exception:
        commit()
        status("failed; cache saved")
        raise


def build_raw_corpus(destination, lock):
    started = time.monotonic()
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        # A previous interrupted run may have completed the final corpus already.
        manifest = verify_raw(destination)
        write_json(destination.parent / "parallel_downloads.json", {"version": "V3.7",
            "recovered_complete_staging": True, "max_workers": DOWNLOAD_WORKERS,
            "peak_active_source_jobs": 0, "source_jobs": 0, "fetch_seconds": 0.,
            "note": "Verified complete staged corpus recovered without source fetching."})
        return manifest
    shared = destination.parent
    cache_root = shared / PREFIX_CACHE_VERSION
    cache_root.mkdir(parents=True, exist_ok=True)
    jobs = source_jobs(lock)
    status_lock, stop = threading.Lock(), threading.Event()
    statuses = {job["key"]: {"stage": "queued", "rows": 0, "candidate_bytes": 0, "seconds": 0.} for job in jobs}
    peak_active = 0
    def status(key, value):
        nonlocal peak_active
        with status_lock:
            statuses[key] = value
            active = sum(item["stage"] in ("opening stream", "reading") for item in statuses.values())
            peak_active = max(peak_active, active)
    # Start the large English stream and all four translations early; interleave Wiki/FW2 thereafter.
    front = [job for job in jobs if job["key"] == "smol_english" or job["repo"] == SOURCE_REPOS[4]]
    wiki_jobs = [job for job in jobs if job["kind"] == "wiki"]
    other_jobs = [job for job in jobs if job not in front and job["kind"] != "wiki"]
    scheduled = front[:]
    for index in range(max(len(wiki_jobs), len(other_jobs))):
        if index < len(wiki_jobs):
            scheduled.append(wiki_jobs[index])
        if index < len(other_jobs):
            scheduled.append(other_jobs[index])
    print(f"V3.7: {len(jobs)} source jobs; {DOWNLOAD_WORKERS} concurrent workers; completed batches persist in Drive.", flush=True)
    reports = {}
    pool = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
    futures = {pool.submit(materialize_prefix, job, cache_root, PREFIX_FACTOR, status, stop): job for job in scheduled}
    pending = set(futures)
    try:
        while pending:
            done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures[future]
                try:
                    report = future.result()
                except Exception:
                    write_json(shared / "parallel_download_failure.json", {"job": job, "error": traceback.format_exc(),
                        "completed_jobs": reports, "training_started": False})
                    raise
                reports[job["key"]] = report
                print(f"Ready {len(reports)}/{len(jobs)}: {job['key']} ({report['seconds']:.1f}s, reused={report['prefix_cache_reused']})", flush=True)
            with status_lock:
                active = [(key, dict(value)) for key, value in statuses.items() if value["stage"] in ("opening stream", "reading")]
            if pending:
                details = "; ".join(f"{key}: {value['stage']}, {value['rows']:,} rows, {value['candidate_bytes']/1024**2:.1f} MiB"
                                    for key, value in active)
                print(f"Sources ready {len(reports)}/{len(jobs)}; active {len(active)}/{DOWNLOAD_WORKERS}. {details}", flush=True)
    finally:
        stop.set()
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    fetch_seconds = time.monotonic() - started

    # Merge in canonical source order so network completion cannot change the corpus.
    rules = recipe()
    seen_hashes, forbidden_groups = set(), set()
    docs = {"train": [], "in_mix": [], "wiki": []}
    counts, rejected, summaries = Counter(), Counter(), {}
    refill_stop = threading.Event()
    for job in jobs:
        targets = job["targets"]
        actual = {key: 0 for key in targets}
        consumed_parts = 0
        factor = PREFIX_FACTOR
        directory = Path(reports[job["key"]]["directory"])
        while True:
            state = read_json(directory / "state.json")
            for part in state["parts"][consumed_parts:]:
                with (directory / part["name"]).open(encoding="utf-8") as stream:
                    for line in stream:
                        candidate = json.loads(line)
                        split, bucket = candidate["split"], candidate["bucket"]
                        stratum = split + ":" + bucket
                        remaining = targets[stratum] - actual[stratum]
                        if remaining <= rules["byte_quota_tolerance"]:
                            continue
                        if split == "train" and candidate["source_group"] in forbidden_groups:
                            rejected["reserved_evaluation_group"] += 1
                            continue
                        span = utf8_prefix(candidate["text"], remaining)
                        if not span:
                            continue
                        digest = hashlib.sha256(span.encode("utf-8")).hexdigest()
                        full_digest = candidate["original_document_sha256"]
                        if full_digest in seen_hashes or digest in seen_hashes:
                            rejected["exact_duplicate"] += 1
                            continue
                        seen_hashes.update((full_digest, digest))
                        if split != "train":
                            forbidden_groups.add(candidate["source_group"])
                        length = len(span.encode("utf-8"))
                        docs[split].append({"text": span, "sha256": digest, "original_document_sha256": full_digest,
                            "source_group": candidate["source_group"], "source": candidate["source"],
                            "language": candidate["language"], "domain": candidate["domain"], "bucket": bucket,
                            "utf8_bytes": length, "source_byte_start": 0, "source_byte_end": length,
                            "clipped": candidate["source_was_clipped"] or span != candidate["text"]})
                        actual[stratum] += length
            consumed_parts = len(state["parts"])
            if all(targets[key] - actual[key] <= rules["byte_quota_tolerance"] for key in targets):
                break
            if state["exhausted"] or state["rows_read"] >= job["row_limit"]:
                raise RuntimeError(f"Data quota shortfall after deduplication: {job['key']}: {actual}/{targets}. No substitution.")
            # Continue the same pinned stream when deduplication consumes the buffer.
            # This is additional data preparation, never another model experiment.
            factor *= 2
            print(f"Extending cached prefix after deduplication: {job['key']}", flush=True)
            prior_rows = state["rows_read"]
            report = materialize_prefix(job, cache_root, factor, status, refill_stop)
            reports[job["key"]]["refilled_prefix"] = report
            if report["rows_read"] == prior_rows:
                raise RuntimeError("No additional source rows available: " + job["key"])
        if job["kind"] == "wiki":
            summaries[job["key"]] = {"bytes": actual["wiki:" + job["key"]], "subset": job["subset"]}
        else:
            for bucket in job["buckets"]:
                key = bucket["key"]
                summaries[key] = {"targets": {split: targets[split + ":" + key] for split in ("train", "in_mix")},
                    "actual": {split: actual[split + ":" + key] for split in ("train", "in_mix")},
                    "repository": job["repo"], "subset": job["subset"]}
        print("Selected byte quotas: " + job["key"] + " " + json.dumps(actual), flush=True)
    by_bucket = defaultdict(list)
    for doc in docs["train"]:
        by_bucket[doc["bucket"]].append(doc)
    rng = np.random.default_rng(42)
    weights = {bucket["key"]: bucket["bp"] / 10000 for bucket in rules["buckets"]}
    for rows in by_bucket.values():
        rng.shuffle(rows)
    used, interleaved = Counter(), []
    while any(by_bucket.values()):
        key = min((key for key, rows in by_bucket.items() if rows), key=lambda key: used[key] / weights[key])
        doc = by_bucket[key].pop()
        interleaved.append(doc)
        used[key] += doc["utf8_bytes"]
    docs["train"] = interleaved
    train_groups = {doc["source_group"] for doc in docs["train"]}
    eval_groups = {doc["source_group"] for split in ("in_mix", "wiki") for doc in docs[split]}
    if train_groups & eval_groups:
        raise ValueError("Source group overlaps training and evaluation.")
    for split, rows in docs.items():
        temporary = destination / (split + ".jsonl.partial")
        with temporary.open("w", encoding="utf-8") as stream:
            for doc in rows:
                stream.write(json.dumps(doc, ensure_ascii=False) + "\n")
        temporary.replace(destination / (split + ".jsonl"))
        counts[split] = len(rows)
    # Preserve the V3 manifest schema so completed V3 raw/token caches remain reusable.
    manifest = {"protocol": PROTOCOL, "recipe_sha256": protocol_fingerprint(), "source_revisions": lock,
        "recipe": rules, "buckets": summaries, "rejected": dict(rejected), "document_counts": dict(counts),
        "original_bytes": {split: sum(doc["utf8_bytes"] for doc in rows) for split, rows in docs.items()},
        "language_bytes": {split: {lang: sum(doc["utf8_bytes"] for doc in rows if doc["language"] == lang) for lang in LANGUAGES}
                           for split, rows in docs.items()},
        "cross_split_source_groups": 0,
        "deduplication_scope": "global exact source/span hashes and source groups; near duplicates are not exhaustively removed",
        "files": {split + ".jsonl": sha256_file(destination / (split + ".jsonl")) for split in docs}}
    write_json(shared / "parallel_downloads.json", {"version": "V3.7", "max_workers": DOWNLOAD_WORKERS,
        "source_jobs": len(jobs), "peak_active_source_jobs": peak_active,
        "fetch_seconds": fetch_seconds, "prepare_raw_seconds": time.monotonic() - started,
        "jobs": reports, "merge_order": [job["key"] for job in jobs],
        "initial_prefix_factor": PREFIX_FACTOR, "minimum_candidate_documents_per_stratum": 4,
        "max_cached_bytes_per_job": PREFIX_JOB_LIMIT_BYTES, "persistent_batch_target_bytes": PREFIX_PART_BYTES,
        "selection": "canonical V3 source order; independent of worker completion order",
        "throughput_speedup_measured": False})
    # The final manifest is the completion marker; it is committed last.
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
        raw_reused = bool(read_json(shared / "parallel_downloads.json").get("recovered_complete_staging", False))
    else:
        raw_reused = True
    raw_manifest = verify_raw(raw)
    prepared = work / "prepared"
    prepared.mkdir(exist_ok=False)
    download_report_path = shared / "parallel_downloads.json"
    if raw_reused:
        download_report = {"version": "V3.7", "raw_cache_reused": True, "source_jobs": 0,
            "max_workers": DOWNLOAD_WORKERS, "peak_active_source_jobs": 0, "fetch_seconds": 0.,
            "note": "Verified completed raw corpus reused; no source fetch was needed."}
    else:
        download_report = read_json(download_report_path)
        download_report["raw_cache_reused"] = False
    if download_report["peak_active_source_jobs"] > DOWNLOAD_WORKERS:
        raise ValueError("Source download concurrency exceeded six jobs.")
    write_json(output / "parallel_downloads.json", download_report)
    preparation_checks = {
        "V37-021_DOWNLOAD_BOUND": download_report["peak_active_source_jobs"] <= 6,
        "V37-022_PREFIX_CACHE": raw_reused or all(Path(item["directory"]).joinpath("state.json").exists()
                                                    for item in download_report.get("jobs", {}).values()),
        "V37-023_MERGE_ORDER": raw_reused or download_report.get("merge_order", [])[:12] == ["wiki_" + lang for lang in LANGUAGES],
        "V37-024_SOURCE_COMPLETE": raw_reused or download_report.get("source_jobs") == 30 and len(download_report.get("jobs", {})) == 30,
    }
    write_json(output / "preparation_requirements.json", {key: "PASS" if passed else "FAIL" for key, passed in preparation_checks.items()})
    if not all(preparation_checks.values()):
        raise ValueError("V3.7 preparation engineering requirement failed.")
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
    exec(MODEL_HELPERS, globals())


def make_model(config):
    from gemma import gm
    model = gm.nn.Gemma3_270M(dtype=jnp.float32)
    return model.clone(config=dataclasses.replace(model.config, num_embed=config["tokenizer_identity"]["model_rows"]))


PILOT_VERSION = 'V3.7'

def make_optimizer(config, params):
    schedule = optax.join_schedules([
        optax.linear_schedule(0., config["learning_rate"], config["warmup_updates"]),
        optax.constant_schedule(config["learning_rate"]),
    ], [config["warmup_updates"]])
    decay = jax.tree.map(lambda value: value.ndim > 1, params)
    transform = optax.chain(
        optax.clip_by_global_norm(config["gradient_clip"]),
        optax.adamw(schedule, b1=.9, b2=.999, eps=1e-8,
                    weight_decay=config["weight_decay"], mask=decay))
    flat = traverse_util.flatten_dict(params)
    count = sum(value.size for value in flat.values())
    routing = [{"path": "/".join(path), "shape": list(value.shape),
                "parameters": value.size, "optimizer": "AdamW",
                "weight_decay": value.ndim > 1}
               for path, value in sorted(flat.items())]
    return transform, {"name": "AdamW", "adamw_parameters": count,
        "muon_parameters": 0, "learning_rate": config["learning_rate"],
        "warmup_updates": config["warmup_updates"], "routing": routing}


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



"""Repair the Tokamax loss gradient boundary without changing its TPU math."""




# Shared native-Gemma runtime assembled into both new notebooks.
# This source is not executed by the artifact builder.

TRAIN_KEYS = ("tokens", "targets", "mask", "positions", "segment_ids")


"""Exact full-vocabulary loss with bounded token-chunk workspace.

Embedded into runner.py by build_notebook.py; JAX imports come from the runner.
"""

HEAD_CHUNK_TOKENS = 128


def cooldown_multiplier(elapsed, budget, start_fraction=0.5, floor=0.3):
    if budget <= 0 or not 0 <= start_fraction < 1 or not 0 < floor <= 1:
        raise ValueError("Invalid elapsed-time learning-rate schedule.")
    progress = max(0.0, min(1.0, float(elapsed) / budget))
    fraction = max(0.0, (progress - start_fraction) / (1.0 - start_fraction))
    return 1.0 - (1.0 - floor) * fraction


def exact_chunked_head(hidden, table, targets, chunk_tokens=HEAD_CHUNK_TOKENS):
    """Same BF16 projection / FP32 logsumexp as native Gemma; no sampling.

    Rematerialize each chunk in backward so scan does not retain every logit
    tensor. The tied table receives both encoding and decoding gradients.
    """
    shape = targets.shape
    hidden = hidden.reshape((-1, hidden.shape[-1]))
    labels = targets.reshape(-1)
    count = labels.shape[0]
    padding = (-count) % chunk_tokens
    x = jnp.pad(hidden, ((0, padding), (0, 0))).reshape(
        (-1, chunk_tokens, hidden.shape[-1]))
    y = jnp.pad(labels, ((0, padding),)).reshape((-1, chunk_tokens))

    @jax.checkpoint
    def body(carry, inputs):
        features, ids = inputs
        with jax.named_scope("vocabulary_projection"):
            logits = jnp.dot(features, table.T).astype(jnp.float32)
        with jax.named_scope("cross_entropy"):
            selected = jnp.take_along_axis(logits, ids[:, None], axis=-1)[:, 0]
            losses = jax.scipy.special.logsumexp(logits, axis=-1) - selected
        return carry, losses

    _, losses = jax.lax.scan(body, None, (x, y))
    return losses.reshape(-1)[:count].reshape(shape)


def native_hidden(model, tokens, positions, attention_mask):
    """Native cache-free text-only Gemma blocks, with unchanged parameter paths."""
    with jax.named_scope("input_embedding"):
        x = model.embedder.encode(tokens)
    for index, block in enumerate(model.blocks):
        with jax.named_scope("transformer_layer_" + str(index)):
            _, x = block(x, positions, None, attention_mask)
    return model.final_norm(x)


def token_losses(model, params, batch):
    compute = jax.tree.map(lambda value: value.astype(jnp.bfloat16), params)
    hidden = model.apply({"params": compute}, batch["tokens"], batch["positions"],
        packed_attention_mask(batch["segment_ids"]), method=native_hidden)
    return exact_chunked_head(hidden, compute["embedder"]["input_embedding"], batch["targets"])


def reference_token_losses(model, params, batch):
    compute = jax.tree.map(lambda value: value.astype(jnp.bfloat16), params)
    logits = model.apply({"params": compute}, tokens=batch["tokens"],
        positions=batch["positions"],
        attention_mask=packed_attention_mask(batch["segment_ids"])).logits.astype(jnp.float32)
    selected = jnp.take_along_axis(logits, batch["targets"][..., None], axis=-1)[..., 0]
    return jax.scipy.special.logsumexp(logits, axis=-1) - selected


def verify_exact_head(model, params, training):
    """Bounded correctness checks, no optimizer updates and no parameter search."""
    begun = time.monotonic()
    key_x, key_w = jax.random.split(jax.random.PRNGKey(370))
    x = jax.random.normal(key_x, (137, 32), dtype=jnp.float32) * .1
    w = jax.random.normal(key_w, (2051, 32), dtype=jnp.float32) * .1
    labels = jnp.arange(137, dtype=jnp.int32) % 2051
    checks = []
    for dtype, tolerance in ((jnp.float32, 2e-5), (jnp.bfloat16, .03)):
        xx, ww = x.astype(dtype), w.astype(dtype)
        def dense(a, b):
            logits = jnp.dot(a, b.T).astype(jnp.float32)
            return jnp.mean(jax.scipy.special.logsumexp(logits, -1)
                - jnp.take_along_axis(logits, labels[:, None], -1)[:, 0])
        def chunked(a, b):
            return jnp.mean(exact_chunked_head(a, b, labels))
        av, ag = jax.block_until_ready(jax.jit(jax.value_and_grad(dense, (0, 1)))(xx, ww))
        bv, bg = jax.block_until_ready(jax.jit(jax.value_and_grad(chunked, (0, 1)))(xx, ww))
        errors = [float(jnp.linalg.norm(a.astype(jnp.float32) - b.astype(jnp.float32))
                        / jnp.maximum(jnp.linalg.norm(a.astype(jnp.float32)), 1e-12))
                  for a, b in zip(ag, bg)]
        passed = abs(float(av) - float(bv)) <= 2e-4 and max(errors) <= tolerance
        checks.append({"dtype": str(dtype), "loss_error": abs(float(av) - float(bv)),
            "gradient_relative_errors": errors, "gradient_tolerance": tolerance, "passed": passed})
    # Verify our native-block entry point against Gemma's ordinary __call__.
    # The real vocabulary is retained; only the probe's token sequence is short.
    batch = {name: values[0, :1, :16] for name, values in training.items()}
    actual = jax.block_until_ready(jax.jit(lambda p: token_losses(model, p, batch))(params))
    expected = jax.block_until_ready(jax.jit(lambda p: reference_token_losses(model, p, batch))(params))
    model_error = float(jnp.max(jnp.abs(actual - expected)))
    passed = all(row["passed"] for row in checks) and model_error <= .005
    report = {"passed": passed, "head_checks": checks, "native_model_loss_max_abs_error": model_error,
        "native_model_tolerance": .005, "chunk_tokens": HEAD_CHUNK_TOKENS,
        "vocabulary": "all rows, exact normalizer", "optimizer_updates": 0,
        "seconds": time.monotonic() - begun}
    if not passed:
        raise ValueError("Exact chunked head failed native loss/gradient verification: " + json.dumps(report))
    return report


def make_training_functions(model, optimizer):
    def one_step(state, batch, lr_multiplier):
        def objective(params):
            losses = token_losses(model, params, batch)
            count = jnp.sum(batch["mask"], dtype=jnp.float32)
            nll = jnp.sum(jnp.where(batch["mask"], losses, 0.))
            return nll / jnp.maximum(count, 1.), count
        (loss, count), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        norm = optax.global_norm(grads)
        finite = jnp.isfinite(loss) & jnp.isfinite(norm) & (count > 0)
        def apply(_):
            with jax.named_scope("optimizer_update"):
                delta, new_opt = optimizer.update(grads, state.opt_state, state.params)
                delta = jax.tree.map(lambda value: value * lr_multiplier, delta)
                next_params = optax.apply_updates(state.params, delta)
            return state.replace(step=state.step + 1,
                params=next_params, opt_state=new_opt,
                data_position=state.data_position + 1)
        state = jax.lax.cond(finite, apply, lambda _: state, operand=None)
        return state, (loss, count, norm, finite)
    def group(state, training, requested, lr_multiplier):
        initial = (state, jnp.array(0, jnp.int32), jnp.array(0., jnp.float32),
                   jnp.array(0, jnp.int32), jnp.array(0., jnp.float32), jnp.array(True))
        def condition(carry):
            return (carry[1] < requested) & carry[5]
        def body(carry):
            state, steps, nll, count, max_norm, finite = carry
            index = state.data_position % training["tokens"].shape[0]
            batch = jax.tree.map(lambda values: jax.lax.dynamic_index_in_dim(
                values, index, keepdims=False), training)
            state, (loss, tokens, norm, good) = one_step(state, batch, lr_multiplier)
            return (state, steps + good.astype(jnp.int32), nll + loss * tokens,
                count + jnp.where(good, tokens, 0).astype(jnp.int32),
                jnp.maximum(max_norm, norm), finite & good)
        state, steps, nll, count, norm, finite = jax.lax.while_loop(condition, body, initial)
        return state, {"updates": steps, "nll_sum": nll, "tokens": count,
                       "max_gradient_norm": norm, "finite": finite}
    return one_step, jax.jit(group, donate_argnums=(0,))


def make_evaluator(model):
    """One executable for both held-out sets; split identity is runtime data."""
    def evaluate(params, arrays):
        initial = (jnp.zeros((2, len(LANGUAGES)), jnp.float32),
                   jnp.zeros((2, len(LANGUAGES)), jnp.float32),
                   jnp.zeros((2, len(DOMAINS)), jnp.float32),
                   jnp.zeros((2, len(DOMAINS)), jnp.float32), jnp.zeros(2, jnp.float32))
        def body(index, totals):
            batch = jax.tree.map(lambda values: jax.lax.dynamic_index_in_dim(
                values, index, keepdims=False), arrays)
            losses = token_losses(model, params, batch)
            nll = jnp.where(batch["text_mask"], losses, 0.).reshape(-1)
            weights = batch["byte_weights"].reshape(-1).astype(jnp.float32)
            language = batch["language_ids"].reshape(-1) + batch["split_id"] * len(LANGUAGES)
            domain = batch["domain_ids"].reshape(-1) + batch["split_id"] * len(DOMAINS)
            ln = jnp.bincount(language, weights=nll, length=2 * len(LANGUAGES)).reshape((2, -1))
            lb = jnp.bincount(language, weights=weights, length=2 * len(LANGUAGES)).reshape((2, -1))
            dn = jnp.bincount(domain, weights=nll, length=2 * len(DOMAINS)).reshape((2, -1))
            db = jnp.bincount(domain, weights=weights, length=2 * len(DOMAINS)).reshape((2, -1))
            eos = jnp.sum(jnp.where(batch["mask"] & ~batch["text_mask"], losses, 0.))
            eos = jax.nn.one_hot(batch["split_id"], 2) * eos
            return tuple(total + value for total, value in zip(totals, (ln, lb, dn, db, eos)))
        return jax.lax.fori_loop(0, arrays["tokens"].shape[0], body, initial)
    return jax.jit(evaluate)


def training_prefix_counts(arrays):
    """CPU preparation only; no per-token source histogram in the training graph."""
    batches = arrays["tokens"].shape[0]
    counts = {"tokens": arrays["mask"].sum(axis=(1, 2), dtype=np.int64)}
    weights = arrays["byte_weights"].reshape(-1)
    for key, field, classes in (("domain_bytes", "domain_ids", len(DOMAINS)),
                                 ("language_bytes", "language_ids", len(LANGUAGES))):
        labels = arrays[field].reshape((batches, -1))
        indices = (labels + np.arange(batches, dtype=np.int64)[:, None] * classes).reshape(-1)
        counts[key] = np.bincount(indices, weights=weights,
            minlength=batches * classes).astype(np.int64).reshape((batches, classes))
    if int(counts["domain_bytes"].sum()) != int(counts["language_bytes"].sum()):
        raise ValueError("Prepared source/language byte accounting differs.")
    return {key: np.concatenate((np.zeros((1, *value.shape[1:]), np.int64),
                np.cumsum(value, axis=0, dtype=np.int64)), axis=0)
            for key, value in counts.items()}


def completed_prefix(prefix, updates):
    batches = prefix.shape[0] - 1
    passes, remainder = divmod(updates, batches)
    return passes * prefix[-1] + prefix[remainder]


def verify_native_protocol():
    started = time.monotonic()
    segments = jnp.array([[0, 0, 1, 1, -1]], jnp.int32)
    actual = np.asarray(jax.jit(packed_attention_mask)(segments))[0]
    expected = np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                         [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]], np.bool_)
    passed = bool(np.array_equal(actual, expected))
    if not passed:
        raise ValueError("Native packed attention mask failed causal/segment/padding verification.")
    return {"passed": passed, "seconds": time.monotonic() - started,
        "optimizer_updates": 0, "attention": "native Gemma", "loss": "native dense exact softmax",
        "custom_kernel_equivalence": "not applicable; no Splash or Tokamax execution",
        "mask_verified": passed}


"""XProf capture and conservative extraction; no utilization estimates.

Embedded into runner.py. Pure JSON parsers also support local fixture checks.
"""

def _number(value):
    if isinstance(value, dict):
        value = value.get("v")
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(str(value).rstrip("%"))
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _normal_key(key):
    return "".join(c for c in str(key).lower() if c.isalnum())


def _property_values(data):
    """Preserve explicit zero; never turn a missing counter into zero."""
    result = {}
    def visit(node):
        if isinstance(node, dict):
            for key, value in node.items():
                number = _number(value)
                if number is not None:
                    result.setdefault(_normal_key(key), number)
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(data)
    return result


def extract_utilization(overview):
    props = _property_values(overview)
    aliases = {
        "matrix_unit_utilization": ("mxu_utilization_percent", "matrix_unit_utilization_percent"),
        "memory_bandwidth_utilization": ("hbm_bw_utilization_percent", "hbm_utilization_percent"),
        "host_idle_time": ("host_idle_time_percent",),
        "device_idle_time": ("device_idle_time_percent",),
    }
    metrics = {}
    for name, keys in aliases.items():
        found = next(((key, props[_normal_key(key)]) for key in keys
                      if _normal_key(key) in props), None)
        valid = found is not None and 0 <= found[1] <= 100
        metrics[name] = {"value": found[1] if valid else None, "unit": "percent",
            "status": "available" if valid else "unavailable",
            "source": "XProf overview_page",
            "source_field": found[0] if valid else None,
            "reason": None if valid else "XProf did not provide a supported percent counter; no FLOP or occupancy proxy used."}
    return metrics


def _datatable_rows(data):
    rows = []
    def visit(node):
        if isinstance(node, dict):
            if isinstance(node.get("cols"), list) and isinstance(node.get("rows"), list):
                columns = [(col.get("id") or col.get("label") or str(i), col.get("label", ""))
                           for i, col in enumerate(node["cols"])]
                for row in node["rows"]:
                    item = {}
                    for (key, label), cell in zip(columns, row.get("c", [])):
                        value = cell.get("v") if isinstance(cell, dict) else cell
                        item[_normal_key(key)] = value
                        if label:
                            item.setdefault(_normal_key(label), value)
                    rows.append(item)
            for value in node.values():
                if isinstance(value, (dict, list)):
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(data)
    return rows


def operation_category(name):
    text = str(name).lower()
    if "optimizer_update" in text or "adam" in text or "optax" in text:
        return "optimizer_updates"
    if "vocabulary_projection" in text:
        return "vocabulary_projection"
    if "cross_entropy" in text or "logsumexp" in text:
        return "cross_entropy"
    if any(key in text for key in ("attention", "/attn/", "q_einsum", "kv_einsum", "attn_vec")):
        return "attention"
    if any(key in text for key in ("mlp", "feedforward", "gating_einsum", "linear_einsum")):
        return "mlp"
    return "other_or_fused"


def extract_operation_timings(hlo_stats):
    """Use exclusive HLO self-time only, never sum overlapping parent scopes."""
    time_keys = ("selftimeus", "selftimeinus", "totalselftimeus", "totalselftimeinus")
    name_keys = ("frameworkopname", "frameworkop", "tfopname", "opname", "hloname",
                 "hloopname", "hloexpression", "hloopexpression", "hlooptext", "expression")
    rows = []
    for row in _datatable_rows(hlo_stats):
        duration = next((_number(row[k]) for k in time_keys if k in row and _number(row[k]) is not None), None)
        names = [str(row[k]) for k in name_keys if row.get(k)]
        if duration is None or duration < 0 or not names:
            continue
        name = " | ".join(dict.fromkeys(names))
        rows.append({"name": name, "self_time_us": duration, "category": operation_category(name)})
    categories = {key: 0.0 for key in ("attention", "vocabulary_projection", "cross_entropy",
                                      "optimizer_updates", "mlp", "other_or_fused")}
    for row in rows:
        categories[row["category"]] += row["self_time_us"]
    total = sum(categories.values())
    return {"status": "available" if rows else "unavailable",
        "reason": None if rows else "XProf did not expose supported exclusive HLO timing columns; inspect the saved HLO/trace artifacts.",
        "source": "XProf hlo_stats", "unit": "microseconds",
        "scope": "Summed HLO self-time over profiled device cores; includes forward/backward; excludes idle. Not elapsed wall time.",
        "classification": "Named HLO metadata; fused/unrecognized operations remain other_or_fused.",
        "categories": {key: {"self_time_us": value if any(row["category"] == key for row in rows) else None,
                               "percent_of_accounted_op_time": 100 * value / total
                                   if total > 0 and any(row["category"] == key for row in rows) else None}
                       for key, value in categories.items()},
        "top_operations": sorted(rows, key=lambda row: row["self_time_us"], reverse=True)[:50]}


def _decode_tool_json(payload):
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload) if isinstance(payload, str) else payload


def summarize_xprof(trace_dir, output):
    files = sorted(Path(trace_dir).rglob("*.xplane.pb")) + sorted(Path(trace_dir).rglob("*.xplane.riegeli"))
    raw = {}
    errors = {}
    if files:
        try:
            from xprof.convert import raw_to_tool_data
            for tool in ("overview_page", "hlo_stats", "roofline_model", "op_profile"):
                try:
                    payload, _ = raw_to_tool_data.xspace_to_tool_data(
                        [str(path) for path in files], tool, {"use_saved_result": False})
                    if payload is None:
                        raise ValueError("XProf returned no tool data")
                    raw[tool] = _decode_tool_json(payload)
                    write_json(output / "profile_tools" / (tool + ".json"), raw[tool])
                except Exception as error:
                    errors[tool] = type(error).__name__ + ": " + str(error)
        except Exception as error:
            errors["converter"] = type(error).__name__ + ": " + str(error)
    else:
        errors["capture"] = "No TPU XPlane files were produced. Host traces alone cannot establish TPU utilization."
    metrics = extract_utilization(raw.get("overview_page", {}))
    # XProf emits default 0.0% strings even when a trace has no hardware counters.
    # Require the underlying counter message, rather than treating those defaults
    # as measured zero utilization. OpStats is generated by overview conversion.
    evidence = {"device_type": None, "hardware_counter_message_present": False,
                "host_timeline_present": False, "device_timeline_present": False}
    try:
        from xprof.protobuf import op_stats_pb2
        candidates = sorted(Path(trace_dir).rglob("ALL_HOSTS.op_stats*.pb"))
        if candidates:
            stats = op_stats_pb2.OpStats()
            stats.ParseFromString(candidates[-1].read_bytes())
            evidence["device_type"] = stats.run_environment.device_type
            tpu = "TPU" in stats.run_environment.device_type.upper()
            evidence["host_timeline_present"] = stats.host_op_metrics_db.total_time_ps > 0
            evidence["device_timeline_present"] = tpu and stats.device_op_metrics_db.total_time_ps > 0
            evidence["hardware_counter_message_present"] = tpu and stats.HasField("performance_counter_result")
            if evidence["hardware_counter_message_present"]:
                counter = stats.performance_counter_result
                for name, field in (("matrix_unit_utilization", "matrix_unit_utilization_percent"),
                                    ("memory_bandwidth_utilization", "hbm_utilization_percent")):
                    value = float(getattr(counter, field))
                    if math.isfinite(value) and 0 <= value <= 100:
                        metrics[name].update(value=value, status="available", reason=None,
                            source="XProf OpStats.performance_counter_result", source_field=field)
    except Exception as error:
        errors["counter_evidence"] = type(error).__name__ + ": " + str(error)
    for name in ("matrix_unit_utilization", "memory_bandwidth_utilization"):
        if not evidence["hardware_counter_message_present"]:
            metrics[name].update(value=None, status="unavailable", source_field=None,
                reason="The profile has no TPU hardware-counter message; XProf default zeros are not measured utilization.")
    for name, flag in (("host_idle_time", "host_timeline_present"), ("device_idle_time", "device_timeline_present")):
        if not evidence[flag]:
            metrics[name].update(value=None, status="unavailable", source_field=None,
                reason="The profile lacks the required timeline evidence; default idle percentages are not measurements.")
    metrics["operation_timings"] = extract_operation_timings(raw.get("hlo_stats", {}))
    return {"measurements": metrics, "extraction_errors": errors,
        "counter_evidence": evidence,
        "xplane_files": [str(path.relative_to(trace_dir)) for path in files]}


def record_diagnostic_profile(state, training, group_jit, output, work, lr_multiplier, dispatches=4, updates_per_dispatch=16):
    """Consume disposable post-checkpoint state; never save diagnostic weights.

    Official scores, counters and checkpoint have already been recorded. This
    is a bounded profile of 64 updates, not a second training candidate.
    """
    begun = time.monotonic()
    trace_dir = Path(work) / "profile"
    trace_dir.mkdir(exist_ok=False)
    report = {"scope": "Post-checkpoint diagnostic; excluded from official training scores/tokens/time.",
        "requested_updates": dispatches * updates_per_dispatch, "completed_updates": 0,
        "learning_rate_multiplier": float(lr_multiplier), "diagnostic_weights_saved": False,
        "host_dispatch_gaps_ms": [], "dispatch_and_device_wait_ms": [],
        "host_wait_note": "Wall time around synchronized dispatch includes device work; it is not host/device idle time.",
        "status": "unavailable", "capture_error": None}
    tracing = False
    previous_end = None
    try:
        # JAX 0.12.0 exposes profiler options through ProfileOptions.
        options = jax.profiler.ProfileOptions()
        options.host_tracer_level = 2
        options.python_tracer_level = 0
        options.enable_hlo_proto = True
        options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC", "tpu_perf_counters": True}
        jax.profiler.start_trace(str(trace_dir), create_perfetto_trace=True, profiler_options=options)
        tracing = True
        for index in range(dispatches):
            begin = time.monotonic()
            if previous_end is not None:
                report["host_dispatch_gaps_ms"].append((begin - previous_end) * 1000)
            with jax.profiler.StepTraceAnnotation("diagnostic_train", step_num=index):
                state, values = jax.block_until_ready(group_jit(state, training,
                    jnp.array(updates_per_dispatch, jnp.int32), jnp.array(lr_multiplier, jnp.float32)))
            end = time.monotonic()
            metrics = jax.device_get(values)
            completed = int(metrics["updates"])
            report["completed_updates"] += completed
            report["dispatch_and_device_wait_ms"].append((end - begin) * 1000)
            if not bool(metrics["finite"]) or completed != updates_per_dispatch:
                raise FloatingPointError("Nonfinite or incomplete diagnostic dispatch")
            previous_end = end
        report["status"] = "captured"
    except Exception as error:
        report["capture_error"] = type(error).__name__ + ": " + str(error)
    finally:
        if tracing:
            try:
                jax.profiler.stop_trace()
            except Exception as error:
                report["stop_error"] = type(error).__name__ + ": " + str(error)
        del state
    report["capture_seconds"] = time.monotonic() - begun
    try:
        report.update(summarize_xprof(trace_dir, output))
    except Exception as error:
        report["extraction_errors"] = {"summary": type(error).__name__ + ": " + str(error)}
        report["measurements"] = extract_utilization({})
        report["measurements"]["operation_timings"] = extract_operation_timings({})
    destination = output / "profile"
    try:
        shutil.copytree(trace_dir, destination)
        report["trace_directory"] = str(destination)
    except Exception as error:
        report["trace_copy_error"] = type(error).__name__ + ": " + str(error)
        report["trace_directory"] = str(trace_dir)
    report["seconds_including_export_and_analysis"] = time.monotonic() - begun
    report["host_idle_note"] = "Host idle can mean normal waiting for TPU work; idle percentages alone do not establish a dispatch bottleneck."
    write_json(output / "profiling.json", report)
    return report


def parameter_digest(params):
    digest = hashlib.sha256()
    for path, value in sorted(traverse_util.flatten_dict(params).items()):
        array = np.asarray(jax.device_get(value))
        digest.update("/".join(path).encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(memoryview(array).cast("B"))
        del array
    return digest.hexdigest()


"""Training-scoped allocator samples, with explicit peak-counter limitations."""

def memory_snapshot(stats, elapsed):
    stats = stats or {}
    keys = ("bytes_in_use", "bytes_limit", "bytes_reserved", "peak_bytes_in_use", "peak_bytes_reserved")
    sample = {"training_elapsed_seconds": float(elapsed)}
    for key in keys:
        value = stats.get(key)
        sample[key] = int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None
    limit = sample["bytes_limit"]
    for key, result in (("bytes_in_use", "active_percent"), ("bytes_reserved", "reserved_percent")):
        sample[result] = 100. * sample[key] / limit if limit and sample[key] is not None else None
    return sample


def summarize_training_memory(before, samples, after, compiler, sampling_seconds):
    observed = [before, *samples, after]
    limits = [row["bytes_limit"] for row in observed if row.get("bytes_limit")]
    limit = limits[-1] if limits else None
    stable_limit = len(set(limits)) <= 1
    def maximum(key):
        values = [row[key] for row in observed if row.get(key) is not None]
        return max(values) if values else None
    active = maximum("bytes_in_use")
    reserved = maximum("bytes_reserved")
    old_peak, new_peak = before.get("peak_bytes_in_use"), after.get("peak_bytes_in_use")
    # Only a new lifetime high-water mark, strictly reached within the training
    # boundaries, establishes the actual allocator peak for that interval.
    isolated = old_peak is not None and new_peak is not None and new_peak > old_peak
    peak = new_peak if isolated else None
    required = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")
    estimate = (compiler[required[0]] + compiler[required[1]] - compiler[required[2]] + compiler[required[3]]) \
        if compiler and all(key in compiler for key in required) else None
    def percent(value):
        return 100. * value / limit if stable_limit and limit and value is not None else None
    return {
        "status": "available" if active is not None and limit and stable_limit else "unavailable",
        "scope": "JAX allocator counters, synchronized training boundaries and approximately 2-second progress samples; before final evaluation, restore, and profiling.",
        "device_limit_bytes": limit, "device_limit_gib": limit / 2**30 if limit else None,
        "device_limit_stable": stable_limit,
        "sampled_peak_active_bytes": active, "sampled_peak_active_gib": active / 2**30 if active is not None else None,
        "sampled_peak_active_percent": percent(active),
        "sampled_peak_reserved_bytes": reserved, "sampled_peak_reserved_percent": percent(reserved),
        "training_allocator_peak_bytes": peak, "training_allocator_peak_percent": percent(peak),
        "training_allocator_peak_status": "isolated_new_high_water_mark" if isolated else "not_isolatable",
        "lifetime_allocator_peak_before_training_bytes": old_peak,
        "lifetime_allocator_peak_after_training_bytes": new_peak,
        "compiled_peak_buffer_estimate_bytes": estimate,
        "compiled_peak_buffer_estimate_percent": percent(estimate),
        "compiler_estimate_formula": "arguments + outputs - aliased buffers + temporary buffers; an estimate, not measured device usage",
        "sampling_seconds_inside_training": sampling_seconds,
        "sample_count": len(observed), "samples": observed,
        "limitations": "Synchronized samples can miss temporary peaks. Lifetime counters include earlier phases unless a new peak was reached during training. Allocator counters may not expose every runtime-owned buffer. Reserved pool bytes are not active tensor bytes; compiler estimates are separate.",
    }


def train_pilot(work, output):
    work, output = Path(work), Path(output)
    version = PILOT_VERSION
    with (output / "training.started").open("x") as marker:
        marker.write(version + ": one training attempt; automatic retry disabled.\n")
    write_json(output / "attempt_status.json", {"stage": "pretraining", "optimizer_updates": 0,
        "timed_training_started": False, "version": version})
    environment = preflight()
    write_json(output / "preflight.json", environment)
    load_model_helpers()
    data_dir = work / "prepared"
    config = read_json(data_dir / "resolved_config.json")
    manifest = verify_raw(Path(config["shared_directory"]) / "raw")
    tokenization = read_json(data_dir / "tokenization.json")
    identity = config["tokenizer_identity"]
    if tokenization["raw_manifest_sha256"] != sha256_file(data_dir / "manifest.json"):
        raise ValueError("Tokenized and raw corpus manifests differ.")
    if shutil.disk_usage(work).free < 3 * 1024**3:
        raise RuntimeError("The weights-only artifact requires at least 3 GiB free local disk.")
    host, orders = {}, {}
    for split in ("train", "in_mix", "wiki"):
        name = split + ".npz"
        if sha256_file(data_dir / name) != tokenization["files"][name]:
            raise ValueError("Prepared input checksum failed: " + name)
        with np.load(data_dir / name, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in INPUT_KEYS}
        if int(arrays["byte_weights"].sum()) != manifest["original_bytes"][split]:
            raise ValueError("Prepared original-byte count differs: " + split)
        host[split], orders[split] = batch_training_arrays(
            arrays, config["local_batch_per_chip"], config["seed"])
    phase = time.monotonic()
    prefix = training_prefix_counts(host["train"])
    write_json(output / "training_accounting.json", {
        "method": "CPU batch prefix sums; completed update cursor; no timed per-token histograms",
        "training_batches": host["train"]["tokens"].shape[0],
        "tokens_per_pass": int(prefix["tokens"][-1]),
        "domain_bytes_per_pass": prefix["domain_bytes"][-1].tolist(),
        "language_bytes_per_pass": prefix["language_bytes"][-1].tolist(),
        "seconds": time.monotonic() - phase})
    training_host = {key: host["train"][key] for key in TRAIN_KEYS}
    eval_host = {key: np.concatenate((host["in_mix"][key], host["wiki"][key]), axis=0)
                 for key in INPUT_KEYS}
    eval_host["split_id"] = np.concatenate((
        np.zeros(host["in_mix"]["tokens"].shape[0], np.int32),
        np.ones(host["wiki"]["tokens"].shape[0], np.int32)))
    input_bytes = sum(value.nbytes for value in training_host.values())
    phase = time.monotonic()
    training, validation = jax.device_put((training_host, eval_host))
    jax.block_until_ready((training, validation))
    input_transfer_seconds = time.monotonic() - phase
    resident = all(jax.devices()[0] in leaf.devices() for leaf in jax.tree.leaves(training))
    del host, arrays, training_host, eval_host
    print(f"Training inputs on TPU: {input_bytes / 1024**2:.1f} MiB; CPU byte accounting prepared.", flush=True)
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment)

    phase = time.monotonic()
    model = make_model(config)
    if model.config.final_logit_softcap is not None:
        raise ValueError("This protocol requires native uncapped Gemma 270M logits.")
    params = jax.jit(lambda key: model.init(key, tokens=jnp.ones((1, 1), jnp.int32))["params"])(
        jax.random.PRNGKey(config["seed"]))
    jax.block_until_ready(params)
    flat = traverse_util.flatten_dict(params)
    parameter_count = sum(value.size for value in flat.values())
    embedding_shape = (identity["model_rows"], 640)
    tied = (flat[("embedder", "input_embedding")].shape == embedding_shape
            and sum(value.shape == embedding_shape for value in flat.values()) == 1)
    if parameter_count != identity["expected_parameters"] or not tied or not all(
            value.dtype == jnp.float32 for value in flat.values()):
        raise ValueError("Parameter count, tied table, or FP32 master weights differ.")
    backbone_digest = initial_backbone_digest(params)
    backbone_key = hashlib.sha256(json.dumps({"gemma": environment["packages"].get("gemma"),
        "jax": environment["jax"], "seed": config["seed"], "backbone": "Gemma3_270M"},
        sort_keys=True).encode()).hexdigest()[:16]
    fingerprint_path = Path(config["shared_directory"]) / ("backbone-" + backbone_key + ".json")
    fingerprint = {"sha256": backbone_digest, "transformer_and_norm_parameters": 100326016,
                   "seed": config["seed"]}
    if fingerprint_path.exists() and read_json(fingerprint_path) != fingerprint:
        raise ValueError("Transformer initialization differs from the shared baseline.")
    write_json(fingerprint_path, fingerprint)
    write_json(output / "backbone_initialization.json", fingerprint)
    probe_path = next(path for path, value in sorted(flat.items())
                      if path[0].startswith("layer_") and value.ndim > 1)
    def probe(parameters):
        return np.asarray(jax.device_get(traverse_util.flatten_dict(parameters)[probe_path].reshape(-1)[:64]))
    initial_probe = probe(params)
    optimizer, partition = make_optimizer(config, params)
    routing = partition.pop("routing")
    routing_ok = (sum(row["parameters"] for row in routing) == parameter_count
                  and all(row["optimizer"] == config["optimizer_label"] for row in routing)
                  and partition["muon_parameters"] + partition["adamw_parameters"] == parameter_count)
    if not routing_ok:
        raise ValueError("Optimizer routing does not cover every parameter with the requested optimizer.")
    write_json(output / "optimizer_parameter_routing.json", {"parameters": parameter_count,
        "optimizer": config["optimizer_label"], "all_parameters_accounted_for": routing_ok,
        "leaves": routing})
    write_json(output / "optimizer_partition.json", partition)
    state = PilotState(jnp.array(0, jnp.int32), params, optimizer.init(params),
        jax.random.PRNGKey(config["seed"]), jnp.array(0, jnp.int32))
    del params, flat
    jax.block_until_ready(state)
    initialization_seconds = time.monotonic() - phase
    protocol_report = verify_native_protocol()
    head_report = verify_exact_head(model, state.params, training)
    write_json(output / "head_verification.json", head_report)
    protocol_report["loss"] = "exact chunked full-vocabulary softmax, 128-token workspace"
    protocol_report["head_verified"] = head_report["passed"]
    write_json(output / "kernel_verification.json", protocol_report)
    _, group_jit = make_training_functions(model, optimizer)
    evaluation_jit = make_evaluator(model)
    print("Compiling one native training loop and one combined held-out evaluator; zero updates...", flush=True)
    phase = time.monotonic()
    executable = group_jit.lower(state, training, jnp.array(0, jnp.int32), jnp.array(1., jnp.float32)).compile()
    evaluator = evaluation_jit.lower(state.params, validation).compile()
    state, warmup_metrics = jax.block_until_ready(group_jit(state, training, jnp.array(0, jnp.int32), jnp.array(1., jnp.float32)))
    if int(warmup_metrics["updates"]) != 0 or int(state.step) != 0 or int(state.data_position) != 0:
        raise ValueError("The zero-update JIT preparation changed optimizer/data state.")
    compilation_seconds = time.monotonic() - phase
    write_json(output / "dispatch_preparation.json", {"dispatch": "jax.jit wrapper",
        "training_warmup_requested_updates": 0, "training_warmup_completed_updates": 0,
        "pretraining_optimizer_step": 0, "pretraining_data_position": 0,
        "training_executables": 1, "evaluation_executables": 1,
        "compiled_objects_used_for": "precompilation and memory inspection only"})
    try:
        analysis = executable.memory_analysis()
        compiled_memory = {name: int(getattr(analysis, name)) for name in
            ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")}
    except (AttributeError, TypeError, NotImplementedError):
        compiled_memory = None
    def evaluate(parameters):
        values = jax.block_until_ready(evaluation_jit(parameters, validation))
        return {split: readable_evaluation(tuple(value[index] for value in values),
                    manifest["original_bytes"][split])
                for index, split in enumerate(("in_mix", "wiki"))}
    phase = time.monotonic()
    before = evaluate(state.params)
    initial_eval_seconds = time.monotonic() - phase
    write_json(output / "timed_training.started.json", {"version": version,
        "stage": "timed training", "training_seconds": config["training_seconds"]})
    write_json(output / "attempt_status.json", {"version": version, "stage": "timed training",
        "timed_training_started": True, "optimizer_updates": None})
    print(f"Initial FineWiki macro BPB={before['wiki']['language_macro_bpb']:.5f}. Starting ONE 60-second {version} run...", flush=True)
    log_rows = []
    updates = tokens_processed = 0
    max_seconds_per_update = last_progress = last_start = 0.
    jax.block_until_ready(state)
    memory_before_training = memory_snapshot(jax.devices()[0].memory_stats(), 0.)
    training_memory_samples = []
    memory_sampling_seconds = 0.
    # Preserve completed dispatches even if a later call fails. Buffered log writes
    # contain scalar metrics only; source/language accounting runs after timing.
    with (output / "training.jsonl").open("x") as stream:
        started = time.monotonic()
        try:
            while True:
                remaining = config["training_seconds"] - (time.monotonic() - started)
                if remaining <= 0:
                    break
                group_size = config["updates_per_dispatch"]
                requested = group_size if updates > 0 and remaining > max(
                    2., 2 * group_size * max_seconds_per_update) else 1
                last_start = time.monotonic() - started
                if last_start >= config["training_seconds"]:
                    break
                lr_multiplier = cooldown_multiplier(last_start, config["training_seconds"],
                    config["cooldown_start_fraction"], config["learning_rate_floor_fraction"])
                call_start = time.monotonic()
                state, values = jax.block_until_ready(group_jit(state, training,
                    jnp.array(requested, jnp.int32), jnp.array(lr_multiplier, jnp.float32)))
                metrics = {key: np.asarray(value).tolist() for key, value in values.items()}
                completed = int(metrics["updates"])
                updates += completed
                tokens_processed += int(metrics["tokens"])
                end = time.monotonic() - started
                row = {**metrics, "update": updates, "requested_updates": requested,
                    "start_seconds": last_start, "end_seconds": end,
                    "learning_rate_multiplier": lr_multiplier,
                    "learning_rate_at_dispatch_start": config["learning_rate"] * lr_multiplier
                        * min((updates - completed) / config["warmup_updates"], 1.),
                    "loss": metrics["nll_sum"] / max(metrics["tokens"], 1)}
                log_rows.append(row)
                safe_row = {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                            for key, value in row.items()}
                stream.write(json.dumps(safe_row, allow_nan=False) + "\n")
                if not metrics["finite"] or completed != requested:
                    stream.flush()
                    raise FloatingPointError("Nonfinite loss/gradient; affected update rejected. No retry.")
                max_seconds_per_update = max(max_seconds_per_update,
                    (time.monotonic() - call_start) / completed)
                if len(log_rows) == 1 or end - last_progress >= 2:
                    sample_started = time.monotonic()
                    training_memory_samples.append(memory_snapshot(jax.devices()[0].memory_stats(), end))
                    memory_sampling_seconds += time.monotonic() - sample_started
                    print(f"updates={updates:,} tokens={tokens_processed:,} loss={row['loss']:.5f} elapsed={end:.2f}s", flush=True)
                    last_progress = end
            jax.block_until_ready(state)
            training_seconds = time.monotonic() - started
        except Exception:
            stream.flush()
            write_json(output / "attempt_status.json", {"version": version,
                "stage": "training failed", "timed_training_started": True,
                "optimizer_updates": updates, "data_position": int(state.data_position)})
            raise
    memory_after_training = memory_snapshot(jax.devices()[0].memory_stats(), training_seconds)
    training_memory = summarize_training_memory(memory_before_training, training_memory_samples,
        memory_after_training, compiled_memory, memory_sampling_seconds)
    write_json(output / "training_memory.json", training_memory)
    if not log_rows:
        raise RuntimeError("No optimizer updates completed.")
    domain_bytes = completed_prefix(prefix["domain_bytes"], updates)
    language_bytes = completed_prefix(prefix["language_bytes"], updates)
    expected_tokens = int(completed_prefix(prefix["tokens"], updates))
    if tokens_processed != expected_tokens:
        raise ValueError("Device token count differs from completed CPU batch prefix.")
    changed = not np.array_equal(initial_probe, probe(state.params))
    final_finite = bool(jax.device_get(jax.jit(lambda parameters: jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(parameters)])))(state.params)))
    phase = time.monotonic()
    after = evaluate(state.params)
    final_eval_seconds = time.monotonic() - phase
    memory_stats = jax.devices()[0].memory_stats()
    phase = time.monotonic()
    metadata = {"version": version, "protocol": PROTOCOL, "config": config,
        "parameter_count": parameter_count, "initialization": "random; zero pretrained weights",
        "backbone_initialization_sha256": backbone_digest, "optimizer": partition,
        "raw_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "tokenization_manifest_sha256": sha256_file(data_dir / "tokenization.json"),
        "packing_version": PACKING_VERSION, "data_order_sha256": orders,
        "training_batches": training["tokens"].shape[0], "updates": updates,
        "data_position": int(state.data_position), "checkpoint_fields": ["params"],
        "checkpoint_kind": "weights_only", "optimizer_resume_supported": False}
    print("Saving ONE FP32 weights-only artifact; optimizer resume is not included...", flush=True)
    local_checkpoint = work / "weights.npz"
    leaves = save_archive(local_checkpoint, {"params": state.params}, metadata)
    digest = sha256_file(local_checkpoint)
    partial = output / "weights.npz.partial"
    shutil.copyfile(local_checkpoint, partial)
    if sha256_file(partial) != digest:
        raise ValueError("Drive weights checksum differs from the local artifact.")
    checkpoint = output / "weights.npz"
    partial.rename(checkpoint)
    with zipfile.ZipFile(checkpoint) as archive:
        saved = json.loads(archive.read("checkpoint.json"))
        schema_verified = (set(saved["tree"]) == {"params"} and saved["metadata"] == metadata
            and sum(name.startswith("arrays/") for name in archive.namelist()) == leaves)
    if not schema_verified:
        raise ValueError("Saved weights schema or array count differs.")
    write_json(output / "checkpoint_manifest.json", {"sha256": digest,
        "bytes": checkpoint.stat().st_size, "array_leaves": leaves,
        "kind": "weights_only", "optimizer_resume_supported": False,
        "verification": "local/Drive SHA256 equality plus weights schema and leaf count",
        "restore_executed": False})
    checkpoint_seconds = time.monotonic() - phase
    restore_started = time.monotonic()
    original_digest = parameter_digest(state.params)
    restored = restore_archive(checkpoint, {"params": state.params})
    restored_digest = parameter_digest(restored["params"])
    exact_restore = original_digest == restored_digest
    reloaded_eval = evaluate(restored["params"])
    reload_bpb_error = max(abs(after[split]["languages"][language]["bpb"]
        - reloaded_eval[split]["languages"][language]["bpb"])
        for split in ("in_mix", "wiki") for language in after[split]["languages"])
    del restored
    restore_seconds = time.monotonic() - restore_started
    checkpoint_report = read_json(output / "checkpoint_manifest.json")
    checkpoint_report.update({"restore_executed": True, "exact_parameter_match": exact_restore,
        "parameter_sha256": original_digest, "restored_parameter_sha256": restored_digest,
        "reload_max_bpb_error": reload_bpb_error, "reload_bpb_tolerance": 1e-6,
        "restore_and_evaluation_seconds": restore_seconds})
    write_json(output / "checkpoint_manifest.json", checkpoint_report)
    bytes_processed = int(domain_bytes.sum())
    prefix_id = config["requirement_prefix"]
    checks = {
        "001_ONE_ATTEMPT": (output / "training.started").exists(),
        "002_ONE_V5E": environment["platform"] == "tpu" and environment["device_count"] == 1,
        "003_SOURCE_PINS": set(manifest["source_revisions"]) == set(SOURCE_REPOS),
        "004_FROZEN_CORPUS": tokenization["raw_manifest_sha256"] == sha256_file(data_dir / "manifest.json"),
        "005_SPLIT_SEPARATION": manifest["cross_split_source_groups"] == 0,
        "006_LOSSLESS_TOKENIZATION": all(tokenization["splits"][split]["original_bytes"] == manifest["original_bytes"][split]
                                          for split in ("train", "in_mix", "wiki")),
        "007_INTACT_PACKING": all(tokenization["splits"][split]["targets_preserved"] and
            tokenization["splits"][split]["context_preserved"] for split in ("train", "in_mix", "wiki")),
        "008_NATIVE_ATTENTION_MASK": protocol_report["passed"],
        "009_RESIDENT_INPUTS": resident,
        "010_GEMMA_BACKBONE": parameter_count == identity["expected_parameters"] and parameter_count < 300000000 and tied,
        "011_OPTIMIZER_ROUTING": routing_ok,
        "012_TIMED_DISPATCH": all(row["start_seconds"] < config["training_seconds"] for row in log_rows),
        "013_DISPATCH_BOUND": all(1 <= row["updates"] <= 16 for row in log_rows),
        "014_FINITE_UPDATES": all(row["finite"] for row in log_rows) and final_finite,
        "015_ZERO_EXTRA_UPDATES": int(state.step) == updates and int(state.data_position) == updates,
        "016_FIXED_EVALUATION": len(after["wiki"]["languages"]) == 12 and all(
            after[split]["original_utf8_bytes"] == before[split]["original_utf8_bytes"] for split in ("in_mix", "wiki")),
        "017_WEIGHTS_ARTIFACT": schema_verified,
        "018_CHANGED_WEIGHTS": changed,
        "019_MINIMUM_UPDATES": updates >= config["minimum_updates"],
        "020_BYTE_ACCOUNTING": bytes_processed == int(language_bytes.sum()) and bytes_processed > 0 and expected_tokens == tokens_processed,
        "025_NATIVE_TRAINING": config["local_batch_per_chip"] == 2 and config["sequence_length"] == 1024,
        "026_COMBINED_EVALUATOR": validation["split_id"].shape[0] == validation["tokens"].shape[0],
        "027_CPU_ACCOUNTING": "byte_weights" not in training and "domain_ids" not in training and "language_ids" not in training,
        "028_WEIGHTS_ONLY": saved["metadata"]["optimizer_resume_supported"] is False,
        "029_EXACT_HEAD": head_report["passed"],
        "030_RESTORED_WEIGHTS": exact_restore and reload_bpb_error <= 1e-6,
        "031_COOLDOWN": all(.3 <= row["learning_rate_multiplier"] <= 1. for row in log_rows),
    }
    checks = {prefix_id + "-" + key: value for key, value in checks.items()}
    checks.update({key: value == "PASS" for key, value in read_json(output / "preparation_requirements.json").items()})
    downloads = read_json(output / "parallel_downloads.json")
    elapsed = time.monotonic() - config["notebook_start_monotonic"]
    rate = config["compute_units_per_hour"]
    baseline_applicable = (identity["tokenizer_sha256"] == "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c"
        and sha256_file(data_dir / "manifest.json") == "74d6d6305b893aa9403e8ac30c10b9d920c23bc7d3c9d020debbd5b3500d707a")
    goals = {"tokens_per_second_at_least_45559": tokens_processed / training_seconds >= 45559.448,
        "compilation_seconds_at_most_60": compilation_seconds <= 60.,
        "notebook_seconds_at_most_300": elapsed <= 300.,
        "occupancy_at_least_95_percent": tokens_processed / (updates * 2 * 1024) >= .95,
        "finewiki_macro_bpb_at_most_v35": after["wiki"]["language_macro_bpb"] <= 2.9347027242183685
            if baseline_applicable else None}
    improvement = ("PASS" if all(value is True for value in goals.values()) else "FAIL") if baseline_applicable else "NOT_COMPARABLE"
    result = {"version": version, "protocol": PROTOCOL, "run_count": 1, "config": config,
        "device": environment["device_kind"], "model_parameters": parameter_count, "optimizer": partition,
        "source_downloads": {key: downloads[key] for key in
            ("source_jobs", "max_workers", "peak_active_source_jobs", "fetch_seconds", "raw_cache_reused")},
        "initial_backbone_sha256": backbone_digest, "input_transfer_seconds": input_transfer_seconds,
        "resident_training_bytes": input_bytes, "initialization_seconds": initialization_seconds,
        "kernel_verification_seconds": protocol_report["seconds"], "compilation_seconds": compilation_seconds,
        "compilation_timing_scope": "one training and one combined evaluation executable plus zero-update JIT dispatch",
        "dispatch_preparation": read_json(output / "dispatch_preparation.json"),
        "initial_eval_seconds": initial_eval_seconds, "final_eval_seconds": final_eval_seconds,
        "training_seconds": training_seconds,
        "training_overrun_seconds": max(0., training_seconds - config["training_seconds"]),
        "last_dispatch_start_seconds": last_start, "host_dispatches": len(log_rows),
        "updates": updates, "scored_tokens": tokens_processed,
        "tokens_per_second": tokens_processed / training_seconds,
        "processed_original_bytes_attributed": bytes_processed,
        "original_bytes_per_second_attributed": bytes_processed / training_seconds,
        "byte_accounting_note": "CPU batch-prefix sums using tokenizer offset attribution; repeated passes included, not unique bytes",
        "training_domain_byte_shares": {name: int(domain_bytes[index]) / bytes_processed for index, name in enumerate(DOMAINS)},
        "training_language_byte_shares": {name: int(language_bytes[index]) / bytes_processed for index, name in enumerate(LANGUAGES)},
        "training_occupancy": tokens_processed / (updates * 2 * 1024),
        "prepared_occupancy": tokenization["splits"]["train"]["occupancy"],
        "completed_data_passes": updates // training["tokens"].shape[0],
        "training_batches": training["tokens"].shape[0], "data_order_sha256": orders,
        "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "initial_eval": before, "final_eval": after,
        "primary_metric": "FineWiki mean of 12 per-language BPB values; lower is better",
        "checkpoint_kind": "weights_only", "optimizer_resume_supported": False,
        "checkpoint_save_verify_seconds": checkpoint_seconds, "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": digest, "checkpoint_array_leaves": leaves,
        "memory_stats": memory_stats, "compiled_memory_estimate": compiled_memory,
        "training_memory": training_memory,
        "memory_scope": "Allocator snapshot after evaluation; compiler estimate separately",
        "notebook_elapsed_seconds": elapsed,
        "compute_units_estimate": {"rate_per_hour": rate, "training": training_seconds * rate / 3600,
            "through_report": elapsed * rate / 3600, "overhead_through_report": (elapsed - training_seconds) * rate / 3600},
        "cost_note": "2.92 CU/hour supplied by user; connected runtime after report adds cost; no USD conversion assumed",
        "historical_v2_tokens_per_second": 45559.448,
        "historical_comparison_note": "V2 used different data and vocabulary; V3.4 used different kernels/batch/optimizer; neither isolates an optimizer effect",
        "v35_baseline_applicable": baseline_applicable, "performance_goals": goals,
        "baseline": {"version": "V3.5", "finewiki_macro_bpb": 2.9347027242183685,
            "tokens_per_second": 45734.29605429257, "training_seconds": 60.014611282999795},
        "head": {"kind": "exact_token_chunked", "chunk_tokens": HEAD_CHUNK_TOKENS,
            "logical_fp32_logits_workspace_bytes": HEAD_CHUNK_TOKENS * identity["model_rows"] * 4,
            "note": "Logical single-chunk logits only; compiler peak memory is reported separately."},
        "learning_rate_schedule": {"warmup_updates": config["warmup_updates"],
            "peak": config["learning_rate"], "cooldown_start_fraction": .5, "floor_fraction": .3,
            "scope": "Elapsed training time, multiplier held constant within each dispatch; scales AdamW update including decay.",
            "last_dispatch_multiplier": log_rows[-1]["learning_rate_multiplier"]},
        "checkpoint_restore_seconds": restore_seconds, "reloaded_eval": reloaded_eval,
        "reload_max_bpb_error": reload_bpb_error, "exact_parameter_restore": exact_restore,
        "head_verification_seconds": head_report["seconds"],
        "report_finalized": False,
        "performance_goal_status": improvement,
        "requirements": {key: "PASS" if passed else "FAIL" for key, passed in checks.items()},
        "engineering_status": "PASS" if all(checks.values()) else "FAIL",
        "learning_status": "PASS" if after["wiki"]["language_macro_bpb"] < before["wiki"]["language_macro_bpb"] else "FAIL"}
    write_json(output / "result.json", result)
    write_json(output / "attempt_status.json", {"version": version, "stage": "result saved",
        "timed_training_started": True, "optimizer_updates": updates})
    print(json.dumps({name: result[name] for name in ("engineering_status", "learning_status",
        "performance_goal_status", "updates", "training_seconds", "tokens_per_second",
        "original_bytes_per_second_attributed")}, indent=2), flush=True)
    print(f"FineWiki macro BPB: {before['wiki']['language_macro_bpb']:.6f} -> {after['wiki']['language_macro_bpb']:.6f}", flush=True)
    if not all(checks.values()):
        raise RuntimeError(version + " engineering requirement failed; result.json was saved.")
    # The official checkpoint, scores and checks are final before consuming this
    # disposable in-memory state. No diagnostic weights are saved or ranked.
    write_json(output / "attempt_status.json", {"version": version, "stage": "profiling",
        "timed_training_started": True, "optimizer_updates": updates})
    profile = record_diagnostic_profile(state, training, group_jit, output, work,
        log_rows[-1]["learning_rate_multiplier"], config["profile_dispatches"],
        config["profile_updates_per_dispatch"])
    del state
    result["profiling"] = profile
    available = sum(value["status"] == "available" for value in profile["measurements"].values())
    result["telemetry_status"] = "COMPLETE" if available == 5 else "PARTIAL" if available else "UNAVAILABLE"
    result["telemetry_available_measurements"] = available
    result["notebook_elapsed_before_profile_seconds"] = elapsed
    elapsed = time.monotonic() - config["notebook_start_monotonic"]
    result["notebook_elapsed_seconds"] = elapsed
    result["compute_units_estimate"].update({"through_report": elapsed * rate / 3600,
        "overhead_through_report": (elapsed - training_seconds) * rate / 3600,
        "diagnostic_profile": profile["seconds_including_export_and_analysis"] * rate / 3600})
    result["performance_goals"]["notebook_seconds_at_most_300"] = elapsed <= 300.
    result["performance_goal_status"] = ("PASS" if all(value is True for value in result["performance_goals"].values())
        else "FAIL") if baseline_applicable else "NOT_COMPARABLE"
    result["report_finalized"] = True
    write_json(output / "result.json", result)
    write_json(output / "attempt_status.json", {"version": version, "stage": "complete",
        "timed_training_started": True, "optimizer_updates": updates,
        "diagnostic_updates": profile["completed_updates"], "telemetry_status": result["telemetry_status"]})
    print("Profiler telemetry:", result["telemetry_status"], "| diagnostic updates:", profile["completed_updates"], flush=True)


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
        if (args.output / "failure.json").exists():
            write_json(args.output / ("failure-secondary-" + str(time.time_ns()) + ".json"),
                {"phase": args.phase, "error": traceback.format_exc(), "automatic_retry": False})
            raise
        write_json(args.output / "failure.json", {"phase": args.phase, "error": traceback.format_exc(), "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()
