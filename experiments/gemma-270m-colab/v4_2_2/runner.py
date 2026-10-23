"""Gemma Colab V4.2.2: one fixed multilingual tokenizer-screening pilot on one v5e TPU.
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

STACK = "bigcode/the-stack-smol"








ENCODING_SCHEMA = "v421-native-byte-document-encoding-1"



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


MODEL_HELPERS = '@struct.dataclass\nclass PilotState:\n    step: jax.Array\n    params: object\n    opt_state: object\n    rng: jax.Array\n    data_position: jax.Array\n    short_rows: jax.Array\n    long_rows: jax.Array\n'

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

PROTOCOL = "V39-expanded-byte-mix-1"
PACKING_VERSION = "V3-intact-chunks-1"
LANGUAGES = ["eng_Latn", "fra_Latn", "spa_Latn", "por_Latn", "cmn_Hani", "arb_Arab",
    "hin_Deva", "swh_Latn", "zul_Latn", "xho_Latn", "tsn_Latn", "sot_Latn",
    "rus_Cyrl", "jpn_Jpan", "deu_Latn", "ibo_Latn"]
WIKI_CODES = ["en", "fr", "es", "pt", "zh", "ar", "hi", "sw", "zu", "xh", "tn", "st", "ru", "ja", "de", "ig"]
DOMAINS = ["pdf_edu", "dclm", "web_edu", "multilingual_web", "math", "code_python",
    "code_java", "code_cpp", "code_javascript", "code_rust", "code_html", "translations", "wiki_holdout"]
SMOL = "HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled"
SOURCE_REPOS = [SMOL, "HuggingFaceFW/fineweb-2", "HuggingFaceTB/finemath",
    "codeparrot/codeparrot-clean", "HuggingFaceFW/finetranslations", STACK, "HuggingFaceFW/finewiki"]
INPUT_KEYS = ("tokens", "targets", "mask", "positions", "segment_ids", "text_mask", "byte_weights", "language_ids", "domain_ids")
EXTRA_KEYS = ("text_mask", "byte_weights", "language_ids", "domain_ids")


def recipe():
    buckets = [dict(key=name, repo=SMOL, subset=None, language="eng_Latn", domain=name, bp=bp)
        for name, bp in (("pdf_edu", 2500), ("dclm", 1500), ("web_edu", 1000))]
    shares = [250, 200, 150, 350, 350, 350, 200, 100, 100, 100, 80, 250, 250, 170, 100]
    buckets += [dict(key="fw2_" + lang, repo=SOURCE_REPOS[1], subset=lang, language=lang,
        domain="multilingual_web", bp=bp) for lang, bp in zip(LANGUAGES[1:], shares)]
    buckets.append(dict(key="math", repo=SOURCE_REPOS[2], subset="finemath-3plus",
        language="eng_Latn", domain="math", bp=500))
    buckets.append(dict(key="code_python", repo=SOURCE_REPOS[3], subset=None,
        language="eng_Latn", domain="code_python", bp=300))
    for key, subset, bp in (("java", "java", 200), ("cpp", "c++", 200),
            ("javascript", "javascript", 150), ("rust", "rust", 100), ("html", "html", 50)):
        buckets.append(dict(key="code_" + key, repo=STACK, subset=subset,
            language="eng_Latn", domain="code_" + key, bp=bp))
    buckets += [dict(key="translations_" + lang, repo=SOURCE_REPOS[4], subset=lang,
        language=lang, domain="translations", bp=125)
        for lang in ("fra_Latn", "arb_Arab", "hin_Deva", "swh_Latn")]
    assert sum(bucket["bp"] for bucket in buckets) == 10000
    return {"protocol": PROTOCOL, "training_bytes_target": 256 * 1024**2,
        "in_mix_eval_bytes_target": 1024**2, "wiki_eval_bytes_per_language": 64 * 1024,
        "max_document_span_bytes": 32 * 1024, "byte_quota_tolerance": 3,
        "source_weight_unit": "original UTF-8 bytes", "buckets": buckets,
        "split_rule": "SHA256(source group) modulo 10: 0 held out, 1..9 train",
        "sampling_scope": "bounded deterministic stream prefixes; not uniform full-corpus sampling",
        "code_language_note": "Code has its own six domain labels; eng_Latn is an accounting placeholder for code, not detected natural-language identity."}


def protocol_fingerprint():
    return hashlib.sha256(json.dumps(recipe(), sort_keys=True).encode()).hexdigest()


def canonical_group(row, text):
    metadata = row.get("metadata")
    url = row.get("url") or (metadata.get("url") if isinstance(metadata, dict) else None)
    if url:
        parsed = urlsplit(str(url))
        return "url:" + urlunsplit(("", parsed.netloc.lower().removeprefix("www."), parsed.path.rstrip("/"), parsed.query, ""))
    if row.get("repo_name") or row.get("repository_name"):
        return "code-repo:" + str(row.get("repo_name") or row["repository_name"])
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



"""V4.2.2 parallel source preparation. Embedded in the notebook; not run locally."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading

DOWNLOAD_WORKERS = 6
PREFIX_CACHE_VERSION = "v39-expanded-source-prefix-1"
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
            texts = [(row.get("content") if domain.startswith("code_") else row.get("text"), bucket["language"])]
    group = canonical_group(row, texts[0][0] or "")
    if job["kind"] != "wiki" and "wikipedia.org" in group:
        return []
    split = "wiki" if job["kind"] == "wiki" else "in_mix" if int(hashlib.sha256(group.encode()).hexdigest()[:16], 16) % 10 == 0 else "train"
    provenance = {key: row.get(key) for key in ("id", "url", "repo_name", "repository_name", "path", "license", "licenses", "lang", "dataset") if row.get(key) is not None}
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


def materialize_prefix(job, root, factor, report_status, stop, required_strata=None):
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
    active = set(job["targets"]) if required_strata is None else set(required_strata)
    if not active <= set(job["targets"]):
        raise ValueError("Unknown source-cache stratum requested.")
    desired = {key: math.ceil(value * factor) for key, value in job["targets"].items() if key in active}
    cache_limit = prefix_cache_limit(job)
    def complete():
        return all(state["candidate_bytes"].get(key, 0) >= quota and state["candidate_documents"].get(key, 0) >= 4
                   for key, quota in desired.items())
    def status(stage):
        report_status(job["key"], {"stage": stage, "rows": state["rows_read"],
            "candidate_bytes": sum(state["candidate_bytes"].values()), "seconds": time.monotonic() - started})
    def result():
        return {"directory": str(directory), "rows_read": state["rows_read"], "candidate_parts": len(state["parts"]),
            "candidate_bytes": sum(state["candidate_bytes"].values()), "prefix_cache_reused": original_parts > 0,
            "cache_limit_bytes": cache_limit, "cached_json_bytes": sum(part["bytes"] for part in state["parts"]),
            "candidate_bytes_by_stratum": dict(state["candidate_bytes"]),
            "selection_policy": "v391-quota-filtered-prefix", "requested_strata": sorted(active),
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
        if job["repo"] == STACK:
            from huggingface_hub import HfApi
            paths = sorted(path for path in HfApi().list_repo_files(STACK, repo_type="dataset", revision=job["revision"])
                if path.startswith("data/" + job["subset"] + "/") and path.endswith(".json"))
            if not paths:
                raise ValueError("Pinned code language files missing: " + job["subset"])
            kwargs.pop("name")
            kwargs.pop("revision")
            kwargs["data_files"] = {"train": ["hf://datasets/" + STACK + "@" + job["revision"] + "/" + path for path in paths]}
            stream = load_dataset("json", **kwargs)
        else:
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
            records = select_prefix_records(candidate_records(job, row), state, desired)
            encoded_records = [(json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8") for record in records]
            projected_json_bytes = committed_bytes + len(buffer) + sum(map(len, encoded_records))
            if projected_json_bytes > cache_limit:
                raise RuntimeError(f"Source-prefix disk cache limit reached: {job['key']}; "
                    f"projected={projected_json_bytes:,}, limit={cache_limit:,} bytes; "
                    f"candidate UTF-8 bytes by stratum={state['candidate_bytes']}. "
                    "Completed source batches preserved; no training started.")
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
        write_json(destination.parent / "parallel_downloads.json", {"version": "V4.2.2",
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
    print(f"V4.2.2: {len(jobs)} source jobs; {DOWNLOAD_WORKERS} concurrent workers; completed batches persist in Drive.", flush=True)
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
            missing = [key for key in targets if targets[key] - actual[key] > rules["byte_quota_tolerance"]]
            report = materialize_prefix(job, cache_root, factor, status, refill_stop, required_strata=missing)
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
    write_json(shared / "parallel_downloads.json", {"version": "V4.2.2", "max_workers": DOWNLOAD_WORKERS,
        "source_jobs": len(jobs), "peak_active_source_jobs": peak_active,
        "fetch_seconds": fetch_seconds, "prepare_raw_seconds": time.monotonic() - started,
        "jobs": reports, "merge_order": [job["key"] for job in jobs],
        "initial_prefix_factor": PREFIX_FACTOR, "minimum_candidate_documents_per_stratum": 4,
        "max_cached_bytes_per_job": max(prefix_cache_limit(job) for job in jobs),
        "cached_bytes_limit_by_job": {job["key"]: prefix_cache_limit(job) for job in jobs},
        "prefix_selection": "stop caching each satisfied stratum; refill deficient strata only",
        "persistent_batch_target_bytes": PREFIX_PART_BYTES,
        "selection": "canonical V3 source order; independent of worker completion order",
        "throughput_speedup_measured": False})
    # The final manifest is the completion marker; it is committed last.
    write_json(destination / "manifest.json", manifest)
    return manifest


def prepare_tokenizer(config, destination):
    import base64
    import tiktoken
    if (config["tokenizer_reference"] != "o200k_base"
            or config["tokenizer_revision"] != "tiktoken-0.14.0"
            or config["model_id"] != "google/gemma-3-270m"
            or importlib.metadata.version("tiktoken") != "0.14.0"):
        raise ValueError("V4.2.2 requires native tiktoken 0.14.0 o200k_base and the Gemma3_270M backbone.")
    destination.mkdir(parents=True, exist_ok=True)
    cache = Path(config["tokenizer_cache_directory"])
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
    tokenizer = tiktoken.get_encoding("o200k_base")
    ranks = tokenizer._mergeable_ranks
    specials = tokenizer._special_tokens
    rank_file = b"".join(base64.b64encode(piece) + b" " + str(rank).encode() + b"\n"
        for piece, rank in sorted(ranks.items(), key=lambda item: item[1]))
    if hashlib.sha256(rank_file).hexdigest() != O200K_RANKS_SHA256:
        raise ValueError("Native o200k_base mergeable ranks differ from the official pinned artifact.")
    asset = {"pat_str": tokenizer._pat_str,
        "mergeable_ranks": [[base64.b64encode(piece).decode(), rank] for piece, rank in ranks.items()],
        "special_tokens": specials}
    payload = (json.dumps(asset, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != O200K_ASSET_SHA256:
        raise ValueError("Native o200k_base pattern, ranks or special-token map differ from the project pin.")
    (destination / "o200k_base.json").write_bytes(payload)
    ids = sorted(set(ranks.values()) | set(specials.values()))
    if len(ranks) != 199998 or len(ids) != 200000 or max(ids) != 200018:
        raise ValueError("Unexpected o200k_base vocabulary inventory.")
    mapping = {old: index + 1 for index, old in enumerate(ids)}
    rows = len(ids) + 3
    total = 100326016 + rows * 640
    if rows != 200003 or total != 228327936:
        raise ValueError("Unexpected model size after native o200k vocabulary adaptation.")
    info = {"reference": "o200k_base", "revision": "tiktoken-0.14.0",
        "implementation": "native tiktoken", "tokenizer_sha256": O200K_ASSET_SHA256,
        "mergeable_ranks_sha256": O200K_RANKS_SHA256, "vocabulary_ids": len(ids),
        "mergeable_token_ids": len(ranks), "native_special_token_ids": specials,
        "original_id_space": max(ids) + 1, "unused_native_ids": max(ids) + 1 - len(ids),
        "model_rows": rows, "pad_id": 0, "bos_id": len(ids) + 1, "eos_id": len(ids) + 2,
        "id_mapping": "sorted defined native IDs map to 1..N; holes have no model row; PAD=0, BOS=N+1, EOS=N+2",
        "expected_parameters": total, "model_type": "native_byte_level_BPE",
        "text_encoding_policy": "encode_ordinary; native special-token spellings remain ordinary text",
        "byte_accounting": "Exact length of each native token byte piece; handles UTF-8 splits without character offsets"}
    info["vocabulary_adaptation"] = {
        "backbone": "Gemma3_270M", "backbone_parameters": 100326016,
        "hidden_size": 640, "native_tokenizer_ids": len(ids), "protocol_tokens": 3,
        "embedding_shape": [rows, 640], "embedding_and_head_tied": True,
        "embedding_parameters": rows * 640, "total_parameters": total,
        "v4_parameters": 268100736, "parameter_delta_from_v4": total - 268100736,
        "automatic_native_special_tokens": False,
        "packing_policy": "Defined native IDs -> 1..200000; PAD=0, BOS=200001, EOS=200002.",
        "initialization": "Fresh random weights; shared seed/backbone digest; no checkpoint transfer.",
        "cache_policy": "Reuse frozen raw bytes; native o200k encoding and context packings use separate hash/schema identities."}
    write_json(destination / "identity.json", info)
    write_json(destination / "native_ids_in_dense_order.json", ids)
    print(f"Native o200k_base: {len(ids):,} defined IDs; tied table {rows:,} x 640; model {total:,} parameters.", flush=True)
    return tokenizer, mapping, info




"""Sequence-aware cache identity and input validation, embedded in the runner."""


def validate_training_config(config):
    if (config["sequence_length"] != 1024 or config["long_sequence_length"] != 2048
            or config["local_batch_per_chip"] != 3 or config["long_batch_per_chip"] != 1
            or config["training_seconds"] != 1200. or config["updates_per_dispatch"] != 16
            or config["short_phase_tokens"] != 3000000 or config["long_phase_tokens"] != 1000000
            or config["batch_discovery_enabled"] is not True
            or config["discovery_start_after_updates"] != 1000
            or config["discovery_window_updates"] != 16 or config["discovery_max_updates"] != 256
            or config["safety_batch_cap"] != 32):
        raise ValueError("V4.2.2 starts at 3x1024 / 1x2048; discovery follows 1000 updates, for at most 256 updates.")
    if not 0 < config["memory_budget_fraction"] < 1:
        raise ValueError("Memory admission must leave device headroom.")


def tokenization_cache_key(tokenizer_sha256, raw_manifest_sha256, sequence_length):
    identity = {"schema": "sequence-aware-token-cache-v3-native-tiktoken-bytes", "tokenizer_sha256": tokenizer_sha256,
        "raw_manifest_sha256": raw_manifest_sha256, "packing_version": PACKING_VERSION,
        "sequence_length": int(sequence_length)}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]


def validate_tokenization_manifest(details, identity, raw_digest, sequence_length):
    expected_key = tokenization_cache_key(identity["tokenizer_sha256"], raw_digest, sequence_length)
    if (details.get("sequence_length") != sequence_length
            or details.get("packing_version") != PACKING_VERSION
            or details.get("cache_key") != expected_key
            or details.get("raw_manifest_sha256") != raw_digest
            or details.get("tokenizer", {}).get("tokenizer_sha256") != identity["tokenizer_sha256"]):
        raise ValueError("Prepared tokenization metadata does not match context, tokenizer, corpus and packing. Existing cache preserved.")


def validate_sequence_arrays(arrays, sequence_length, split):
    shape = arrays["tokens"].shape
    if len(shape) != 2 or shape[0] < 1 or shape[1] != sequence_length:
        raise ValueError(f"{split}: expected nonempty [rows, {sequence_length}] inputs, found {shape}.")
    if any(arrays[key].shape != shape for key in INPUT_KEYS):
        raise ValueError(f"{split}: token, mask, position or accounting array shapes disagree.")


def training_specification_matches(config, training):
    return (config["local_batch_per_chip"] == 3 and config["long_batch_per_chip"] == 1
        and config["training_seconds"] == 1200. and config["batch_discovery_enabled"] is True
        and config["discovery_start_after_updates"] == 1000
        and training["short"]["tokens"].shape[1:] == (1024,)
        and training["long"]["tokens"].shape[1:] == (2048,))


def prepare_cached(config, work, output):
    validate_training_config(config)
    started = time.monotonic()
    work, output = Path(work), Path(output)
    shared = output.parent.parent / "V3-shared" / protocol_fingerprint()[:16]
    shared.mkdir(parents=True, exist_ok=True)
    lock = immutable_sources(shared)
    raw = shared / "raw"
    raw_reused = raw.exists()
    if not raw_reused:
        check_source_access(lock)
        staging = shared / "raw.partial"
        build_raw_corpus(staging, lock)
        staging.rename(raw)
    manifest = verify_raw(raw)
    prepared = work / "prepared"
    prepared.mkdir(exist_ok=False)
    if raw_reused:
        downloads = {"version": "V4.2.2", "raw_cache_reused": True, "source_jobs": 0,
            "max_workers": DOWNLOAD_WORKERS, "peak_active_source_jobs": 0, "fetch_seconds": 0.}
    else:
        downloads = read_json(shared / "parallel_downloads.json")
        downloads["raw_cache_reused"] = False
    write_json(output / "parallel_downloads.json", downloads)
    expected_jobs = len(LANGUAGES) + len({(b["repo"], b["subset"]) for b in recipe()["buckets"]})
    checks = {"V4.2.2-021_DOWNLOAD_BOUND": downloads["peak_active_source_jobs"] <= 6,
        "V4.2.2-022_PREFIX_CACHE": raw_reused or all((Path(item["directory"]) / "state.json").exists()
            for item in downloads.get("jobs", {}).values()),
        "V4.2.2-023_MERGE_ORDER": raw_reused or downloads["merge_order"][:len(LANGUAGES)] == ["wiki_" + lang for lang in LANGUAGES],
        "V4.2.2-024_SOURCE_COMPLETE": raw_reused or downloads["source_jobs"] == expected_jobs == len(downloads["jobs"])}
    write_json(output / "preparation_requirements.json", {key: "PASS" if value else "FAIL" for key, value in checks.items()})
    if not all(checks.values()):
        raise ValueError("Expanded data preparation failed validation.")
    tokenizer, mapping, identity = prepare_tokenizer(config, prepared / "tokenizer")
    raw_digest = sha256_file(raw / "manifest.json")
    contexts = {}
    missing_packing = any(not (shared / "tokenized" / tokenization_cache_key(
        identity["tokenizer_sha256"], raw_digest, length)).exists() for length in (1024, 2048))
    encoding_directory = None
    encoding_report = {"status": "not_needed", "reason": "Both matching packed context caches already exist.",
        "schema": ENCODING_SCHEMA, "encode_passes_this_run": 0}
    if missing_packing:
        encoding_directory, encoding_details, encoding_reused = prepared_encoding(
            shared, raw, work, tokenizer, mapping, identity, raw_digest)
        for split, counts in encoding_details["splits"].items():
            if counts["original_bytes"] != manifest["original_bytes"][split]:
                raise ValueError("Encoded original byte total differs: " + split)
        encoding_report = {"status": "available", "schema": ENCODING_SCHEMA,
            "cache_reused": encoding_reused, "encode_passes_this_run": 0 if encoding_reused else 1,
            "encoding_manifest": encoding_details}
        write_json(output / "encoding_manifest.json", encoding_details)
    write_json(output / "encoding_preparation.json", encoding_report)
    for length in (1024, 2048):
        destination = prepared / str(length)
        destination.mkdir(exist_ok=False)
        key = tokenization_cache_key(identity["tokenizer_sha256"], raw_digest, length)
        cache = shared / "tokenized" / key
        reused = cache.exists()
        if reused:
            details = read_json(cache / "tokenization.json")
            validate_tokenization_manifest(details, identity, raw_digest, length)
            for name, digest in details["files"].items():
                if sha256_file(cache / name) != digest:
                    raise ValueError("Context tokenization cache checksum mismatch: " + name)
                shutil.copyfile(cache / name, destination / name)
        else:
            details = {"tokenizer": identity, "raw_manifest_sha256": raw_digest, "sequence_length": length,
                "packing_version": PACKING_VERSION, "cache_key": key, "splits": {}, "files": {}}
            # Evaluate only at the fixed 1024 context. Long packing is training-only.
            for split in (("train", "in_mix", "wiki") if length == 1024 else ("train",)):
                arrays, packing = pack_encoded_split(encoding_directory, split, identity, length)
                validate_sequence_arrays(arrays, length, split)
                name = split + ".npz"
                np.savez_compressed(destination / name, **arrays)
                details["splits"][split] = {k: v for k, v in packing.items() if not k.startswith("assignments_")}
                details["files"][name] = sha256_file(destination / name)
                print(f"{length}: {split}: {packing['text_tokens']:,} text targets, occupancy={packing['occupancy']:.2%}", flush=True)
                del arrays
            write_json(destination / "tokenization.json", details)
            temporary = cache.with_name(key + ".partial-" + str(os.getpid()))
            temporary.mkdir(parents=True, exist_ok=False)
            for name in [*details["files"], "tokenization.json"]:
                shutil.copyfile(destination / name, temporary / name)
            temporary.rename(cache)
        write_json(destination / "tokenization.json", details)
        contexts[str(length)] = {"manifest": details, "cache_reused": reused, "cache_key": key}
        write_json(output / f"tokenization_{length}.json", details)
    # Top-level metadata remains the fixed-context evaluation metadata.
    shutil.copyfile(prepared / "1024/tokenization.json", prepared / "tokenization.json")
    config = {**config, "tokenizer_identity": identity, "shared_directory": str(shared)}
    write_json(prepared / "resolved_config.json", config)
    shutil.copyfile(raw / "manifest.json", prepared / "manifest.json")
    shutil.copyfile(raw / "manifest.json", output / "data_manifest.json")
    shutil.copyfile(shared / "source-lock.json", output / "source-lock.json")
    shutil.copytree(prepared / "tokenizer", output / "tokenizer")
    shutil.copyfile(prepared / "tokenizer/identity.json", output / "tokenizer_identity.json")
    write_json(output / "vocabulary_adaptation.json", identity["vocabulary_adaptation"])
    write_json(output / "preparation.json", {"raw_cache_reused": raw_reused, "contexts": contexts,
        "encoding": encoding_report,
        "preparation_seconds": time.monotonic() - started, "raw_manifest_sha256": raw_digest,
        "actual_training_bytes": manifest["original_bytes"]["train"], "target_training_bytes": 256 * 1024**2,
        "tokenizer_sha256": identity["tokenizer_sha256"], "shared_directory": str(shared)})


def load_model_helpers(cpu_test=False):
    accelerator_imports(cpu_test)
    exec(MODEL_HELPERS, globals())


def make_model(config):
    # Only the tied vocabulary table changes; retain the V4 transformer architecture.
    from gemma import gm
    model = gm.nn.Gemma3_270M(dtype=jnp.float32)
    return model.clone(config=dataclasses.replace(model.config, num_embed=config["tokenizer_identity"]["model_rows"]))


PILOT_VERSION = 'V4.2.2'

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


def readable_evaluation(values, raw_bytes, sequence_length):
    ln, lb, dn, db, eos = [np.asarray(value) for value in values]
    if int(round(float(lb.sum()))) != raw_bytes or not np.isfinite(ln).all():
        raise ValueError("Evaluation original-byte denominator or finite-loss check failed.")
    languages = {language: {"bpb": float(ln[i] / (lb[i] * math.log(2))), "nll_nats": float(ln[i]),
                            "original_utf8_bytes": int(lb[i])} for i, language in enumerate(LANGUAGES) if lb[i] > 0}
    domains = {domain: {"bpb": float(dn[i] / (db[i] * math.log(2))), "original_utf8_bytes": int(db[i])}
               for i, domain in enumerate(DOMAINS) if db[i] > 0}
    return {"pooled_bpb": float(ln.sum() / (lb.sum() * math.log(2))),
        "language_macro_bpb": float(np.mean([value["bpb"] for value in languages.values()])),
        "legacy_12_language_macro_bpb": float(np.mean([languages[lang]["bpb"] for lang in LANGUAGES[:12] if lang in languages])),
        "languages": languages, "domains": domains, "original_utf8_bytes": raw_bytes,
        "eos_nll_nats_separate": float(eos), "scoring": f"all original text targets once; virtual EOS excluded from BPB; fixed {sequence_length}-token chunk context"}


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


def token_losses(model, params, batch):
    compute = jax.tree.map(lambda value: value.astype(jnp.bfloat16), params)
    logits = model.apply({"params": compute}, tokens=batch["tokens"],
        positions=batch["positions"],
        attention_mask=packed_attention_mask(batch["segment_ids"])).logits.astype(jnp.float32)
    with jax.named_scope("cross_entropy"):
        selected = jnp.take_along_axis(logits, batch["targets"][..., None], axis=-1)[..., 0]
        return jax.scipy.special.logsumexp(logits, axis=-1) - selected


def make_training_functions(model, optimizer, batch_size=1, mode="short"):
    cursor_field = "short_rows" if mode == "short" else "long_rows"
    def one_step(state, batch):
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
                next_params = optax.apply_updates(state.params, delta)
            return state.replace(step=state.step + 1, params=next_params, opt_state=new_opt,
                data_position=state.data_position + 1,
                **{cursor_field: getattr(state, cursor_field) + batch_size})
        return jax.lax.cond(finite, apply, lambda _: state, operand=None), (loss, count, norm, finite)
    def group(state, training, requested):
        initial = (state, jnp.array(0, jnp.int32), jnp.array(0., jnp.float32),
                   jnp.array(0, jnp.int32), jnp.array(0., jnp.float32), jnp.array(True))
        def condition(carry):
            return (carry[1] < requested) & carry[5]
        def body(carry):
            state, steps, nll, count, max_norm, finite = carry
            indices = (getattr(state, cursor_field) + jnp.arange(batch_size)) % training["tokens"].shape[0]
            batch = jax.tree.map(lambda values: values[indices], training)
            state, (loss, tokens, norm, good) = one_step(state, batch)
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
        "scope": "JAX allocator counters, synchronized training boundaries and approximately 2-second progress samples; before final evaluation and profiling.",
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


"""XProf measured-metric parsing; compiler estimates are reported separately.

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
    if "vocabulary_projection" in text or ("embedder" in text and "dot_general" in text) or "/decode/" in text:
        return "vocabulary_projection"
    if "cross_entropy" in text or "logsumexp" in text:
        return "cross_entropy"
    if any(key in text for key in ("/mlp/", "feedforward", "gating_einsum", "linear_einsum")):
        return "mlp"
    if any(key in text for key in ("/attn/", "q_einsum", "kv_einsum", "attn_vec")):
        return "attention"
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


def historical_comparison(config, tokenizer_hash, corpus_hash, throughput, bpb):
    same_data = (tokenizer_hash == "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c"
        and corpus_hash == "74d6d6305b893aa9403e8ac30c10b9d920c23bc7d3c9d020debbd5b3500d707a")
    return {"reference_version": "V3.5", "reference_tokens_per_second": 45734.29605429257,
        "reference_finewiki_macro_bpb": 2.9347027242183685,
        "reference_batch": 2, "reference_sequence_length": 1024,
        "reference_training_seconds": 60.014611282999795,
        "current_sequence_length": config["sequence_length"],
        "same_evaluation_context_length": config["sequence_length"] == 1024,
        "current_training_contexts": [1024, 2048], "current_corpus": "expanded 256 MiB",
        "current_batch": config["local_batch_per_chip"], "current_budget_seconds": config["training_seconds"],
        "same_tokenizer_and_corpus": same_data,
        "controlled_same_budget_comparison": False,
        "throughput_ratio_to_reference": throughput / 45734.29605429257 if same_data else None,
        "bpb_difference_from_reference": bpb - 2.9347027242183685 if same_data else None,
        "note": "Descriptive historical comparison only. Corpus, language coverage, batch schedule, training context and duration differ. BPB and throughput do not isolate implementation efficiency."}


def extract_roofline_estimates(roofline):
    """Use the aggregate Program row and its own peak units, never sum rates."""
    tables = roofline if isinstance(roofline, list) else [roofline]
    table = next((x for x in tables if isinstance(x, dict) and "cols" in x
                  and "peak_flop_rate" in x.get("p", {})), None)
    missing = {"status": "unavailable", "kind": "compiler_cost_model_estimate",
        "reason": "No supported aggregate Program row and hardware peaks in roofline export."}
    if table is None:
        return missing
    row = next((r for r in _datatable_rows(table) if r.get("category") == "Program"
                and r.get("operation") == "Program" and r.get("step") == "Total"), None)
    if row is None:
        return missing
    flop = _number(row.get("modelfloprate"))
    hbm = _number(row.get("hbmbw"))
    peak_flop = _number(table["p"].get("peak_flop_rate"))
    peak_hbm = _number(table["p"].get("peak_hbm_bw"))
    if any(x is None or x <= 0 for x in (peak_flop, peak_hbm)) or flop is None or hbm is None:
        return missing
    return {"status": "available", "kind": "compiler_cost_model_estimate", "source": "XProf roofline_model aggregate Program row",
        "estimated_compute_tflops_per_second": flop / 1000., "peak_compute_tflops_per_second": peak_flop / 1000.,
        "estimated_compute_percent_of_peak": 100 * flop / peak_flop,
        "estimated_hbm_gib_per_second": hbm, "peak_hbm_gib_per_second": peak_hbm,
        "estimated_hbm_percent_of_peak": 100 * hbm / peak_hbm,
        "estimated_bound_by": row.get("boundby"),
        "note": "Compiler FLOP/byte costs divided by observed duration. Not hardware-counter measurements or MXU busy time. Estimates may exceed physical peak; never clamp them."}


def raw_counter_evidence(profile_data):
    rows = []
    if profile_data is not None:
        for plane in profile_data.planes:
            for line in plane.lines:
                if "counter" in line.name.lower():
                    rows.append({"plane": plane.name, "line": line.name,
                                 "event_count": sum(1 for _ in line.events)})
    return {"tracks": rows, "counter_events": sum(row["event_count"] for row in rows)}


def profiling_requirements(metrics, training_memory):
    def available(name):
        return metrics.get(name, {}).get("status") == "available"
    ops = metrics.get("operation_timings", {})
    categories = ops.get("categories", {})
    checks = {
        "matrix_unit_utilization": available("matrix_unit_utilization"),
        "memory_bandwidth_utilization": available("memory_bandwidth_utilization"),
        "host_and_device_idle_time": available("host_idle_time") and available("device_idle_time"),
        "operation_timings": available("operation_timings") and all(
            categories.get(key, {}).get("self_time_us") is not None
            for key in ("attention", "vocabulary_projection", "optimizer_updates")),
        "training_memory": training_memory.get("status") == "available",
    }
    count = sum(checks.values())
    return {"checks": {k: "PASS" if v else "FAIL" for k, v in checks.items()},
        "requirements_status": "PASS" if all(checks.values()) else "FAIL",
        "telemetry_status": "COMPLETE" if all(checks.values()) else "PARTIAL" if count else "UNAVAILABLE",
        "available_requirement_groups": count, "required_groups": len(checks)}


def summarize_xprof(trace_dir, output, profile_data, completed_updates):
    files = sorted(Path(trace_dir).rglob("*.xplane.pb")) + sorted(Path(trace_dir).rglob("*.xplane.riegeli"))
    raw, errors = {}, {}
    if files:
        try:
            from xprof.convert import raw_to_tool_data
            # Reuse OpStats created by the first conversion; do not rebuild it
            # four times or generate the redundant op_profile/Perfetto exports.
            for index, tool in enumerate(("overview_page", "hlo_stats", "roofline_model")):
                try:
                    payload, _ = raw_to_tool_data.xspace_to_tool_data(
                        [str(p) for p in files], tool, {"use_saved_result": index > 0})
                    if payload is None:
                        raise ValueError("XProf returned no tool data")
                    raw[tool] = _decode_tool_json(payload)
                    write_json(output / "profile_tools" / (tool + ".json"), raw[tool])
                except Exception as error:
                    errors[tool] = type(error).__name__ + ": " + str(error)
        except Exception as error:
            errors["converter"] = type(error).__name__ + ": " + str(error)
    else:
        errors["capture"] = "No raw TPU XPlane files were produced."
    metrics = extract_utilization(raw.get("overview_page", {}))
    evidence = {"hardware_counter_message_present": False, "device_timeline_present": False,
                "device_type": None, "raw_counters": None}
    try:
        if profile_data is None and files and files[0].suffix == ".pb":
            profile_data = jax.profiler.ProfileData.from_file(str(files[0]))
        evidence["raw_counters"] = raw_counter_evidence(profile_data)
    except Exception as error:
        errors["raw_counter_audit"] = type(error).__name__ + ": " + str(error)
    try:
        from xprof.protobuf import op_stats_pb2
        candidates = sorted(Path(trace_dir).rglob("ALL_HOSTS.op_stats*.pb"))
        if candidates:
            stats = op_stats_pb2.OpStats()
            stats.ParseFromString(candidates[-1].read_bytes())
            evidence["device_type"] = stats.run_environment.device_type
            tpu = "TPU" in evidence["device_type"].upper()
            evidence["device_timeline_present"] = tpu and stats.device_op_metrics_db.total_time_ps > 0
            evidence["hardware_counter_message_present"] = tpu and stats.HasField("performance_counter_result")
            if evidence["hardware_counter_message_present"]:
                for name, field in (("matrix_unit_utilization", "matrix_unit_utilization_percent"),
                                    ("memory_bandwidth_utilization", "hbm_utilization_percent")):
                    value = float(getattr(stats.performance_counter_result, field))
                    if math.isfinite(value) and 0 <= value <= 100:
                        metrics[name].update(value=value, status="available", reason=None,
                            source="XProf OpStats.performance_counter_result", source_field=field)
    except Exception as error:
        errors["counter_evidence"] = type(error).__name__ + ": " + str(error)
    for name in ("matrix_unit_utilization", "memory_bandwidth_utilization"):
        if not evidence["hardware_counter_message_present"]:
            metrics[name].update(value=None, status="unavailable", source_field=None,
                reason="No supported TPU utilization-counter message was exported. Default zeros are not measurements; inspect raw-counter audit. Runtime cause is not established.")
    if not evidence["device_timeline_present"]:
        metrics["device_idle_time"].update(value=None, status="unavailable", source_field=None,
            reason="No TPU device timeline supports the overview idle percentage.")
    metrics["device_idle_time"]["scope"] = "XProf device-operation timeline idle; not unused MXU capacity inside active operations."
    reported_host_idle = metrics["host_idle_time"].get("value")
    metrics["host_idle_time"].update(value=None, status="unavailable",
        reason="XProf host-op idle accounting does not establish Python/CPU idle or blocked dispatch time. See separately measured host wait and dispatch gaps.")
    metrics["host_idle_time"]["xprof_host_op_idle_percent_context_only"] = reported_host_idle
    ops = extract_operation_timings(raw.get("hlo_stats", {}))
    for category in ops["categories"].values():
        duration = category["self_time_us"]
        category["milliseconds_per_update"] = duration / (1000 * completed_updates) if duration is not None and completed_updates else None
    metrics["operation_timings"] = ops
    return {"measurements": metrics, "estimated_utilization": extract_roofline_estimates(raw.get("roofline_model", {})),
        "counter_evidence": evidence, "extraction_errors": errors,
        "xplane_files": [str(p.relative_to(trace_dir)) for p in files]}


def record_diagnostic_profile(state, training, group_jit, output, work, dispatches=4, updates_per_dispatch=8):
    """One bounded capture of the existing executable; official weights stay saved."""
    begun = time.monotonic()
    trace_dir = Path(work) / "profile"
    trace_dir.mkdir(exist_ok=False)
    report = {"scope": "Post-checkpoint diagnostic, excluded from official training scores/tokens/time. No diagnostic weights saved.",
        "requested_updates": dispatches * updates_per_dispatch, "completed_updates": 0,
        "capture_status": "FAIL", "capture_error": None, "diagnostic_weights_saved": False,
        "host_dispatch_gaps_ms": [], "host_enqueue_ms": [], "host_sync_wait_ms": [],
        "dispatch_and_device_wait_ms": [], "capture_options": {
            "host_tracer_level": 2, "python_tracer_level": 0, "enable_hlo_proto": True,
            "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC", "tpu_perf_counters": True,
            "create_perfetto_trace": False}}
    tracing, previous_end, profile_data = False, None, None
    try:
        options = jax.profiler.ProfileOptions()
        options.host_tracer_level = 2
        options.python_tracer_level = 0
        options.enable_hlo_proto = True
        options.raise_error_on_start_failure = True
        options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC", "tpu_perf_counters": True}
        jax.profiler.start_trace(str(trace_dir), create_perfetto_trace=False, profiler_options=options)
        tracing = True
        window_start = time.monotonic()
        for index in range(dispatches):
            begin = time.monotonic()
            if previous_end is not None:
                report["host_dispatch_gaps_ms"].append((begin - previous_end) * 1000)
            with jax.profiler.StepTraceAnnotation("diagnostic_train", step_num=index):
                enqueue_start = time.monotonic()
                pending = group_jit(state, training, jnp.array(updates_per_dispatch, jnp.int32))
                enqueue_end = time.monotonic()
                state, values = jax.block_until_ready(pending)
                end = time.monotonic()
            report["host_enqueue_ms"].append((enqueue_end - enqueue_start) * 1000)
            report["host_sync_wait_ms"].append((end - enqueue_end) * 1000)
            report["dispatch_and_device_wait_ms"].append((end - begin) * 1000)
            metrics = jax.device_get(values)
            completed = int(metrics["updates"])
            report["completed_updates"] += completed
            if not bool(metrics["finite"]) or completed != updates_per_dispatch:
                raise FloatingPointError("Nonfinite or incomplete diagnostic dispatch")
            previous_end = end
        report["diagnostic_window_seconds"] = time.monotonic() - window_start
        report["capture_status"] = "PASS"
    except Exception as error:
        report["capture_error"] = type(error).__name__ + ": " + str(error)
    finally:
        if tracing:
            try:
                profile_data = jax.profiler.stop_trace()
            except Exception as error:
                report["stop_error"] = type(error).__name__ + ": " + str(error)
                report["capture_status"] = "FAIL"
        del state
    report["capture_seconds"] = time.monotonic() - begun
    try:
        report.update(summarize_xprof(trace_dir, output, profile_data, report["completed_updates"]))
    except Exception as error:
        report["extraction_errors"] = {"summary": type(error).__name__ + ": " + str(error)}
        report["measurements"] = extract_utilization({})
        report["measurements"]["operation_timings"] = extract_operation_timings({})
        report["estimated_utilization"] = {"status": "unavailable", "reason": "Profile summary failed."}
    window = report.get("diagnostic_window_seconds", 0.)
    report["host_timing"] = {
        "enqueue_wall_ms": sum(report["host_enqueue_ms"]),
        "synchronization_call_wall_ms": sum(report["host_sync_wait_ms"]),
        "between_dispatch_wall_ms": sum(report["host_dispatch_gaps_ms"]),
        "synchronization_call_percent_of_window": 100 * sum(report["host_sync_wait_ms"]) / (1000 * window) if window else None,
        "note": "Host wall-clock durations around enqueue and synchronous completion APIs, including API overhead. These are not OS CPU-idle percentages; synchronization can overlap TPU execution."}
    destination = output / "profile"
    try:
        shutil.copytree(trace_dir, destination)
        report["trace_directory"] = str(destination)
    except Exception as error:
        report["trace_copy_error"] = type(error).__name__ + ": " + str(error)
        report["trace_directory"] = str(trace_dir)
    training_memory = read_json(output / "training_memory.json")
    report.update(profiling_requirements(report["measurements"], training_memory))
    if report["capture_status"] != "PASS":
        report["requirements_status"] = "FAIL"
    report["seconds_including_export_and_analysis"] = time.monotonic() - begun
    write_json(output / "profiling.json", report)
    return report




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
    validate_training_config(config)
    manifest = verify_raw(Path(config["shared_directory"]) / "raw")
    tokenization = read_json(data_dir / "tokenization.json")
    identity = config["tokenizer_identity"]
    validate_tokenization_manifest(tokenization, identity, sha256_file(data_dir / "manifest.json"), config["sequence_length"])
    if tokenization["raw_manifest_sha256"] != sha256_file(data_dir / "manifest.json"):
        raise ValueError("Tokenized and raw corpus manifests differ.")
    if shutil.disk_usage(work).free < 3 * 1024**3:
        raise RuntimeError("The weights-only artifact requires at least 3 GiB free local disk.")
    host, orders, prefixes, context_metadata = {}, {}, {}, {}
    for mode, length in (("short", 1024), ("long", 2048)):
        directory = data_dir / str(length)
        details = read_json(directory / "tokenization.json")
        validate_tokenization_manifest(details, identity, sha256_file(data_dir / "manifest.json"), length)
        context_metadata[mode] = details
        if sha256_file(directory / "train.npz") != details["files"]["train.npz"]:
            raise ValueError("Training input checksum mismatch: " + mode)
        with np.load(directory / "train.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in INPUT_KEYS}
        validate_sequence_arrays(arrays, length, mode)
        if int(arrays["byte_weights"].sum()) != manifest["original_bytes"]["train"]:
            raise ValueError("Training byte attribution mismatch: " + mode)
        arrays, orders[mode] = row_training_arrays(arrays, config["seed"])
        prefixes[mode] = row_prefix_counts(arrays)
        host[mode] = {key: arrays[key] for key in TRAIN_KEYS}
        del arrays
    eval_parts = {}
    for split in ("in_mix", "wiki"):
        directory = data_dir / "1024"
        name = split + ".npz"
        if sha256_file(directory / name) != tokenization["files"][name]:
            raise ValueError("Evaluation checksum mismatch: " + split)
        with np.load(directory / name, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in INPUT_KEYS}
        validate_sequence_arrays(arrays, 1024, split)
        if int(arrays["byte_weights"].sum()) != manifest["original_bytes"][split]:
            raise ValueError("Evaluation byte attribution mismatch: " + split)
        eval_parts[split], _ = batch_training_arrays(arrays, 1, config["seed"])
    eval_host = {key: np.concatenate((eval_parts["in_mix"][key], eval_parts["wiki"][key]), axis=0)
        for key in INPUT_KEYS}
    eval_host["split_id"] = np.concatenate((np.zeros(eval_parts["in_mix"]["tokens"].shape[0], np.int32),
        np.ones(eval_parts["wiki"]["tokens"].shape[0], np.int32)))
    input_bytes_by_mode = {mode: sum(value.nbytes for value in values.values()) for mode, values in host.items()}
    input_bytes = sum(input_bytes_by_mode.values())
    eval_bytes = sum(value.nbytes for value in eval_host.values())
    phase = time.monotonic()
    training, validation = jax.device_put((host, eval_host))
    jax.block_until_ready((training, validation))
    input_transfer_seconds = time.monotonic() - phase
    resident = all(jax.devices()[0] in leaf.devices() for leaf in jax.tree.leaves(training))
    del host, arrays, eval_parts, eval_host
    config["other_resident_bytes_short"] = input_bytes_by_mode["long"] + eval_bytes
    config["other_resident_bytes_long"] = input_bytes_by_mode["short"] + eval_bytes
    print(f"Both context inputs on TPU: {input_bytes / 1024**2:.1f} MiB; exact CPU row accounting prepared.", flush=True)
    write_json(output / "training_accounting.json", {"method": "Independent context row cursors, fixed shuffled order; no batch padding or cursor reset",
        "contexts": {mode: {"rows_per_pass": training[mode]["tokens"].shape[0],
            "tokens_per_pass": int(prefixes[mode]["tokens"][-1]), "training_input_bytes": input_bytes_by_mode[mode]}
            for mode in training}, "evaluation_context": 1024,
        "note": "Both contexts reuse the same frozen text. Byte exposure includes repeats, not unique corpus bytes."})
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
        jax.random.PRNGKey(config["seed"]), jnp.array(0, jnp.int32),
        jnp.array(0, jnp.int32), jnp.array(0, jnp.int32))
    del params, flat
    jax.block_until_ready(state)
    initialization_seconds = time.monotonic() - phase
    protocol_report = verify_native_protocol()
    write_json(output / "kernel_verification.json", protocol_report)
    functions, analyses, all_analyses, compilation_events = {}, {"short": {}, "long": {}}, {}, []
    def compile_shape(mode, batch, current_state, scope):
        if scope == "inside_training_budget":
            if int(current_state.step) < config["discovery_start_after_updates"] and not config.get("early_memory_discovery"):
                raise ValueError("Discovery compilation attempted before 1000 updates without a memory fallback.")
        elif scope != "before_training" or ((mode, batch) not in (("short", 3), ("long", 1))
                and not (config.get("early_memory_discovery") and batch == 1)):
            raise ValueError("Unexpected initial compilation shape or scope.")
        key = (mode, batch)
        if key in functions:
            return all_analyses[key]
        begun = time.monotonic()
        _, group = make_training_functions(model, optimizer, batch, mode)
        executable = group.lower(current_state, training[mode], jnp.array(0, jnp.int32)).compile()
        analysis = compiler_memory(executable)
        functions[key] = executable
        all_analyses[key] = analysis
        analyses[mode][batch] = analysis
        peak = compiler_peak(analysis)
        other = config["other_resident_bytes_" + mode]
        compilation_events.append({"mode": mode, "batch": batch, "sequence_length": 1024 if mode == "short" else 2048,
            "seconds": time.monotonic() - begun, "scope": scope, "official_step_at_compile": int(current_state.step),
            "compiler_memory": analysis, "compiler_peak_plus_other_resident_bytes": peak + other if peak is not None else None})
        write_json(output / "compilation_events.json", compilation_events)
        return analysis
    print("Compiling fixed 3x1024, fixed 1x2048 and 1024-context evaluator; zero optimizer updates...", flush=True)
    phase = time.monotonic()
    for mode, batch in (("short", 3), ("long", 1)):
        def admit_initial(selected_batch):
            analysis = compile_shape(mode, selected_batch, state, "before_training")
            limit = (jax.devices()[0].memory_stats() or {}).get("bytes_limit")
            peak = compiler_peak(analysis)
            if peak is not None and limit and peak + config["other_resident_bytes_" + mode] > limit * config["memory_budget_fraction"]:
                raise BatchMemoryError("Required initial context exceeds the conservative memory budget: " + mode)
            return analysis
        try:
            admit_initial(batch)
        except Exception as error:
            memory_discovery_fallback(config, mode, batch, error, "initial_compilation", int(state.step))
            admit_initial(1)
    evaluation_jit = make_evaluator(model)
    evaluator = evaluation_jit.lower(state.params, validation).compile()
    initial_short_batch = config.get("discovery_initial_batches", {"short": 3})["short"]
    state, warmup_metrics = jax.block_until_ready(functions[("short", initial_short_batch)](state, training["short"], jnp.array(0, jnp.int32)))
    if int(state.step) != 0 or int(state.short_rows) != 0 or int(state.long_rows) != 0:
        raise ValueError("Compilation preparation performed an optimizer update.")
    compilation_seconds = time.monotonic() - phase
    write_json(output / "dispatch_preparation.json", {"dispatch": "compiled executable",
        "training_warmup_requested_updates": 0, "training_warmup_completed_updates": int(warmup_metrics["updates"]),
        "pretraining_optimizer_step": int(state.step), "initial_training_executables": len(functions),
        "evaluation_executables": 1, "initial_requested_shapes": [[3, 1024], [1, 2048]], "adaptive_compilation": "after 1000 updates, or earlier after a recoverable memory failure; inside training budget"})
    def evaluate(parameters):
        values = jax.block_until_ready(evaluator(parameters, validation))
        return {split: readable_evaluation(tuple(value[index] for value in values),
                    manifest["original_bytes"][split], config["sequence_length"])
                for index, split in enumerate(("in_mix", "wiki"))}
    phase = time.monotonic()
    before = evaluate(state.params)
    initial_eval_seconds = time.monotonic() - phase
    write_json(output / "timed_training.started.json", {"version": version,
        "stage": "timed training", "training_seconds": config["training_seconds"]})
    write_json(output / "attempt_status.json", {"version": version, "stage": "timed training",
        "timed_training_started": True, "optimizer_updates": None})
    print(f"Initial FineWiki macro BPB={before['wiki']['language_macro_bpb']:.5f}. Starting ONE {config['training_seconds']:g}-second {version} run, batch {initial_short_batch} x {config['sequence_length']}...", flush=True)
    state, adaptive = delayed_train(state, training, prefixes, functions, analyses, compile_shape, config, output)
    log_rows = adaptive["rows"]
    updates, tokens_processed = adaptive["updates"], adaptive["tokens"]
    training_seconds, last_start = adaptive["training_seconds"], adaptive["last_start"]
    used_keys = {(row["mode"], row["batch"]) for row in log_rows}
    used_analyses = [all_analyses[key] for key in used_keys if all_analyses[key] is not None]
    compiled_memory = max(used_analyses, key=compiler_peak) if used_analyses else None
    training_memory = summarize_training_memory(adaptive["memory_before"], adaptive["samples"],
        adaptive["memory_after"], compiled_memory, adaptive["sampling_seconds"])
    peaks_with_other = [compiler_peak(all_analyses[key]) + config["other_resident_bytes_" + key[0]]
        for key in used_keys if all_analyses[key] is not None]
    training_memory["compiled_peak_plus_other_resident_bytes"] = max(peaks_with_other) if peaks_with_other else None
    limit = training_memory["device_limit_bytes"]
    training_memory["compiled_peak_plus_other_resident_percent"] = (
        max(peaks_with_other) * 100 / limit if peaks_with_other and limit else None)
    training_memory["compiler_estimates_by_shape"] = compilation_events
    write_json(output / "training_memory.json", training_memory)
    if not log_rows:
        raise RuntimeError("No optimizer updates completed.")
    domain_bytes = sum(completed_prefix(prefixes[mode]["domain_bytes"], rows)
        for mode, rows in adaptive["rows_used"].items())
    language_bytes = sum(completed_prefix(prefixes[mode]["language_bytes"], rows)
        for mode, rows in adaptive["rows_used"].items())
    expected_tokens = sum(int(completed_prefix(prefixes[mode]["tokens"], rows))
        for mode, rows in adaptive["rows_used"].items())
    if (tokens_processed != expected_tokens or int(state.short_rows) != adaptive["rows_used"]["short"]
            or int(state.long_rows) != adaptive["rows_used"]["long"]):
        raise ValueError("CPU/device token or context cursor accounting differs.")
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
        "training_rows_per_context": {mode: values["tokens"].shape[0] for mode, values in training.items()}, "updates": updates,
        "context_rows_consumed": adaptive["rows_used"], "final_short_batch": adaptive["short_batch"],
        "final_long_batch": adaptive["long_batch"], "batch_policy": adaptive["batch_policy"],
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
        "016_FIXED_EVALUATION": len(after["wiki"]["languages"]) == len(LANGUAGES) and all(
            after[split]["original_utf8_bytes"] == before[split]["original_utf8_bytes"] for split in ("in_mix", "wiki")),
        "017_WEIGHTS_ARTIFACT": schema_verified,
        "018_CHANGED_WEIGHTS": changed,
        "019_MINIMUM_UPDATES": updates >= config["minimum_updates"],
        "020_BYTE_ACCOUNTING": bytes_processed == int(language_bytes.sum()) and bytes_processed > 0 and expected_tokens == tokens_processed,
        "025_NATIVE_TRAINING": training_specification_matches(config, training),
        "026_COMBINED_EVALUATOR": validation["split_id"].shape[0] == validation["tokens"].shape[0],
        "027_CPU_ACCOUNTING": all(not set(EXTRA_KEYS) & set(values) for values in training.values()),
        "030_INITIAL_BATCHES": all(row["batch"] == (3 if row["mode"] == "short" else 1)
            and row["update"] <= 1000 for row in log_rows if row["phase"] == "initial_batches"),
        "031_TOKEN_SCHEDULE": all(row["mode"] == context_phase(row["total_tokens"] - row["tokens"], row["short_batch"])
            for row in log_rows if row["phase"] != "discovery"),
        "033_DELAYED_BOUNDED_DISCOVERY": (not adaptive["batch_policy"]["discovery_started"] or
            ((adaptive["batch_policy"]["started_at_update"] == 1000 or adaptive["batch_policy"]["early_memory_trigger"])
             and adaptive["batch_policy"]["discovery"]["discovery_updates"] <= 256)),
        "034_LOCKED_BATCHES": all(row["batch"] == adaptive[row["mode"] + "_batch"]
            for row in log_rows if row["phase"] == "locked_training"),
        "035_COMPILE_TRIGGER": all(event["scope"] == "before_training" or event["official_step_at_compile"] >= 1000
            or adaptive["batch_policy"]["early_memory_trigger"] for event in compilation_events),
        "032_CONTEXT_CURSORS": int(state.short_rows) == adaptive["rows_used"]["short"] and int(state.long_rows) == adaptive["rows_used"]["long"],
        "028_WEIGHTS_ONLY": saved["metadata"]["optimizer_resume_supported"] is False,
        "029_NO_RESTORE_CHECK": read_json(output / "checkpoint_manifest.json")["restore_executed"] is False,
    }
    checks = {prefix_id + "-" + key: value for key, value in checks.items()}
    checks.update({key: value == "PASS" for key, value in read_json(output / "preparation_requirements.json").items()})
    downloads = read_json(output / "parallel_downloads.json")
    elapsed = time.monotonic() - config["notebook_start_monotonic"]
    rate = config["compute_units_per_hour"]
    comparison = historical_comparison(config, identity["tokenizer_sha256"],
        sha256_file(data_dir / "manifest.json"), tokens_processed / training_seconds,
        after["wiki"]["language_macro_bpb"])
    goals = {"throughput_at_least_v35_reference": tokens_processed / training_seconds >= 45734.29605429257,
        "pretraining_compilation_seconds_at_most_180": compilation_seconds <= 180.,
        "notebook_seconds_at_most_1800": elapsed <= 1800.,
        "occupancy_at_least_95_percent": tokens_processed / adaptive["positions"] >= .95}
    improvement = "PASS" if all(goals.values()) else "FAIL"
    result = {"version": version, "protocol": PROTOCOL, "run_count": 1, "config": config,
        "device": environment["device_kind"], "model_parameters": parameter_count, "optimizer": partition,
        "tokenizer_identity": identity, "vocabulary_adaptation": identity["vocabulary_adaptation"],
        "source_downloads": {key: downloads[key] for key in
            ("source_jobs", "max_workers", "peak_active_source_jobs", "fetch_seconds", "raw_cache_reused")},
        "initial_backbone_sha256": backbone_digest, "input_transfer_seconds": input_transfer_seconds,
        "resident_training_bytes": input_bytes, "initialization_seconds": initialization_seconds,
        "kernel_verification_seconds": protocol_report["seconds"], "compilation_seconds": compilation_seconds,
        "compilation_timing_scope": "initial 3x1024/1x2048 and evaluator before training; discovery compilation after 1000 updates or recoverable memory trigger counts inside training",
        "compilation_events": compilation_events,
        "adaptive_compilation_seconds_inside_training": sum(e["seconds"] for e in compilation_events if e["scope"] == "inside_training_budget"),
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
        "training_occupancy": tokens_processed / adaptive["positions"],
        "prepared_occupancy": tokenization["splits"]["train"]["occupancy"],
        "completed_data_passes_by_context": {mode: rows / training[mode]["tokens"].shape[0] for mode, rows in adaptive["rows_used"].items()},
        "final_short_batch": adaptive["short_batch"], "final_long_batch": adaptive["long_batch"],
        "batch_policy": adaptive["batch_policy"],
        "maximum_short_batch_executed": max((row["batch"] for row in log_rows if row["mode"] == "short"), default=0),
        "maximum_long_batch_executed": max((row["batch"] for row in log_rows if row["mode"] == "long"), default=0),
        "context_rows_consumed": adaptive["rows_used"], "batch_checks": adaptive["checks"],
        "context_transitions": adaptive["transitions"], "data_order_sha256": orders,
        "tokens_by_context": {mode: sum(row["tokens"] for row in log_rows if row["mode"] == mode) for mode in training},
        "updates_by_context": {mode: sum(row["updates"] for row in log_rows if row["mode"] == mode) for mode in training},
        "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "initial_eval": before, "final_eval": after,
        "primary_metric": "FineWiki mean of 16 per-language BPB values at fixed 1024-token context; lower is better",
        "legacy_12_language_macro_bpb": {"initial": before["wiki"]["legacy_12_language_macro_bpb"], "final": after["wiki"]["legacy_12_language_macro_bpb"]},
        "checkpoint_kind": "weights_only", "optimizer_resume_supported": False,
        "checkpoint_save_verify_seconds": checkpoint_seconds, "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": digest, "checkpoint_array_leaves": leaves,
        "memory_stats": memory_stats, "compiled_memory_estimate": compiled_memory,
        "training_memory": training_memory,
        "head": {"kind": "native_dense_exact_softmax", "chunked": False},
        "learning_rate_schedule": {"kind": "five_update_warmup_then_constant", "cooldown": False,
            "warmup_updates": config["warmup_updates"], "peak": config["learning_rate"]},
        "checkpoint_restoration_check": "disabled_by_user", "report_finalized": False,
        "memory_scope": "Training allocator samples and high-water counters; compiler estimate separately",
        "notebook_elapsed_seconds": elapsed,
        "compute_units_estimate": {"rate_per_hour": rate, "training": training_seconds * rate / 3600,
            "through_report": elapsed * rate / 3600, "overhead_through_report": (elapsed - training_seconds) * rate / 3600},
        "cost_note": "2.92 CU/hour supplied by user; connected runtime after report adds cost; no USD conversion assumed",
        "historical_v2_tokens_per_second": 45559.448,
        "historical_comparison_note": "Native o200k, expanded corpus and 16-language holdout, delayed batch discovery and 1200 seconds differ from V3.5. No controlled speed/quality claim.",
        "historical_comparison": comparison, "performance_goals": goals,
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
    # Scores and saved weights are final before disposable diagnostic updates.
    write_json(output / "attempt_status.json", {"version": version, "stage": "profiling",
        "timed_training_started": True, "optimizer_updates": updates})
    diagnostic_shapes = []
    def profile_dispatch(current, ignored, requested):
        mode = "short" if len(diagnostic_shapes) < 2 else "long"
        batch = adaptive[mode + "_batch"]
        diagnostic_shapes.append({"mode": mode, "batch": batch, "sequence_length": 1024 if mode == "short" else 2048})
        return functions[(mode, batch)](current, training[mode], requested)
    profile = record_diagnostic_profile(state, training, profile_dispatch, output, work,
        config["profile_dispatches"], config["profile_updates_per_dispatch"])
    profile["diagnostic_shapes"] = diagnostic_shapes
    profile["operation_timing_note"] = (f"Mixed diagnostic: 16 updates at {adaptive['short_batch']}x1024, then "
        f"16 at {adaptive['long_batch']}x2048. Category ms/update average these shapes; not per-context timings.")
    write_json(output / "profiling.json", profile)
    del state
    result["profiling"] = profile
    result["profiling_status"] = profile["requirements_status"]
    result["telemetry_status"] = profile["telemetry_status"]
    result["notebook_elapsed_before_profile_seconds"] = elapsed
    elapsed = time.monotonic() - config["notebook_start_monotonic"]
    result["notebook_elapsed_seconds"] = elapsed
    result["compute_units_estimate"].update({"through_report": elapsed * rate / 3600,
        "overhead_through_report": (elapsed - training_seconds) * rate / 3600,
        "diagnostic_profile": profile["seconds_including_export_and_analysis"] * rate / 3600})
    result["performance_goals"]["notebook_seconds_at_most_1800"] = elapsed <= 1800.
    result["performance_goal_status"] = "PASS" if all(result["performance_goals"].values()) else "FAIL"
    result["report_finalized"] = True
    write_json(output / "result.json", result)
    write_json(output / "attempt_status.json", {"version": version, "stage": "complete",
        "timed_training_started": True, "optimizer_updates": updates,
        "diagnostic_updates": profile["completed_updates"], "profiling_status": result["profiling_status"]})
    print("Profiling requirements:", result["profiling_status"], "| telemetry:", result["telemetry_status"], flush=True)
    for name, measurement in profile["measurements"].items():
        if measurement["status"] != "available":
            print("MISSING:", name, "|", measurement.get("reason"), flush=True)




def check_source_access(lock):
    """Check gated code access before launching expensive parallel corpus preparation."""
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    files = HfApi().list_repo_files(STACK, repo_type="dataset", revision=lock[STACK])
    selected = {}
    for language in ("java", "c++", "javascript", "rust", "html"):
        paths = [path for path in files if path.startswith("data/" + language + "/") and path.endswith(".json")]
        if not paths:
            raise ValueError("Pinned Stack-smol has no JSON files for " + language)
        # HEAD, not a full language-file download. Never accept dataset terms on the user's behalf.
        try:
            get_hf_file_metadata(hf_hub_url(STACK, paths[0], repo_type="dataset", revision=lock[STACK]),
                token=os.environ.get("HF_TOKEN"))
        except Exception as error:
            raise RuntimeError("The expanded corpus requires access to https://huggingface.co/datasets/bigcode/the-stack-smol . "
                "Accept its access conditions with your Hugging Face account and use a permitted HF_TOKEN. "
                "No corpus substitution or model training occurred.") from error
        selected[language] = paths
    return selected


def context_phase(tokens, short_batch):
    # Both fixed shapes are compiled before training; schedule starts at token zero.
    return "long" if tokens % 4000000 >= 3000000 else "short"

def tokens_to_boundary(tokens, short_batch):
    position = tokens % 4000000
    return (3000000 if position < 3000000 else 4000000) - position

def bounded_dispatch(prefix, rows, batch, proposed, tokens, short_batch):
    """Switch on the first complete optimizer update crossing a token boundary."""
    remaining = tokens_to_boundary(tokens, short_batch)
    if remaining is None:
        return proposed
    start = int(completed_prefix(prefix, rows))
    for count in range(1, proposed + 1):
        if int(completed_prefix(prefix, rows + count * batch)) - start >= remaining:
            return count
    return proposed

def compiler_memory(executable):
    try:
        analysis = executable.memory_analysis()
        values = {key: int(getattr(analysis, key)) for key in
            ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")}
        if any(value < 0 for value in values.values()):
            return None
        return values
    except (AttributeError, TypeError, NotImplementedError):
        return None

def compiler_peak(values):
    if values is None:
        return None
    return (values["argument_size_in_bytes"] + values["output_size_in_bytes"]
        - values["alias_size_in_bytes"] + values["temp_size_in_bytes"])



def row_training_arrays(arrays, seed):
    order = np.random.default_rng(seed).permutation(arrays["tokens"].shape[0])
    return {key: value[order] for key, value in arrays.items()}, hashlib.sha256(order.tobytes()).hexdigest()

def row_prefix_counts(arrays):
    return training_prefix_counts({key: value[:, None, :] for key, value in arrays.items()})






"""Bound useful candidate storage independently of the upstream source mix."""


def prefix_cache_limit(job):
    # The 256 MiB corpus is final text. Cached JSON also contains eval, spare
    # candidates, escaping and provenance; it needs an independent disk budget.
    return max(PREFIX_JOB_LIMIT_BYTES, 4 * sum(job["targets"].values()))


def select_prefix_records(records, state, desired):
    """Stage an entire input row without modifying the committed cursor/counts."""
    counts = dict(state["candidate_bytes"])
    documents = dict(state["candidate_documents"])
    selected = []
    for record in records:
        stratum = record["split"] + ":" + record["bucket"]
        if stratum not in desired:
            continue
        if counts.get(stratum, 0) >= desired[stratum] and documents.get(stratum, 0) >= 4:
            continue
        selected.append(record)
        counts[stratum] = counts.get(stratum, 0) + record["utf8_bytes"]
        documents[stratum] = documents.get(stratum, 0) + 1
    return selected



def encoding_cache_key(tokenizer_sha256, raw_digest):
    value = {"schema": ENCODING_SCHEMA, "tokenizer_sha256": tokenizer_sha256,
        "raw_manifest_sha256": raw_digest, "languages": LANGUAGES, "domains": DOMAINS,
        "mapping": "sorted native IDs -> 1..N; virtual BOS/EOS added during packing"}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]

def verify_encoding(directory, identity, raw_digest):
    details = read_json(directory / "encoding.json")
    if (details.get("schema") != ENCODING_SCHEMA
            or details.get("cache_key") != encoding_cache_key(identity["tokenizer_sha256"], raw_digest)
            or details.get("tokenizer_sha256") != identity["tokenizer_sha256"]
            or details.get("raw_manifest_sha256") != raw_digest
            or details.get("languages") != LANGUAGES or details.get("domains") != DOMAINS):
        raise ValueError("Document encoding cache identity differs; cache preserved.")
    for name, digest in details["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError("Document encoding checksum differs: " + name)
    for split, counts in details["splits"].items():
        for name in ("ids.bin", "byte_weights.bin"):
            if (directory / split / name).stat().st_size != counts["text_tokens"] * 4:
                raise ValueError("Encoded binary size differs: " + split + "/" + name)
    return details

def encode_corpus_once(raw, destination, tokenizer, mapping, identity, raw_digest):
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    details = {"schema": ENCODING_SCHEMA,
        "cache_key": encoding_cache_key(identity["tokenizer_sha256"], raw_digest),
        "tokenizer_sha256": identity["tokenizer_sha256"], "raw_manifest_sha256": raw_digest,
        "text_encoding_policy": identity["text_encoding_policy"], "byte_accounting": identity["byte_accounting"],
        "languages": LANGUAGES, "domains": DOMAINS, "splits": {}, "files": {}}
    for split in ("train", "in_mix", "wiki"):
        folder = destination / split
        folder.mkdir()
        token_count = byte_count = document_count = 0
        last_progress = time.monotonic()
        with (raw / (split + ".jsonl")).open(encoding="utf-8") as source, \
                (folder / "ids.bin").open("wb", buffering=8 * 1024**2) as ids_file, \
                (folder / "byte_weights.bin").open("wb", buffering=8 * 1024**2) as weights_file, \
                (folder / "documents.jsonl").open("w", encoding="utf-8") as documents_file:
            for line in source:
                doc = json.loads(line)
                native_ids = tokenizer.encode_ordinary(doc["text"])
                pieces = tokenizer.decode_tokens_bytes(native_ids)
                original = doc["text"].encode("utf-8")
                if b"".join(pieces) != original or len(original) != doc["utf8_bytes"]:
                    raise ValueError("Native tokenizer is not byte-lossless on fixed input " + doc["sha256"])
                ids = np.array([mapping[index] for index in native_ids], dtype="<i4")
                weights = np.fromiter((len(piece) for piece in pieces), dtype="<i4", count=len(pieces))
                if not len(ids) or len(ids) != len(weights) or int(weights.sum()) != doc["utf8_bytes"]:
                    raise ValueError("Native token/byte attribution differs: " + doc["sha256"])
                ids_file.write(ids.tobytes())
                weights_file.write(weights.tobytes())
                documents_file.write(json.dumps({"offset": token_count, "tokens": len(ids),
                    "original_bytes": doc["utf8_bytes"], "sha256": doc["sha256"],
                    "language_id": LANGUAGES.index(doc["language"]), "domain_id": DOMAINS.index(doc["domain"])}) + "\n")
                token_count += len(ids)
                byte_count += doc["utf8_bytes"]
                document_count += 1
                if time.monotonic() - last_progress >= 5.:
                    print(f"Native encoding once: {split}, {document_count:,} documents, {token_count:,} tokens", flush=True)
                    last_progress = time.monotonic()
        details["splits"][split] = {"text_tokens": token_count, "original_bytes": byte_count,
            "documents": document_count, "encode_passes": 1, "roundtrip_verification_passes": 1}
        for name in ("ids.bin", "byte_weights.bin", "documents.jsonl"):
            relative = split + "/" + name
            details["files"][relative] = sha256_file(destination / relative)
        print(f"Encoded {split} once: {token_count:,} tokens; {byte_count:,} original bytes", flush=True)
    details["encoding_seconds"] = time.monotonic() - started
    write_json(destination / "encoding.json", details)
    return details

def prepared_encoding(shared, raw, work, tokenizer, mapping, identity, raw_digest):
    key = encoding_cache_key(identity["tokenizer_sha256"], raw_digest)
    cache = shared / "encoded" / key
    if cache.exists():
        details = verify_encoding(cache, identity, raw_digest)
        return cache, details, True
    local = work / ("encoding-" + key)
    details = encode_corpus_once(raw, local, tokenizer, mapping, identity, raw_digest)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(key + ".partial-" + str(time.time_ns()))
    shutil.copytree(local, temporary)
    verify_encoding(temporary, identity, raw_digest)
    temporary.rename(cache)
    # Cold packing reads the fast local files; later runs can use the verified cache.
    return local, details, False

def pack_encoded_split(directory, split, identity, sequence_length):
    started = time.monotonic()
    details = read_json(directory / "encoding.json")["splits"][split]
    documents = read_jsonl(directory / split / "documents.jsonl")
    if len(documents) != details["documents"] or not documents:
        raise ValueError("Encoded document count differs: " + split)
    ids = np.memmap(directory / split / "ids.bin", mode="r", dtype="<i4")
    weights = np.memmap(directory / split / "byte_weights.bin", mode="r", dtype="<i4")
    chunks = sum((doc["tokens"] + sequence_length) // sequence_length for doc in documents)
    shape = (chunks, sequence_length)
    result = {key: np.zeros(shape, np.bool_ if key in ("mask", "text_mask") else np.int32)
        for key in ("tokens", "targets", "mask", "text_mask", "byte_weights", "language_ids", "domain_ids")}
    row = expected_offset = 0
    last_progress = time.monotonic()
    for doc in documents:
        offset, count = doc["offset"], doc["tokens"]
        if offset != expected_offset or count < 1:
            raise ValueError("Encoded document offsets are not contiguous.")
        expected_offset += count
        sequence = np.concatenate(([identity["bos_id"]], ids[offset:offset + count], [identity["eos_id"]]))
        byte_weights = np.concatenate((weights[offset:offset + count], [0]))
        for start in range(0, count + 1, sequence_length):
            size = min(sequence_length, count + 1 - start)
            result["tokens"][row, :size] = sequence[start:start + size]
            result["targets"][row, :size] = sequence[start + 1:start + size + 1]
            result["mask"][row, :size] = True
            result["text_mask"][row, :size] = start + np.arange(size) < count
            result["byte_weights"][row, :size] = byte_weights[start:start + size]
            result["language_ids"][row, :size] = doc["language_id"]
            result["domain_ids"][row, :size] = doc["domain_id"]
            row += 1
        if time.monotonic() - last_progress >= 5.:
            print(f"Packing {sequence_length}: {split}, {row:,}/{chunks:,} chunks", flush=True)
            last_progress = time.monotonic()
    if (row != chunks or expected_offset != details["text_tokens"] or expected_offset != len(ids)
            or len(weights) != len(ids) or int(result["byte_weights"].sum()) != details["original_bytes"]):
        raise ValueError("Encoded chunks differ from original targets/bytes.")
    packed, packing = pack_chunks(result, 42)
    for key in EXTRA_KEYS:
        packed[key] = np.zeros_like(packed["tokens"], dtype=result[key].dtype)
        for source, target, start, length in packing["assignments_source_bin_start_length"]:
            packed[key][target, start:start + length] = result[key][source, :length]
    packing.update(text_tokens=details["text_tokens"], original_bytes=int(packed["byte_weights"].sum()),
        packing_seconds=time.monotonic() - started, encoding_schema=ENCODING_SCHEMA)
    return packed, packing




O200K_ASSET_SHA256 = "97973057009b00298ee5c575a23b895c912518ded47f9dc1559032e9f2532187"

O200K_RANKS_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"



def projected_candidate_peak(analyses, candidate, other_resident_bytes):
    """Conservative compilation gate, distinct from the next executable's estimate."""
    known = sorted((b, compiler_peak(v)) for b, v in analyses.items() if v is not None)
    if not known:
        return None
    b, peak = known[-1]
    if len(known) > 1:
        earlier_b, earlier_peak = known[-2]
        slope = max(0., (peak - earlier_peak) / (b - earlier_b))
        # A tiny/negative adjacent slope must not justify an unlimited jump.
        slope = max(slope, analyses[b]["temp_size_in_bytes"] / b)
        projected = peak + (candidate - b) * slope * 1.10
    else:
        projected = peak + analyses[b]["temp_size_in_bytes"] * (candidate - b) / b * 1.10
    return int(projected) + other_resident_bytes

class BatchDiscovery:
    def __init__(self, config):
        self.config = config
        self.updates = 0
        self.turn = "short"
        self.finished_at_update = None
        self.contexts = {mode: {"current_batch": config.get("discovery_initial_batches", {"short": 3, "long": 1})[mode],
            "selected_batch": config.get("discovery_initial_batches", {"short": 3, "long": 1})[mode], "selected_tps": None,
            "locked": False, "lock_reason": None, "observations": [],
            "window_updates": 0, "window_tokens": 0, "window_seconds": 0.}
            for mode in ("short", "long")}

    def finished(self):
        return all(row["locked"] for row in self.contexts.values())

    def mode(self):
        if self.finished():
            raise ValueError("The batch controller is locked and must not be called again.")
        if self.contexts[self.turn]["locked"]:
            self.turn = "long" if self.turn == "short" else "short"
        return self.turn

    def remaining_in_window(self, mode):
        return min(self.config["discovery_window_updates"] - self.contexts[mode]["window_updates"],
            self.config["discovery_max_updates"] - self.updates)

    def observe(self, mode, metrics, seconds):
        row = self.contexts[mode]
        row["window_updates"] += int(metrics["updates"])
        row["window_tokens"] += int(metrics["tokens"])
        row["window_seconds"] += seconds
        self.updates += int(metrics["updates"])
        return row["window_updates"] == self.config["discovery_window_updates"]

    def lock(self, mode, reason):
        row = self.contexts[mode]
        row["locked"], row["lock_reason"] = True, reason

    def lock_all(self, reason):
        for mode in self.contexts:
            if not self.contexts[mode]["locked"]:
                self.lock(mode, reason)

    def choose(self, mode, state, analyses, compile_shape, remaining_seconds, output, elapsed):
        row = self.contexts[mode]
        batch = row["current_batch"]
        tps = row["window_tokens"] / row["window_seconds"]
        check = {"at_update": int(state.step), "discovery_updates": self.updates, "mode": mode,
            "batch": batch, "window_updates": row["window_updates"], "window_tokens": row["window_tokens"],
            "dispatch_seconds": row["window_seconds"], "tokens_per_second": tps,
            "previous_selected_tokens_per_second": row["selected_tps"], "elapsed_seconds": elapsed}
        improves = row["selected_tps"] is None or tps >= row["selected_tps"] * 1.02
        if improves:
            row["selected_batch"], row["selected_tps"] = batch, tps
        else:
            self.lock(mode, "Larger batch did not improve useful dispatch tokens/s by at least 2%.")
        row["observations"].append({"batch": batch, "updates": row["window_updates"],
            "tokens": row["window_tokens"], "dispatch_seconds": row["window_seconds"],
            "tokens_per_second": tps, "accepted": improves})
        if self.updates >= self.config["discovery_max_updates"]:
            self.lock_all("The 256-update discovery budget is exhausted; retain measured batches.")
        if not row["locked"]:
            stats = jax.devices()[0].memory_stats() or {}
            limit = stats.get("bytes_limit")
            other = self.config["other_resident_bytes_" + mode]
            current = analyses[mode].get(batch)
            untracked = max(0, stats.get("bytes_in_use", 0)
                - (current["argument_size_in_bytes"] if current else 0) - other)
            projected = projected_candidate_peak(analyses[mode], batch + 1, other + untracked)
            check.update(device_capacity_bytes=limit, projected_next_peak_bytes=projected,
                additional_observed_resident_bytes=untracked,
                allocator_snapshot=memory_snapshot(stats, elapsed))
            if batch >= self.config.get("discovery_memory_caps", {}).get(mode, self.config["safety_batch_cap"]):
                self.lock(mode, "Memory-failure upper bound or safety cap reached; retain measured batch.")
            elif batch >= self.config["safety_batch_cap"]:
                self.lock(mode, "Safety batch cap reached; no claim of a global optimum.")
            elif projected is None or limit is None:
                self.lock(mode, "Memory estimate or device capacity unavailable; retain measured batch.")
            elif remaining_seconds < 60.:
                self.lock_all("Under 60 seconds remain in the training wall budget; end discovery.")
            else:
                # Inspect the next compiled shape instead of declaring a maximum
                # from an extrapolation. Compilation performs no optimizer update.
                candidate = batch + 1
                try:
                    analysis = compile_shape(mode, candidate, state, "inside_training_budget")
                except Exception as error:
                    if not is_memory_error(error):
                        raise
                    analysis = None
                    check["candidate_memory_error"] = str(error)
                peak = compiler_peak(analysis)
                check["candidate_compiler_peak_plus_other_resident_bytes"] = (
                    peak + other + untracked if peak is not None else None)
                if peak is None or peak + other + untracked > limit * self.config["memory_budget_fraction"]:
                    self.lock(mode, "Candidate executable exceeds the memory budget or has no estimate; never executed.")
                else:
                    # Compiled is not selected: it must complete its own measured window first.
                    row["current_batch"] = candidate
                    check["next_candidate_batch"] = candidate
        check.update(action="lock" if row["locked"] else "increase", selected_batch=row["selected_batch"],
            reason=row["lock_reason"] or "Measure the next admitted batch using real training updates.")
        row["window_updates"] = row["window_tokens"] = 0
        row["window_seconds"] = 0.
        self.turn = "long" if mode == "short" else "short"
        return check

    def report(self):
        return {"maximum_discovery_updates": self.config["discovery_max_updates"],
            "discovery_updates": self.updates, "window_updates": self.config["discovery_window_updates"],
            "both_locked": self.finished(), "finished_at_update": self.finished_at_update,
            "selected_batches": {mode: row["selected_batch"] for mode, row in self.contexts.items()},
            "contexts": self.contexts,
            "scope": "Largest measured reasonable batch within a bounded real-training search; not a proof of the physical OOM limit."}

class BatchMemoryError(RuntimeError):
    """A compiled batch exceeds the configured memory admission budget."""

def is_memory_error(error):
    if isinstance(error, (MemoryError, BatchMemoryError)):
        return True
    message = str(error).lower()
    return any(marker in message for marker in ("resource_exhausted", "out of memory", "outofmemory",
        "failed to allocate", "memory allocation", "bad_alloc"))

def memory_discovery_fallback(config, mode, batch, error, stage, update):
    if not is_memory_error(error):
        raise error
    if batch <= 1:
        raise RuntimeError(f"Memory failure at 1x{1024 if mode == 'short' else 2048}; batch discovery cannot reduce below one sequence.") from error
    config["early_memory_discovery"] = True
    config.setdefault("discovery_initial_batches", {"short": 3, "long": 1})[mode] = 1
    config.setdefault("discovery_memory_caps", {})[mode] = batch - 1
    config.setdefault("memory_discovery_events", []).append({"mode": mode, "failed_batch": batch,
        "fallback_batch": 1, "stage": stage, "at_update": update,
        "error_type": type(error).__name__, "error": str(error)})
    print(f"MEMORY FALLBACK: {mode} batch {batch} -> 1; start discovery early, excluding failed batch {batch}.", flush=True)

def verify_surviving_state(state, updates, rows_used):
    if any(getattr(leaf, "is_deleted", lambda: False)() for leaf in jax.tree.leaves(state)):
        raise RuntimeError("Memory failure invalidated donated training buffers; cannot safely retry. No weights reset or checkpoint restore.")
    jax.block_until_ready(state)
    if (int(state.step) != updates or int(state.short_rows) != rows_used["short"]
            or int(state.long_rows) != rows_used["long"] or int(state.data_position) != updates):
        raise RuntimeError("Training state is not the last committed update after memory failure; recovery stopped.")

def delayed_train(state, training, prefixes, functions, analyses, compile_shape, config, output):
    initial_batches = {"short": 3, "long": 1}
    threshold = config["discovery_start_after_updates"]
    finder = None
    started_at_update = None
    log_rows, decisions, transitions, samples = [], [], [], []
    rows_used = {"short": 0, "long": 0}
    updates = tokens = positions = 0
    maximum_step_seconds = last_progress = last_start = sampling_seconds = 0.
    last_context = None

    def policy():
        report = finder.report() if finder is not None else None
        return {"mode": "delayed_discovery", "discovery_enabled": True,
            "discovery_start_after_updates": threshold,
            "discovery_started": finder is not None,
            "started_at_update": started_at_update,
            "initial_batches": initial_batches,
            "selected_batches": report["selected_batches"] if report else dict(initial_batches),
            "early_memory_trigger": bool(config.get("early_memory_discovery")),
            "memory_events": config.get("memory_discovery_events", []),
            "discovery": report,
            "context_schedule": "3M short / 1M long cumulative tokens; discovery temporarily measures both contexts."}

    write_json(output / "batch_policy.json", policy())
    print(f"Initial batches 3x1024 / 1x2048; discovery starts after {threshold:,} completed updates.", flush=True)
    memory_before = memory_snapshot(jax.devices()[0].memory_stats(), 0.)
    with (output / "training.jsonl").open("x") as stream, (output / "batch_checks.jsonl").open("x") as checks:
        started = time.monotonic()
        try:
            while True:
                elapsed = time.monotonic() - started
                remaining = config["training_seconds"] - elapsed
                if remaining <= 0:
                    break
                if finder is None and (updates >= threshold or config.get("early_memory_discovery")):
                    if updates != threshold and not config.get("early_memory_discovery"):
                        raise ValueError("A dispatch crossed the exact discovery boundary.")
                    finder = BatchDiscovery(config)
                    started_at_update = updates
                    write_json(output / "batch_policy.json", policy())
                    print(f"BATCH DISCOVERY START: {updates:,} completed updates; same model, Adam state and data cursors.", flush=True)
                phase = "initial_batches" if finder is None else ("locked_training" if finder.finished() else "discovery")
                batches = dict(initial_batches) if finder is None else finder.report()["selected_batches"]
                requested = config["updates_per_dispatch"]
                if phase == "discovery":
                    mode = finder.mode()
                    used_batch = finder.contexts[mode]["current_batch"]
                    requested = min(requested, finder.remaining_in_window(mode))
                    if requested <= 0:
                        finder.lock_all("Discovery update budget reached.")
                        continue
                else:
                    mode = context_phase(tokens, batches["short"])
                    used_batch = batches[mode]
                    if finder is None:
                        requested = min(requested, threshold - updates)
                length = 1024 if mode == "short" else 2048
                if (phase, mode) != last_context:
                    transitions.append({"phase": phase, "mode": mode, "batch": used_batch,
                        "tokens": tokens, "update": updates, "elapsed_seconds": elapsed,
                        "short_batch": batches["short"], "long_batch": batches["long"], "cycle": tokens // 4000000})
                    print(f"Context: {phase}, {used_batch} x {length}; tokens={tokens:,}", flush=True)
                    last_context = (phase, mode)
                if updates == 0 or remaining < max(2., 2 * requested * maximum_step_seconds):
                    requested = 1
                if phase != "discovery":
                    requested = bounded_dispatch(prefixes[mode]["tokens"], rows_used[mode], used_batch,
                        requested, tokens, batches["short"])
                last_start = time.monotonic() - started
                if last_start >= config["training_seconds"]:
                    break
                begin = time.monotonic()
                try:
                    pending = functions[(mode, used_batch)](state, training[mode], jnp.array(requested, jnp.int32))
                    enqueued = time.monotonic()
                    next_state, values = jax.block_until_ready(pending)
                except Exception as error:
                    if finder is not None or not is_memory_error(error):
                        raise
                    verify_surviving_state(state, updates, rows_used)
                    memory_discovery_fallback(config, mode, used_batch, error, "initial_training", updates)
                    write_json(output / "batch_policy.json", policy())
                    compile_shape(mode, 1, state, "inside_training_budget")
                    continue
                state = next_state
                end_call = time.monotonic()
                metrics = {key: np.asarray(value).tolist() for key, value in values.items()}
                completed = int(metrics["updates"])
                previous_rows = rows_used[mode]
                rows_used[mode] += completed * used_batch
                expected = int(completed_prefix(prefixes[mode]["tokens"], rows_used[mode])
                    - completed_prefix(prefixes[mode]["tokens"], previous_rows))
                updates += completed
                tokens += int(metrics["tokens"])
                positions += completed * used_batch * length
                elapsed = end_call - started
                row = {**metrics, "update": updates, "total_tokens": tokens, "phase": phase,
                    "requested_updates": requested, "mode": mode, "batch": used_batch,
                    "short_batch": batches["short"], "long_batch": batches["long"], "sequence_length": length,
                    "start_seconds": last_start, "end_seconds": elapsed, "dispatch_seconds": end_call - begin,
                    "enqueue_seconds": enqueued - begin, "synchronization_call_seconds": end_call - enqueued,
                    "loss": metrics["nll_sum"] / max(metrics["tokens"], 1),
                    "short_rows": rows_used["short"], "long_rows": rows_used["long"]}
                log_rows.append(row)
                stream.write(json.dumps({key: None if isinstance(value, float) and not math.isfinite(value) else value
                    for key, value in row.items()}, allow_nan=False) + "\n")
                if not metrics["finite"] or completed != requested or expected != int(metrics["tokens"]):
                    raise FloatingPointError("Nonfinite/incomplete update or CPU/device accounting mismatch. No retry.")
                maximum_step_seconds = max(maximum_step_seconds, (end_call - begin) / completed)
                if phase == "discovery" and finder.observe(mode, metrics, end_call - begin):
                    check = finder.choose(mode, state, analyses, compile_shape,
                        config["training_seconds"] - elapsed, output, elapsed)
                    check["decision_complete_seconds"] = time.monotonic() - started
                    decisions.append(check)
                    checks.write(json.dumps(check, allow_nan=False) + "\n")
                    checks.flush()
                    stream.flush()
                    print(f"Discovery @{updates:,}: {mode} {check['action']}, selected batch={check['selected_batch']}; {check['reason']}", flush=True)
                    write_json(output / "batch_policy.json", policy())
                if finder is not None and finder.finished() and finder.finished_at_update is None:
                    finder.finished_at_update = updates
                    write_json(output / "batch_policy.json", policy())
                    print("BATCHES LOCKED:", finder.report()["selected_batches"], "No further discovery.", flush=True)
                if len(log_rows) == 1 or elapsed - last_progress >= 5.:
                    stream.flush()
                    sample_started = time.monotonic()
                    samples.append(memory_snapshot(jax.devices()[0].memory_stats(), elapsed))
                    sampling_seconds += time.monotonic() - sample_started
                    print(f"updates={updates:,} tokens={tokens:,} batch={used_batch}x{length} loss={row['loss']:.5f} elapsed={elapsed:.2f}s", flush=True)
                    last_progress = elapsed
            jax.block_until_ready(state)
        except Exception:
            stream.flush()
            checks.flush()
            write_json(output / "batch_policy.json", policy())
            write_json(output / "attempt_status.json", {"version": "V4.2.2", "stage": "training failed",
                "timed_training_started": True, "optimizer_updates": updates, "tokens": tokens,
                "completed_short_rows": rows_used["short"], "completed_long_rows": rows_used["long"]})
            raise
        seconds = time.monotonic() - started
    if finder is not None and not finder.finished():
        finder.lock_all("Training wall-clock budget ended; retain measured selections.")
        finder.finished_at_update = updates
    after = memory_snapshot(jax.devices()[0].memory_stats(), seconds)
    write_json(output / "context_transitions.json", transitions)
    final_policy = policy()
    write_json(output / "batch_policy.json", final_policy)
    batches = final_policy["selected_batches"]
    return state, {"rows": log_rows, "checks": decisions, "transitions": transitions,
        "updates": updates, "tokens": tokens, "positions": positions, "rows_used": rows_used,
        "short_batch": batches["short"], "long_batch": batches["long"], "batch_policy": final_policy,
        "training_seconds": seconds, "last_start": last_start, "memory_before": memory_before,
        "memory_after": after, "samples": samples, "sampling_seconds": sampling_seconds}

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
