"""V3.2 parallel source preparation. Embedded in the notebook; not run locally."""
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
        write_json(destination.parent / "parallel_downloads.json", {"version": "V3.2",
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
    print(f"V3.2: {len(jobs)} source jobs; {DOWNLOAD_WORKERS} concurrent workers; completed batches persist in Drive.", flush=True)
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
    write_json(shared / "parallel_downloads.json", {"version": "V3.2", "max_workers": DOWNLOAD_WORKERS,
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
