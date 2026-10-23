"""Expanded byte-defined corpus and independently cached context packings."""

PROTOCOL = "V39-expanded-byte-mix-1"
LANGUAGES = ["eng_Latn", "fra_Latn", "spa_Latn", "por_Latn", "cmn_Hani", "arb_Arab",
    "hin_Deva", "swh_Latn", "zul_Latn", "xho_Latn", "tsn_Latn", "sot_Latn",
    "rus_Cyrl", "jpn_Jpan", "deu_Latn", "ibo_Latn"]
WIKI_CODES = ["en", "fr", "es", "pt", "zh", "ar", "hi", "sw", "zu", "xh", "tn", "st", "ru", "ja", "de", "ig"]
DOMAINS = ["pdf_edu", "dclm", "web_edu", "multilingual_web", "math", "code_python",
    "code_java", "code_cpp", "code_javascript", "code_rust", "code_html", "translations", "wiki_holdout"]
STACK = "bigcode/the-stack-smol"
SOURCE_REPOS = [SMOL, "HuggingFaceFW/fineweb-2", "HuggingFaceTB/finemath",
    "codeparrot/codeparrot-clean", "HuggingFaceFW/finetranslations", STACK, "HuggingFaceFW/finewiki"]
PREFIX_CACHE_VERSION = "v39-expanded-source-prefix-1"


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
        downloads = {"version": "V3.9", "raw_cache_reused": True, "source_jobs": 0,
            "max_workers": DOWNLOAD_WORKERS, "peak_active_source_jobs": 0, "fetch_seconds": 0.}
    else:
        downloads = read_json(shared / "parallel_downloads.json")
        downloads["raw_cache_reused"] = False
    write_json(output / "parallel_downloads.json", downloads)
    expected_jobs = len(LANGUAGES) + len({(b["repo"], b["subset"]) for b in recipe()["buckets"]})
    checks = {"V39-021_DOWNLOAD_BOUND": downloads["peak_active_source_jobs"] <= 6,
        "V39-022_PREFIX_CACHE": raw_reused or all((Path(item["directory"]) / "state.json").exists()
            for item in downloads.get("jobs", {}).values()),
        "V39-023_MERGE_ORDER": raw_reused or downloads["merge_order"][:len(LANGUAGES)] == ["wiki_" + lang for lang in LANGUAGES],
        "V39-024_SOURCE_COMPLETE": raw_reused or downloads["source_jobs"] == expected_jobs == len(downloads["jobs"])}
    write_json(output / "preparation_requirements.json", {key: "PASS" if value else "FAIL" for key, value in checks.items()})
    if not all(checks.values()):
        raise ValueError("Expanded data preparation failed validation.")
    tokenizer, mapping, identity = prepare_tokenizer(config, prepared / "tokenizer")
    raw_digest = sha256_file(raw / "manifest.json")
    contexts = {}
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
                arrays, packing = tokenize_split(read_jsonl(raw / (split + ".jsonl")), tokenizer, mapping, identity, length)
                validate_sequence_arrays(arrays, length, split)
                name = split + ".npz"
                np.savez_compressed(destination / name, **arrays)
                details["splits"][split] = {k: v for k, v in packing.items() if not k.startswith("assignments_")}
                details["files"][name] = sha256_file(destination / name)
                print(f"{length}: {split}: {packing['text_tokens']:,} text targets, occupancy={packing['occupancy']:.2%}", flush=True)
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
    write_json(output / "preparation.json", {"raw_cache_reused": raw_reused, "contexts": contexts,
        "preparation_seconds": time.monotonic() - started, "raw_manifest_sha256": raw_digest,
        "actual_training_bytes": manifest["original_bytes"]["train"], "target_training_bytes": 256 * 1024**2,
        "tokenizer_sha256": identity["tokenizer_sha256"], "shared_directory": str(shared)})
