"""Native o200k_base preparation and exact byte accounting; embedded in notebook."""
O200K_ASSET_SHA256 = "97973057009b00298ee5c575a23b895c912518ded47f9dc1559032e9f2532187"
O200K_RANKS_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"


def prepare_tokenizer(config, destination):
    import base64
    import tiktoken
    if (config["tokenizer_reference"] != "o200k_base"
            or config["tokenizer_revision"] != "tiktoken-0.14.0"
            or config["model_id"] != "google/gemma-3-270m"
            or importlib.metadata.version("tiktoken") != "0.14.0"):
        raise ValueError("V4.2.1 requires native tiktoken 0.14.0 o200k_base and the Gemma3_270M backbone.")
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
