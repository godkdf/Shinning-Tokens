"""Context-independent, byte-verified encoding; sequential context packing."""
ENCODING_SCHEMA = "v4-dense-document-encoding-1"


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
                text = doc["text"]
                encoded = tokenizer.encode(text, add_special_tokens=False)
                if tokenizer.decode(encoded.ids, skip_special_tokens=False).encode("utf-8") != text.encode("utf-8"):
                    raise ValueError("Tokenizer is not byte-lossless on fixed input " + doc["sha256"])
                ids = np.array([mapping[index] for index in encoded.ids], dtype="<i4")
                if not len(ids):
                    raise ValueError("Nonempty source text produced zero tokens.")
                # Same monotonic byte-frontier rule as V3.9.1, calculated once.
                char_bytes = np.concatenate(([0], np.cumsum([len(c.encode("utf-8")) for c in text])))
                frontier = np.maximum.accumulate(np.array([0] + [end for _, end in encoded.offsets], np.int64))
                frontier = np.clip(frontier, 0, len(text))
                frontier[-1] = len(text)
                weights = np.diff(char_bytes[frontier]).astype("<i4")
                if int(weights.sum()) != doc["utf8_bytes"]:
                    raise ValueError("Original-byte attribution differs: " + doc["sha256"])
                ids_file.write(ids.tobytes())
                weights_file.write(weights.tobytes())
                documents_file.write(json.dumps({"offset": token_count, "tokens": len(ids),
                    "original_bytes": doc["utf8_bytes"], "sha256": doc["sha256"],
                    "language_id": LANGUAGES.index(doc["language"]), "domain_id": DOMAINS.index(doc["domain"])}) + "\n")
                token_count += len(ids)
                byte_count += doc["utf8_bytes"]
                document_count += 1
                if time.monotonic() - last_progress >= 5.:
                    print(f"Encoding once: {split}, {document_count:,} documents, {token_count:,} text tokens", flush=True)
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
