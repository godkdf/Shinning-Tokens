"""Original Llama 3 native tiktoken path, verified against Meta's original ranks."""
LLAMA3_TOKENIZER = {
    "reference": "NousResearch/Meta-Llama-3-8B",
    "revision": "315b20096dc791d381d514deb5f8bd9c8d6d3061",
    "sha256": "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7",
    "official_reference": "meta-llama/Meta-Llama-3-8B",
    "official_revision": "8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920",
    "official_ranks_sha256": "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55",
    "license_sha256": "475211637354ce4c14b9c3dacccbefbaba735fe1b0db97d6c4fd11cd1356819f",
    "use_policy_sha256": "003513f595a20b7d63c609dfe80879531acb9caef6376fcc109bf0827471f3d1",
    "vocabulary_ids": 128256,
    "model_rows": 128259,
    "expected_parameters": 182411776,
}
LLAMA3_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def llama3_text_chunks(text):
    """Preserve Meta's 400000-character slices and 25000-character run limits."""
    for start in range(0, len(text), 400000):
        part = text[start:start + 400000]
        boundary, run_length, was_space = 0, 0, None
        for position, character in enumerate(part):
            is_space = character.isspace()
            run_length = run_length + 1 if is_space == was_space else 1
            was_space = is_space
            if run_length > 25000:
                yield part[boundary:position]
                boundary, run_length = position, 1
        if boundary < len(part):
            yield part[boundary:]


class Llama3Tokenizer:
    def __init__(self, ranks, specials):
        import tiktoken
        self.engine = tiktoken.Encoding(name="llama3", pat_str=LLAMA3_PATTERN,
            mergeable_ranks=ranks, special_tokens=specials, explicit_n_vocab=128256)

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError("This raw-text protocol adds only its separate packing BOS/EOS.")
        ids = []
        for part in llama3_text_chunks(text):
            ids.extend(self.engine.encode_ordinary(part))
        return ids

    def decode_tokens_bytes(self, ids):
        return self.engine.decode_tokens_bytes(ids)

    def decode(self, ids):
        return self.engine.decode(ids)


def prepare_tokenizer(config, destination):
    import base64
    from huggingface_hub import hf_hub_download
    if (config["tokenizer_reference"] != LLAMA3_TOKENIZER["reference"]
            or config["tokenizer_revision"] != LLAMA3_TOKENIZER["revision"]
            or config["model_id"] != "google/gemma-3-270m"
            or importlib.metadata.version("tiktoken") != "0.14.0"):
        raise ValueError("V4.4.1 requires pinned original Llama 3, native tiktoken 0.14.0 and the Gemma3_270M backbone.")
    destination.mkdir(parents=True, exist_ok=True)
    for name, digest in (("tokenizer.json", LLAMA3_TOKENIZER["sha256"]),
            ("LICENSE", LLAMA3_TOKENIZER["license_sha256"]),
            ("USE_POLICY.md", LLAMA3_TOKENIZER["use_policy_sha256"])):
        source = Path(hf_hub_download(repo_id=LLAMA3_TOKENIZER["reference"], filename=name,
            revision=LLAMA3_TOKENIZER["revision"], cache_dir=config["tokenizer_cache_directory"]))
        if sha256_file(source) != digest:
            raise ValueError("Pinned Llama 3 source checksum differs: " + name)
        shutil.copyfile(source, destination / name)
    raw = json.loads((destination / "tokenizer.json").read_text())
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    byte_values, codepoints = list(visible), list(visible)
    extra = 0
    for value in range(256):
        if value not in visible:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    byte_decoder = {chr(codepoint): value for value, codepoint in zip(byte_values, codepoints)}
    ranks = {bytes(byte_decoder[c] for c in piece): index for piece, index in raw["model"]["vocab"].items()}
    rank_file = b"".join(base64.b64encode(piece) + b" " + str(rank).encode() + b"\n"
        for piece, rank in sorted(ranks.items(), key=lambda item: item[1]))
    if len(ranks) != 128000 or hashlib.sha256(rank_file).hexdigest() != LLAMA3_TOKENIZER["official_ranks_sha256"]:
        raise ValueError("Reconstructed Llama 3 ranks differ from Meta's original tokenizer.model.")
    (destination / "tokenizer.model").write_bytes(rank_file)
    names = ["<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>", "<|reserved_special_token_2|>", "<|reserved_special_token_3|>",
        "<|start_header_id|>", "<|end_header_id|>", "<|reserved_special_token_4|>", "<|eot_id|>"]
    names.extend(f"<|reserved_special_token_{i}|>" for i in range(5, 251))
    specials = {name: 128000 + index for index, name in enumerate(names)}
    if {item["content"]: item["id"] for item in raw["added_tokens"]} != specials:
        raise ValueError("Llama 3 special-token IDs differ from the original Meta table.")
    tokenizer = Llama3Tokenizer(ranks, specials)
    ids = sorted(set(ranks.values()) | set(specials.values()))
    if ids != list(range(128256)):
        raise ValueError("Expected exactly 128,256 contiguous original Llama 3 IDs.")
    mapping = {old: index + 1 for index, old in enumerate(ids)}
    rows = len(ids) + 3
    total = 100326016 + rows * 640
    if rows != LLAMA3_TOKENIZER["model_rows"] or total != LLAMA3_TOKENIZER["expected_parameters"]:
        raise ValueError("Unexpected model size after Llama 3 vocabulary adaptation.")
    write_json(destination / "native_tiktoken.json", {"name": "llama3", "pat_str": LLAMA3_PATTERN,
        "mergeable_ranks_file": "tokenizer.model", "mergeable_ranks_sha256": LLAMA3_TOKENIZER["official_ranks_sha256"],
        "special_tokens": specials, "text_policy": "ordinary text; no automatic BOS/EOS",
        "max_encode_chars": 400000, "max_consecutive_chars": 25000})
    info = {"reference": LLAMA3_TOKENIZER["reference"], "revision": LLAMA3_TOKENIZER["revision"],
        "tokenizer_family": "original Llama 3", "implementation": "native tiktoken 0.14.0 / Meta reference pattern and chunking",
        "official_reference": LLAMA3_TOKENIZER["official_reference"], "official_revision": LLAMA3_TOKENIZER["official_revision"],
        "tokenizer_sha256": LLAMA3_TOKENIZER["sha256"], "mergeable_ranks_sha256": LLAMA3_TOKENIZER["official_ranks_sha256"],
        "vocabulary_ids": len(ids), "mergeable_token_ids": len(ranks), "native_special_token_ids": specials,
        "original_id_space": max(ids) + 1, "unused_native_ids": 0, "model_rows": rows,
        "pad_id": 0, "bos_id": len(ids) + 1, "eos_id": len(ids) + 2,
        "id_mapping": "sorted native IDs map to 1..N; PAD=0, BOS=N+1, EOS=N+2",
        "expected_parameters": total, "model_type": "native_Llama3_byte_BPE",
        "text_encoding_policy": "Meta reference chunking; encode_ordinary; special spellings remain ordinary text; no normalization/BOS/EOS/chat template",
        "byte_accounting": "Exact native token byte lengths, including UTF-8 fragments"}
    info["vocabulary_adaptation"] = {
        "backbone": "Gemma3_270M", "backbone_parameters": 100326016,
        "hidden_size": 640, "native_tokenizer_ids": len(ids), "protocol_tokens": 3,
        "embedding_shape": [rows, 640], "embedding_and_head_tied": True,
        "embedding_parameters": rows * 640, "total_parameters": total,
        "v4_parameters": 268100736, "parameter_delta_from_v4": total - 268100736,
        "v4_3_1_parameters": 260763136, "parameter_delta_from_v4_3_1": total - 260763136,
        "automatic_native_special_tokens": False,
        "packing_policy": "Native IDs -> 1..128256; PAD=0, BOS=128257, EOS=128258.",
        "initialization": "Fresh random weights; shared seed/backbone digest; no checkpoint transfer.",
        "cache_policy": "Reuse frozen raw bytes; original Llama 3 native encoding and context packings use separate hash/schema identities."}
    write_json(destination / "identity.json", info)
    write_json(destination / "native_ids_in_dense_order.json", ids)
    print(f"Original Llama 3: {len(ids):,} IDs; tied table {rows:,} x 640; model {total:,} parameters.", flush=True)
    return tokenizer, mapping, info
