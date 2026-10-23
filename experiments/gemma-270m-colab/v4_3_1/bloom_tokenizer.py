"""Official BLOOM tokenizer with exact native ByteLevel token-byte accounting."""
BLOOM_TOKENIZER = {
    "reference": "bigscience/bloom",
    "revision": "7f10a99ce7c08f03c7719a586cb2cbda1433ac05",
    "sha256": "3fa39cd4b1500feb205bcce3b9703a4373414cafe4970e0657b413f7ddd2a9d3",
    "vocabulary_ids": 250680,
    "model_rows": 250683,
    "expected_parameters": 260763136,
}


class BloomByteTokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.engine = Tokenizer.from_file(str(path))
        self.engine.no_padding()
        self.engine.no_truncation()
        raw = json.loads(Path(path).read_text())
        if raw["model"]["type"] != "BPE" or raw.get("normalizer") is not None or raw["decoder"]["type"] != "ByteLevel":
            raise ValueError("Expected the pinned unnormalized BLOOM ByteLevel BPE pipeline.")
        # Invert the tokenizer's byte-to-Unicode alphabet. This is not a new encoder.
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        byte_values, codepoints = list(visible), list(visible)
        extra = 0
        for value in range(256):
            if value not in visible:
                byte_values.append(value)
                codepoints.append(256 + extra)
                extra += 1
        byte_decoder = {chr(codepoint): value for value, codepoint in zip(byte_values, codepoints)}
        special = {item["id"]: item["content"].encode("utf-8") for item in raw["added_tokens"] if item["special"]}
        self.byte_vocab = {index: special[index] if index in special else bytes(byte_decoder[c] for c in piece)
            for piece, index in self.engine.get_vocab(with_added_tokens=True).items()}

    def encode(self, text, add_special_tokens=False):
        if "\x0b" in text:
            raise ValueError("The pinned BLOOM vocabulary cannot represent vertical tab U+000B. Original corpus bytes are preserved; no document was skipped or normalized.")
        return self.engine.encode(text, add_special_tokens=add_special_tokens)

    def decode_tokens_bytes(self, ids):
        return [self.byte_vocab[index] for index in ids]

    def decode(self, ids, skip_special_tokens=False):
        return self.engine.decode(ids, skip_special_tokens=skip_special_tokens)


def prepare_tokenizer(config, destination):
    from huggingface_hub import hf_hub_download
    if (config["tokenizer_reference"] != BLOOM_TOKENIZER["reference"]
            or config["tokenizer_revision"] != BLOOM_TOKENIZER["revision"]
            or config["model_id"] != "google/gemma-3-270m"):
        raise ValueError("V4.3.1 requires the pinned BLOOM tokenizer and the existing Gemma3_270M backbone.")
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(hf_hub_download(repo_id=BLOOM_TOKENIZER["reference"], filename="tokenizer.json",
        revision=BLOOM_TOKENIZER["revision"], cache_dir=config["tokenizer_cache_directory"]))
    if sha256_file(source) != BLOOM_TOKENIZER["sha256"]:
        raise ValueError("BLOOM tokenizer SHA256 differs from the pinned project identity.")
    shutil.copyfile(source, destination / "tokenizer.json")
    tokenizer = BloomByteTokenizer(destination / "tokenizer.json")
    ids = sorted(set(tokenizer.engine.get_vocab(with_added_tokens=True).values()))
    if ids != list(range(BLOOM_TOKENIZER["vocabulary_ids"])):
        raise ValueError("Expected exactly 250,680 contiguous native BLOOM IDs.")
    mapping = {old: index + 1 for index, old in enumerate(ids)}
    rows = len(ids) + 3
    total = 100326016 + rows * 640
    if rows != BLOOM_TOKENIZER["model_rows"] or total != BLOOM_TOKENIZER["expected_parameters"]:
        raise ValueError("Unexpected model size after BLOOM vocabulary adaptation.")
    raw = json.loads((destination / "tokenizer.json").read_text())
    specials = {item["content"]: item["id"] for item in raw["added_tokens"] if item["special"]}
    info = {"reference": BLOOM_TOKENIZER["reference"], "revision": BLOOM_TOKENIZER["revision"],
        "implementation": "official BLOOM tokenizer.json / Hugging Face tokenizers",
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "tokenizer_sha256": BLOOM_TOKENIZER["sha256"], "vocabulary_ids": len(ids),
        "native_special_token_ids": specials, "original_id_space": max(ids) + 1,
        "unused_native_ids": 0, "model_rows": rows, "pad_id": 0,
        "bos_id": len(ids) + 1, "eos_id": len(ids) + 2,
        "id_mapping": "sorted native IDs map to 1..N; PAD=0, BOS=N+1, EOS=N+2",
        "expected_parameters": total, "model_type": "BLOOM_ByteLevel_BPE",
        "unsupported_input_codepoints": ["U+000B"],
        "text_encoding_policy": "Official BLOOM encode(add_special_tokens=False); native added-token matching retained; no chat template",
        "byte_accounting": "Exact inverse-ByteLevel token bytes; native special spellings use their literal UTF-8 bytes"}
    info["vocabulary_adaptation"] = {
        "backbone": "Gemma3_270M", "backbone_parameters": 100326016,
        "hidden_size": 640, "native_tokenizer_ids": len(ids), "protocol_tokens": 3,
        "embedding_shape": [rows, 640], "embedding_and_head_tied": True,
        "embedding_parameters": rows * 640, "total_parameters": total,
        "v4_parameters": 268100736, "parameter_delta_from_v4": total - 268100736,
        "v4_2_2_parameters": 228327936, "parameter_delta_from_v4_2_2": total - 228327936,
        "automatic_native_special_tokens": False,
        "packing_policy": "Native IDs -> 1..250680; PAD=0, BOS=250681, EOS=250682.",
        "initialization": "Fresh random weights; shared seed/backbone digest; no checkpoint transfer.",
        "cache_policy": "Reuse frozen raw bytes; checksum-protected BLOOM encoding and both context packings have separate identities."}
    write_json(destination / "identity.json", info)
    write_json(destination / "native_ids_in_dense_order.json", ids)
    print(f"BLOOM tokenizer: {len(ids):,} IDs; tied table {rows:,} x 640; model {total:,} parameters.", flush=True)
    return tokenizer, mapping, info
