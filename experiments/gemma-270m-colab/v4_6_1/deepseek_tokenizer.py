"""Pinned official DeepSeek-V3-Base tokenizer with exact original-byte accounting."""
DEEPSEEK_TOKENIZER = {
    "reference": "deepseek-ai/DeepSeek-V3-Base",
    "revision": "afb92e1fa402c2be2a9eb085312bb02e0384d6c7",
    "sha256": "621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41",
    "vocabulary_ids": 128815,
    "model_rows": 128818,
    "expected_parameters": 182769536,
}


class DeepSeekTokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.engine = Tokenizer.from_file(str(path))
        self.engine.no_padding()
        self.engine.no_truncation()
        raw = json.loads(Path(path).read_text())
        if (raw["model"]["type"] != "BPE"
                or raw.get("normalizer") != {"type": "Sequence", "normalizers": []}
                or raw["decoder"]["type"] != "ByteLevel"
                or any(item["lstrip"] or item["rstrip"] for item in raw["added_tokens"])):
            raise ValueError("Expected the pinned identity-normalization DeepSeek-V3-Base ByteLevel BPE pipeline.")
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        byte_values, codepoints = list(visible), list(visible)
        extra = 0
        for value in range(256):
            if value not in visible:
                byte_values.append(value)
                codepoints.append(256 + extra)
                extra += 1
        byte_decoder = {chr(codepoint): value for value, codepoint in zip(byte_values, codepoints)}
        added = {item["id"]: item["content"].encode("utf-8") for item in raw["added_tokens"]}
        self.byte_vocab = {index: added[index] if index in added else bytes(byte_decoder[c] for c in piece)
            for piece, index in self.engine.get_vocab(with_added_tokens=True).items()}

    def encode(self, text, add_special_tokens=False):
        return self.engine.encode(text, add_special_tokens=add_special_tokens).ids

    def decode_tokens_bytes(self, ids):
        return [self.byte_vocab[index] for index in ids]

    def decode(self, ids):
        return self.engine.decode(ids, skip_special_tokens=False)


def prepare_tokenizer(config, destination):
    from huggingface_hub import hf_hub_download
    if (config["tokenizer_reference"] != DEEPSEEK_TOKENIZER["reference"]
            or config["tokenizer_revision"] != DEEPSEEK_TOKENIZER["revision"]
            or config["model_id"] != "google/gemma-3-270m"):
        raise ValueError("V4.6.1 requires the pinned DeepSeek-V3-Base tokenizer and Gemma3_270M backbone.")
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(hf_hub_download(repo_id=DEEPSEEK_TOKENIZER["reference"], filename="tokenizer.json",
        revision=DEEPSEEK_TOKENIZER["revision"], cache_dir=config["tokenizer_cache_directory"]))
    if sha256_file(source) != DEEPSEEK_TOKENIZER["sha256"]:
        raise ValueError("DeepSeek-V3-Base tokenizer SHA256 differs from the pinned identity.")
    shutil.copyfile(source, destination / "tokenizer.json")
    tokenizer = DeepSeekTokenizer(destination / "tokenizer.json")
    ids = sorted(set(tokenizer.engine.get_vocab(with_added_tokens=True).values()))
    if ids != list(range(128815)):
        raise ValueError("Expected exactly 128,815 contiguous native DeepSeek-V3-Base token IDs.")
    mapping = {old: index + 1 for index, old in enumerate(ids)}
    rows = len(ids) + 3
    total = 100326016 + rows * 640
    if rows != DEEPSEEK_TOKENIZER["model_rows"] or total != DEEPSEEK_TOKENIZER["expected_parameters"]:
        raise ValueError("Unexpected model size after DeepSeek vocabulary adaptation.")
    raw = json.loads((destination / "tokenizer.json").read_text())
    specials = {item["content"]: item["id"] for item in raw["added_tokens"] if item["special"]}
    info = {"reference": DEEPSEEK_TOKENIZER["reference"], "revision": DEEPSEEK_TOKENIZER["revision"],
        "implementation": "official DeepSeek-V3-Base tokenizer.json / Hugging Face tokenizers",
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "tokenizer_sha256": DEEPSEEK_TOKENIZER["sha256"], "vocabulary_ids": len(ids),
        "base_vocabulary_ids": len(raw["model"]["vocab"]), "native_added_token_count": len(raw["added_tokens"]),
        "native_special_token_ids": specials, "original_id_space": max(ids) + 1,
        "unused_native_ids": 0, "model_rows": rows, "pad_id": 0,
        "bos_id": len(ids) + 1, "eos_id": len(ids) + 2,
        "id_mapping": "sorted native IDs -> 1..N; PAD=0, BOS=N+1, EOS=N+2",
        "expected_parameters": total, "model_type": "DeepSeek_V3_Base_official_ByteLevel_BPE",
        "normalization": "official identity (empty normalizer sequence)",
        "scoring_text_form": "original frozen UTF-8 bytes",
        "text_encoding_policy": "Official added-token matching; add_special_tokens=False; no chat template or normalization",
        "byte_accounting": "Exact original UTF-8 token-byte lengths, including added-token spellings and partial UTF-8 pieces"}
    info["vocabulary_adaptation"] = {"backbone": "Gemma3_270M", "backbone_parameters": 100326016,
        "hidden_size": 640, "native_tokenizer_ids": len(ids), "protocol_tokens": 3,
        "embedding_shape": [rows, 640], "embedding_and_head_tied": True,
        "embedding_parameters": rows * 640, "total_parameters": total,
        "v4_parameters": 268100736, "parameter_delta_from_v4": total - 268100736,
        "v4_4_1_parameters": 182411776, "parameter_delta_from_v4_4_1": total - 182411776,
        "automatic_native_special_tokens": False,
        "packing_policy": "Native IDs -> 1..128815; PAD=0, BOS=128816, EOS=128817.",
        "initialization": "Fresh random weights; shared seed/backbone digest; no checkpoint transfer.",
        "cache_policy": "Reuse frozen raw corpus; separate DeepSeek encoding and context packing identities."}
    write_json(destination / "identity.json", info)
    write_json(destination / "native_ids_in_dense_order.json", ids)
    print(f"DeepSeek-V3-Base: {len(ids):,} IDs; tied table {rows:,} x 640; model {total:,} parameters.", flush=True)
    return tokenizer, mapping, info
