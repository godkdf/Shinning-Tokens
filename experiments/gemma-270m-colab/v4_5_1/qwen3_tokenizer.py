"""Official Qwen3 with NFC enabled; byte counts refer to its normalized text."""
QWEN3_TOKENIZER = {
    "reference": "Qwen/Qwen3-0.6B",
    "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "vocabulary_ids": 151669,
    "model_rows": 151672,
    "expected_parameters": 197396096,
}


class Qwen3Tokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.engine = Tokenizer.from_file(str(path))
        self.engine.no_padding()
        self.engine.no_truncation()
        raw = json.loads(Path(path).read_text())
        if (raw["model"]["type"] != "BPE" or raw.get("normalizer") != {"type": "NFC"}
                or raw["decoder"]["type"] != "ByteLevel"):
            raise ValueError("Expected the pinned official Qwen3 NFC ByteLevel BPE pipeline.")
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

    def normalize(self, text):
        return self.engine.normalizer.normalize_str(text)

    def encode(self, text, add_special_tokens=False):
        return self.engine.encode(text, add_special_tokens=add_special_tokens).ids

    def decode_tokens_bytes(self, ids):
        return [self.byte_vocab[index] for index in ids]

    def decode(self, ids):
        return self.engine.decode(ids, skip_special_tokens=False)


def prepare_tokenizer(config, destination):
    from huggingface_hub import hf_hub_download
    if (config["tokenizer_reference"] != QWEN3_TOKENIZER["reference"]
            or config["tokenizer_revision"] != QWEN3_TOKENIZER["revision"]
            or config["model_id"] != "google/gemma-3-270m"):
        raise ValueError("V4.5.1 requires the pinned official Qwen3 NFC tokenizer and Gemma3_270M backbone.")
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(hf_hub_download(repo_id=QWEN3_TOKENIZER["reference"], filename="tokenizer.json",
        revision=QWEN3_TOKENIZER["revision"], cache_dir=config["tokenizer_cache_directory"]))
    if sha256_file(source) != QWEN3_TOKENIZER["sha256"]:
        raise ValueError("Official Qwen3 tokenizer SHA256 differs from the pinned identity.")
    shutil.copyfile(source, destination / "tokenizer.json")
    tokenizer = Qwen3Tokenizer(destination / "tokenizer.json")
    ids = sorted(set(tokenizer.engine.get_vocab(with_added_tokens=True).values()))
    if ids != list(range(151669)):
        raise ValueError("Expected exactly 151,669 contiguous native Qwen3 token IDs.")
    mapping = {old: index + 1 for index, old in enumerate(ids)}
    rows = len(ids) + 3
    total = 100326016 + rows * 640
    if rows != QWEN3_TOKENIZER["model_rows"] or total != QWEN3_TOKENIZER["expected_parameters"]:
        raise ValueError("Unexpected model size after Qwen3 vocabulary adaptation.")
    raw = json.loads((destination / "tokenizer.json").read_text())
    specials = {item["content"]: item["id"] for item in raw["added_tokens"] if item["special"]}
    info = {"reference": QWEN3_TOKENIZER["reference"], "revision": QWEN3_TOKENIZER["revision"],
        "implementation": "official Qwen3 tokenizer.json / Hugging Face tokenizers",
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "tokenizer_sha256": QWEN3_TOKENIZER["sha256"], "vocabulary_ids": len(ids),
        "base_vocabulary_ids": len(raw["model"]["vocab"]), "native_added_token_count": len(raw["added_tokens"]),
        "native_special_token_ids": specials, "original_id_space": max(ids) + 1,
        "unused_native_ids": 0, "model_rows": rows, "pad_id": 0,
        "bos_id": len(ids) + 1, "eos_id": len(ids) + 2,
        "id_mapping": "sorted native IDs -> 1..N; PAD=0, BOS=N+1, EOS=N+2",
        "expected_parameters": total, "model_type": "Qwen3_official_NFC_ByteLevel_BPE",
        "normalization": "official NFC enabled", "scoring_text_form": "NFC-normalized UTF-8",
        "text_encoding_policy": "Official NFC and added-token matching; add_special_tokens=False; no chat template",
        "byte_accounting": "Exact token-byte lengths of NFC-normalized scoring text; not original pre-NFC source bytes"}
    info["vocabulary_adaptation"] = {"backbone": "Gemma3_270M", "backbone_parameters": 100326016,
        "hidden_size": 640, "native_tokenizer_ids": len(ids), "protocol_tokens": 3,
        "embedding_shape": [rows, 640], "embedding_and_head_tied": True,
        "embedding_parameters": rows * 640, "total_parameters": total,
        "v4_parameters": 268100736, "parameter_delta_from_v4": total - 268100736,
        "v4_4_1_parameters": 182411776, "parameter_delta_from_v4_4_1": total - 182411776,
        "automatic_native_special_tokens": False,
        "packing_policy": "Native IDs -> 1..151669; PAD=0, BOS=151670, EOS=151671.",
        "initialization": "Fresh random weights; shared seed/backbone digest; no checkpoint transfer.",
        "cache_policy": "Frozen raw corpus preserved; separate official-NFC text view, encoding and context packings."}
    write_json(destination / "identity.json", info)
    write_json(destination / "native_ids_in_dense_order.json", ids)
    print(f"Qwen3 official NFC: {len(ids):,} IDs; tied table {rows:,} x 640; model {total:,} parameters.", flush=True)
    return tokenizer, mapping, info
