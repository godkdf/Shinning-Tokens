"""Sequence-aware cache identity and input validation, embedded in the runner."""


def validate_training_config(config):
    batch = config["local_batch_per_chip"]
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError("local_batch_per_chip must be a positive integer.")
    if config["sequence_length"] != 2048 or config["training_seconds"] != 180.:
        raise ValueError("V3.8.5 requires 2048-token sequences and one 180-second training interval.")


def tokenization_cache_key(tokenizer_sha256, raw_manifest_sha256, sequence_length):
    identity = {"schema": "sequence-aware-token-cache-v2", "tokenizer_sha256": tokenizer_sha256,
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
    return (config["sequence_length"] == 2048 and config["training_seconds"] == 180.
        and training["tokens"].shape[1:] == (config["local_batch_per_chip"], config["sequence_length"]))
