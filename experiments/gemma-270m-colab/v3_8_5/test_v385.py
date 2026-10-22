"""Context, cache and notebook checks without a TPU training experiment."""
import ast
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("build_v385", HERE / "build_notebook.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
NOTEBOOK, SOURCE = builder.build_notebook()
ORIGINAL = builder.embedded_literals(json.loads(builder.BASE.read_text()))["PROGRAM_SOURCE"]


def helpers():
    env = {"np": np, "math": math, "json": json, "hashlib": hashlib,
           "defaultdict": defaultdict, "Path": Path}
    names = {"validate_training_config", "tokenization_cache_key", "validate_tokenization_manifest",
        "validate_sequence_arrays", "training_specification_matches", "pack_chunks", "tokenize_split",
        "operation_category", "readable_evaluation"}
    constants = {"INPUT_KEYS", "EXTRA_KEYS", "LANGUAGES", "DOMAINS", "PACKING_VERSION"}
    tree = ast.parse(SOURCE)
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
        or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in constants)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "fixture", "exec"), env)
    return env


H = helpers()


def test_notebook_defaults_and_all_code_parses():
    values = builder.embedded_literals(NOTEBOOK)
    c = values["CONFIG"]
    assert c["version"] == "V3.8.5"
    assert c["local_batch_per_chip"] == 2
    assert c["sequence_length"] == 2048
    assert c["training_seconds"] == 180
    assert c["learning_rate"] == .0003
    assert c["warmup_updates"] == 5
    assert c["profile_dispatches"] * c["profile_updates_per_dispatch"] == 32
    assert values["PROGRAM_SOURCE"] == SOURCE
    assert "restore_archive" not in SOURCE and "reloaded_eval" not in SOURCE
    assert 'config["local_batch_per_chip"] == 3' not in SOURCE
    assert "fixed 1024-token chunk context" not in SOURCE
    assert 'pilot_V3_8_5.py' in "".join(NOTEBOOK["cells"][7]["source"])
    assert 'colab-v5e1/V3.8.5' in "".join(NOTEBOOK["cells"][3]["source"])
    assert 'gemma_colab_pilot_V3_8_5_reports.zip' in "".join(NOTEBOOK["cells"][15]["source"])
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_context_changes_cache_identity_but_not_raw_protocol_or_model_math():
    key = H["tokenization_cache_key"]
    assert key("tokenizer", "raw", 1024) != key("tokenizer", "raw", 2048)
    assert key("tokenizer", "raw", 2048) == key("tokenizer", "raw", 2048)
    assert key("tokenizer-2", "raw", 2048) != key("tokenizer", "raw", 2048)
    assert key("tokenizer", "raw-2", 2048) != key("tokenizer", "raw", 2048)
    def function(source, name):
        return ast.dump(next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name))
    for name in ("recipe", "protocol_fingerprint", "tokenize_split", "pack_chunks", "make_model",
                 "make_optimizer", "token_losses", "make_training_functions"):
        assert function(SOURCE, name) == function(ORIGINAL, name)


def test_cache_metadata_rejects_legacy_context():
    identity, raw = {"tokenizer_sha256": "tok"}, "raw"
    valid = {"tokenizer": identity, "raw_manifest_sha256": raw, "sequence_length": 2048,
        "packing_version": H["PACKING_VERSION"], "cache_key": H["tokenization_cache_key"]("tok", raw, 2048)}
    H["validate_tokenization_manifest"](valid, identity, raw, 2048)
    for change in ({"sequence_length": 1024}, {"packing_version": "wrong"},
                   {"cache_key": "old"}, {"raw_manifest_sha256": "other"}):
        with pytest.raises(ValueError, match="metadata does not match"):
            H["validate_tokenization_manifest"](dict(valid, **change), identity, raw, 2048)
    legacy = {"tokenizer": identity, "raw_manifest_sha256": raw}
    with pytest.raises(ValueError):
        H["validate_tokenization_manifest"](legacy, identity, raw, 2048)


def test_training_contract_follows_actual_configured_batch():
    for batch in (1, 2, 3, 5):
        config = {"local_batch_per_chip": batch, "sequence_length": 2048, "training_seconds": 180.}
        H["validate_training_config"](config)
        assert H["training_specification_matches"](config, {"tokens": np.empty((2, batch, 2048), np.int32)})
        assert not H["training_specification_matches"](config, {"tokens": np.empty((2, batch, 1024), np.int32)})
    for batch in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            H["validate_training_config"]({"local_batch_per_chip": batch, "sequence_length": 2048, "training_seconds": 180.})


class CharacterTokenizer:
    def encode(self, text, add_special_tokens):
        return SimpleNamespace(ids=[ord(c) for c in text], offsets=[(i, i + 1) for i in range(len(text))])
    def decode(self, ids, skip_special_tokens):
        return "".join(chr(x) for x in ids)


def test_rechunking_preserves_all_targets_and_unicode_bytes():
    texts = ["a" * 2049 + "é中" + "b" * 17, "c" * 11]
    documents = [{"text": t, "sha256": hashlib.sha256(t.encode()).hexdigest(),
        "language": H["LANGUAGES"][0], "domain": H["DOMAINS"][0], "utf8_bytes": len(t.encode())} for t in texts]
    chars = sorted(set("".join(texts)))
    mapping = {ord(c): i + 3 for i, c in enumerate(chars)}
    identity = {"bos_id": 1, "eos_id": 2}
    observed = {}
    for length in (1024, 2048):
        arrays, report = H["tokenize_split"](documents, CharacterTokenizer(), mapping, identity, length)
        H["validate_sequence_arrays"](arrays, length, "fixture")
        targets = arrays["targets"][arrays["mask"]]
        observed[length] = np.sort(targets)
        assert int(arrays["text_mask"].sum()) == sum(map(len, texts))
        assert int(arrays["byte_weights"].sum()) == sum(len(t.encode()) for t in texts)
        assert np.count_nonzero(targets == 2) == len(texts)
        assert report["targets_preserved"] and report["context_preserved"]
        assert arrays["tokens"].shape[-1] == length
        assert int(arrays["positions"].max()) == length - 1
        if length == 1024:
            with pytest.raises(ValueError, match="2048"):
                H["validate_sequence_arrays"](arrays, 2048, "stale-cache")
    np.testing.assert_array_equal(observed[1024], observed[2048])


def test_evaluation_reports_actual_context_and_preserves_byte_denominator():
    languages, domains = len(H["LANGUAGES"]), len(H["DOMAINS"])
    ln, lb = np.ones(languages), np.ones(languages) * 10
    dn, db = np.ones(domains), np.ones(domains) * 15
    result = H["readable_evaluation"]((ln, lb, dn, db, 0.), 120, 2048)
    assert result["original_utf8_bytes"] == 120
    assert "2048-token" in result["scoring"]
    assert result["pooled_bpb"] == pytest.approx(12 / (120 * math.log(2)))


def test_mlp_is_not_mislabeled_by_parent_attention_scope():
    category = H["operation_category"]
    assert category("Gemma3_270M._apply_attention/layer_14/mlp/Einsum_0/dot_general") == "mlp"
    assert category("Gemma3_270M._apply_attention/layer_14/attn/q_einsum") == "attention"
    assert category("Gemma3_270M._apply_attention/layer_14/pre_attention_norm/rsqrt") == "other_or_fused"
    assert category("Gemma3_270M/embedder.decode/dot_general") == "vocabulary_projection"


def test_dependency_lock_stays_identical_and_existing_notebook_cannot_be_overwritten(tmp_path, monkeypatch):
    def lock_values(notebook):
        tree = ast.parse("".join(notebook["cells"][3]["source"]))
        return {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ("REQUIREMENTS_INPUT", "RESOLVED_DEPENDENCY_LOCK")}
    assert lock_values(NOTEBOOK) == lock_values(json.loads(builder.BASE.read_text()))
    existing = tmp_path / "existing.ipynb"
    existing.write_text("preserve me")
    monkeypatch.setattr(builder, "TARGET", existing)
    with pytest.raises(FileExistsError):
        builder.main()
    assert existing.read_text() == "preserve me"
