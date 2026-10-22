"""Derive V3.8.5 from the user's V3.8 notebook; never overwrite either artifact."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
BASE = PARENT / "gemma_270m_colab_v5e1_V3_8.ipynb"
TARGET = PARENT / "gemma_270m_colab_v5e1_V3_8_5.ipynb"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one source anchor; found {source.count(old)}: {old[:100]}")
    return source.replace(old, new, 1)


def embedded_literals(notebook):
    cell = next(c for c in notebook["cells"] if "PROGRAM_SOURCE =" in "".join(c["source"]))
    values = {}
    for node in ast.parse("".join(cell["source"])).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("PROGRAM_SOURCE", "CONFIG"):
                values[node.targets[0].id] = ast.literal_eval(node.value)
    return values


def build_source(original):
    source = original.replace("V3.8", "V3.8.5").replace("V38-", "V385-")
    # The source attached by the user has relaxed guards; replace the whole
    # guard rather than relying on its old numeric thresholds.
    start = source.index('    if config["local_batch_per_chip"]', source.index("def train_pilot("))
    end = source.index('    manifest = verify_raw(', start)
    source = source[:start] + '    validate_training_config(config)\n' + source[end:]
    anchor = 'def prepare_cached(config, work, output):\n'
    source = replace_once(source, anchor, (HERE / "context.py").read_text() + '\n\n' + anchor +
        '    validate_training_config(config)\n')
    source = replace_once(source,
        '    token_key = hashlib.sha256((identity["tokenizer_sha256"] + sha256_file(raw / "manifest.json") + PACKING_VERSION).encode()).hexdigest()[:24]',
        '    raw_digest = sha256_file(raw / "manifest.json")\n'
        '    token_key = tokenization_cache_key(identity["tokenizer_sha256"], raw_digest, config["sequence_length"])')
    source = replace_once(source, '        cached = read_json(token_cache / "tokenization.json")',
        '        cached = read_json(token_cache / "tokenization.json")\n'
        '        validate_tokenization_manifest(cached, identity, raw_digest, config["sequence_length"])')
    source = replace_once(source,
        '        details = {"tokenizer": identity, "raw_manifest_sha256": sha256_file(raw / "manifest.json"), "splits": {}, "files": {}}',
        '        details = {"tokenizer": identity, "raw_manifest_sha256": raw_digest,\n'
        '            "sequence_length": config["sequence_length"], "packing_version": PACKING_VERSION,\n'
        '            "cache_key": token_key, "splits": {}, "files": {}}')
    source = replace_once(source, '            name = split + ".npz"\n',
        '            validate_sequence_arrays(arrays, config["sequence_length"], split)\n'
        '            name = split + ".npz"\n')
    source = replace_once(source, '        "tokenizer_sha256": identity["tokenizer_sha256"], "shared_directory": str(shared)})',
        '        "tokenizer_sha256": identity["tokenizer_sha256"], "shared_directory": str(shared),\n'
        '        "sequence_length": config["sequence_length"], "tokenization_cache_key": token_key})')
    source = replace_once(source, '    identity = config["tokenizer_identity"]\n',
        '    identity = config["tokenizer_identity"]\n'
        '    validate_tokenization_manifest(tokenization, identity, sha256_file(data_dir / "manifest.json"), config["sequence_length"])\n')
    source = replace_once(source, '            arrays = {key: archive[key] for key in INPUT_KEYS}\n',
        '            arrays = {key: archive[key] for key in INPUT_KEYS}\n'
        '        validate_sequence_arrays(arrays, config["sequence_length"], split)\n')
    source = replace_once(source,
        '        "025_NATIVE_TRAINING": config["local_batch_per_chip"] == 3 and config["sequence_length"] == 1024 and config["training_seconds"] == 180.,',
        '        "025_NATIVE_TRAINING": training_specification_matches(config, training),')
    source = replace_once(source, 'def readable_evaluation(values, raw_bytes):',
        'def readable_evaluation(values, raw_bytes, sequence_length):')
    source = replace_once(source,
        '"scoring": "all original text targets once; virtual EOS excluded from BPB; fixed 1024-token chunk context"',
        '"scoring": f"all original text targets once; virtual EOS excluded from BPB; fixed {sequence_length}-token chunk context"')
    source = replace_once(source, '                    manifest["original_bytes"][split])',
        '                    manifest["original_bytes"][split], config["sequence_length"])')
    source = replace_once(source,
        '        "reference_batch": 2, "reference_training_seconds": 60.014611282999795,',
        '        "reference_batch": 2, "reference_sequence_length": 1024,\n'
        '        "reference_training_seconds": 60.014611282999795,\n'
        '        "current_sequence_length": config["sequence_length"],\n'
        '        "same_context_length": config["sequence_length"] == 1024,')
    source = replace_once(source,
        '        "note": "Descriptive historical comparison only. Batch and training duration changed; BPB does not isolate implementation efficiency."',
        '        "note": "Descriptive historical comparison only. Context length, packing, batch and training duration differ. BPB and throughput do not isolate implementation efficiency."')
    source = replace_once(source,
        '        "historical_comparison_note": "V3.5 used batch 2 and 60 seconds; V3.8.5 uses batch 3 and 180 seconds. Historical differences do not isolate implementation efficiency.",',
        '        "historical_comparison_note": "V3.5 used batch 2 x 1024 and 60 seconds. This run uses 2048-token context and 180 seconds; context, packing and batch affect the comparison.",')
    # Avoid attributing every native Gemma MLP to the parent _apply_attention.
    source = replace_once(source,
        '    if any(key in text for key in ("attention", "/attn/", "q_einsum", "kv_einsum", "attn_vec")):\n'
        '        return "attention"\n'
        '    if any(key in text for key in ("mlp", "feedforward", "gating_einsum", "linear_einsum")):\n'
        '        return "mlp"',
        '    if any(key in text for key in ("/mlp/", "feedforward", "gating_einsum", "linear_einsum")):\n'
        '        return "mlp"\n'
        '    if any(key in text for key in ("/attn/", "q_einsum", "kv_einsum", "attn_vec")):\n'
        '        return "attention"')
    ast.parse(source)
    return source


def markdown(text):
    return {"cell_type": "markdown", "id": hashlib.sha256(text.encode()).hexdigest()[:8],
            "metadata": {}, "source": text.splitlines(True)}


def build_notebook():
    notebook = json.loads(BASE.read_text())
    values = embedded_literals(notebook)
    source = build_source(values["PROGRAM_SOURCE"])
    config = values["CONFIG"]
    config.update(experiment="GEM-COLAB-V3.8.5", version="V3.8.5", requirement_prefix="V385",
        implementation_revision="V3.8.5-native-b2-s2048-180s-1", sequence_length=2048,
        local_batch_per_chip=2, training_seconds=180.)
    notebook["metadata"]["colab"]["name"] = TARGET.name
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "PROGRAM_SOURCE =" in text:
            tail = text[text.index('CONFIG["tokenizer_reference"]'):]
            tail = tail.replace("V3.8", "V3.8.5").replace("V3_8", "V3_8_5")
            text = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + "\n" + tail
        else:
            text = text.replace("V3.8", "V3.8.5").replace("V3_8", "V3_8_5").replace("V38-", "V385-")
        cell["source"] = text.splitlines(True)
        if cell["cell_type"] == "code":
            cell.update(execution_count=None, outputs=[])
            ast.parse(text)
    notebook["cells"][0] = markdown((HERE / "INTRO.md").read_text())
    notebook["cells"][1] = markdown((HERE / "REQUIREMENTS.md").read_text())
    notebook["cells"][8] = markdown("""## 4. Prepare 2,048-token inputs from the frozen corpus
Reuse verified raw text and the pinned tokenizer. The tokenization cache key
includes context length. Build new 2,048-token chunks and packing; never load
the old 1,024-token input cache. Future matching 2,048-token caches are reused.
""")
    notebook["cells"][10] = markdown("""## 5. Train once for 180 seconds — batch 2 × 2,048
4,096 positions/update. Native dense full-vocabulary loss, AdamW, five warmup
updates then learning rate 0.0003. Final evaluation uses the same 2,048-token
context as training. No checkpoint restoration or reload evaluation.
Save weights before a separate 32-update diagnostic. Missing required hardware
measurements produce profiling FAIL while preserving the completed model.
""")
    data = "".join(notebook["cells"][16]["source"])
    data = data.replace("1,024-token context", "2,048-token context")
    data = data.replace("1,024-token chunk context", "2,048-token chunk context")
    data = data.replace("V3.8.5 uses batch 3 × 1,024 and 180 seconds.", "V3.8.5 defaults to batch 2 × 2,048 and 180 seconds.")
    data += "\nThe sequence-aware cache rebuilds chunks from the same frozen source text. Source bytes and tokenizer remain fixed; chunk boundaries, packing order, and effective evaluation context change. Historical 1,024-context BPB scores are not matched-context comparisons.\n"
    notebook["cells"][16]["source"] = data.splitlines(True)
    return notebook, source


def main():
    if TARGET.exists() or (HERE / "runner.py").exists():
        raise FileExistsError("Preserving existing V3.8.5 artifacts; create a new version.")
    notebook, source = build_notebook()
    with (HERE / "runner.py").open("x") as stream:
        stream.write(source)
    with TARGET.open("x") as stream:
        json.dump(notebook, stream, indent=1, ensure_ascii=False)
        stream.write("\n")
    print(TARGET)


if __name__ == "__main__":
    main()
