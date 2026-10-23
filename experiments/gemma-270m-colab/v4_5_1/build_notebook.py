"""Create V4.5.1 with official Qwen3 NFC and explicitly normalized-text scoring."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "gemma_270m_colab_v5e1_V4_4_1.ipynb"
TARGET = HERE.parent / "gemma_270m_colab_v5e1_V4_5_1.ipynb"


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one anchor; found {source.count(old)}: {old[:100]}")
    return source.replace(old, new, 1)


def literals(notebook):
    cell = next(c for c in notebook["cells"] if "PROGRAM_SOURCE =" in "".join(c["source"]))
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse("".join(cell["source"])).body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("PROGRAM_SOURCE", "CONFIG")}


def tokenizer_pin():
    tree = ast.parse((HERE / "qwen3_tokenizer.py").read_text())
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and n.targets[0].id == "QWEN3_TOKENIZER"))


def build_source(source):
    source = source.replace("V4.4.1", "V4.5.1")
    additions = "\n\n".join((HERE / name).read_text() for name in ("qwen3_tokenizer.py", "nfc_view.py"))
    replacements = {}
    for n in ast.parse(additions).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            replacements[n.name] = ast.get_source_segment(additions, n)
        elif isinstance(n, ast.Assign):
            replacements[n.targets[0].id] = ast.get_source_segment(additions, n)
    lines = source.splitlines(True)
    for n in reversed(ast.parse(source).body):
        name = n.name if isinstance(n, (ast.FunctionDef, ast.ClassDef)) else (
            n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None)
        if name in ("LLAMA3_TOKENIZER", "LLAMA3_PATTERN", "Llama3Tokenizer", "llama3_text_chunks"):
            lines[n.lineno - 1:n.end_lineno] = []
        elif name in replacements:
            lines[n.lineno - 1:n.end_lineno] = [replacements.pop(name) + "\n"]
    source = once("".join(lines), "\ndef main():", "\n\n" + "\n\n".join(replacements.values()) + "\n\ndef main():")
    source = once(source, 'ENCODING_SCHEMA = "v441-native-llama3-byte-document-encoding-1"',
        'ENCODING_SCHEMA = "v451-qwen3-nfc-document-encoding-1"')
    source = once(source, '"schema": "sequence-aware-token-cache-v5-native-llama3-bytes"',
        '"schema": "sequence-aware-token-cache-v6-qwen3-nfc-bytes"')
    source = once(source, '    tokenizer, mapping, identity = prepare_tokenizer(config, prepared / "tokenizer")',
        '''    tokenizer, mapping, identity = prepare_tokenizer(config, prepared / "tokenizer")
    source_raw = raw
    raw, manifest, normalization = prepare_nfc_view(shared, source_raw, tokenizer, identity, output)
    config["scoring_text_directory"] = str(raw)
    config["frozen_raw_manifest_sha256"] = normalization["source_manifest_sha256"]
    config["scoring_text_form"] = "NFC-normalized UTF-8"''')
    source = once(source,
        '    manifest = verify_raw(Path(config["shared_directory"]) / "raw")',
        '''    manifest = verify_raw(Path(config["scoring_text_directory"]))
    source_raw = Path(config["shared_directory"]) / "raw"
    verify_raw(source_raw)
    if (sha256_file(source_raw / "manifest.json") != config["frozen_raw_manifest_sha256"]
            or manifest["text_normalization"]["source_manifest_sha256"] != config["frozen_raw_manifest_sha256"]
            or manifest["text_normalization"]["scoring_text_form"] != "NFC-normalized UTF-8"):
        raise ValueError("NFC scoring view no longer matches the unchanged frozen source corpus.")''')
    source = once(source, '        "actual_training_bytes": manifest["original_bytes"]["train"], "target_training_bytes": 256 * 1024**2,',
        '        "actual_training_bytes": manifest["original_bytes"]["train"], "target_training_bytes": 256 * 1024**2,\n'
        '        "training_bytes_scope": "NFC-normalized text; target refers to frozen source bytes",\n'
        '        "frozen_source_training_bytes": normalization["source_raw_bytes"]["train"], "normalization": normalization,')
    source = once(source, '    return {"pooled_bpb": float(ln.sum() / (lb.sum() * math.log(2))),',
        '    return {"scoring_text_form": "NFC-normalized UTF-8", "scored_utf8_bytes": raw_bytes,\n'
        '        "legacy_byte_field_note": "original_utf8_bytes contains normalized scoring bytes for compatibility",\n'
        '        "pooled_bpb": float(ln.sum() / (lb.sum() * math.log(2))),')
    source = once(source, '        "006_LOSSLESS_TOKENIZATION":', '        "006_NFC_TEXT_TOKENIZATION":')
    source = once(source, '        "029_NO_RESTORE_CHECK": read_json(output / "checkpoint_manifest.json")["restore_executed"] is False,',
        '        "029_NO_RESTORE_CHECK": read_json(output / "checkpoint_manifest.json")["restore_executed"] is False,\n'
        '        "036_NFC_SCORING_VIEW": manifest["text_normalization"]["source_corpus_modified"] is False\n'
        '            and manifest["text_normalization"]["normalized_train_eval_span_overlap"] == 0,')
    source = once(source, '        "byte_accounting_note": "CPU batch-prefix sums using tokenizer offset attribution; repeated passes included, not unique bytes",',
        '        "byte_accounting_note": "Exact NFC token bytes and CPU row-prefix sums; repeats included. Legacy original-byte fields contain NFC scoring bytes.",\n'
        '        "scoring_text_form": "NFC-normalized UTF-8",\n'
        '        "normalization": read_json(output / "normalization_report.json"),\n'
        '        "frozen_raw_data_manifest_sha256": config["frozen_raw_manifest_sha256"],\n'
        '        "processed_scoring_bytes": bytes_processed, "scoring_bytes_per_second": bytes_processed / training_seconds,')
    source = source.replace('"FineWiki mean of 16 per-language BPB values at fixed 1024-token context; lower is better"',
        '"FineWiki mean of 16 per-language BPB on official Qwen3 NFC-normalized UTF-8 text at context 1024; denominator is NFC bytes"')
    source = source.replace("Original Llama 3 tokenizer, expanded corpus", "Official Qwen3 NFC-normalized scoring, expanded corpus")
    source = source.replace("Llama 3 native encoding once:", "Qwen3 NFC encoding once:")
    source = once(source, '{byte_count:,} original bytes', '{byte_count:,} NFC-scoring bytes')
    source = once(source, '"Evaluation original-byte denominator or finite-loss check failed."',
        '"Evaluation NFC-byte denominator or finite-loss check failed."')
    source = once(source, '        "original_bytes_per_second_attributed")}, indent=2), flush=True)',
        '        "scoring_bytes_per_second")}, indent=2), flush=True)')
    source = source.replace("Initial FineWiki macro BPB=", "Initial FineWiki NFC macro BPB=").replace("FineWiki macro BPB:", "FineWiki NFC macro BPB:")
    source = source.replace('"note": "Both contexts reuse the same frozen text. Byte exposure includes repeats, not unique corpus bytes."',
        '"note": "Both contexts reuse the same NFC-normalized view of the frozen source documents. Byte exposure is normalized text including repeats."')
    ast.parse(source)
    return source


def markdown(text):
    return {"cell_type": "markdown", "id": hashlib.sha256(text.encode()).hexdigest()[:8],
        "metadata": {}, "source": text.splitlines(True)}


def build_notebook():
    notebook = json.loads(BASE.read_text())
    values = literals(notebook)
    source = build_source(values["PROGRAM_SOURCE"])
    config = values["CONFIG"]
    pin = tokenizer_pin()
    config.update(version="V4.5.1", experiment="GEM-COLAB-V4.5.1", requirement_prefix="V4.5.1",
        implementation_revision="V4.5.1-qwen3-official-nfc-delayed-discovery-1",
        tokenizer_reference=pin["reference"], tokenizer_revision=pin["revision"],
        scoring_text_form="NFC-normalized UTF-8")
    notebook["metadata"]["colab"]["name"] = TARGET.name
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "PROGRAM_SOURCE =" in text:
            tail = text[text.index('CONFIG["tokenizer_reference"]'):]
            tail = once(tail, "CONFIG[\"tokenizer_revision\"] = '315b20096dc791d381d514deb5f8bd9c8d6d3061'",
                'CONFIG["tokenizer_revision"] = ' + repr(pin["revision"]))
            tail = once(tail, '"V3-tokenizers" / "llama3"', '"V3-tokenizers" / "qwen3"')
            tail = tail.replace("V4_4_1", "V4_5_1").replace("V4.4.1", "V4.5.1")
            text = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + "\n" + tail
        else:
            text = text.replace("V4.4.1", "V4.5.1").replace("gemma_colab_pilot_V4_4_1", "gemma_colab_pilot_V4_5_1")
        cell["source"] = text.splitlines(True)
        if cell["cell_type"] == "code":
            ast.parse(text)
            cell.update(outputs=[], execution_count=None)
    for index, name in ((0, "INTRO.md"), (1, "REQUIREMENTS.md")):
        notebook["cells"][index] = markdown((HERE / name).read_text())
    notebook["cells"][4] = markdown("## 2. Official Qwen3 tokenizer and corpus access\nLoad the pinned Qwen/Qwen3-0.6B tokenizer with its official NFC normalizer enabled. Scores use NFC-normalized UTF-8 bytes. HF_TOKEN remains for corpus sources including Stack-smol. No model weights or remote Python code.\n")
    auth = "".join(notebook["cells"][5]["source"])
    auth = once(auth, 'TOKENIZER_REFERENCE = "NousResearch/Meta-Llama-3-8B"', 'TOKENIZER_REFERENCE = "Qwen/Qwen3-0.6B"')
    auth = auth.replace("the pinned tokenizer redistribution is public", "Qwen3's tokenizer is public")
    notebook["cells"][5]["source"] = auth.splitlines(True)
    notebook["cells"][8] = markdown("## 4. Preserve raw data; build an official-NFC scoring view\nKeep the frozen 256 MiB source corpus untouched. Create a checksum-protected NFC copy with identical documents, split assignments, order and source provenance; report changed documents and byte counts. Encode the NFC text once and pack 1,024/2,048 sequentially. BPB divides by normalized UTF-8 bytes.\n")
    display = "".join(notebook["cells"][13]["source"])
    display = display.replace('("Attributed original bytes/s", fmt(result["original_bytes_per_second_attributed"], 1))',
        '("NFC-normalized scoring bytes/s", fmt(result["scoring_bytes_per_second"], 1))')
    display = display.replace("Initial → final FineWiki macro BPB", "Initial → final FineWiki macro BPB (NFC text)").replace("Initial → final in-mixture pooled BPB", "Initial → final in-mixture pooled BPB (NFC text)")
    display += '\nprint("NFC normalization report:", json.dumps(result["normalization"], indent=2))\nprint("Metric scope: NFC-normalized text/bytes; earlier original-byte BPB is not a controlled comparison.")\n'
    notebook["cells"][13]["source"] = display.splitlines(True)
    data = "".join(notebook["cells"][16]["source"])
    data = data[:data.index("\nV4.5.1 retains the frozen raw corpus")]
    data = once(data, 'BPB excludes virtual EOS and divides original-text negative log likelihood by\noriginal UTF-8 bytes ×ln(2).',
        'V4.5.1 BPB excludes virtual EOS and divides NFC-text negative log likelihood by\nNFC-normalized UTF-8 bytes ×ln(2). The source quotas above apply before normalization.')
    data += "\nV4.5.1 keeps the frozen source corpus intact and uses a separate official-Qwen3-NFC scoring view of the same documents, order, splits and provenance. Report raw and normalized byte totals, changed-document counts and hashes; reject new normalized train/held-out exact-span collisions. The NFC byte count may differ from the 256 MiB source target. Official normalizer, BPE, regex and added-token matching remain enabled. Encode once and pack both contexts under separate cache identities. Initial batches, delayed discovery, 20-minute budget and 3M/1M context schedule remain unchanged. This NFC-normalized BPB is explicitly a different text/scoring basis from previous original-byte reports.\n"
    notebook["cells"][16] = markdown(data)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    return notebook, source


def main():
    runner, provenance = HERE / "runner.py", HERE / "build_provenance.json"
    if any(path.exists() for path in (TARGET, runner, provenance)):
        raise FileExistsError("Preserving delivered artifacts; create a new version.")
    notebook, source = build_notebook()
    with runner.open("x") as stream:
        stream.write(source)
    with TARGET.open("x") as stream:
        json.dump(notebook, stream, indent=1, ensure_ascii=False)
        stream.write("\n")
    with provenance.open("x") as stream:
        json.dump({"base_notebook": BASE.name, "base_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
            "notebook_sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(source.encode()).hexdigest(), "tokenizer": tokenizer_pin(),
            "scoring_text_form": "official Qwen3 NFC-normalized UTF-8", "source_corpus_modified": False,
            "discovery_start_after_updates": 1000, "training_seconds": 1200,
            "device_tests_executed": False}, stream, indent=2)
        stream.write("\n")
    print(TARGET)


if __name__ == "__main__":
    main()
