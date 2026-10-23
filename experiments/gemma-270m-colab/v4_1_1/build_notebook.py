"""Derive V4.1.1 from the preserved V4 notebook; no accelerator execution."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "gemma_270m_colab_v5e1_V4.ipynb"
TARGET = HERE.parent / "gemma_270m_colab_v5e1_V4_1_1.ipynb"
TOKENIZER = {
    "reference": "google/gemma-4-E4B",
    "revision": "411aa17b749aa952df1359d2dcea73917a544d9a",
    "sha256": "12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6",
    "vocabulary_ids": 262144,
    "model_rows": 262147,
    "expected_parameters": 268100096,
}


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one anchor, found {source.count(old)}: {old[:100]}")
    return source.replace(old, new, 1)


def literals(notebook):
    cell = next(c for c in notebook["cells"] if "PROGRAM_SOURCE =" in "".join(c["source"]))
    return {node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("PROGRAM_SOURCE", "CONFIG")}


def markdown(text):
    return {"cell_type": "markdown", "id": hashlib.sha256(text.encode()).hexdigest()[:8],
        "metadata": {}, "source": text.splitlines(True)}


def build_source(source):
    source = source.replace("V4", "V4.1.1")
    source = once(source, 'STACK = "bigcode/the-stack-smol"',
        'STACK = "bigcode/the-stack-smol"\n\nGEMMA4_TOKENIZER = ' + repr(TOKENIZER))
    source = once(source, '    reference = config["tokenizer_reference"]',
        '''    reference = config["tokenizer_reference"]
    if (reference != GEMMA4_TOKENIZER["reference"]
            or config["tokenizer_revision"] != GEMMA4_TOKENIZER["revision"]
            or config["model_id"] != "google/gemma-3-270m"):
        raise ValueError("V4.1.1 requires the pinned Gemma 4 tokenizer and the existing Gemma3_270M backbone.")''')
    source = once(source, '    tokenizer = Tokenizer.from_file(str(destination / "tokenizer.json"))',
        '''    tokenizer_digest = sha256_file(destination / "tokenizer.json")
    if tokenizer_digest != GEMMA4_TOKENIZER["sha256"] or revision != GEMMA4_TOKENIZER["revision"]:
        raise ValueError("Gemma 4 tokenizer artifact/revision differs from the pinned identity.")
    tokenizer = Tokenizer.from_file(str(destination / "tokenizer.json"))''')
    source = once(source, '    rows = len(ids) + 3',
        '''    if ids != list(range(GEMMA4_TOKENIZER["vocabulary_ids"])):
        raise ValueError("Expected exactly 262,144 contiguous native Gemma 4 token IDs.")
    rows = len(ids) + 3''')
    source = once(source, '    write_json(destination / "identity.json", info)',
        '''    info["vocabulary_adaptation"] = {
        "backbone": "Gemma3_270M", "backbone_parameters": 100326016,
        "hidden_size": 640, "native_tokenizer_ids": len(ids),
        "protocol_tokens": 3, "embedding_shape": [rows, 640],
        "embedding_and_head_tied": True, "embedding_parameters": rows * 640,
        "total_parameters": total, "v4_parameters": 268100736,
        "parameter_delta_from_v4": total - 268100736,
        "automatic_native_special_tokens": False,
        "packing_policy": "Native IDs remapped to 1..262144; PAD=0, BOS=262145, EOS=262146.",
        "initialization": "Fresh random weights; shared seed/backbone digest; no checkpoint transfer.",
        "cache_policy": "Reuse frozen raw bytes; encode and pack under the Gemma 4 tokenizer SHA256.",
    }
    if rows != GEMMA4_TOKENIZER["model_rows"] or total != GEMMA4_TOKENIZER["expected_parameters"]:
        raise ValueError("Gemma 4 model vocabulary adaptation has an unexpected parameter count.")
    print(f"Gemma 4 tokenizer: {len(ids):,} IDs; tied table {rows:,} x 640; model {total:,} parameters.", flush=True)
    write_json(destination / "identity.json", info)''')
    source = once(source,
        '    shutil.copyfile(prepared / "tokenizer/identity.json", output / "tokenizer_identity.json")',
        '    shutil.copyfile(prepared / "tokenizer/identity.json", output / "tokenizer_identity.json")\n'
        '    write_json(output / "vocabulary_adaptation.json", identity["vocabulary_adaptation"])')
    source = once(source, 'def make_model(config):\n    from gemma import gm',
        '''def make_model(config):
    # Only the tied vocabulary table changes; retain the V4 transformer architecture.
    from gemma import gm''')
    source = once(source,
        '        "device": environment["device_kind"], "model_parameters": parameter_count, "optimizer": partition,',
        '        "device": environment["device_kind"], "model_parameters": parameter_count, "optimizer": partition,\n'
        '        "tokenizer_identity": identity, "vocabulary_adaptation": identity["vocabulary_adaptation"],')
    ast.parse(source)
    return source


def build_notebook():
    notebook = json.loads(BASE.read_text())
    values = literals(notebook)
    source = build_source(values["PROGRAM_SOURCE"])
    config = values["CONFIG"]
    config.update(version="V4.1.1", experiment="GEM-COLAB-V4.1.1", requirement_prefix="V4.1.1",
        implementation_revision="V4.1.1-gemma4-tokenizer-1",
        tokenizer_reference=TOKENIZER["reference"], tokenizer_revision=TOKENIZER["revision"])
    notebook["metadata"]["colab"]["name"] = TARGET.name
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "PROGRAM_SOURCE =" in text:
            tail = text[text.index('CONFIG["tokenizer_reference"]'):]
            tail = once(tail,
                'CONFIG["tokenizer_revision"] = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1" if CONFIG["tokenizer_reference"] == "google/gemma-3-270m" else "main"',
                'CONFIG["tokenizer_revision"] = ' + repr(TOKENIZER["revision"]))
            tail = tail.replace("V4", "V4_1_1")
            text = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + "\n" + tail
        else:
            text = text.replace("V4", "V4.1.1")
            # Filenames use underscores, display names use dots.
            text = text.replace("gemma_colab_pilot_V4.1.1", "gemma_colab_pilot_V4_1_1")
        cell["source"] = text.splitlines(True)
        if cell["cell_type"] == "code":
            ast.parse(text)
            cell.update(outputs=[], execution_count=None)
    for index, filename in ((0, "INTRO.md"), (1, "REQUIREMENTS.md")):
        notebook["cells"][index] = markdown((HERE / filename).read_text())
    notebook["cells"][4] = markdown("## 2. Pinned Gemma 4 tokenizer and data access\nThe backbone remains Gemma3_270M. Only the tokenizer and vocabulary table change. Use HF_TOKEN with access to the tokenizer repository and Stack-smol.\n")
    auth = "".join(notebook["cells"][5]["source"])
    auth = once(auth, 'TOKENIZER_REFERENCE = "google/gemma-3-270m" #@param {type:"string"}',
        'TOKENIZER_REFERENCE = ' + repr(TOKENIZER["reference"]))
    auth = auth.replace("Gemma and Stack-smol access", "Gemma 4 tokenizer and Stack-smol access")
    notebook["cells"][5]["source"] = auth.splitlines(True)
    notebook["cells"][8] = markdown("## 4. Encode Gemma 4 once; pack both contexts\nReuse V4's frozen 256 MiB raw corpus and holdouts. Gemma 4's tokenizer SHA256 selects its own encoded and packed caches. On first use, encode and verify each document once, then pack 1,024 and 2,048 sequentially.\n")
    display = "".join(notebook["cells"][13]["source"])
    display = once(display,
        '    ("Version / model parameters", f\'{result["version"]} / {result["model_parameters"]:,}\'),',
        '    ("Version / model parameters", f\'{result["version"]} / {result["model_parameters"]:,}\'),\n'
        '    ("Tokenizer / native vocabulary IDs", f\'{result["tokenizer_identity"]["reference"]} / {result["tokenizer_identity"]["vocabulary_ids"]:,}\'),\n'
        '    ("Tied embedding/head rows × width", f\'{result["tokenizer_identity"]["model_rows"]:,} × 640\'),\n'
        '    ("Parameter change from V4", result["vocabulary_adaptation"]["parameter_delta_from_v4"]),')
    notebook["cells"][13]["source"] = display.splitlines(True)
    data = "".join(notebook["cells"][16]["source"])
    data += "\nV4.1.1 changes the tokenizer to the pinned Gemma 4 artifact. The frozen raw corpus/source identities remain V4-compatible; tokenizer SHA256 prevents reuse of Gemma 3 encodings or packings. Native automatic BOS insertion is disabled; the existing 3 virtual packing tokens are retained. Native control-token IDs remain vocabulary entries; no chat template or multimodal input is added.\n"
    notebook["cells"][16] = markdown(data)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    return notebook, source


def main():
    runner = HERE / "runner.py"
    provenance = HERE / "build_provenance.json"
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
            "runner_sha256": hashlib.sha256(source.encode()).hexdigest(), "tokenizer": TOKENIZER,
            "validation_scope": "Static only; no device tests or training."}, stream, indent=2)
        stream.write("\n")
    print(TARGET)


if __name__ == "__main__":
    main()
