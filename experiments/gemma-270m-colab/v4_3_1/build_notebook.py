"""Create V4.3.1 from preserved V4.2.2; change tokenizer and vocabulary only."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "gemma_270m_colab_v5e1_V4_2_2.ipynb"
TARGET = HERE.parent / "gemma_270m_colab_v5e1_V4_3_1.ipynb"


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
    tree = ast.parse((HERE / "bloom_tokenizer.py").read_text())
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and n.targets[0].id == "BLOOM_TOKENIZER"))


def build_source(source):
    source = source.replace("V4.2.2", "V4.3.1")
    additions = (HERE / "bloom_tokenizer.py").read_text()
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
        if name in ("O200K_ASSET_SHA256", "O200K_RANKS_SHA256"):
            lines[n.lineno - 1:n.end_lineno] = []
        elif name in replacements:
            lines[n.lineno - 1:n.end_lineno] = [replacements.pop(name) + "\n"]
    source = once("".join(lines), "\ndef main():", "\n\n" + "\n\n".join(replacements.values()) + "\n\ndef main():")
    source = once(source, 'ENCODING_SCHEMA = "v421-native-byte-document-encoding-1"',
        'ENCODING_SCHEMA = "v431-bloom-byte-document-encoding-1"')
    source = once(source, '"schema": "sequence-aware-token-cache-v3-native-tiktoken-bytes"',
        '"schema": "sequence-aware-token-cache-v4-bloom-bytes"')
    source = once(source, '                native_ids = tokenizer.encode_ordinary(doc["text"])',
        '                native_ids = tokenizer.encode(doc["text"], add_special_tokens=False).ids')
    source = source.replace("Native o200k, expanded corpus", "BLOOM tokenizer, expanded corpus")
    source = source.replace("Native encoding once:", "BLOOM encoding once:")
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
    config.update(version="V4.3.1", experiment="GEM-COLAB-V4.3.1", requirement_prefix="V4.3.1",
        implementation_revision="V4.3.1-bloom-delayed-discovery-memory-trigger-1",
        tokenizer_reference=pin["reference"], tokenizer_revision=pin["revision"])
    notebook["metadata"]["colab"]["name"] = TARGET.name
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "PROGRAM_SOURCE =" in text:
            tail = text[text.index('CONFIG["tokenizer_reference"]'):]
            tail = once(tail, 'CONFIG["tokenizer_revision"] = "tiktoken-0.14.0"',
                'CONFIG["tokenizer_revision"] = ' + repr(pin["revision"]))
            tail = once(tail, '"V3-tokenizers" / "tiktoken"', '"V3-tokenizers" / "bloom"')
            tail = tail.replace("V4_2_2", "V4_3_1").replace("V4.2.2", "V4.3.1")
            text = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + "\n" + tail
        else:
            # Reuse the identical pinned package environment; only output paths change.
            text = text.replace("V4.2.2", "V4.3.1").replace("gemma_colab_pilot_V4_2_2", "gemma_colab_pilot_V4_3_1")
        cell["source"] = text.splitlines(True)
        if cell["cell_type"] == "code":
            ast.parse(text)
            cell.update(outputs=[], execution_count=None)
    for index, name in ((0, "INTRO.md"), (1, "REQUIREMENTS.md")):
        notebook["cells"][index] = markdown((HERE / name).read_text())
    notebook["cells"][4] = markdown("## 2. Pinned BLOOM tokenizer and corpus access\nLoad the official bigscience/bloom tokenizer.json at its fixed revision and verify SHA256. The model remains Gemma3_270M. HF_TOKEN is for corpus sources including Stack-smol; BLOOM's tokenizer is public.\n")
    auth = "".join(notebook["cells"][5]["source"])
    auth = once(auth, 'TOKENIZER_REFERENCE = "o200k_base"', 'TOKENIZER_REFERENCE = "bigscience/bloom"')
    auth = auth.replace("o200k itself is public", "BLOOM's tokenizer is public")
    notebook["cells"][5]["source"] = auth.splitlines(True)
    notebook["cells"][8] = markdown("## 4. Encode BLOOM once; pack both contexts\nReuse the frozen 256 MiB corpus and holdouts. Use the official BLOOM ByteLevel BPE, verify exact token-byte reconstruction, then pack 1,024 and 2,048 sequentially. Separate BLOOM cache identities preserve previous tokenizer caches.\n")
    data = "".join(notebook["cells"][16]["source"])
    data = data[:data.index("\nV4.3.1 retains the frozen raw corpus")]
    data += "\nV4.3.1 retains the frozen raw corpus and source identities. BLOOM uses its official ByteLevel BPE pipeline with add_special_tokens=False and native added-token matching. Exact inverse-ByteLevel token bytes preserve original byte counts, including split UTF-8 characters and literal native special tokens. Encoding and packing schemas/hash identities are separate from o200k and Gemma. Start at 3 ×1,024 /1 ×2,048, discover both batches after 1,000 updates (earlier on a recoverable memory failure), lock once, and retain the cumulative 3M/1M schedule outside discovery.\n"
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
            "discovery_start_after_updates": 1000, "training_seconds": 1200,
            "device_tests_executed": False}, stream, indent=2)
        stream.write("\n")
    print(TARGET)


if __name__ == "__main__":
    main()
