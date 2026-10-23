"""Create V4.6.1 from preserved V4.4.1 with official DeepSeek-V3-Base tokenization."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "gemma_270m_colab_v5e1_V4_4_1.ipynb"
TARGET = HERE.parent / "gemma_270m_colab_v5e1_V4_6_1.ipynb"


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
    tree = ast.parse((HERE / "deepseek_tokenizer.py").read_text())
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and n.targets[0].id == "DEEPSEEK_TOKENIZER"))


def build_source(source):
    source = source.replace("V4.4.1", "V4.6.1")
    additions = (HERE / "deepseek_tokenizer.py").read_text()
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
        if name in ("LLAMA3_TOKENIZER", "LLAMA3_PATTERN", "llama3_text_chunks", "Llama3Tokenizer"):
            lines[n.lineno - 1:n.end_lineno] = []
        elif name in replacements:
            lines[n.lineno - 1:n.end_lineno] = [replacements.pop(name) + "\n"]
    source = once("".join(lines), "\ndef main():", "\n\n" + "\n\n".join(replacements.values()) + "\n\ndef main():")
    source = once(source, 'ENCODING_SCHEMA = "v441-native-llama3-byte-document-encoding-1"',
        'ENCODING_SCHEMA = "v461-deepseek-v3-base-byte-document-encoding-1"')
    source = once(source, '"schema": "sequence-aware-token-cache-v5-native-llama3-bytes"',
        '"schema": "sequence-aware-token-cache-v7-deepseek-v3-base-bytes"')
    source = source.replace("Original Llama 3 tokenizer, expanded corpus", "DeepSeek-V3-Base tokenizer, expanded corpus")
    source = source.replace("Llama 3 native encoding once:", "DeepSeek-V3-Base encoding once:")
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
    config.update(version="V4.6.1", experiment="GEM-COLAB-V4.6.1", requirement_prefix="V4.6.1",
        implementation_revision="V4.6.1-deepseek-v3-base-delayed-discovery-memory-trigger-1",
        tokenizer_reference=pin["reference"], tokenizer_revision=pin["revision"])
    notebook["metadata"]["colab"]["name"] = TARGET.name
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "PROGRAM_SOURCE =" in text:
            tail = text[text.index('CONFIG["tokenizer_reference"]'):]
            tail = once(tail, "CONFIG[\"tokenizer_revision\"] = '315b20096dc791d381d514deb5f8bd9c8d6d3061'",
                'CONFIG["tokenizer_revision"] = ' + repr(pin["revision"]))
            tail = once(tail, '"V3-tokenizers" / "llama3"', '"V3-tokenizers" / "deepseek_v3_base"')
            tail = tail.replace("V4_4_1", "V4_6_1").replace("V4.4.1", "V4.6.1")
            text = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + "\n" + tail
        else:
            # Keep the identical pinned native-tokenizer environment and cache marker.
            text = text.replace("V4.4.1", "V4.6.1").replace("gemma_colab_pilot_V4_4_1", "gemma_colab_pilot_V4_6_1")
        cell["source"] = text.splitlines(True)
        if cell["cell_type"] == "code":
            ast.parse(text)
            cell.update(outputs=[], execution_count=None)
    for index, name in ((0, "INTRO.md"), (1, "REQUIREMENTS.md")):
        notebook["cells"][index] = markdown((HERE / name).read_text())
    notebook["cells"][4] = markdown("## 2. DeepSeek-V3-Base tokenizer and corpus access\nDownload the pinned official public tokenizer.json and verify its SHA256. Preserve its BPE pipeline and added-token matching. No automatic native BOS/EOS or chat template. HF_TOKEN remains for corpus sources including Stack-smol. No model weights or remote Python code are loaded.\n")
    auth = "".join(notebook["cells"][5]["source"])
    auth = once(auth, 'TOKENIZER_REFERENCE = "NousResearch/Meta-Llama-3-8B"', 'TOKENIZER_REFERENCE = "deepseek-ai/DeepSeek-V3-Base"')
    auth = auth.replace("the pinned tokenizer redistribution is public", "the official DeepSeek-V3-Base tokenizer is public")
    notebook["cells"][5]["source"] = auth.splitlines(True)
    notebook["cells"][8] = markdown("## 4. Encode DeepSeek-V3-Base once; pack both contexts\nReuse the frozen 256 MiB corpus and holdouts. Verify exact original UTF-8 reconstruction, then pack 1,024 and 2,048 sequentially. DeepSeek encoding and packing caches are separate. Official normalization is an identity operation; BPB uses original source bytes. Preserve native added-token matching, with no automatic native BOS/EOS or chat template.\n")
    data = "".join(notebook["cells"][16]["source"])
    data = data[:data.index("\nV4.6.1 retains the frozen raw corpus")]
    data += "\nV4.6.1 retains the frozen raw corpus/source identities. The pinned official DeepSeek-V3-Base tokenizer uses its original BPE, pretokenizer and added-token matching, with add_special_tokens=False. Its empty normalizer sequence preserves original text. Native added-token spellings retain literal UTF-8 byte lengths; ordinary byte-level tokens may contain partial UTF-8 pieces. Verify every original document byte before packing. Encode once with separate DeepSeek cache schemas. Start at 3 ×1,024 /1 ×2,048, discover both batches after 1,000 updates (earlier on a recoverable memory failure), lock once, and retain the cumulative 3M/1M schedule outside discovery.\n"
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
