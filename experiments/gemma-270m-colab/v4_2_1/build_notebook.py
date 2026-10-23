"""Create V4.2.1 from preserved V4.1.1: native o200k, two fixed training shapes."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "gemma_270m_colab_v5e1_V4_1_1.ipynb"
TARGET = HERE.parent / "gemma_270m_colab_v5e1_V4_2_1.ipynb"


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one anchor, found {source.count(old)}: {old[:100]}")
    return source.replace(old, new, 1)


def literals(notebook):
    cell = next(c for c in notebook["cells"] if "PROGRAM_SOURCE =" in "".join(c["source"]))
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse("".join(cell["source"])).body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("PROGRAM_SOURCE", "CONFIG")}


def replace_helpers(source):
    additions = "\n\n".join((HERE / name).read_text() for name in ("native_tokenizer.py", "fixed_training.py"))
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
        if name in ("GEMMA4_TOKENIZER", "BatchDiscovery", "adaptive_train", "projected_candidate_peak"):
            lines[n.lineno - 1:n.end_lineno] = []
        elif name in replacements:
            lines[n.lineno - 1:n.end_lineno] = [replacements.pop(name) + "\n"]
    source = "".join(lines)
    return once(source, "\ndef main():", "\n\n" + "\n\n".join(replacements.values()) + "\n\ndef main():")


def build_source(source):
    source = source.replace("V4.1.1", "V4.2.1")
    source = replace_helpers(source)
    source = once(source, 'ENCODING_SCHEMA = "v4-dense-document-encoding-1"',
        'ENCODING_SCHEMA = "v421-native-byte-document-encoding-1"')
    source = once(source, '"schema": "sequence-aware-token-cache-v2"',
        '"schema": "sequence-aware-token-cache-v3-native-tiktoken-bytes"')
    source = once(source, '    # Both lengths have been measured before this is used for locked training.',
        '    # Both fixed shapes are compiled before training; schedule starts at token zero.')
    source = once(source, '    def compile_shape(mode, batch, current_state, scope):\n        key = (mode, batch)',
        '''    def compile_shape(mode, batch, current_state, scope):
        if (mode, batch) not in (("short", 3), ("long", 1)) or scope != "before_training":
            raise ValueError("Only fixed 3x1024 and 1x2048 may compile, before training starts.")
        key = (mode, batch)''')
    source = once(source, '    print("Compiling initial short, fixed long and 1024-context evaluator; zero optimizer updates...", flush=True)',
        '    print("Compiling fixed 3x1024, fixed 1x2048 and 1024-context evaluator; zero optimizer updates...", flush=True)')
    source = once(source,
        '    for mode in ("short", "long"):\n        analysis = compile_shape(mode, 1, state, "before_training")',
        '    for mode, batch in (("short", 3), ("long", 1)):\n        analysis = compile_shape(mode, batch, state, "before_training")')
    source = once(source,
        '    state, warmup_metrics = jax.block_until_ready(functions[("short", 1)](state, training["short"], jnp.array(0, jnp.int32)))',
        '    state, warmup_metrics = jax.block_until_ready(functions[("short", 3)](state, training["short"], jnp.array(0, jnp.int32)))')
    source = once(source, '"adaptive_compilation": "inside the 1200-second training wall-clock budget"',
        '"fixed_training_shapes": [[3, 1024], [1, 2048]], "adaptive_compilation": "disabled; no discovery"')
    source = once(source,
        '    state, adaptive = adaptive_train(state, training, prefixes, compile_shape, functions, analyses, config, output)',
        '    state, adaptive = fixed_train(state, training, prefixes, functions, config, output)')
    source = source.replace('"batch_discovery": adaptive["discovery"]', '"batch_policy": adaptive["batch_policy"]')
    begin = source.index('        "030_BOUNDED_DISCOVERY":')
    end = source.index('        "032_CONTEXT_CURSORS":', begin)
    source = source[:begin] + '''        "030_FIXED_BATCHES": all(row["batch"] == (3 if row["mode"] == "short" else 1) for row in log_rows),
        "031_TOKEN_SCHEDULE": all(row["mode"] == context_phase(row["total_tokens"] - row["tokens"], 3) for row in log_rows),
        "033_NO_DISCOVERY": adaptive["batch_policy"]["discovery_enabled"] is False and not adaptive["checks"],
        "034_ONLY_REQUESTED_SHAPES": set(functions) == {("short", 3), ("long", 1)},
        "035_NO_TRAINING_COMPILATION": all(event["scope"] == "before_training" for event in compilation_events),
''' + source[end:]
    source = source.replace('"initial short, fixed long and fixed-context evaluator; later compilations counted inside training wall time"',
        '"fixed 3x1024, fixed 1x2048 and fixed-context evaluator, all before training; no training-time compilation"')
    source = source.replace('"Expanded corpus and 16-language holdout, adaptive batches, mixed training context and 1200 seconds differ from V3.5. No controlled speed/quality claim."',
        '"Native o200k, expanded corpus and 16-language holdout, fixed 3x1024/1x2048 and 1200 seconds differ from V3.5. No controlled speed/quality claim."')
    for mode in ("short", "long"):
        source = once(source, f'max(row["batch"] for row in log_rows if row["mode"] == "{mode}")',
            f'max((row["batch"] for row in log_rows if row["mode"] == "{mode}"), default=0)')
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
    for key in ("discovery_max_updates", "discovery_window_updates", "safety_batch_cap"):
        config.pop(key)
    config.update(version="V4.2.1", experiment="GEM-COLAB-V4.2.1", requirement_prefix="V4.2.1",
        implementation_revision="V4.2.1-native-o200k-fixed-3x1024-1x2048-1",
        tokenizer_reference="o200k_base", tokenizer_revision="tiktoken-0.14.0",
        local_batch_per_chip=3, long_batch_per_chip=1, batch_discovery_enabled=False)
    notebook["metadata"]["colab"]["name"] = TARGET.name
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "PROGRAM_SOURCE =" in text:
            tail = text[text.index('CONFIG["tokenizer_reference"]'):]
            tail = once(tail, "CONFIG[\"tokenizer_revision\"] = '411aa17b749aa952df1359d2dcea73917a544d9a'",
                'CONFIG["tokenizer_revision"] = "tiktoken-0.14.0"\n'
                'CONFIG["tokenizer_cache_directory"] = str(OUTPUT.parent.parent / "V3-tokenizers" / "tiktoken")')
            tail = tail.replace("V4_1_1", "V4_2_1").replace("V4.1.1", "V4.2.1")
            text = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + "\n" + tail
        else:
            text = text.replace("V4_1_1", "V4_2_1").replace("V4.1.1", "V4.2.1")
        cell["source"] = text.splitlines(True)
        if cell["cell_type"] == "code":
            ast.parse(text)
            cell.update(outputs=[], execution_count=None)
    setup = "".join(notebook["cells"][3]["source"])
    for node in reversed(ast.parse(setup).body):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("REQUIREMENTS_INPUT", "RESOLVED_DEPENDENCY_LOCK"):
            value = ast.literal_eval(node.value)
            if any(line.startswith(("tiktoken==", "regex==")) for line in value.splitlines()):
                raise ValueError("Unexpected existing native-tokenizer pins; inspect before adding dependencies.")
            lines = setup.splitlines(True)
            lines[node.lineno - 1:node.end_lineno] = [node.targets[0].id + " = " + repr(value + "\ntiktoken==0.14.0\nregex==2026.9.29\n") + "\n"]
            setup = "".join(lines)
    setup = setup.replace('BASE / ".venv-V3"', 'BASE / ".venv-V4_2_1"').replace('".v3-lock-sha256"', '".v4_2_1-lock-sha256"')
    setup = setup.replace("Reusing matching V3.4 environment", "Reusing matching V4.2.1 environment")
    setup = setup.replace("Installing the embedded successful V3.4 lock", "Installing the V3.4 pins plus native tiktoken/regex pins")
    setup = setup.replace('"source": "successful user-supplied V3.4 report lock"',
        '"source": "successful V3.4 report lock plus tiktoken==0.14.0 and regex==2026.9.29"')
    notebook["cells"][3]["source"] = setup.splitlines(True)
    for index, name in ((0, "INTRO.md"), (1, "REQUIREMENTS.md")):
        notebook["cells"][index] = markdown((HERE / name).read_text())
    notebook["cells"][4] = markdown("## 2. Native o200k tokenizer and source-data access\nNative `tiktoken` loads `o200k_base`. HF_TOKEN is for corpus access, including Stack-smol; the tokenizer itself needs no Hugging Face model access.\n")
    notebook["cells"][5]["source"] = '''from google.colab import userdata
from getpass import getpass

TOKENIZER_REFERENCE = "o200k_base"
try:
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = None
if not HF_TOKEN:
    HF_TOKEN = getpass("Hugging Face token for corpus access, including Stack-smol: ")
if not HF_TOKEN.strip():
    raise ValueError("HF_TOKEN is required for the corpus sources; o200k itself is public.")
'''.splitlines(True)
    notebook["cells"][8] = markdown("## 4. Encode native o200k once; pack both contexts\nReuse the frozen 256 MiB corpus and holdouts. Verify each document by exact native token-byte reconstruction, then pack 1,024 and 2,048 sequentially. o200k encoding/packing identities are separate from the Gemma caches.\n")
    notebook["cells"][10] = markdown("## 5. One 1,200-second run with fixed batches\nCompile only 3 ×1,024 and 1 ×2,048 before the timer. Start at 3 ×1,024; repeat 3M short /1M long cumulative scored-token intervals. No discovery, ramp, batch checks or candidate compilation. All updates train the same model and optimizer.\n")
    display = "".join(notebook["cells"][13]["source"])
    display = once(display, '    ("Discovery updates / maximum", f\'{result["batch_discovery"]["discovery_updates"]} / 256\'),',
        '    ("Batch discovery", "Disabled; fixed batches from update 1"),')
    display = display.replace("Locked batches: short / long", "Fixed batches: short / long").replace("Adaptive compilation seconds inside training", "Compilation seconds inside training")
    display = once(display, 'print("Locked batch discovery:", json.dumps(result["batch_discovery"], indent=2))',
        'print("Fixed batch policy:", json.dumps(result["batch_policy"], indent=2))')
    notebook["cells"][13]["source"] = display.splitlines(True)
    data = "".join(notebook["cells"][16]["source"])
    data = data[:data.index("\nV4.2.1 retains this exact corpus recipe")]
    data += "\nV4.2.1 retains the frozen raw corpus and source identities. Native tiktoken encode_ordinary preserves special-token spellings as ordinary text. Exact per-token byte lengths replace character-offset attribution. Its encoding/packing schema and tokenizer hash keep caches separate. Both training batches are fixed from update 1: 3 ×1,024 and 1 ×2,048, with the repeating 3M/1M token schedule.\n"
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
            "runner_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "tokenizer": "native tiktoken 0.14.0 / o200k_base", "fixed_batches": [[3, 1024], [1, 2048]],
            "training_seconds": 1200, "device_tests_executed": False}, stream, indent=2)
        stream.write("\n")
    print(TARGET)


if __name__ == "__main__":
    main()
