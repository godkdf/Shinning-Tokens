"""Build a new V3.9 artifact exclusively; earlier notebooks are immutable."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "gemma_270m_colab_v5e1_V3_8_5.ipynb"
TARGET = HERE.parent / "gemma_270m_colab_v5e1_V3_9.ipynb"


def literals(notebook):
    cell = next(c for c in notebook["cells"] if "PROGRAM_SOURCE =" in "".join(c["source"]))
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse("".join(cell["source"])).body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("PROGRAM_SOURCE", "CONFIG")}


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one anchor, found {source.count(old)}: {old[:80]}")
    return source.replace(old, new, 1)


def replace_definitions(source, additions):
    replacement = {}
    for node in ast.parse(additions).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            replacement[node.name] = ast.get_source_segment(additions, node)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            replacement[node.targets[0].id] = ast.get_source_segment(additions, node)
    lines = source.splitlines(True)
    edits = []
    for node in ast.parse(source).body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name in replacement:
            edits.append((node.lineno - 1, node.end_lineno, replacement.pop(name) + "\n"))
    for start, end, value in reversed(edits):
        lines[start:end] = [value]
    source = "".join(lines)
    constants = [value for value in replacement.values() if isinstance(ast.parse(value).body[0], ast.Assign)]
    helpers = [value for value in replacement.values() if not isinstance(ast.parse(value).body[0], ast.Assign)]
    source = source.replace("def read_json(path):", "\n\n".join(constants) + "\n\ndef read_json(path):", 1)
    # New helpers must be defined before main() can call the program.
    index = source.index("\ndef main():")
    return source[:index] + "\n\n" + "\n\n".join(helpers) + "\n" + source[index:]


def build_source(original):
    source = original.replace("V3.8.5", "V3.9").replace("V385", "V39")
    for name in ("data.py", "adaptive.py", "loop.py", "train.py"):
        source = replace_definitions(source, (HERE / name).read_text())
    old = ast.literal_eval(next(n.value for n in ast.parse(source).body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "MODEL_HELPERS"))
    new = old + "    short_rows: jax.Array\n    long_rows: jax.Array\n"
    source = once(source, repr(old), repr(new))
    source = once(source, '    if row.get("repo_name"):\n        return "code-repo:" + str(row["repo_name"])',
        '    if row.get("repo_name") or row.get("repository_name"):\n'
        '        return "code-repo:" + str(row.get("repo_name") or row["repository_name"])')
    source = once(source, 'row.get("content") if domain == "code" else row.get("text")',
        'row.get("content") if domain.startswith("code_") else row.get("text")')
    source = once(source, '("id", "url", "repo_name", "path", "license", "dataset")',
        '("id", "url", "repo_name", "repository_name", "path", "license", "licenses", "lang", "dataset")')
    source = once(source, '        stream = load_dataset(job["repo"], **kwargs)',
        '''        if job["repo"] == STACK:
            from huggingface_hub import HfApi
            paths = sorted(path for path in HfApi().list_repo_files(STACK, repo_type="dataset", revision=job["revision"])
                if path.startswith("data/" + job["subset"] + "/") and path.endswith(".json"))
            if not paths:
                raise ValueError("Pinned code language files missing: " + job["subset"])
            kwargs.pop("name")
            kwargs.pop("revision")
            kwargs["data_files"] = {"train": ["hf://datasets/" + STACK + "@" + job["revision"] + "/" + path for path in paths]}
            stream = load_dataset("json", **kwargs)
        else:
            stream = load_dataset(job["repo"], **kwargs)''')
    source = once(source, '        "language_macro_bpb": float(np.mean([value["bpb"] for value in languages.values()])),',
        '        "language_macro_bpb": float(np.mean([value["bpb"] for value in languages.values()])),\n'
        '        "legacy_12_language_macro_bpb": float(np.mean([languages[lang]["bpb"] for lang in LANGUAGES[:12] if lang in languages])),')
    source = once(source, '        "same_context_length": config["sequence_length"] == 1024,',
        '        "same_evaluation_context_length": config["sequence_length"] == 1024,\n'
        '        "current_training_contexts": [1024, 2048], "current_corpus": "expanded 256 MiB",')
    source = source.replace('Context length, packing, batch and training duration differ.',
        'Corpus, language coverage, batch schedule, training context and duration differ.')
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
    config.update(experiment="GEM-COLAB-V3.9", version="V3.9", requirement_prefix="V39",
        implementation_revision="V3.9-expanded-adaptive-mixed-context-1", sequence_length=1024,
        long_sequence_length=2048, local_batch_per_chip=1, batch_check_updates=100,
        training_seconds=1200., memory_budget_fraction=.85, safety_batch_cap=32,
        short_phase_tokens=3000000, long_phase_tokens=1000000)
    notebook["metadata"]["colab"]["name"] = TARGET.name
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "PROGRAM_SOURCE =" in text:
            tail = text[text.index('CONFIG["tokenizer_reference"]'):]
            text = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + "\n" + tail
        text = text.replace("V3.8.5", "V3.9").replace("V3_8_5", "V3_9").replace("V385", "V39")
        cell["source"] = text.splitlines(True)
        if cell["cell_type"] == "code":
            cell.update(execution_count=None, outputs=[])
            ast.parse(text)
    for index, name in ((0, "INTRO.md"), (1, "REQUIREMENTS.md"), (16, "DATA.md")):
        notebook["cells"][index] = markdown((HERE / name).read_text())
    notebook["cells"][4] = markdown("## 2. Authorize tokenizer and expanded code corpus\nUse HF_TOKEN with access to the Gemma tokenizer and [Stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol). Accept the dataset's access conditions on Hugging Face before running. The notebook checks access before large data preparation.\n")
    notebook["cells"][8] = markdown("## 4. Prepare the frozen 256 MiB expanded corpus\nPrepare both 1,024 and 2,048 training packings. Fixed evaluation uses 1,024. Checksums and cache keys include corpus, tokenizer and context. Earlier 64 MiB caches remain untouched.\n")
    notebook["cells"][10] = markdown("## 5. Train once for 1,200 seconds\nStart 1 × 1,024 for 100 updates. Check every 100 official updates; increase by one when memory and measured throughput justify it. Candidate compilation counts inside the timer. Once short batch reaches 3, repeat 3M short / 1M long tokens, with long fixed at 1 × 2,048. Save official scores and weights before 32 disposable profiling updates.\n")
    code = "".join(notebook["cells"][5]["source"])
    notebook["cells"][5]["source"] = code.replace("with Gemma repository access", "with Gemma and Stack-smol access").splitlines(True)
    code = "".join(notebook["cells"][13]["source"])
    code = once(code, '("Batch × context", f\'{result["config"]["local_batch_per_chip"]} × {result["config"]["sequence_length"]:,}\'),',
        '("Initial → final short batch; long batch", f\'1 → {result["final_short_batch"]} × 1,024; long 1 × 2,048\'),\n'
        '    ("Short / long scored tokens", str(result["tokens_by_context"])),\n'
        '    ("Adaptive compilation seconds inside training", fmt(result["adaptive_compilation_seconds_inside_training"])),')
    code = code.replace('("Compiler buffer estimate % capacity", fmt(memory["compiled_peak_buffer_estimate_percent"]) + "% (estimate)"),',
        '("Compiler + inactive context/evaluation estimate % capacity", fmt(memory["compiled_peak_plus_other_resident_percent"]) + "% (estimate)"),')
    code += '\nprint("Batch decisions:", json.dumps(result["batch_checks"], indent=2))\nprint("Context transitions:", json.dumps(result["context_transitions"], indent=2))\nprint(result["profiling"]["operation_timing_note"])\n'
    notebook["cells"][13]["source"] = code.splitlines(True)
    return notebook, source


def main():
    if TARGET.exists() or (HERE / "runner.py").exists():
        raise FileExistsError("Preserving delivered artifacts; create a new version.")
    notebook, source = build_notebook()
    with (HERE / "runner.py").open("x") as stream:
        stream.write(source)
    with TARGET.open("x") as stream:
        json.dump(notebook, stream, indent=1, ensure_ascii=False)
        stream.write("\n")
    print(TARGET)


if __name__ == "__main__":
    main()
