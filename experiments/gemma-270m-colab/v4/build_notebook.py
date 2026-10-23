"""Create V4 from V3.9.1 without changing previously delivered notebooks."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "gemma_270m_colab_v5e1_V3_9_1.ipynb"
TARGET = HERE.parent / "gemma_270m_colab_v5e1_V4.ipynb"
BASE_COMMIT = "d1143a10cf5ecef2ed4b8eeb8cdfe5f2edc74a81"


def load_base():
    """Allow pre-V4 cleanup without losing the exact historical build input."""
    if BASE.exists():
        return json.loads(BASE.read_text())
    repository = HERE.parents[2]
    relative = BASE.relative_to(repository).as_posix()
    try:
        payload = subprocess.check_output(
            ["git", "show", BASE_COMMIT + ":" + relative], cwd=repository)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "The historical base is absent. Fetch full Git history to rebuild V4; "
            "the delivered notebook and runner need no historical files to run."
        ) from error
    return json.loads(payload)


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one anchor; found {source.count(old)}: {old[:100]}")
    return source.replace(old, new, 1)


def embedded_literals(notebook):
    cell = next(c for c in notebook["cells"] if "PROGRAM_SOURCE =" in "".join(c["source"]))
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse("".join(cell["source"])).body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("PROGRAM_SOURCE", "CONFIG")}


def replace_definitions(source, additions):
    replacements = {}
    for node in ast.parse(additions).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            replacements[node.name] = ast.get_source_segment(additions, node)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            replacements[node.targets[0].id] = ast.get_source_segment(additions, node)
    edits = []
    lines = source.splitlines(True)
    for node in ast.parse(source).body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name in replacements:
            edits.append((node.lineno - 1, node.end_lineno, replacements.pop(name) + "\n"))
    for start, end, replacement in reversed(edits):
        lines[start:end] = [replacement]
    source = "".join(lines)
    constants = [v for v in replacements.values() if isinstance(ast.parse(v).body[0], ast.Assign)]
    helpers = [v for v in replacements.values() if not isinstance(ast.parse(v).body[0], ast.Assign)]
    source = once(source, "def read_json(path):", "\n\n".join(constants) + "\n\ndef read_json(path):")
    return once(source, "\ndef main():", "\n\n" + "\n\n".join(helpers) + "\n\ndef main():")


def update_preparation(source):
    source = once(source, '    contexts = {}\n    for length in (1024, 2048):',
        '''    contexts = {}
    missing_packing = any(not (shared / "tokenized" / tokenization_cache_key(
        identity["tokenizer_sha256"], raw_digest, length)).exists() for length in (1024, 2048))
    encoding_directory = None
    encoding_report = {"status": "not_needed", "reason": "Both matching packed context caches already exist.",
        "schema": ENCODING_SCHEMA, "encode_passes_this_run": 0}
    if missing_packing:
        encoding_directory, encoding_details, encoding_reused = prepared_encoding(
            shared, raw, work, tokenizer, mapping, identity, raw_digest)
        for split, counts in encoding_details["splits"].items():
            if counts["original_bytes"] != manifest["original_bytes"][split]:
                raise ValueError("Encoded original byte total differs: " + split)
        encoding_report = {"status": "available", "schema": ENCODING_SCHEMA,
            "cache_reused": encoding_reused, "encode_passes_this_run": 0 if encoding_reused else 1,
            "encoding_manifest": encoding_details}
        write_json(output / "encoding_manifest.json", encoding_details)
    write_json(output / "encoding_preparation.json", encoding_report)
    for length in (1024, 2048):''')
    source = once(source,
        '                arrays, packing = tokenize_split(read_jsonl(raw / (split + ".jsonl")), tokenizer, mapping, identity, length)',
        '                arrays, packing = pack_encoded_split(encoding_directory, split, identity, length)')
    source = once(source,
        '                print(f"{length}: {split}: {packing[\'text_tokens\']:,} text targets, occupancy={packing[\'occupancy\']:.2%}", flush=True)',
        '                print(f"{length}: {split}: {packing[\'text_tokens\']:,} text targets, occupancy={packing[\'occupancy\']:.2%}", flush=True)\n'
        '                del arrays')
    source = once(source,
        '    write_json(output / "preparation.json", {"raw_cache_reused": raw_reused, "contexts": contexts,',
        '    write_json(output / "preparation.json", {"raw_cache_reused": raw_reused, "contexts": contexts,\n'
        '        "encoding": encoding_report,')
    return source


def build_source(original):
    source = original.replace("V3.9.1", "V4").replace('"V39-0', '"V4-0')
    for name in ("encoding.py", "discovery.py"):
        source = replace_definitions(source, (HERE / name).read_text())
    source = update_preparation(source)
    source = once(source,
        '    functions, analyses, all_analyses, compilation_events = {}, {}, {}, []',
        '    functions, analyses, all_analyses, compilation_events = {}, {"short": {}, "long": {}}, {}, []')
    source = once(source,
        '        if mode == "short":\n            analyses[batch] = analysis',
        '        analyses[mode][batch] = analysis')
    source = once(source,
        '        "030_INITIAL_BATCH": all(row["batch"] == 1 and row["sequence_length"] == 1024 for row in log_rows if row["update"] <= 100),\n'
        '        "031_TOKEN_SCHEDULE": all(row["mode"] == context_phase(row["total_tokens"] - row["tokens"], row["short_batch"]) for row in log_rows),',
        '        "030_BOUNDED_DISCOVERY": adaptive["discovery"]["discovery_updates"] <= config["discovery_max_updates"],\n'
        '        "031_TOKEN_SCHEDULE": all(row["mode"] == context_phase(row["total_tokens"] - row["tokens"], row["short_batch"])\n'
        '            for row in log_rows if row["phase"] == "locked_training"),\n'
        '        "033_BOTH_CONTEXTS_MEASURED": all(adaptive["discovery"]["contexts"][mode]["observations"] for mode in ("short", "long")),\n'
        '        "034_BATCHES_LOCKED": adaptive["discovery"]["both_locked"] and all(\n'
        '            row["batch"] == adaptive[mode + "_batch"] for row in log_rows if row["phase"] == "locked_training" for mode in (row["mode"],)),\n'
        '        "035_NO_RECHECKS": all(check["at_update"] <= adaptive["discovery"]["finished_at_update"] for check in adaptive["checks"])\n'
        '            if adaptive["discovery"]["finished_at_update"] is not None else False,')
    source = once(source,
        '        "final_short_batch": adaptive["short_batch"], "maximum_short_batch_executed": max(row["batch"] for row in log_rows if row["mode"] == "short"),',
        '        "final_short_batch": adaptive["short_batch"], "final_long_batch": adaptive["long_batch"],\n'
        '        "batch_discovery": adaptive["discovery"],\n'
        '        "maximum_short_batch_executed": max(row["batch"] for row in log_rows if row["mode"] == "short"),\n'
        '        "maximum_long_batch_executed": max(row["batch"] for row in log_rows if row["mode"] == "long"),')
    source = once(source,
        '        "context_rows_consumed": adaptive["rows_used"], "final_short_batch": adaptive["short_batch"],',
        '        "context_rows_consumed": adaptive["rows_used"], "final_short_batch": adaptive["short_batch"],\n'
        '        "final_long_batch": adaptive["long_batch"], "batch_discovery": adaptive["discovery"],')
    source = once(source, '        batch = adaptive["short_batch"] if mode == "short" else 1',
        '        batch = adaptive[mode + "_batch"]')
    source = once(source,
        '    profile["operation_timing_note"] = "Mixed diagnostic: 16 updates at final short batch x1024, then 16 at 1x2048. Category ms/update average these shapes; not per-context timings."',
        '    profile["operation_timing_note"] = (f"Mixed diagnostic: 16 updates at {adaptive[\'short_batch\']}x1024, then "\n'
        '        f"16 at {adaptive[\'long_batch\']}x2048. Category ms/update average these shapes; not per-context timings.")')
    # Remove obsolete entry points so there is one preparation and one selection path.
    lines = source.splitlines(True)
    for node in reversed(ast.parse(source).body):
        if isinstance(node, ast.FunctionDef) and node.name in ("tokenize_split", "batch_decision"):
            lines[node.lineno - 1:node.end_lineno] = []
    source = "".join(lines)
    ast.parse(source)
    return source


def markdown(text):
    return {"cell_type": "markdown", "id": hashlib.sha256(text.encode()).hexdigest()[:8],
        "metadata": {}, "source": text.splitlines(True)}


def build_notebook():
    notebook = load_base()
    values = embedded_literals(notebook)
    source = build_source(values["PROGRAM_SOURCE"])
    config = values["CONFIG"]
    config.pop("batch_check_updates")
    config.update(version="V4", experiment="GEM-COLAB-V4", requirement_prefix="V4",
        implementation_revision="V4-encode-once-two-context-discovery-1",
        discovery_max_updates=256, discovery_window_updates=16)
    notebook["metadata"]["colab"]["name"] = TARGET.name
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "PROGRAM_SOURCE =" in text:
            tail = text[text.index('CONFIG["tokenizer_reference"]'):]
            tail = tail.replace("V3.9.1", "V4").replace("V3_9_1", "V4")
            text = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + "\n" + tail
        else:
            text = text.replace("V3.9.1", "V4").replace("V3_9_1", "V4")
        cell["source"] = text.splitlines(True)
        if cell["cell_type"] == "code":
            ast.parse(text)
            cell.update(outputs=[], execution_count=None)
    for index, name in ((0,"INTRO.md"),(1,"REQUIREMENTS.md")):
        notebook["cells"][index] = markdown((HERE / name).read_text())
    notebook["cells"][8] = markdown("## 4. Prepare both contexts from one verified encoding\nReuse the frozen 256 MiB corpus. Encode IDs, verify round trips and calculate byte attribution once per document; pack 1,024 and 2,048 sequentially. Reuse existing matching V3.9.1 packed caches without rebuilding.\n")
    notebook["cells"][10] = markdown("## 5. One 1,200-second run: discover both batches, then lock\nUse up to the first 256 real optimizer updates, in 16-update measurement blocks, to select an independent batch for each context. All updates and discovery compilation count in the training budget. Once both are locked, repeat the original 3M-short/1M-long token schedule with those two batch sizes. No later rechecks. Initial/final evaluation only; one mixed post-save diagnostic.\n")
    display = "".join(notebook["cells"][13]["source"])
    display = once(display,
        '("Initial → final short batch; long batch", f\'1 → {result["final_short_batch"]} × 1,024; long 1 × 2,048\'),',
        '("Locked batches: short / long", f\'{result["final_short_batch"]} × 1,024 / {result["final_long_batch"]} × 2,048\'),\n'
        '    ("Discovery updates / maximum", f\'{result["batch_discovery"]["discovery_updates"]} / 256\'),')
    display += '\nprint("Locked batch discovery:", json.dumps(result["batch_discovery"], indent=2))\nprint("Encoding preparation:", (OUTPUT / "encoding_preparation.json").read_text())\n'
    notebook["cells"][13]["source"] = display.splitlines(True)
    data = "".join(notebook["cells"][16]["source"])
    data += "\nV4 retains this exact corpus recipe and source-cache identities. Document encoding is now context-independent and byte-verified once, with an additional checksum-protected encoded-document cache. Context packings remain compatible with V3.9.1. The early discovery phase deliberately trains both contexts before the normal 3M/1M cycle; all discovery tokens count toward the cumulative schedule.\n"
    notebook["cells"][16] = markdown(data)
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
