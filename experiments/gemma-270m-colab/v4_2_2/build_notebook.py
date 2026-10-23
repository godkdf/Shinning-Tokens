"""Preserve V4.2.1 and create V4.2.2 with discovery delayed until update 1000."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "gemma_270m_colab_v5e1_V4_2_1.ipynb"
CONTROLLER_BASE = HERE.parent / "v4_1_1/runner.py"
TARGET = HERE.parent / "gemma_270m_colab_v5e1_V4_2_2.ipynb"


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one anchor; found {source.count(old)}: {old[:100]}")
    return source.replace(old, new, 1)


def literals(notebook):
    cell = next(c for c in notebook["cells"] if "PROGRAM_SOURCE =" in "".join(c["source"]))
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse("".join(cell["source"])).body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("PROGRAM_SOURCE", "CONFIG")}


def helpers():
    original = CONTROLLER_BASE.read_text()
    definitions = {n.name: ast.get_source_segment(original, n) for n in ast.parse(original).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    controller = definitions["BatchDiscovery"]
    controller = once(controller, '{"current_batch": 1, "selected_batch": 1, "selected_tps": None,',
        '{"current_batch": config.get("discovery_initial_batches", {"short": 3, "long": 1})[mode],\n'
        '            "selected_batch": config.get("discovery_initial_batches", {"short": 3, "long": 1})[mode], "selected_tps": None,')
    controller = once(controller, '            if batch >= self.config["safety_batch_cap"]:',
        '            if batch >= self.config.get("discovery_memory_caps", {}).get(mode, self.config["safety_batch_cap"]):\n'
        '                self.lock(mode, "Memory-failure upper bound or safety cap reached; retain measured batch.")\n'
        '            elif batch >= self.config["safety_batch_cap"]:')
    controller = once(controller,
        '                analysis = compile_shape(mode, candidate, state, "inside_training_budget")\n                peak = compiler_peak(analysis)',
        '''                try:
                    analysis = compile_shape(mode, candidate, state, "inside_training_budget")
                except Exception as error:
                    if not is_memory_error(error):
                        raise
                    analysis = None
                    check["candidate_memory_error"] = str(error)
                peak = compiler_peak(analysis)''')
    return "\n\n".join((definitions["projected_candidate_peak"], controller,
        (HERE / "delayed_training.py").read_text()))


def build_source(source):
    source = source.replace("V4.2.1", "V4.2.2")
    additions = helpers()
    replacements = {n.name: ast.get_source_segment(additions, n) for n in ast.parse(additions).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    lines = source.splitlines(True)
    for n in reversed(ast.parse(source).body):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            if n.name == "fixed_train":
                lines[n.lineno - 1:n.end_lineno] = []
            elif n.name in replacements:
                lines[n.lineno - 1:n.end_lineno] = [replacements.pop(n.name) + "\n"]
    source = once("".join(lines), "\ndef main():", "\n\n" + "\n\n".join(replacements.values()) + "\n\ndef main():")
    source = once(source,
        '        if (mode, batch) not in (("short", 3), ("long", 1)) or scope != "before_training":\n'
        '            raise ValueError("Only fixed 3x1024 and 1x2048 may compile, before training starts.")',
        '''        if scope == "inside_training_budget":
            if int(current_state.step) < config["discovery_start_after_updates"] and not config.get("early_memory_discovery"):
                raise ValueError("Discovery compilation attempted before 1000 updates without a memory fallback.")
        elif scope != "before_training" or ((mode, batch) not in (("short", 3), ("long", 1))
                and not (config.get("early_memory_discovery") and batch == 1)):
            raise ValueError("Unexpected initial compilation shape or scope.")''')
    source = once(source,
        '        analysis = compile_shape(mode, batch, state, "before_training")\n'
        '        limit = (jax.devices()[0].memory_stats() or {}).get("bytes_limit")\n'
        '        peak = compiler_peak(analysis)\n'
        '        if peak is not None and limit and peak + config["other_resident_bytes_" + mode] > limit * config["memory_budget_fraction"]:\n'
        '            raise RuntimeError("Required initial context exceeds the conservative memory budget: " + mode)',
        '''        def admit_initial(selected_batch):
            analysis = compile_shape(mode, selected_batch, state, "before_training")
            limit = (jax.devices()[0].memory_stats() or {}).get("bytes_limit")
            peak = compiler_peak(analysis)
            if peak is not None and limit and peak + config["other_resident_bytes_" + mode] > limit * config["memory_budget_fraction"]:
                raise BatchMemoryError("Required initial context exceeds the conservative memory budget: " + mode)
            return analysis
        try:
            admit_initial(batch)
        except Exception as error:
            memory_discovery_fallback(config, mode, batch, error, "initial_compilation", int(state.step))
            admit_initial(1)''')
    source = once(source,
        '    state, warmup_metrics = jax.block_until_ready(functions[("short", 3)](state, training["short"], jnp.array(0, jnp.int32)))',
        '    initial_short_batch = config.get("discovery_initial_batches", {"short": 3})["short"]\n'
        '    state, warmup_metrics = jax.block_until_ready(functions[("short", initial_short_batch)](state, training["short"], jnp.array(0, jnp.int32)))')
    source = once(source, '"adaptive_compilation": "disabled; no discovery"',
        '"adaptive_compilation": "after 1000 updates, or earlier after a recoverable memory failure; inside training budget"')
    source = once(source, '"initial_training_executables": 2,', '"initial_training_executables": len(functions),')
    source = once(source, '"fixed_training_shapes": [[3, 1024], [1, 2048]]',
        '"initial_requested_shapes": [[3, 1024], [1, 2048]]')
    source = once(source, "batch {config['local_batch_per_chip']} x {config['sequence_length']}...",
        "batch {initial_short_batch} x {config['sequence_length']}...")
    source = once(source,
        '    state, adaptive = fixed_train(state, training, prefixes, functions, config, output)',
        '    state, adaptive = delayed_train(state, training, prefixes, functions, analyses, compile_shape, config, output)')
    start = source.index('        "030_FIXED_BATCHES":')
    end = source.index('        "032_CONTEXT_CURSORS":', start)
    source = source[:start] + '''        "030_INITIAL_BATCHES": all(row["batch"] == (3 if row["mode"] == "short" else 1)
            and row["update"] <= 1000 for row in log_rows if row["phase"] == "initial_batches"),
        "031_TOKEN_SCHEDULE": all(row["mode"] == context_phase(row["total_tokens"] - row["tokens"], row["short_batch"])
            for row in log_rows if row["phase"] != "discovery"),
        "033_DELAYED_BOUNDED_DISCOVERY": (not adaptive["batch_policy"]["discovery_started"] or
            ((adaptive["batch_policy"]["started_at_update"] == 1000 or adaptive["batch_policy"]["early_memory_trigger"])
             and adaptive["batch_policy"]["discovery"]["discovery_updates"] <= 256)),
        "034_LOCKED_BATCHES": all(row["batch"] == adaptive[row["mode"] + "_batch"]
            for row in log_rows if row["phase"] == "locked_training"),
        "035_COMPILE_TRIGGER": all(event["scope"] == "before_training" or event["official_step_at_compile"] >= 1000
            or adaptive["batch_policy"]["early_memory_trigger"] for event in compilation_events),
''' + source[end:]
    source = source.replace('"fixed 3x1024, fixed 1x2048 and fixed-context evaluator, all before training; no training-time compilation"',
        '"initial 3x1024/1x2048 and evaluator before training; discovery compilation after 1000 updates or recoverable memory trigger counts inside training"')
    source = source.replace('"Native o200k, expanded corpus and 16-language holdout, fixed 3x1024/1x2048 and 1200 seconds differ from V3.5. No controlled speed/quality claim."',
        '"Native o200k, expanded corpus and 16-language holdout, delayed batch discovery and 1200 seconds differ from V3.5. No controlled speed/quality claim."')
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
    config.update(version="V4.2.2", experiment="GEM-COLAB-V4.2.2", requirement_prefix="V4.2.2",
        implementation_revision="V4.2.2-o200k-delayed-discovery-memory-trigger-1",
        batch_discovery_enabled=True, discovery_start_after_updates=1000,
        discovery_max_updates=256, discovery_window_updates=16, safety_batch_cap=32)
    notebook["metadata"]["colab"]["name"] = TARGET.name
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "PROGRAM_SOURCE =" in text:
            tail = text[text.index('CONFIG["tokenizer_reference"]'):].replace("V4_2_1", "V4_2_2").replace("V4.2.1", "V4.2.2")
            text = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + "\n" + tail
        else:
            # Keep the matching V4.2.1 native-tokenizer environment and its cache marker.
            text = text.replace("V4.2.1", "V4.2.2").replace("gemma_colab_pilot_V4_2_1", "gemma_colab_pilot_V4_2_2")
        cell["source"] = text.splitlines(True)
        if cell["cell_type"] == "code":
            ast.parse(text)
            cell.update(outputs=[], execution_count=None)
    for index, name in ((0, "INTRO.md"), (1, "REQUIREMENTS.md")):
        notebook["cells"][index] = markdown((HERE / name).read_text())
    notebook["cells"][10] = markdown("## 5. One 1,200-second run; discovery after 1,000 updates\nStart at 3 ×1,024 /1 ×2,048 with the 3M/1M token cycle. End a dispatch exactly at update 1,000. Then discover both batches in at most 256 real updates; lock both and resume the same cycle. Recoverable memory errors can trigger discovery earlier. Candidate compilation counts inside the training timer.\n")
    display = "".join(notebook["cells"][13]["source"])
    display = display.replace("Fixed batches: short / long", "Final batches: short / long")
    display = once(display, '    ("Batch discovery", "Disabled; fixed batches from update 1"),',
        '    ("Discovery trigger / actual start update", f\'1,000, or memory failure / {result["batch_policy"]["started_at_update"]}\'),\n'
        '    ("Discovery updates / maximum", f\'{(result["batch_policy"]["discovery"] or {}).get("discovery_updates", 0)} / 256\'),')
    display = display.replace('print("Fixed batch policy:"', 'print("Delayed batch discovery policy:"')
    notebook["cells"][13]["source"] = display.splitlines(True)
    data = "".join(notebook["cells"][16]["source"])
    data = once(data, 'Both training batches are fixed from update 1: 3 ×1,024 and 1 ×2,048, with the repeating 3M/1M token schedule.',
        'Start at 3 ×1,024 and 1 ×2,048 for exactly 1,000 updates, then discover both batches once (earlier on a recoverable memory failure). Discovery temporarily measures both contexts; afterwards resume the cumulative 3M/1M schedule with the selected batches.')
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
            "controller_source_sha256": hashlib.sha256(CONTROLLER_BASE.read_bytes()).hexdigest(),
            "notebook_sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "discovery_start_after_updates": 1000, "early_trigger": "recoverable memory failure",
            "device_tests_executed": False}, stream, indent=2)
        stream.write("\n")
    print(TARGET)


if __name__ == "__main__":
    main()
