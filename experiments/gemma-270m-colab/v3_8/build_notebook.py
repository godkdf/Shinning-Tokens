"""Build V3.8 exclusively from V3.5; preserve all delivered notebooks."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
TARGET = PARENT / "gemma_270m_colab_v5e1_V3_8.ipynb"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one anchor, found {source.count(old)}: {old[:100]}")
    return source.replace(old, new, 1)


def build_source():
    source = (PARENT / "v3_5/runner.py").read_text().replace("V3.5", "V3.8").replace("V35", "V38")
    # User explicitly excluded restoration checks. Remove the unused loader too.
    start = source.index("def restore_archive(")
    end = source.index("MODEL_HELPERS =", start)
    source = source[:start] + source[end:]
    source = replace_once(source,
        '    selected = jnp.take_along_axis(logits, batch["targets"][..., None], axis=-1)[..., 0]\n'
        '    return jax.scipy.special.logsumexp(logits, axis=-1) - selected',
        '    with jax.named_scope("cross_entropy"):\n'
        '        selected = jnp.take_along_axis(logits, batch["targets"][..., None], axis=-1)[..., 0]\n'
        '        return jax.scipy.special.logsumexp(logits, axis=-1) - selected')
    source = replace_once(source,
        '            delta, new_opt = optimizer.update(grads, state.opt_state, state.params)\n'
        '            return state.replace(step=state.step + 1,\n'
        '                params=optax.apply_updates(state.params, delta), opt_state=new_opt,',
        '            with jax.named_scope("optimizer_update"):\n'
        '                delta, new_opt = optimizer.update(grads, state.opt_state, state.params)\n'
        '                next_params = optax.apply_updates(state.params, delta)\n'
        '            return state.replace(step=state.step + 1,\n'
        '                params=next_params, opt_state=new_opt,')
    start = source.index("def train_pilot(")
    helpers = "\n\n".join((HERE / name).read_text() for name in ("memory.py", "profiling.py"))
    source = source[:start] + helpers + "\n\n" + source[start:]
    source = replace_once(source, '    manifest = verify_raw(Path(config["shared_directory"]) / "raw")',
        '    if config["local_batch_per_chip"] <= 1 or config["sequence_length"] <= 1024 or config["training_seconds"] <= 18.:\n'
        '        raise ValueError("V3.8 requires batch 3 x 1024 and one 180-second training interval.")\n'
        '    manifest = verify_raw(Path(config["shared_directory"]) / "raw")')
    source = replace_once(source,
        'Starting ONE 60-second {version} run...',
        'Starting ONE {config["training_seconds"]:g}-second {version} run, batch {config["local_batch_per_chip"]} x {config["sequence_length"]}...')
    # The containing f-string uses double quotes: keep config keys single quoted.
    source = source.replace('Starting ONE {config["training_seconds"]:g}-second {version} run, batch {config["local_batch_per_chip"]} x {config["sequence_length"]}',
        "Starting ONE {config['training_seconds']:g}-second {version} run, batch {config['local_batch_per_chip']} x {config['sequence_length']}")
    source = replace_once(source, '    # Preserve completed dispatches even if a later call fails.',
        '    memory_before_training = memory_snapshot(jax.devices()[0].memory_stats(), 0.)\n'
        '    training_memory_samples = []\n'
        '    memory_sampling_seconds = 0.\n'
        '    # Preserve completed dispatches even if a later call fails.')
    source = replace_once(source, '                if len(log_rows) == 1 or end - last_progress >= 2:\n',
        '                if len(log_rows) == 1 or end - last_progress >= 2:\n'
        '                    sample_started = time.monotonic()\n'
        '                    training_memory_samples.append(memory_snapshot(jax.devices()[0].memory_stats(), end))\n'
        '                    memory_sampling_seconds += time.monotonic() - sample_started\n')
    source = replace_once(source, '    if not log_rows:\n',
        '    memory_after_training = memory_snapshot(jax.devices()[0].memory_stats(), training_seconds)\n'
        '    training_memory = summarize_training_memory(memory_before_training, training_memory_samples,\n'
        '        memory_after_training, compiled_memory, memory_sampling_seconds)\n'
        '    write_json(output / "training_memory.json", training_memory)\n'
        '    if not log_rows:\n')
    source = source.replace('updates * 2 * 1024', 'updates * config["local_batch_per_chip"] * config["sequence_length"]')
    source = replace_once(source, 'config["local_batch_per_chip"] == 2 and config["sequence_length"] == 1024,',
        'config["local_batch_per_chip"] == 3 and config["sequence_length"] == 1024 and config["training_seconds"] == 180.,')
    source = replace_once(source, '        "028_WEIGHTS_ONLY": saved["metadata"]["optimizer_resume_supported"] is False,',
        '        "028_WEIGHTS_ONLY": saved["metadata"]["optimizer_resume_supported"] is False,\n'
        '        "029_NO_RESTORE_CHECK": read_json(output / "checkpoint_manifest.json")["restore_executed"] is False,')
    start = source.index('    goals = {"tokens_per_second_at_least_45559"')
    end = source.index('    result = {', start)
    source = source[:start] + '''    comparison = historical_comparison(config, identity["tokenizer_sha256"],
        sha256_file(data_dir / "manifest.json"), tokens_processed / training_seconds,
        after["wiki"]["language_macro_bpb"])
    goals = {"throughput_at_least_v35_reference": tokens_processed / training_seconds >= 45734.29605429257,
        "compilation_seconds_at_most_60": compilation_seconds <= 60.,
        "notebook_seconds_at_most_480": elapsed <= 480.,
        "occupancy_at_least_95_percent": tokens_processed / (updates * config["local_batch_per_chip"] * config["sequence_length"]) >= .95}
    improvement = "PASS" if all(goals.values()) else "FAIL"
''' + source[end:]
    source = replace_once(source, '        "memory_stats": memory_stats, "compiled_memory_estimate": compiled_memory,',
        '        "memory_stats": memory_stats, "compiled_memory_estimate": compiled_memory,\n'
        '        "training_memory": training_memory,\n'
        '        "head": {"kind": "native_dense_exact_softmax", "chunked": False},\n'
        '        "learning_rate_schedule": {"kind": "five_update_warmup_then_constant", "cooldown": False,\n'
        '            "warmup_updates": config["warmup_updates"], "peak": config["learning_rate"]},\n'
        '        "checkpoint_restoration_check": "disabled_by_user", "report_finalized": False,')
    source = replace_once(source, '        "memory_scope": "Allocator snapshot after evaluation; compiler estimate separately",',
        '        "memory_scope": "Training allocator samples and high-water counters; compiler estimate separately",')
    source = replace_once(source,
        '        "historical_comparison_note": "V2 used different data and vocabulary; V3.4 used different kernels/batch/optimizer; neither isolates an optimizer effect",',
        '        "historical_comparison_note": "V3.5 used batch 2 and 60 seconds; V3.8 uses batch 3 and 180 seconds. Historical differences do not isolate implementation efficiency.",')
    source = replace_once(source, '        "v34_baseline_applicable": baseline_applicable, "performance_goals": goals,',
        '        "historical_comparison": comparison, "performance_goals": goals,')
    source = replace_once(source,
        '    if not all(checks.values()):\n'
        '        raise RuntimeError(version + " engineering requirement failed; result.json was saved.")',
        '''    if not all(checks.values()):
        raise RuntimeError(version + " engineering requirement failed; result.json was saved.")
    # Scores and saved weights are final before disposable diagnostic updates.
    write_json(output / "attempt_status.json", {"version": version, "stage": "profiling",
        "timed_training_started": True, "optimizer_updates": updates})
    profile = record_diagnostic_profile(state, training, group_jit, output, work,
        config["profile_dispatches"], config["profile_updates_per_dispatch"])
    del state
    result["profiling"] = profile
    result["profiling_status"] = profile["requirements_status"]
    result["telemetry_status"] = profile["telemetry_status"]
    result["notebook_elapsed_before_profile_seconds"] = elapsed
    elapsed = time.monotonic() - config["notebook_start_monotonic"]
    result["notebook_elapsed_seconds"] = elapsed
    result["compute_units_estimate"].update({"through_report": elapsed * rate / 3600,
        "overhead_through_report": (elapsed - training_seconds) * rate / 3600,
        "diagnostic_profile": profile["seconds_including_export_and_analysis"] * rate / 3600})
    result["performance_goals"]["notebook_seconds_at_most_480"] = elapsed <= 480.
    result["performance_goal_status"] = "PASS" if all(result["performance_goals"].values()) else "FAIL"
    result["report_finalized"] = True
    write_json(output / "result.json", result)
    write_json(output / "attempt_status.json", {"version": version, "stage": "complete",
        "timed_training_started": True, "optimizer_updates": updates,
        "diagnostic_updates": profile["completed_updates"], "profiling_status": result["profiling_status"]})
    print("Profiling requirements:", result["profiling_status"], "| telemetry:", result["telemetry_status"], flush=True)
    for name, measurement in profile["measurements"].items():
        if measurement["status"] != "available":
            print("MISSING:", name, "|", measurement.get("reason"), flush=True)
''')
    # Obsolete, unused baseline applicability block is not a V3.8 acceptance gate.
    start = source.index('    baseline_applicable = (')
    end = source.index('    comparison = historical_comparison', start)
    source = source[:start] + source[end:]
    ast.parse(source)
    return source


def cell(kind, text):
    value = {"cell_type": kind, "id": hashlib.sha256(text.encode()).hexdigest()[:8],
             "metadata": {}, "source": text.splitlines(True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def build_notebook(source):
    notebook = json.loads((PARENT / "gemma_270m_colab_v5e1_V3_5.ipynb").read_text())
    notebook["metadata"]["colab"]["name"] = TARGET.name
    for item in notebook["cells"]:
        text = "".join(item["source"])
        if "PROGRAM_SOURCE =" in text:
            tree = ast.parse(text)
            assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets))
            config = ast.literal_eval(assignment.value)
            config.update(experiment="GEM-COLAB-V3.8", version="V3.8", requirement_prefix="V38",
                implementation_revision="V3.8-native-b3-180s-1", local_batch_per_chip=3,
                training_seconds=180., profile_dispatches=4, profile_updates_per_dispatch=8)
            tail = text[text.index('CONFIG["tokenizer_reference"]'):]
            text = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + "\n" + tail
        text = text.replace("V3.5", "V3.8").replace("V3_5", "V3_8").replace("V35-", "V38-")
        # Embedded source contains historical V3.5 labels; do not rewrite them.
        if "PROGRAM_SOURCE =" in text:
            tail = text[text.index('CONFIG["tokenizer_reference"]'):]
            text = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + "\n" + tail
        item["source"] = text.splitlines(True)
        if item["cell_type"] == "code":
            item.update(execution_count=None, outputs=[])
    notebook["cells"][0] = cell("markdown", (HERE / "INTRO.md").read_text())
    notebook["cells"][1] = cell("markdown", (HERE / "REQUIREMENTS.md").read_text())
    notebook["cells"][10] = cell("markdown", """## 5. Train once for 180 seconds
Batch 3 × 1,024. Native dense full-vocabulary loss, AdamW, five warmup updates,
then learning rate 0.0003. Compilation and initial evaluation precede timing.
Final evaluation and weights saving follow it. No restoration or reload evaluation.
A separate 32-update diagnostic profiles disposable state after weights are saved.
Missing required measurements produce a profiling FAIL; training results remain saved.
""")
    notebook["cells"][12] = cell("markdown", "## 6. Read training results and measurement completeness\nHistorical runs used different batch sizes and training durations.\n")
    notebook["cells"][13] = cell("code", (HERE / "display_results.py").read_text())
    notebook["cells"][15] = cell("code", (HERE / "download_reports.py").read_text())
    data = "".join(notebook["cells"][16]["source"])
    data = data.replace("Realized 60-second source shares", "Realized 180-second source shares")
    data = data.replace("This one-seed, one-minute result", "This one-seed, three-minute result")
    data += "\nV3.8 uses batch 3 × 1,024 and 180 seconds. V3.5 used batch 2 × 1,024 and 60 seconds; BPB differences do not isolate implementation efficiency.\n"
    notebook["cells"][16]["source"] = data.splitlines(True)
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            ast.parse("".join(item["source"]))
    return notebook


def main():
    if TARGET.exists() or (HERE / "runner.py").exists():
        raise FileExistsError("Preserving an existing V3.8 artifact; create a new version.")
    source = build_source()
    notebook = build_notebook(source)
    with (HERE / "runner.py").open("x") as stream:
        stream.write(source)
    with TARGET.open("x") as stream:
        json.dump(notebook, stream, indent=1, ensure_ascii=False)
        stream.write("\n")
    print(TARGET)


if __name__ == "__main__":
    main()
