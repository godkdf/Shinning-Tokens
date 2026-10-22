"""Build only V3.7. Earlier notebooks and source versions remain untouched."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
BASE = PARENT / "v3_5" / "runner.py"
TARGET = PARENT / "gemma_270m_colab_v5e1_V3_7.ipynb"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one source anchor, found {text.count(old)}: {old[:100]}")
    return text.replace(old, new, 1)


def build_source():
    source = BASE.read_text().replace("V3.5", "V3.7").replace("V35", "V37")
    start = source.index("def token_losses(")
    end = source.index("def make_training_functions(", start)
    source = source[:start] + (HERE / "head.py").read_text() + "\n\n" + source[end:]
    start = source.index("def train_pilot(")
    source = source[:start] + (HERE / "profiling.py").read_text() + "\n\n" + source[start:]
    source = replace_once(source, "def one_step(state, batch):", "def one_step(state, batch, lr_multiplier):")
    source = replace_once(source, "def group(state, training, requested):", "def group(state, training, requested, lr_multiplier):")
    source = replace_once(source, "one_step(state, batch)", "one_step(state, batch, lr_multiplier)")
    source = replace_once(source,
        '            delta, new_opt = optimizer.update(grads, state.opt_state, state.params)\n'
        '            return state.replace(step=state.step + 1,\n'
        '                params=optax.apply_updates(state.params, delta), opt_state=new_opt,',
        '            with jax.named_scope("optimizer_update"):\n'
        '                delta, new_opt = optimizer.update(grads, state.opt_state, state.params)\n'
        '                delta = jax.tree.map(lambda value: value * lr_multiplier, delta)\n'
        '                next_params = optax.apply_updates(state.params, delta)\n'
        '            return state.replace(step=state.step + 1,\n'
        '                params=next_params, opt_state=new_opt,')
    for call in ("group_jit.lower(state, training, jnp.array(0, jnp.int32))",
                 "group_jit(state, training, jnp.array(0, jnp.int32))"):
        source = replace_once(source, call, call[:-1] + ", jnp.array(1., jnp.float32))")
    source = replace_once(source,
        '    write_json(output / "kernel_verification.json", protocol_report)',
        '    head_report = verify_exact_head(model, state.params, training)\n'
        '    write_json(output / "head_verification.json", head_report)\n'
        '    protocol_report["loss"] = "exact chunked full-vocabulary softmax, 128-token workspace"\n'
        '    protocol_report["head_verified"] = head_report["passed"]\n'
        '    write_json(output / "kernel_verification.json", protocol_report)')
    source = replace_once(source,
        '                call_start = time.monotonic()\n'
        '                state, values = jax.block_until_ready(group_jit(state, training, jnp.array(requested, jnp.int32)))',
        '                lr_multiplier = cooldown_multiplier(last_start, config["training_seconds"],\n'
        '                    config["cooldown_start_fraction"], config["learning_rate_floor_fraction"])\n'
        '                call_start = time.monotonic()\n'
        '                state, values = jax.block_until_ready(group_jit(state, training,\n'
        '                    jnp.array(requested, jnp.int32), jnp.array(lr_multiplier, jnp.float32)))')
    source = replace_once(source, '                    "start_seconds": last_start, "end_seconds": end,',
        '                    "start_seconds": last_start, "end_seconds": end,\n'
        '                    "learning_rate_multiplier": lr_multiplier,\n'
        '                    "learning_rate_at_dispatch_start": config["learning_rate"] * lr_multiplier\n'
        '                        * min((updates - completed) / config["warmup_updates"], 1.),')
    source = replace_once(source, '    checkpoint_seconds = time.monotonic() - phase',
        '''    checkpoint_seconds = time.monotonic() - phase
    restore_started = time.monotonic()
    original_digest = parameter_digest(state.params)
    restored = restore_archive(checkpoint, {"params": state.params})
    restored_digest = parameter_digest(restored["params"])
    exact_restore = original_digest == restored_digest
    reloaded_eval = evaluate(restored["params"])
    reload_bpb_error = max(abs(after[split]["languages"][language]["bpb"]
        - reloaded_eval[split]["languages"][language]["bpb"])
        for split in ("in_mix", "wiki") for language in after[split]["languages"])
    del restored
    restore_seconds = time.monotonic() - restore_started
    checkpoint_report = read_json(output / "checkpoint_manifest.json")
    checkpoint_report.update({"restore_executed": True, "exact_parameter_match": exact_restore,
        "parameter_sha256": original_digest, "restored_parameter_sha256": restored_digest,
        "reload_max_bpb_error": reload_bpb_error, "reload_bpb_tolerance": 1e-6,
        "restore_and_evaluation_seconds": restore_seconds})
    write_json(output / "checkpoint_manifest.json", checkpoint_report)''')
    source = replace_once(source, '        "028_WEIGHTS_ONLY": saved["metadata"]["optimizer_resume_supported"] is False,',
        '        "028_WEIGHTS_ONLY": saved["metadata"]["optimizer_resume_supported"] is False,\n'
        '        "029_EXACT_HEAD": head_report["passed"],\n'
        '        "030_RESTORED_WEIGHTS": exact_restore and reload_bpb_error <= 1e-6,\n'
        '        "031_COOLDOWN": all(.3 <= row["learning_rate_multiplier"] <= 1. for row in log_rows),')
    source = replace_once(source, '"finewiki_macro_bpb_at_most_v34": after["wiki"]["language_macro_bpb"] <= 2.782397528489431',
        '"finewiki_macro_bpb_at_most_v35": after["wiki"]["language_macro_bpb"] <= 2.9347027242183685')
    source = replace_once(source, '        "v34_baseline_applicable": baseline_applicable, "performance_goals": goals,',
        '        "v35_baseline_applicable": baseline_applicable, "performance_goals": goals,\n'
        '        "baseline": {"version": "V3.5", "finewiki_macro_bpb": 2.9347027242183685,\n'
        '            "tokens_per_second": 45734.29605429257, "training_seconds": 60.014611282999795},\n'
        '        "head": {"kind": "exact_token_chunked", "chunk_tokens": HEAD_CHUNK_TOKENS,\n'
        '            "logical_fp32_logits_workspace_bytes": HEAD_CHUNK_TOKENS * identity["model_rows"] * 4,\n'
        '            "note": "Logical single-chunk logits only; compiler peak memory is reported separately."},\n'
        '        "learning_rate_schedule": {"warmup_updates": config["warmup_updates"],\n'
        '            "peak": config["learning_rate"], "cooldown_start_fraction": .5, "floor_fraction": .3,\n'
        '            "scope": "Elapsed training time, multiplier held constant within each dispatch; scales AdamW update including decay.",\n'
        '            "last_dispatch_multiplier": log_rows[-1]["learning_rate_multiplier"]},\n'
        '        "checkpoint_restore_seconds": restore_seconds, "reloaded_eval": reloaded_eval,\n'
        '        "reload_max_bpb_error": reload_bpb_error, "exact_parameter_restore": exact_restore,\n'
        '        "head_verification_seconds": head_report["seconds"],\n'
        '        "report_finalized": False,')
    source = replace_once(source,
        '    if not all(checks.values()):\n'
        '        raise RuntimeError(version + " engineering requirement failed; result.json was saved.")',
        '''    if not all(checks.values()):
        raise RuntimeError(version + " engineering requirement failed; result.json was saved.")
    # The official checkpoint, scores and checks are final before consuming this
    # disposable in-memory state. No diagnostic weights are saved or ranked.
    write_json(output / "attempt_status.json", {"version": version, "stage": "profiling",
        "timed_training_started": True, "optimizer_updates": updates})
    profile = record_diagnostic_profile(state, training, group_jit, output, work,
        log_rows[-1]["learning_rate_multiplier"], config["profile_dispatches"],
        config["profile_updates_per_dispatch"])
    del state
    result["profiling"] = profile
    available = sum(value["status"] == "available" for value in profile["measurements"].values())
    result["telemetry_status"] = "COMPLETE" if available == 5 else "PARTIAL" if available else "UNAVAILABLE"
    result["telemetry_available_measurements"] = available
    result["notebook_elapsed_before_profile_seconds"] = elapsed
    elapsed = time.monotonic() - config["notebook_start_monotonic"]
    result["notebook_elapsed_seconds"] = elapsed
    result["compute_units_estimate"].update({"through_report": elapsed * rate / 3600,
        "overhead_through_report": (elapsed - training_seconds) * rate / 3600,
        "diagnostic_profile": profile["seconds_including_export_and_analysis"] * rate / 3600})
    result["performance_goals"]["notebook_seconds_at_most_300"] = elapsed <= 300.
    result["performance_goal_status"] = ("PASS" if all(value is True for value in result["performance_goals"].values())
        else "FAIL") if baseline_applicable else "NOT_COMPARABLE"
    result["report_finalized"] = True
    write_json(output / "result.json", result)
    write_json(output / "attempt_status.json", {"version": version, "stage": "complete",
        "timed_training_started": True, "optimizer_updates": updates,
        "diagnostic_updates": profile["completed_updates"], "telemetry_status": result["telemetry_status"]})
    print("Profiler telemetry:", result["telemetry_status"], "| diagnostic updates:", profile["completed_updates"], flush=True)''')
    return source


def cell(kind, text):
    result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    if TARGET.exists():
        raise FileExistsError("Preserving existing notebook: " + str(TARGET))
    source = build_source()
    notebook = json.loads((PARENT / "gemma_270m_colab_v5e1_V3_5.ipynb").read_text())
    for item in notebook["cells"]:
        text = "".join(item["source"])
        if item["cell_type"] == "code":
            item.update(execution_count=None, outputs=[])
            if "PROGRAM_SOURCE =" in text:
                tree = ast.parse(text)
                config_node = next(node for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in node.targets))
                config = ast.literal_eval(config_node.value)
                config.update(experiment="GEM-COLAB-V3.7", version="V3.7", requirement_prefix="V37",
                    implementation_revision="V3.7-exact-head-profile-1", cooldown_start_fraction=.5,
                    learning_rate_floor_fraction=.3, head_chunk_tokens=128,
                    profile_dispatches=4, profile_updates_per_dispatch=16)
                tail = text[text.index('CONFIG["tokenizer_reference"]'):].replace("V3.5", "V3.7").replace("V3_5", "V3_7")
                text = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + "\n" + tail
            else:
                text = text.replace("V3.5", "V3.7").replace("V3_5", "V3_7")
        else:
            text = text.replace("V3.5", "V3.7").replace("V35-", "V37-")
        item["source"] = text.splitlines(True)
    notebook["cells"][0] = cell("markdown", (HERE / "INTRO.md").read_text())
    notebook["cells"][1] = cell("markdown", (HERE / "REQUIREMENTS.md").read_text())
    # Replace the results cell with a display that includes all requested telemetry.
    notebook["cells"][13] = cell("code", (HERE / "display_results.py").read_text())
    notebook["cells"][10] = cell("markdown", """## 5. Train once, verify the saved model, then collect a bounded profile
The official run trains for 60 seconds with an elapsed-time cooldown. Exact-head
value/gradient checks precede timing. After saving, weights are restored and
verified. A separate 64-update diagnostic profile consumes disposable state;
its weights are discarded, its compute cost is reported, and it cannot change
the official checkpoint or score. Missing TPU counters appear as unavailable.
""")
    notebook["cells"][16]["source"] = ("".join(notebook["cells"][16]["source"])
        + "\nV3.7 preserves the V3.5 corpus and tokenizer protocol. It changes the exact head implementation and adds a timed cooldown. Diagnostic state is discarded after checkpoint verification.\n").splitlines(True)
    download = "".join(notebook["cells"][15]["source"])
    download = download.replace('"kernel_verification.json", "training_accounting.json",',
        '"kernel_verification.json", "head_verification.json", "profiling.json", "training_accounting.json",')
    download = download.replace('shutil.copyfile(REPORT_ZIP, OUTPUT / REPORT_ZIP.name)',
        '''    for directory in ("profile", "profile_tools"):
        for path in sorted((OUTPUT / directory).rglob("*")):
            if path.is_file():
                bundle.write(path, str(path.relative_to(OUTPUT)))
shutil.copyfile(REPORT_ZIP, OUTPUT / REPORT_ZIP.name)''')
    notebook["cells"][15]["source"] = download.splitlines(True)
    with (HERE / "runner.py").open("x") as stream:
        stream.write(source)
    with TARGET.open("x") as stream:
        json.dump(notebook, stream, indent=1, ensure_ascii=False)
        stream.write("\n")
    print(TARGET)
    print("Notebook SHA256:", hashlib.sha256(TARGET.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
