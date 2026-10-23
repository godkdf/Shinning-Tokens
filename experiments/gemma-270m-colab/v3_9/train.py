def train_pilot(work, output):
    work, output = Path(work), Path(output)
    version = PILOT_VERSION
    with (output / "training.started").open("x") as marker:
        marker.write(version + ": one training attempt; automatic retry disabled.\n")
    write_json(output / "attempt_status.json", {"stage": "pretraining", "optimizer_updates": 0,
        "timed_training_started": False, "version": version})
    environment = preflight()
    write_json(output / "preflight.json", environment)
    load_model_helpers()
    data_dir = work / "prepared"
    config = read_json(data_dir / "resolved_config.json")
    validate_training_config(config)
    manifest = verify_raw(Path(config["shared_directory"]) / "raw")
    tokenization = read_json(data_dir / "tokenization.json")
    identity = config["tokenizer_identity"]
    validate_tokenization_manifest(tokenization, identity, sha256_file(data_dir / "manifest.json"), config["sequence_length"])
    if tokenization["raw_manifest_sha256"] != sha256_file(data_dir / "manifest.json"):
        raise ValueError("Tokenized and raw corpus manifests differ.")
    if shutil.disk_usage(work).free < 3 * 1024**3:
        raise RuntimeError("The weights-only artifact requires at least 3 GiB free local disk.")
    host, orders, prefixes, context_metadata = {}, {}, {}, {}
    for mode, length in (("short", 1024), ("long", 2048)):
        directory = data_dir / str(length)
        details = read_json(directory / "tokenization.json")
        validate_tokenization_manifest(details, identity, sha256_file(data_dir / "manifest.json"), length)
        context_metadata[mode] = details
        if sha256_file(directory / "train.npz") != details["files"]["train.npz"]:
            raise ValueError("Training input checksum mismatch: " + mode)
        with np.load(directory / "train.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in INPUT_KEYS}
        validate_sequence_arrays(arrays, length, mode)
        if int(arrays["byte_weights"].sum()) != manifest["original_bytes"]["train"]:
            raise ValueError("Training byte attribution mismatch: " + mode)
        arrays, orders[mode] = row_training_arrays(arrays, config["seed"])
        prefixes[mode] = row_prefix_counts(arrays)
        host[mode] = {key: arrays[key] for key in TRAIN_KEYS}
        del arrays
    eval_parts = {}
    for split in ("in_mix", "wiki"):
        directory = data_dir / "1024"
        name = split + ".npz"
        if sha256_file(directory / name) != tokenization["files"][name]:
            raise ValueError("Evaluation checksum mismatch: " + split)
        with np.load(directory / name, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in INPUT_KEYS}
        validate_sequence_arrays(arrays, 1024, split)
        if int(arrays["byte_weights"].sum()) != manifest["original_bytes"][split]:
            raise ValueError("Evaluation byte attribution mismatch: " + split)
        eval_parts[split], _ = batch_training_arrays(arrays, 1, config["seed"])
    eval_host = {key: np.concatenate((eval_parts["in_mix"][key], eval_parts["wiki"][key]), axis=0)
        for key in INPUT_KEYS}
    eval_host["split_id"] = np.concatenate((np.zeros(eval_parts["in_mix"]["tokens"].shape[0], np.int32),
        np.ones(eval_parts["wiki"]["tokens"].shape[0], np.int32)))
    input_bytes_by_mode = {mode: sum(value.nbytes for value in values.values()) for mode, values in host.items()}
    input_bytes = sum(input_bytes_by_mode.values())
    eval_bytes = sum(value.nbytes for value in eval_host.values())
    phase = time.monotonic()
    training, validation = jax.device_put((host, eval_host))
    jax.block_until_ready((training, validation))
    input_transfer_seconds = time.monotonic() - phase
    resident = all(jax.devices()[0] in leaf.devices() for leaf in jax.tree.leaves(training))
    del host, arrays, eval_parts, eval_host
    config["other_resident_bytes_short"] = input_bytes_by_mode["long"] + eval_bytes
    config["other_resident_bytes_long"] = input_bytes_by_mode["short"] + eval_bytes
    print(f"Both context inputs on TPU: {input_bytes / 1024**2:.1f} MiB; exact CPU row accounting prepared.", flush=True)
    write_json(output / "training_accounting.json", {"method": "Independent context row cursors, fixed shuffled order; no batch padding or cursor reset",
        "contexts": {mode: {"rows_per_pass": training[mode]["tokens"].shape[0],
            "tokens_per_pass": int(prefixes[mode]["tokens"][-1]), "training_input_bytes": input_bytes_by_mode[mode]}
            for mode in training}, "evaluation_context": 1024,
        "note": "Both contexts reuse the same frozen text. Byte exposure includes repeats, not unique corpus bytes."})
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment)

    phase = time.monotonic()
    model = make_model(config)
    if model.config.final_logit_softcap is not None:
        raise ValueError("This protocol requires native uncapped Gemma 270M logits.")
    params = jax.jit(lambda key: model.init(key, tokens=jnp.ones((1, 1), jnp.int32))["params"])(
        jax.random.PRNGKey(config["seed"]))
    jax.block_until_ready(params)
    flat = traverse_util.flatten_dict(params)
    parameter_count = sum(value.size for value in flat.values())
    embedding_shape = (identity["model_rows"], 640)
    tied = (flat[("embedder", "input_embedding")].shape == embedding_shape
            and sum(value.shape == embedding_shape for value in flat.values()) == 1)
    if parameter_count != identity["expected_parameters"] or not tied or not all(
            value.dtype == jnp.float32 for value in flat.values()):
        raise ValueError("Parameter count, tied table, or FP32 master weights differ.")
    backbone_digest = initial_backbone_digest(params)
    backbone_key = hashlib.sha256(json.dumps({"gemma": environment["packages"].get("gemma"),
        "jax": environment["jax"], "seed": config["seed"], "backbone": "Gemma3_270M"},
        sort_keys=True).encode()).hexdigest()[:16]
    fingerprint_path = Path(config["shared_directory"]) / ("backbone-" + backbone_key + ".json")
    fingerprint = {"sha256": backbone_digest, "transformer_and_norm_parameters": 100326016,
                   "seed": config["seed"]}
    if fingerprint_path.exists() and read_json(fingerprint_path) != fingerprint:
        raise ValueError("Transformer initialization differs from the shared baseline.")
    write_json(fingerprint_path, fingerprint)
    write_json(output / "backbone_initialization.json", fingerprint)
    probe_path = next(path for path, value in sorted(flat.items())
                      if path[0].startswith("layer_") and value.ndim > 1)
    def probe(parameters):
        return np.asarray(jax.device_get(traverse_util.flatten_dict(parameters)[probe_path].reshape(-1)[:64]))
    initial_probe = probe(params)
    optimizer, partition = make_optimizer(config, params)
    routing = partition.pop("routing")
    routing_ok = (sum(row["parameters"] for row in routing) == parameter_count
                  and all(row["optimizer"] == config["optimizer_label"] for row in routing)
                  and partition["muon_parameters"] + partition["adamw_parameters"] == parameter_count)
    if not routing_ok:
        raise ValueError("Optimizer routing does not cover every parameter with the requested optimizer.")
    write_json(output / "optimizer_parameter_routing.json", {"parameters": parameter_count,
        "optimizer": config["optimizer_label"], "all_parameters_accounted_for": routing_ok,
        "leaves": routing})
    write_json(output / "optimizer_partition.json", partition)
    state = PilotState(jnp.array(0, jnp.int32), params, optimizer.init(params),
        jax.random.PRNGKey(config["seed"]), jnp.array(0, jnp.int32),
        jnp.array(0, jnp.int32), jnp.array(0, jnp.int32))
    del params, flat
    jax.block_until_ready(state)
    initialization_seconds = time.monotonic() - phase
    protocol_report = verify_native_protocol()
    write_json(output / "kernel_verification.json", protocol_report)
    functions, analyses, all_analyses, compilation_events = {}, {}, {}, []
    def compile_shape(mode, batch, current_state, scope):
        key = (mode, batch)
        if key in functions:
            return all_analyses[key]
        begun = time.monotonic()
        _, group = make_training_functions(model, optimizer, batch, mode)
        executable = group.lower(current_state, training[mode], jnp.array(0, jnp.int32)).compile()
        analysis = compiler_memory(executable)
        functions[key] = executable
        all_analyses[key] = analysis
        if mode == "short":
            analyses[batch] = analysis
        peak = compiler_peak(analysis)
        other = config["other_resident_bytes_" + mode]
        compilation_events.append({"mode": mode, "batch": batch, "sequence_length": 1024 if mode == "short" else 2048,
            "seconds": time.monotonic() - begun, "scope": scope, "official_step_at_compile": int(current_state.step),
            "compiler_memory": analysis, "compiler_peak_plus_other_resident_bytes": peak + other if peak is not None else None})
        write_json(output / "compilation_events.json", compilation_events)
        return analysis
    print("Compiling initial short, fixed long and 1024-context evaluator; zero optimizer updates...", flush=True)
    phase = time.monotonic()
    for mode in ("short", "long"):
        analysis = compile_shape(mode, 1, state, "before_training")
        limit = (jax.devices()[0].memory_stats() or {}).get("bytes_limit")
        peak = compiler_peak(analysis)
        if peak is not None and limit and peak + config["other_resident_bytes_" + mode] > limit * config["memory_budget_fraction"]:
            raise RuntimeError("Required initial context exceeds the conservative memory budget: " + mode)
    evaluation_jit = make_evaluator(model)
    evaluator = evaluation_jit.lower(state.params, validation).compile()
    state, warmup_metrics = jax.block_until_ready(functions[("short", 1)](state, training["short"], jnp.array(0, jnp.int32)))
    if int(state.step) != 0 or int(state.short_rows) != 0 or int(state.long_rows) != 0:
        raise ValueError("Compilation preparation performed an optimizer update.")
    compilation_seconds = time.monotonic() - phase
    write_json(output / "dispatch_preparation.json", {"dispatch": "compiled executable",
        "training_warmup_requested_updates": 0, "training_warmup_completed_updates": int(warmup_metrics["updates"]),
        "pretraining_optimizer_step": int(state.step), "initial_training_executables": 2,
        "evaluation_executables": 1, "adaptive_compilation": "inside the 1200-second training wall-clock budget"})
    def evaluate(parameters):
        values = jax.block_until_ready(evaluator(parameters, validation))
        return {split: readable_evaluation(tuple(value[index] for value in values),
                    manifest["original_bytes"][split], config["sequence_length"])
                for index, split in enumerate(("in_mix", "wiki"))}
    phase = time.monotonic()
    before = evaluate(state.params)
    initial_eval_seconds = time.monotonic() - phase
    write_json(output / "timed_training.started.json", {"version": version,
        "stage": "timed training", "training_seconds": config["training_seconds"]})
    write_json(output / "attempt_status.json", {"version": version, "stage": "timed training",
        "timed_training_started": True, "optimizer_updates": None})
    print(f"Initial FineWiki macro BPB={before['wiki']['language_macro_bpb']:.5f}. Starting ONE {config['training_seconds']:g}-second {version} run, batch {config['local_batch_per_chip']} x {config['sequence_length']}...", flush=True)
    state, adaptive = adaptive_train(state, training, prefixes, compile_shape, functions, analyses, config, output)
    log_rows = adaptive["rows"]
    updates, tokens_processed = adaptive["updates"], adaptive["tokens"]
    training_seconds, last_start = adaptive["training_seconds"], adaptive["last_start"]
    used_keys = {(row["mode"], row["batch"]) for row in log_rows}
    used_analyses = [all_analyses[key] for key in used_keys if all_analyses[key] is not None]
    compiled_memory = max(used_analyses, key=compiler_peak) if used_analyses else None
    training_memory = summarize_training_memory(adaptive["memory_before"], adaptive["samples"],
        adaptive["memory_after"], compiled_memory, adaptive["sampling_seconds"])
    peaks_with_other = [compiler_peak(all_analyses[key]) + config["other_resident_bytes_" + key[0]]
        for key in used_keys if all_analyses[key] is not None]
    training_memory["compiled_peak_plus_other_resident_bytes"] = max(peaks_with_other) if peaks_with_other else None
    limit = training_memory["device_limit_bytes"]
    training_memory["compiled_peak_plus_other_resident_percent"] = (
        max(peaks_with_other) * 100 / limit if peaks_with_other and limit else None)
    training_memory["compiler_estimates_by_shape"] = compilation_events
    write_json(output / "training_memory.json", training_memory)
    if not log_rows:
        raise RuntimeError("No optimizer updates completed.")
    domain_bytes = sum(completed_prefix(prefixes[mode]["domain_bytes"], rows)
        for mode, rows in adaptive["rows_used"].items())
    language_bytes = sum(completed_prefix(prefixes[mode]["language_bytes"], rows)
        for mode, rows in adaptive["rows_used"].items())
    expected_tokens = sum(int(completed_prefix(prefixes[mode]["tokens"], rows))
        for mode, rows in adaptive["rows_used"].items())
    if (tokens_processed != expected_tokens or int(state.short_rows) != adaptive["rows_used"]["short"]
            or int(state.long_rows) != adaptive["rows_used"]["long"]):
        raise ValueError("CPU/device token or context cursor accounting differs.")
    changed = not np.array_equal(initial_probe, probe(state.params))
    final_finite = bool(jax.device_get(jax.jit(lambda parameters: jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(parameters)])))(state.params)))
    phase = time.monotonic()
    after = evaluate(state.params)
    final_eval_seconds = time.monotonic() - phase
    memory_stats = jax.devices()[0].memory_stats()
    phase = time.monotonic()
    metadata = {"version": version, "protocol": PROTOCOL, "config": config,
        "parameter_count": parameter_count, "initialization": "random; zero pretrained weights",
        "backbone_initialization_sha256": backbone_digest, "optimizer": partition,
        "raw_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "tokenization_manifest_sha256": sha256_file(data_dir / "tokenization.json"),
        "packing_version": PACKING_VERSION, "data_order_sha256": orders,
        "training_rows_per_context": {mode: values["tokens"].shape[0] for mode, values in training.items()}, "updates": updates,
        "context_rows_consumed": adaptive["rows_used"], "final_short_batch": adaptive["short_batch"],
        "data_position": int(state.data_position), "checkpoint_fields": ["params"],
        "checkpoint_kind": "weights_only", "optimizer_resume_supported": False}
    print("Saving ONE FP32 weights-only artifact; optimizer resume is not included...", flush=True)
    local_checkpoint = work / "weights.npz"
    leaves = save_archive(local_checkpoint, {"params": state.params}, metadata)
    digest = sha256_file(local_checkpoint)
    partial = output / "weights.npz.partial"
    shutil.copyfile(local_checkpoint, partial)
    if sha256_file(partial) != digest:
        raise ValueError("Drive weights checksum differs from the local artifact.")
    checkpoint = output / "weights.npz"
    partial.rename(checkpoint)
    with zipfile.ZipFile(checkpoint) as archive:
        saved = json.loads(archive.read("checkpoint.json"))
        schema_verified = (set(saved["tree"]) == {"params"} and saved["metadata"] == metadata
            and sum(name.startswith("arrays/") for name in archive.namelist()) == leaves)
    if not schema_verified:
        raise ValueError("Saved weights schema or array count differs.")
    write_json(output / "checkpoint_manifest.json", {"sha256": digest,
        "bytes": checkpoint.stat().st_size, "array_leaves": leaves,
        "kind": "weights_only", "optimizer_resume_supported": False,
        "verification": "local/Drive SHA256 equality plus weights schema and leaf count",
        "restore_executed": False})
    checkpoint_seconds = time.monotonic() - phase
    bytes_processed = int(domain_bytes.sum())
    prefix_id = config["requirement_prefix"]
    checks = {
        "001_ONE_ATTEMPT": (output / "training.started").exists(),
        "002_ONE_V5E": environment["platform"] == "tpu" and environment["device_count"] == 1,
        "003_SOURCE_PINS": set(manifest["source_revisions"]) == set(SOURCE_REPOS),
        "004_FROZEN_CORPUS": tokenization["raw_manifest_sha256"] == sha256_file(data_dir / "manifest.json"),
        "005_SPLIT_SEPARATION": manifest["cross_split_source_groups"] == 0,
        "006_LOSSLESS_TOKENIZATION": all(tokenization["splits"][split]["original_bytes"] == manifest["original_bytes"][split]
                                          for split in ("train", "in_mix", "wiki")),
        "007_INTACT_PACKING": all(tokenization["splits"][split]["targets_preserved"] and
            tokenization["splits"][split]["context_preserved"] for split in ("train", "in_mix", "wiki")),
        "008_NATIVE_ATTENTION_MASK": protocol_report["passed"],
        "009_RESIDENT_INPUTS": resident,
        "010_GEMMA_BACKBONE": parameter_count == identity["expected_parameters"] and parameter_count < 300000000 and tied,
        "011_OPTIMIZER_ROUTING": routing_ok,
        "012_TIMED_DISPATCH": all(row["start_seconds"] < config["training_seconds"] for row in log_rows),
        "013_DISPATCH_BOUND": all(1 <= row["updates"] <= 16 for row in log_rows),
        "014_FINITE_UPDATES": all(row["finite"] for row in log_rows) and final_finite,
        "015_ZERO_EXTRA_UPDATES": int(state.step) == updates and int(state.data_position) == updates,
        "016_FIXED_EVALUATION": len(after["wiki"]["languages"]) == len(LANGUAGES) and all(
            after[split]["original_utf8_bytes"] == before[split]["original_utf8_bytes"] for split in ("in_mix", "wiki")),
        "017_WEIGHTS_ARTIFACT": schema_verified,
        "018_CHANGED_WEIGHTS": changed,
        "019_MINIMUM_UPDATES": updates >= config["minimum_updates"],
        "020_BYTE_ACCOUNTING": bytes_processed == int(language_bytes.sum()) and bytes_processed > 0 and expected_tokens == tokens_processed,
        "025_NATIVE_TRAINING": training_specification_matches(config, training),
        "026_COMBINED_EVALUATOR": validation["split_id"].shape[0] == validation["tokens"].shape[0],
        "027_CPU_ACCOUNTING": all(not set(EXTRA_KEYS) & set(values) for values in training.values()),
        "030_INITIAL_BATCH": all(row["batch"] == 1 and row["sequence_length"] == 1024 for row in log_rows if row["update"] <= 100),
        "031_TOKEN_SCHEDULE": all(row["mode"] == context_phase(row["total_tokens"] - row["tokens"], row["short_batch"]) for row in log_rows),
        "032_CONTEXT_CURSORS": int(state.short_rows) == adaptive["rows_used"]["short"] and int(state.long_rows) == adaptive["rows_used"]["long"],
        "028_WEIGHTS_ONLY": saved["metadata"]["optimizer_resume_supported"] is False,
        "029_NO_RESTORE_CHECK": read_json(output / "checkpoint_manifest.json")["restore_executed"] is False,
    }
    checks = {prefix_id + "-" + key: value for key, value in checks.items()}
    checks.update({key: value == "PASS" for key, value in read_json(output / "preparation_requirements.json").items()})
    downloads = read_json(output / "parallel_downloads.json")
    elapsed = time.monotonic() - config["notebook_start_monotonic"]
    rate = config["compute_units_per_hour"]
    comparison = historical_comparison(config, identity["tokenizer_sha256"],
        sha256_file(data_dir / "manifest.json"), tokens_processed / training_seconds,
        after["wiki"]["language_macro_bpb"])
    goals = {"throughput_at_least_v35_reference": tokens_processed / training_seconds >= 45734.29605429257,
        "pretraining_compilation_seconds_at_most_180": compilation_seconds <= 180.,
        "notebook_seconds_at_most_1800": elapsed <= 1800.,
        "occupancy_at_least_95_percent": tokens_processed / adaptive["positions"] >= .95}
    improvement = "PASS" if all(goals.values()) else "FAIL"
    result = {"version": version, "protocol": PROTOCOL, "run_count": 1, "config": config,
        "device": environment["device_kind"], "model_parameters": parameter_count, "optimizer": partition,
        "source_downloads": {key: downloads[key] for key in
            ("source_jobs", "max_workers", "peak_active_source_jobs", "fetch_seconds", "raw_cache_reused")},
        "initial_backbone_sha256": backbone_digest, "input_transfer_seconds": input_transfer_seconds,
        "resident_training_bytes": input_bytes, "initialization_seconds": initialization_seconds,
        "kernel_verification_seconds": protocol_report["seconds"], "compilation_seconds": compilation_seconds,
        "compilation_timing_scope": "initial short, fixed long and fixed-context evaluator; later compilations counted inside training wall time",
        "compilation_events": compilation_events,
        "adaptive_compilation_seconds_inside_training": sum(e["seconds"] for e in compilation_events if e["scope"] == "inside_training_budget"),
        "dispatch_preparation": read_json(output / "dispatch_preparation.json"),
        "initial_eval_seconds": initial_eval_seconds, "final_eval_seconds": final_eval_seconds,
        "training_seconds": training_seconds,
        "training_overrun_seconds": max(0., training_seconds - config["training_seconds"]),
        "last_dispatch_start_seconds": last_start, "host_dispatches": len(log_rows),
        "updates": updates, "scored_tokens": tokens_processed,
        "tokens_per_second": tokens_processed / training_seconds,
        "processed_original_bytes_attributed": bytes_processed,
        "original_bytes_per_second_attributed": bytes_processed / training_seconds,
        "byte_accounting_note": "CPU batch-prefix sums using tokenizer offset attribution; repeated passes included, not unique bytes",
        "training_domain_byte_shares": {name: int(domain_bytes[index]) / bytes_processed for index, name in enumerate(DOMAINS)},
        "training_language_byte_shares": {name: int(language_bytes[index]) / bytes_processed for index, name in enumerate(LANGUAGES)},
        "training_occupancy": tokens_processed / adaptive["positions"],
        "prepared_occupancy": tokenization["splits"]["train"]["occupancy"],
        "completed_data_passes_by_context": {mode: rows / training[mode]["tokens"].shape[0] for mode, rows in adaptive["rows_used"].items()},
        "final_short_batch": adaptive["short_batch"], "maximum_short_batch_executed": max(row["batch"] for row in log_rows if row["mode"] == "short"),
        "context_rows_consumed": adaptive["rows_used"], "batch_checks": adaptive["checks"],
        "context_transitions": adaptive["transitions"], "data_order_sha256": orders,
        "tokens_by_context": {mode: sum(row["tokens"] for row in log_rows if row["mode"] == mode) for mode in training},
        "updates_by_context": {mode: sum(row["updates"] for row in log_rows if row["mode"] == mode) for mode in training},
        "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "initial_eval": before, "final_eval": after,
        "primary_metric": "FineWiki mean of 16 per-language BPB values at fixed 1024-token context; lower is better",
        "legacy_12_language_macro_bpb": {"initial": before["wiki"]["legacy_12_language_macro_bpb"], "final": after["wiki"]["legacy_12_language_macro_bpb"]},
        "checkpoint_kind": "weights_only", "optimizer_resume_supported": False,
        "checkpoint_save_verify_seconds": checkpoint_seconds, "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": digest, "checkpoint_array_leaves": leaves,
        "memory_stats": memory_stats, "compiled_memory_estimate": compiled_memory,
        "training_memory": training_memory,
        "head": {"kind": "native_dense_exact_softmax", "chunked": False},
        "learning_rate_schedule": {"kind": "five_update_warmup_then_constant", "cooldown": False,
            "warmup_updates": config["warmup_updates"], "peak": config["learning_rate"]},
        "checkpoint_restoration_check": "disabled_by_user", "report_finalized": False,
        "memory_scope": "Training allocator samples and high-water counters; compiler estimate separately",
        "notebook_elapsed_seconds": elapsed,
        "compute_units_estimate": {"rate_per_hour": rate, "training": training_seconds * rate / 3600,
            "through_report": elapsed * rate / 3600, "overhead_through_report": (elapsed - training_seconds) * rate / 3600},
        "cost_note": "2.92 CU/hour supplied by user; connected runtime after report adds cost; no USD conversion assumed",
        "historical_v2_tokens_per_second": 45559.448,
        "historical_comparison_note": "Expanded corpus and 16-language holdout, adaptive batches, mixed training context and 1200 seconds differ from V3.5. No controlled speed/quality claim.",
        "historical_comparison": comparison, "performance_goals": goals,
        "performance_goal_status": improvement,
        "requirements": {key: "PASS" if passed else "FAIL" for key, passed in checks.items()},
        "engineering_status": "PASS" if all(checks.values()) else "FAIL",
        "learning_status": "PASS" if after["wiki"]["language_macro_bpb"] < before["wiki"]["language_macro_bpb"] else "FAIL"}
    write_json(output / "result.json", result)
    write_json(output / "attempt_status.json", {"version": version, "stage": "result saved",
        "timed_training_started": True, "optimizer_updates": updates})
    print(json.dumps({name: result[name] for name in ("engineering_status", "learning_status",
        "performance_goal_status", "updates", "training_seconds", "tokens_per_second",
        "original_bytes_per_second_attributed")}, indent=2), flush=True)
    print(f"FineWiki macro BPB: {before['wiki']['language_macro_bpb']:.6f} -> {after['wiki']['language_macro_bpb']:.6f}", flush=True)
    if not all(checks.values()):
        raise RuntimeError(version + " engineering requirement failed; result.json was saved.")
    # Scores and saved weights are final before disposable diagnostic updates.
    write_json(output / "attempt_status.json", {"version": version, "stage": "profiling",
        "timed_training_started": True, "optimizer_updates": updates})
    diagnostic_shapes = []
    def profile_dispatch(current, ignored, requested):
        mode = "short" if len(diagnostic_shapes) < 2 else "long"
        batch = adaptive["short_batch"] if mode == "short" else 1
        diagnostic_shapes.append({"mode": mode, "batch": batch, "sequence_length": 1024 if mode == "short" else 2048})
        return functions[(mode, batch)](current, training[mode], requested)
    profile = record_diagnostic_profile(state, training, profile_dispatch, output, work,
        config["profile_dispatches"], config["profile_updates_per_dispatch"])
    profile["diagnostic_shapes"] = diagnostic_shapes
    profile["operation_timing_note"] = "Mixed diagnostic: 16 updates at final short batch x1024, then 16 at 1x2048. Category ms/update average these shapes; not per-context timings."
    write_json(output / "profiling.json", profile)
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
    result["performance_goals"]["notebook_seconds_at_most_1800"] = elapsed <= 1800.
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


