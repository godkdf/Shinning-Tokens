# Shared native-Gemma runtime assembled into both new notebooks.
# This source is not executed by the artifact builder.

TRAIN_KEYS = ("tokens", "targets", "mask", "positions", "segment_ids")


def token_losses(model, params, batch):
    compute = jax.tree.map(lambda value: value.astype(jnp.bfloat16), params)
    logits = model.apply({"params": compute}, tokens=batch["tokens"],
        positions=batch["positions"],
        attention_mask=packed_attention_mask(batch["segment_ids"])).logits.astype(jnp.float32)
    selected = jnp.take_along_axis(logits, batch["targets"][..., None], axis=-1)[..., 0]
    return jax.scipy.special.logsumexp(logits, axis=-1) - selected


def make_training_functions(model, optimizer):
    def one_step(state, batch):
        def objective(params):
            losses = token_losses(model, params, batch)
            count = jnp.sum(batch["mask"], dtype=jnp.float32)
            nll = jnp.sum(jnp.where(batch["mask"], losses, 0.))
            return nll / jnp.maximum(count, 1.), count
        (loss, count), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        norm = optax.global_norm(grads)
        finite = jnp.isfinite(loss) & jnp.isfinite(norm) & (count > 0)
        def apply(_):
            delta, new_opt = optimizer.update(grads, state.opt_state, state.params)
            return state.replace(step=state.step + 1,
                params=optax.apply_updates(state.params, delta), opt_state=new_opt,
                data_position=state.data_position + 1)
        state = jax.lax.cond(finite, apply, lambda _: state, operand=None)
        return state, (loss, count, norm, finite)
    def group(state, training, requested):
        initial = (state, jnp.array(0, jnp.int32), jnp.array(0., jnp.float32),
                   jnp.array(0, jnp.int32), jnp.array(0., jnp.float32), jnp.array(True))
        def condition(carry):
            return (carry[1] < requested) & carry[5]
        def body(carry):
            state, steps, nll, count, max_norm, finite = carry
            index = state.data_position % training["tokens"].shape[0]
            batch = jax.tree.map(lambda values: jax.lax.dynamic_index_in_dim(
                values, index, keepdims=False), training)
            state, (loss, tokens, norm, good) = one_step(state, batch)
            return (state, steps + good.astype(jnp.int32), nll + loss * tokens,
                count + jnp.where(good, tokens, 0).astype(jnp.int32),
                jnp.maximum(max_norm, norm), finite & good)
        state, steps, nll, count, norm, finite = jax.lax.while_loop(condition, body, initial)
        return state, {"updates": steps, "nll_sum": nll, "tokens": count,
                       "max_gradient_norm": norm, "finite": finite}
    return one_step, jax.jit(group, donate_argnums=(0,))


def make_evaluator(model):
    """One executable for both held-out sets; split identity is runtime data."""
    def evaluate(params, arrays):
        initial = (jnp.zeros((2, len(LANGUAGES)), jnp.float32),
                   jnp.zeros((2, len(LANGUAGES)), jnp.float32),
                   jnp.zeros((2, len(DOMAINS)), jnp.float32),
                   jnp.zeros((2, len(DOMAINS)), jnp.float32), jnp.zeros(2, jnp.float32))
        def body(index, totals):
            batch = jax.tree.map(lambda values: jax.lax.dynamic_index_in_dim(
                values, index, keepdims=False), arrays)
            losses = token_losses(model, params, batch)
            nll = jnp.where(batch["text_mask"], losses, 0.).reshape(-1)
            weights = batch["byte_weights"].reshape(-1).astype(jnp.float32)
            language = batch["language_ids"].reshape(-1) + batch["split_id"] * len(LANGUAGES)
            domain = batch["domain_ids"].reshape(-1) + batch["split_id"] * len(DOMAINS)
            ln = jnp.bincount(language, weights=nll, length=2 * len(LANGUAGES)).reshape((2, -1))
            lb = jnp.bincount(language, weights=weights, length=2 * len(LANGUAGES)).reshape((2, -1))
            dn = jnp.bincount(domain, weights=nll, length=2 * len(DOMAINS)).reshape((2, -1))
            db = jnp.bincount(domain, weights=weights, length=2 * len(DOMAINS)).reshape((2, -1))
            eos = jnp.sum(jnp.where(batch["mask"] & ~batch["text_mask"], losses, 0.))
            eos = jax.nn.one_hot(batch["split_id"], 2) * eos
            return tuple(total + value for total, value in zip(totals, (ln, lb, dn, db, eos)))
        return jax.lax.fori_loop(0, arrays["tokens"].shape[0], body, initial)
    return jax.jit(evaluate)


def training_prefix_counts(arrays):
    """CPU preparation only; no per-token source histogram in the training graph."""
    batches = arrays["tokens"].shape[0]
    counts = {"tokens": arrays["mask"].sum(axis=(1, 2), dtype=np.int64)}
    weights = arrays["byte_weights"].reshape(-1)
    for key, field, classes in (("domain_bytes", "domain_ids", len(DOMAINS)),
                                 ("language_bytes", "language_ids", len(LANGUAGES))):
        labels = arrays[field].reshape((batches, -1))
        indices = (labels + np.arange(batches, dtype=np.int64)[:, None] * classes).reshape(-1)
        counts[key] = np.bincount(indices, weights=weights,
            minlength=batches * classes).astype(np.int64).reshape((batches, classes))
    if int(counts["domain_bytes"].sum()) != int(counts["language_bytes"].sum()):
        raise ValueError("Prepared source/language byte accounting differs.")
    return {key: np.concatenate((np.zeros((1, *value.shape[1:]), np.int64),
                np.cumsum(value, axis=0, dtype=np.int64)), axis=0)
            for key, value in counts.items()}


def completed_prefix(prefix, updates):
    batches = prefix.shape[0] - 1
    passes, remainder = divmod(updates, batches)
    return passes * prefix[-1] + prefix[remainder]


def verify_native_protocol():
    started = time.monotonic()
    segments = jnp.array([[0, 0, 1, 1, -1]], jnp.int32)
    actual = np.asarray(jax.jit(packed_attention_mask)(segments))[0]
    expected = np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                         [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]], np.bool_)
    passed = bool(np.array_equal(actual, expected))
    if not passed:
        raise ValueError("Native packed attention mask failed causal/segment/padding verification.")
    return {"passed": passed, "seconds": time.monotonic() - started,
        "optimizer_updates": 0, "attention": "native Gemma", "loss": "native dense exact softmax",
        "custom_kernel_equivalence": "not applicable; no Splash or Tokamax execution",
        "mask_verified": passed}


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
    manifest = verify_raw(Path(config["shared_directory"]) / "raw")
    tokenization = read_json(data_dir / "tokenization.json")
    identity = config["tokenizer_identity"]
    if tokenization["raw_manifest_sha256"] != sha256_file(data_dir / "manifest.json"):
        raise ValueError("Tokenized and raw corpus manifests differ.")
    if shutil.disk_usage(work).free < 3 * 1024**3:
        raise RuntimeError("The weights-only artifact requires at least 3 GiB free local disk.")
    host, orders = {}, {}
    for split in ("train", "in_mix", "wiki"):
        name = split + ".npz"
        if sha256_file(data_dir / name) != tokenization["files"][name]:
            raise ValueError("Prepared input checksum failed: " + name)
        with np.load(data_dir / name, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in INPUT_KEYS}
        if int(arrays["byte_weights"].sum()) != manifest["original_bytes"][split]:
            raise ValueError("Prepared original-byte count differs: " + split)
        host[split], orders[split] = batch_training_arrays(
            arrays, config["local_batch_per_chip"], config["seed"])
    phase = time.monotonic()
    prefix = training_prefix_counts(host["train"])
    write_json(output / "training_accounting.json", {
        "method": "CPU batch prefix sums; completed update cursor; no timed per-token histograms",
        "training_batches": host["train"]["tokens"].shape[0],
        "tokens_per_pass": int(prefix["tokens"][-1]),
        "domain_bytes_per_pass": prefix["domain_bytes"][-1].tolist(),
        "language_bytes_per_pass": prefix["language_bytes"][-1].tolist(),
        "seconds": time.monotonic() - phase})
    training_host = {key: host["train"][key] for key in TRAIN_KEYS}
    eval_host = {key: np.concatenate((host["in_mix"][key], host["wiki"][key]), axis=0)
                 for key in INPUT_KEYS}
    eval_host["split_id"] = np.concatenate((
        np.zeros(host["in_mix"]["tokens"].shape[0], np.int32),
        np.ones(host["wiki"]["tokens"].shape[0], np.int32)))
    input_bytes = sum(value.nbytes for value in training_host.values())
    phase = time.monotonic()
    training, validation = jax.device_put((training_host, eval_host))
    jax.block_until_ready((training, validation))
    input_transfer_seconds = time.monotonic() - phase
    resident = all(jax.devices()[0] in leaf.devices() for leaf in jax.tree.leaves(training))
    del host, arrays, training_host, eval_host
    print(f"Training inputs on TPU: {input_bytes / 1024**2:.1f} MiB; CPU byte accounting prepared.", flush=True)
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
        jax.random.PRNGKey(config["seed"]), jnp.array(0, jnp.int32))
    del params, flat
    jax.block_until_ready(state)
    initialization_seconds = time.monotonic() - phase
    protocol_report = verify_native_protocol()
    write_json(output / "kernel_verification.json", protocol_report)
    _, group_jit = make_training_functions(model, optimizer)
    evaluation_jit = make_evaluator(model)
    print("Compiling one native training loop and one combined held-out evaluator; zero updates...", flush=True)
    phase = time.monotonic()
    executable = group_jit.lower(state, training, jnp.array(0, jnp.int32)).compile()
    evaluator = evaluation_jit.lower(state.params, validation).compile()
    state, warmup_metrics = jax.block_until_ready(group_jit(state, training, jnp.array(0, jnp.int32)))
    if int(warmup_metrics["updates"]) != 0 or int(state.step) != 0 or int(state.data_position) != 0:
        raise ValueError("The zero-update JIT preparation changed optimizer/data state.")
    compilation_seconds = time.monotonic() - phase
    write_json(output / "dispatch_preparation.json", {"dispatch": "jax.jit wrapper",
        "training_warmup_requested_updates": 0, "training_warmup_completed_updates": 0,
        "pretraining_optimizer_step": 0, "pretraining_data_position": 0,
        "training_executables": 1, "evaluation_executables": 1,
        "compiled_objects_used_for": "precompilation and memory inspection only"})
    try:
        analysis = executable.memory_analysis()
        compiled_memory = {name: int(getattr(analysis, name)) for name in
            ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")}
    except (AttributeError, TypeError, NotImplementedError):
        compiled_memory = None
    def evaluate(parameters):
        values = jax.block_until_ready(evaluation_jit(parameters, validation))
        return {split: readable_evaluation(tuple(value[index] for value in values),
                    manifest["original_bytes"][split])
                for index, split in enumerate(("in_mix", "wiki"))}
    phase = time.monotonic()
    before = evaluate(state.params)
    initial_eval_seconds = time.monotonic() - phase
    write_json(output / "timed_training.started.json", {"version": version,
        "stage": "timed training", "training_seconds": config["training_seconds"]})
    write_json(output / "attempt_status.json", {"version": version, "stage": "timed training",
        "timed_training_started": True, "optimizer_updates": None})
    print(f"Initial FineWiki macro BPB={before['wiki']['language_macro_bpb']:.5f}. Starting ONE 60-second {version} run...", flush=True)
    log_rows = []
    updates = tokens_processed = 0
    max_seconds_per_update = last_progress = last_start = 0.
    jax.block_until_ready(state)
    # Preserve completed dispatches even if a later call fails. Buffered log writes
    # contain scalar metrics only; source/language accounting runs after timing.
    with (output / "training.jsonl").open("x") as stream:
        started = time.monotonic()
        try:
            while True:
                remaining = config["training_seconds"] - (time.monotonic() - started)
                if remaining <= 0:
                    break
                group_size = config["updates_per_dispatch"]
                requested = group_size if updates > 0 and remaining > max(
                    2., 2 * group_size * max_seconds_per_update) else 1
                last_start = time.monotonic() - started
                if last_start >= config["training_seconds"]:
                    break
                call_start = time.monotonic()
                state, values = jax.block_until_ready(group_jit(state, training, jnp.array(requested, jnp.int32)))
                metrics = {key: np.asarray(value).tolist() for key, value in values.items()}
                completed = int(metrics["updates"])
                updates += completed
                tokens_processed += int(metrics["tokens"])
                end = time.monotonic() - started
                row = {**metrics, "update": updates, "requested_updates": requested,
                    "start_seconds": last_start, "end_seconds": end,
                    "loss": metrics["nll_sum"] / max(metrics["tokens"], 1)}
                log_rows.append(row)
                safe_row = {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                            for key, value in row.items()}
                stream.write(json.dumps(safe_row, allow_nan=False) + "\n")
                if not metrics["finite"] or completed != requested:
                    stream.flush()
                    raise FloatingPointError("Nonfinite loss/gradient; affected update rejected. No retry.")
                max_seconds_per_update = max(max_seconds_per_update,
                    (time.monotonic() - call_start) / completed)
                if len(log_rows) == 1 or end - last_progress >= 2:
                    print(f"updates={updates:,} tokens={tokens_processed:,} loss={row['loss']:.5f} elapsed={end:.2f}s", flush=True)
                    last_progress = end
            jax.block_until_ready(state)
            training_seconds = time.monotonic() - started
        except Exception:
            stream.flush()
            write_json(output / "attempt_status.json", {"version": version,
                "stage": "training failed", "timed_training_started": True,
                "optimizer_updates": updates, "data_position": int(state.data_position)})
            raise
    if not log_rows:
        raise RuntimeError("No optimizer updates completed.")
    domain_bytes = completed_prefix(prefix["domain_bytes"], updates)
    language_bytes = completed_prefix(prefix["language_bytes"], updates)
    expected_tokens = int(completed_prefix(prefix["tokens"], updates))
    if tokens_processed != expected_tokens:
        raise ValueError("Device token count differs from completed CPU batch prefix.")
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
        "training_batches": training["tokens"].shape[0], "updates": updates,
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
        "016_FIXED_EVALUATION": len(after["wiki"]["languages"]) == 12 and all(
            after[split]["original_utf8_bytes"] == before[split]["original_utf8_bytes"] for split in ("in_mix", "wiki")),
        "017_WEIGHTS_ARTIFACT": schema_verified,
        "018_CHANGED_WEIGHTS": changed,
        "019_MINIMUM_UPDATES": updates >= config["minimum_updates"],
        "020_BYTE_ACCOUNTING": bytes_processed == int(language_bytes.sum()) and bytes_processed > 0 and expected_tokens == tokens_processed,
        "025_NATIVE_TRAINING": config["local_batch_per_chip"] == 2 and config["sequence_length"] == 1024,
        "026_COMBINED_EVALUATOR": validation["split_id"].shape[0] == validation["tokens"].shape[0],
        "027_CPU_ACCOUNTING": "byte_weights" not in training and "domain_ids" not in training and "language_ids" not in training,
        "028_WEIGHTS_ONLY": saved["metadata"]["optimizer_resume_supported"] is False,
    }
    checks = {prefix_id + "-" + key: value for key, value in checks.items()}
    checks.update({key: value == "PASS" for key, value in read_json(output / "preparation_requirements.json").items()})
    downloads = read_json(output / "parallel_downloads.json")
    elapsed = time.monotonic() - config["notebook_start_monotonic"]
    rate = config["compute_units_per_hour"]
    baseline_applicable = (identity["tokenizer_sha256"] == "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c"
        and sha256_file(data_dir / "manifest.json") == "74d6d6305b893aa9403e8ac30c10b9d920c23bc7d3c9d020debbd5b3500d707a")
    goals = {"tokens_per_second_at_least_45559": tokens_processed / training_seconds >= 45559.448,
        "compilation_seconds_at_most_60": compilation_seconds <= 60.,
        "notebook_seconds_at_most_300": elapsed <= 300.,
        "occupancy_at_least_95_percent": tokens_processed / (updates * 2 * 1024) >= .95,
        "finewiki_macro_bpb_at_most_v34": after["wiki"]["language_macro_bpb"] <= 2.782397528489431
            if baseline_applicable else None}
    improvement = ("PASS" if all(value is True for value in goals.values()) else "FAIL") if baseline_applicable else "NOT_COMPARABLE"
    result = {"version": version, "protocol": PROTOCOL, "run_count": 1, "config": config,
        "device": environment["device_kind"], "model_parameters": parameter_count, "optimizer": partition,
        "source_downloads": {key: downloads[key] for key in
            ("source_jobs", "max_workers", "peak_active_source_jobs", "fetch_seconds", "raw_cache_reused")},
        "initial_backbone_sha256": backbone_digest, "input_transfer_seconds": input_transfer_seconds,
        "resident_training_bytes": input_bytes, "initialization_seconds": initialization_seconds,
        "kernel_verification_seconds": protocol_report["seconds"], "compilation_seconds": compilation_seconds,
        "compilation_timing_scope": "one training and one combined evaluation executable plus zero-update JIT dispatch",
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
        "training_occupancy": tokens_processed / (updates * 2 * 1024),
        "prepared_occupancy": tokenization["splits"]["train"]["occupancy"],
        "completed_data_passes": updates // training["tokens"].shape[0],
        "training_batches": training["tokens"].shape[0], "data_order_sha256": orders,
        "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "initial_eval": before, "final_eval": after,
        "primary_metric": "FineWiki mean of 12 per-language BPB values; lower is better",
        "checkpoint_kind": "weights_only", "optimizer_resume_supported": False,
        "checkpoint_save_verify_seconds": checkpoint_seconds, "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": digest, "checkpoint_array_leaves": leaves,
        "memory_stats": memory_stats, "compiled_memory_estimate": compiled_memory,
        "memory_scope": "Allocator snapshot after evaluation; compiler estimate separately",
        "notebook_elapsed_seconds": elapsed,
        "compute_units_estimate": {"rate_per_hour": rate, "training": training_seconds * rate / 3600,
            "through_report": elapsed * rate / 3600, "overhead_through_report": (elapsed - training_seconds) * rate / 3600},
        "cost_note": "2.92 CU/hour supplied by user; connected runtime after report adds cost; no USD conversion assumed",
        "historical_v2_tokens_per_second": 45559.448,
        "historical_comparison_note": "V2 used different data and vocabulary; V3.4 used different kernels/batch/optimizer; neither isolates an optimizer effect",
        "v34_baseline_applicable": baseline_applicable, "performance_goals": goals,
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
