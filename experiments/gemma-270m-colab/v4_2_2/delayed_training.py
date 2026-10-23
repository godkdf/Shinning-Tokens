"""Fixed first 1000 updates, bounded discovery, then permanent batch locks."""


class BatchMemoryError(RuntimeError):
    """A compiled batch exceeds the configured memory admission budget."""


def is_memory_error(error):
    if isinstance(error, (MemoryError, BatchMemoryError)):
        return True
    message = str(error).lower()
    return any(marker in message for marker in ("resource_exhausted", "out of memory", "outofmemory",
        "failed to allocate", "memory allocation", "bad_alloc"))


def memory_discovery_fallback(config, mode, batch, error, stage, update):
    if not is_memory_error(error):
        raise error
    if batch <= 1:
        raise RuntimeError(f"Memory failure at 1x{1024 if mode == 'short' else 2048}; batch discovery cannot reduce below one sequence.") from error
    config["early_memory_discovery"] = True
    config.setdefault("discovery_initial_batches", {"short": 3, "long": 1})[mode] = 1
    config.setdefault("discovery_memory_caps", {})[mode] = batch - 1
    config.setdefault("memory_discovery_events", []).append({"mode": mode, "failed_batch": batch,
        "fallback_batch": 1, "stage": stage, "at_update": update,
        "error_type": type(error).__name__, "error": str(error)})
    print(f"MEMORY FALLBACK: {mode} batch {batch} -> 1; start discovery early, excluding failed batch {batch}.", flush=True)


def verify_surviving_state(state, updates, rows_used):
    if any(getattr(leaf, "is_deleted", lambda: False)() for leaf in jax.tree.leaves(state)):
        raise RuntimeError("Memory failure invalidated donated training buffers; cannot safely retry. No weights reset or checkpoint restore.")
    jax.block_until_ready(state)
    if (int(state.step) != updates or int(state.short_rows) != rows_used["short"]
            or int(state.long_rows) != rows_used["long"] or int(state.data_position) != updates):
        raise RuntimeError("Training state is not the last committed update after memory failure; recovery stopped.")


def validate_training_config(config):
    if (config["sequence_length"] != 1024 or config["long_sequence_length"] != 2048
            or config["local_batch_per_chip"] != 3 or config["long_batch_per_chip"] != 1
            or config["training_seconds"] != 1200. or config["updates_per_dispatch"] != 16
            or config["short_phase_tokens"] != 3000000 or config["long_phase_tokens"] != 1000000
            or config["batch_discovery_enabled"] is not True
            or config["discovery_start_after_updates"] != 1000
            or config["discovery_window_updates"] != 16 or config["discovery_max_updates"] != 256
            or config["safety_batch_cap"] != 32):
        raise ValueError("V4.2.2 starts at 3x1024 / 1x2048; discovery follows 1000 updates, for at most 256 updates.")
    if not 0 < config["memory_budget_fraction"] < 1:
        raise ValueError("Memory admission must leave device headroom.")


def training_specification_matches(config, training):
    return (config["local_batch_per_chip"] == 3 and config["long_batch_per_chip"] == 1
        and config["training_seconds"] == 1200. and config["batch_discovery_enabled"] is True
        and config["discovery_start_after_updates"] == 1000
        and training["short"]["tokens"].shape[1:] == (1024,)
        and training["long"]["tokens"].shape[1:] == (2048,))


def delayed_train(state, training, prefixes, functions, analyses, compile_shape, config, output):
    initial_batches = {"short": 3, "long": 1}
    threshold = config["discovery_start_after_updates"]
    finder = None
    started_at_update = None
    log_rows, decisions, transitions, samples = [], [], [], []
    rows_used = {"short": 0, "long": 0}
    updates = tokens = positions = 0
    maximum_step_seconds = last_progress = last_start = sampling_seconds = 0.
    last_context = None

    def policy():
        report = finder.report() if finder is not None else None
        return {"mode": "delayed_discovery", "discovery_enabled": True,
            "discovery_start_after_updates": threshold,
            "discovery_started": finder is not None,
            "started_at_update": started_at_update,
            "initial_batches": initial_batches,
            "selected_batches": report["selected_batches"] if report else dict(initial_batches),
            "early_memory_trigger": bool(config.get("early_memory_discovery")),
            "memory_events": config.get("memory_discovery_events", []),
            "discovery": report,
            "context_schedule": "3M short / 1M long cumulative tokens; discovery temporarily measures both contexts."}

    write_json(output / "batch_policy.json", policy())
    print(f"Initial batches 3x1024 / 1x2048; discovery starts after {threshold:,} completed updates.", flush=True)
    memory_before = memory_snapshot(jax.devices()[0].memory_stats(), 0.)
    with (output / "training.jsonl").open("x") as stream, (output / "batch_checks.jsonl").open("x") as checks:
        started = time.monotonic()
        try:
            while True:
                elapsed = time.monotonic() - started
                remaining = config["training_seconds"] - elapsed
                if remaining <= 0:
                    break
                if finder is None and (updates >= threshold or config.get("early_memory_discovery")):
                    if updates != threshold and not config.get("early_memory_discovery"):
                        raise ValueError("A dispatch crossed the exact discovery boundary.")
                    finder = BatchDiscovery(config)
                    started_at_update = updates
                    write_json(output / "batch_policy.json", policy())
                    print(f"BATCH DISCOVERY START: {updates:,} completed updates; same model, Adam state and data cursors.", flush=True)
                phase = "initial_batches" if finder is None else ("locked_training" if finder.finished() else "discovery")
                batches = dict(initial_batches) if finder is None else finder.report()["selected_batches"]
                requested = config["updates_per_dispatch"]
                if phase == "discovery":
                    mode = finder.mode()
                    used_batch = finder.contexts[mode]["current_batch"]
                    requested = min(requested, finder.remaining_in_window(mode))
                    if requested <= 0:
                        finder.lock_all("Discovery update budget reached.")
                        continue
                else:
                    mode = context_phase(tokens, batches["short"])
                    used_batch = batches[mode]
                    if finder is None:
                        requested = min(requested, threshold - updates)
                length = 1024 if mode == "short" else 2048
                if (phase, mode) != last_context:
                    transitions.append({"phase": phase, "mode": mode, "batch": used_batch,
                        "tokens": tokens, "update": updates, "elapsed_seconds": elapsed,
                        "short_batch": batches["short"], "long_batch": batches["long"], "cycle": tokens // 4000000})
                    print(f"Context: {phase}, {used_batch} x {length}; tokens={tokens:,}", flush=True)
                    last_context = (phase, mode)
                if updates == 0 or remaining < max(2., 2 * requested * maximum_step_seconds):
                    requested = 1
                if phase != "discovery":
                    requested = bounded_dispatch(prefixes[mode]["tokens"], rows_used[mode], used_batch,
                        requested, tokens, batches["short"])
                last_start = time.monotonic() - started
                if last_start >= config["training_seconds"]:
                    break
                begin = time.monotonic()
                try:
                    pending = functions[(mode, used_batch)](state, training[mode], jnp.array(requested, jnp.int32))
                    enqueued = time.monotonic()
                    next_state, values = jax.block_until_ready(pending)
                except Exception as error:
                    if finder is not None or not is_memory_error(error):
                        raise
                    verify_surviving_state(state, updates, rows_used)
                    memory_discovery_fallback(config, mode, used_batch, error, "initial_training", updates)
                    write_json(output / "batch_policy.json", policy())
                    compile_shape(mode, 1, state, "inside_training_budget")
                    continue
                state = next_state
                end_call = time.monotonic()
                metrics = {key: np.asarray(value).tolist() for key, value in values.items()}
                completed = int(metrics["updates"])
                previous_rows = rows_used[mode]
                rows_used[mode] += completed * used_batch
                expected = int(completed_prefix(prefixes[mode]["tokens"], rows_used[mode])
                    - completed_prefix(prefixes[mode]["tokens"], previous_rows))
                updates += completed
                tokens += int(metrics["tokens"])
                positions += completed * used_batch * length
                elapsed = end_call - started
                row = {**metrics, "update": updates, "total_tokens": tokens, "phase": phase,
                    "requested_updates": requested, "mode": mode, "batch": used_batch,
                    "short_batch": batches["short"], "long_batch": batches["long"], "sequence_length": length,
                    "start_seconds": last_start, "end_seconds": elapsed, "dispatch_seconds": end_call - begin,
                    "enqueue_seconds": enqueued - begin, "synchronization_call_seconds": end_call - enqueued,
                    "loss": metrics["nll_sum"] / max(metrics["tokens"], 1),
                    "short_rows": rows_used["short"], "long_rows": rows_used["long"]}
                log_rows.append(row)
                stream.write(json.dumps({key: None if isinstance(value, float) and not math.isfinite(value) else value
                    for key, value in row.items()}, allow_nan=False) + "\n")
                if not metrics["finite"] or completed != requested or expected != int(metrics["tokens"]):
                    raise FloatingPointError("Nonfinite/incomplete update or CPU/device accounting mismatch. No retry.")
                maximum_step_seconds = max(maximum_step_seconds, (end_call - begin) / completed)
                if phase == "discovery" and finder.observe(mode, metrics, end_call - begin):
                    check = finder.choose(mode, state, analyses, compile_shape,
                        config["training_seconds"] - elapsed, output, elapsed)
                    check["decision_complete_seconds"] = time.monotonic() - started
                    decisions.append(check)
                    checks.write(json.dumps(check, allow_nan=False) + "\n")
                    checks.flush()
                    stream.flush()
                    print(f"Discovery @{updates:,}: {mode} {check['action']}, selected batch={check['selected_batch']}; {check['reason']}", flush=True)
                    write_json(output / "batch_policy.json", policy())
                if finder is not None and finder.finished() and finder.finished_at_update is None:
                    finder.finished_at_update = updates
                    write_json(output / "batch_policy.json", policy())
                    print("BATCHES LOCKED:", finder.report()["selected_batches"], "No further discovery.", flush=True)
                if len(log_rows) == 1 or elapsed - last_progress >= 5.:
                    stream.flush()
                    sample_started = time.monotonic()
                    samples.append(memory_snapshot(jax.devices()[0].memory_stats(), elapsed))
                    sampling_seconds += time.monotonic() - sample_started
                    print(f"updates={updates:,} tokens={tokens:,} batch={used_batch}x{length} loss={row['loss']:.5f} elapsed={elapsed:.2f}s", flush=True)
                    last_progress = elapsed
            jax.block_until_ready(state)
        except Exception:
            stream.flush()
            checks.flush()
            write_json(output / "batch_policy.json", policy())
            write_json(output / "attempt_status.json", {"version": "V4.2.2", "stage": "training failed",
                "timed_training_started": True, "optimizer_updates": updates, "tokens": tokens,
                "completed_short_rows": rows_used["short"], "completed_long_rows": rows_used["long"]})
            raise
        seconds = time.monotonic() - started
    if finder is not None and not finder.finished():
        finder.lock_all("Training wall-clock budget ended; retain measured selections.")
        finder.finished_at_update = updates
    after = memory_snapshot(jax.devices()[0].memory_stats(), seconds)
    write_json(output / "context_transitions.json", transitions)
    final_policy = policy()
    write_json(output / "batch_policy.json", final_policy)
    batches = final_policy["selected_batches"]
    return state, {"rows": log_rows, "checks": decisions, "transitions": transitions,
        "updates": updates, "tokens": tokens, "positions": positions, "rows_used": rows_used,
        "short_batch": batches["short"], "long_batch": batches["long"], "batch_policy": final_policy,
        "training_seconds": seconds, "last_start": last_start, "memory_before": memory_before,
        "memory_after": after, "samples": samples, "sampling_seconds": sampling_seconds}
