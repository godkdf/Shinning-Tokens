"""One timed attempt; compilation during adaptation counts against 1200 seconds."""


def adaptive_train(state, training, prefixes, compile_shape, functions, analyses, config, output):
    log_rows, decisions, transitions, samples = [], [], [], []
    updates = tokens = positions = 0
    rows_used = {"short": 0, "long": 0}
    batch = 1
    previous_tps = None
    held = False
    window = {"updates": 0, "tokens": 0, "dispatch_seconds": 0.}
    maximum_step_seconds = last_progress = last_start = sampling_seconds = 0.
    last_mode = None
    limit = (jax.devices()[0].memory_stats() or {}).get("bytes_limit")
    memory_before = memory_snapshot(jax.devices()[0].memory_stats(), 0.)
    with (output / "training.jsonl").open("x") as stream, (output / "batch_checks.jsonl").open("x") as checks:
        started = time.monotonic()
        try:
            while True:
                remaining = config["training_seconds"] - (time.monotonic() - started)
                if remaining <= 0:
                    break
                mode = context_phase(tokens, batch)
                used_batch, length = (1, 2048) if mode == "long" else (batch, 1024)
                if mode != last_mode:
                    transitions.append({"mode": mode, "tokens": tokens, "update": updates,
                        "elapsed_seconds": time.monotonic() - started, "short_batch": batch,
                        "cycle": tokens // 4000000,
                        "boundary_policy": "first completed update; no partial gradient or discarded targets"})
                    print(f"Context: {mode}, {used_batch} x {length}; tokens={tokens:,}", flush=True)
                    last_mode = mode
                requested = min(config["updates_per_dispatch"], 100 - updates % 100)
                if updates == 0 or remaining < max(2., 2 * requested * maximum_step_seconds):
                    requested = 1
                requested = bounded_dispatch(prefixes[mode]["tokens"], rows_used[mode], used_batch,
                    requested, tokens, batch)
                last_start = time.monotonic() - started
                if last_start >= config["training_seconds"]:
                    break
                begin = time.monotonic()
                pending = functions[(mode, used_batch)](state, training[mode], jnp.array(requested, jnp.int32))
                enqueued = time.monotonic()
                state, values = jax.block_until_ready(pending)
                end_call = time.monotonic()
                metrics = {key: np.asarray(value).tolist() for key, value in values.items()}
                completed = int(metrics["updates"])
                old_rows = rows_used[mode]
                rows_used[mode] += completed * used_batch
                expected = int(completed_prefix(prefixes[mode]["tokens"], rows_used[mode])
                    - completed_prefix(prefixes[mode]["tokens"], old_rows))
                updates += completed
                tokens += int(metrics["tokens"])
                positions += completed * used_batch * length
                elapsed = end_call - started
                row = {**metrics, "update": updates, "total_tokens": tokens,
                    "requested_updates": requested, "mode": mode, "batch": used_batch, "short_batch": batch, "sequence_length": length,
                    "start_seconds": last_start, "end_seconds": elapsed,
                    "dispatch_seconds": end_call - begin, "enqueue_seconds": enqueued - begin,
                    "synchronization_call_seconds": end_call - enqueued,
                    "loss": metrics["nll_sum"] / max(metrics["tokens"], 1),
                    "short_rows": rows_used["short"], "long_rows": rows_used["long"]}
                log_rows.append(row)
                stream.write(json.dumps({k: None if isinstance(v, float) and not math.isfinite(v) else v
                    for k, v in row.items()}, allow_nan=False) + "\n")
                if not metrics["finite"] or completed != requested or expected != int(metrics["tokens"]):
                    raise FloatingPointError("Nonfinite/incomplete update or CPU/device accounting mismatch. No retry.")
                maximum_step_seconds = max(maximum_step_seconds, (end_call - begin) / completed)
                if mode == "short":
                    window["updates"] += completed
                    window["tokens"] += int(metrics["tokens"])
                    window["dispatch_seconds"] += end_call - begin
                if updates % 100 == 0:
                    tps = window["tokens"] / window["dispatch_seconds"] if window["dispatch_seconds"] else None
                    stats_at_check = jax.devices()[0].memory_stats() or {}
                    current_analysis = analyses.get(batch)
                    tracked_arguments = current_analysis["argument_size_in_bytes"] if current_analysis else 0
                    untracked_resident = max(0, stats_at_check.get("bytes_in_use", 0)
                        - tracked_arguments - config["other_resident_bytes_short"])
                    other_bytes = config["other_resident_bytes_short"] + untracked_resident
                    projected = projected_candidate_peak(analyses, batch + 1, other_bytes)
                    check = {"at_update": updates, "tokens": tokens, "short_batch": batch,
                        "current_mode": mode, "short_window": dict(window), "useful_dispatch_tokens_per_second": tps,
                        "previous_batch_dispatch_tokens_per_second": previous_tps,
                        "projected_next_peak_bytes": projected, "device_capacity_bytes": limit,
                        "allocator_snapshot_at_check": memory_snapshot(stats_at_check, time.monotonic() - started),
                        "additional_observed_resident_bytes": untracked_resident,
                        "memory_budget_fraction": config["memory_budget_fraction"],
                        "elapsed_seconds": time.monotonic() - started}
                    if mode != "short" or window["updates"] < 100:
                        decision = {"action": "defer", "batch": batch, "reason": "Need 100 official short-context updates at this batch; long batch stays 1."}
                    elif held:
                        decision = {"action": "hold", "batch": batch, "reason": "Ramp stopped at a prior memory/throughput/cap boundary; continue real training."}
                    else:
                        decision = batch_decision(batch, tps, previous_tps, projected, limit,
                            config["memory_budget_fraction"], config["safety_batch_cap"])
                        if decision["action"] == "compile_candidate":
                            # Do not start a costly compile at the tail of the paid training interval.
                            if config["training_seconds"] - (time.monotonic() - started) < 60.:
                                decision = {"action": "hold", "batch": batch, "reason": "Less than 60 seconds remain; preserve useful training time."}
                            else:
                                candidate = batch + 1
                                analysis = compile_shape("short", candidate, state, "inside_training_budget")
                                actual_peak = compiler_peak(analysis)
                                check["candidate_compiler_peak_plus_other_resident_bytes"] = (
                                    actual_peak + other_bytes if actual_peak is not None else None)
                                if actual_peak is None or actual_peak + other_bytes > limit * config["memory_budget_fraction"]:
                                    decision = {"action": "hold", "batch": batch, "reason": "Candidate executable's memory estimate exceeds budget or is unavailable; never executed."}
                                elif time.monotonic() - started >= config["training_seconds"]:
                                    decision = {"action": "hold", "batch": batch, "reason": "Training budget expired during compilation; candidate never executed."}
                                else:
                                    decision = {"action": "increase", "batch": candidate,
                                        "reason": "Compiler estimate fits the 85% budget; assess throughput using its next real training window."}
                                    previous_tps = tps
                        if decision["action"] == "rollback_and_hold":
                            # Only the shape changes. All completed updates and optimizer moments are retained.
                            batch = decision["batch"]
                            held = True
                        elif decision["action"] == "increase":
                            batch = decision["batch"]
                        elif decision["action"] == "hold":
                            held = True
                        if decision["action"] != "defer":
                            window = {"updates": 0, "tokens": 0, "dispatch_seconds": 0.}
                    check.update(decision)
                    check["decision_complete_seconds"] = time.monotonic() - started
                    if decision["action"] == "hold":
                        window = {"updates": 0, "tokens": 0, "dispatch_seconds": 0.}
                    decisions.append(check)
                    checks.write(json.dumps(check, allow_nan=False) + "\n")
                    checks.flush()
                    stream.flush()
                    print(f"Batch check @{updates:,}: {check['action']}, short batch={batch}; {check['reason']}", flush=True)
                if len(log_rows) == 1 or elapsed - last_progress >= 5.:
                    sample_start = time.monotonic()
                    samples.append(memory_snapshot(jax.devices()[0].memory_stats(), elapsed))
                    sampling_seconds += time.monotonic() - sample_start
                    print(f"updates={updates:,} tokens={tokens:,} batch={used_batch}x{length} loss={row['loss']:.5f} elapsed={elapsed:.2f}s", flush=True)
                    last_progress = elapsed
            jax.block_until_ready(state)
        except Exception:
            stream.flush()
            checks.flush()
            write_json(output / "attempt_status.json", {"version": "V3.9", "stage": "training failed",
                "timed_training_started": True, "optimizer_updates": updates, "tokens": tokens,
                "completed_short_rows": rows_used["short"], "completed_long_rows": rows_used["long"]})
            raise
        seconds = time.monotonic() - started
    after = memory_snapshot(jax.devices()[0].memory_stats(), seconds)
    write_json(output / "context_transitions.json", transitions)
    return state, {"rows": log_rows, "checks": decisions, "transitions": transitions,
        "updates": updates, "tokens": tokens, "positions": positions, "rows_used": rows_used,
        "short_batch": batch, "training_seconds": seconds, "last_start": last_start,
        "memory_before": memory_before, "memory_after": after, "samples": samples,
        "sampling_seconds": sampling_seconds}
