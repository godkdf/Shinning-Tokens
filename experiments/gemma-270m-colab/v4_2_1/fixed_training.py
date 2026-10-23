"""Only the two user-selected batch shapes; no discovery or training-time compilation."""


def validate_training_config(config):
    if (config["sequence_length"] != 1024 or config["long_sequence_length"] != 2048
            or config["local_batch_per_chip"] != 3 or config["long_batch_per_chip"] != 1
            or config["training_seconds"] != 1200. or config["updates_per_dispatch"] != 16
            or config["short_phase_tokens"] != 3000000 or config["long_phase_tokens"] != 1000000
            or config["batch_discovery_enabled"] is not False):
        raise ValueError("V4.2.1 requires fixed 3x1024 / 1x2048, a 3M/1M token cycle and 1200 seconds.")
    if not 0 < config["memory_budget_fraction"] < 1:
        raise ValueError("Memory admission must leave device headroom.")


def training_specification_matches(config, training):
    return (config["local_batch_per_chip"] == 3 and config["long_batch_per_chip"] == 1
        and config["training_seconds"] == 1200. and config["batch_discovery_enabled"] is False
        and training["short"]["tokens"].shape[1:] == (1024,)
        and training["long"]["tokens"].shape[1:] == (2048,))


def fixed_train(state, training, prefixes, functions, config, output):
    batches = {"short": 3, "long": 1}
    policy = {"mode": "fixed", "selected_batches": batches, "discovery_enabled": False,
        "candidate_compilation": False, "training_time_compilation": False,
        "context_schedule": "Repeat 3M short / 1M long cumulative scored tokens from token zero."}
    write_json(output / "batch_policy.json", policy)
    print("FIXED BATCHES: 3 x 1024 / 1 x 2048; no discovery or batch changes.", flush=True)
    log_rows, transitions, samples = [], [], []
    rows_used = {"short": 0, "long": 0}
    updates = tokens = positions = 0
    maximum_step_seconds = last_progress = last_start = sampling_seconds = 0.
    last_context = None
    memory_before = memory_snapshot(jax.devices()[0].memory_stats(), 0.)
    with (output / "training.jsonl").open("x") as stream:
        started = time.monotonic()
        try:
            while True:
                elapsed = time.monotonic() - started
                remaining = config["training_seconds"] - elapsed
                if remaining <= 0:
                    break
                mode = context_phase(tokens, batches["short"])
                used_batch = batches[mode]
                length = 1024 if mode == "short" else 2048
                requested = config["updates_per_dispatch"]
                if mode != last_context:
                    transitions.append({"phase": "fixed_batches", "mode": mode, "batch": used_batch,
                        "tokens": tokens, "update": updates, "elapsed_seconds": elapsed,
                        "short_batch": 3, "long_batch": 1, "cycle": tokens // 4000000})
                    print(f"Context: fixed {used_batch} x {length}; tokens={tokens:,}", flush=True)
                    last_context = mode
                if updates == 0 or remaining < max(2., 2 * requested * maximum_step_seconds):
                    requested = 1
                requested = bounded_dispatch(prefixes[mode]["tokens"], rows_used[mode], used_batch,
                    requested, tokens, batches["short"])
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
                previous_rows = rows_used[mode]
                rows_used[mode] += completed * used_batch
                expected = int(completed_prefix(prefixes[mode]["tokens"], rows_used[mode])
                    - completed_prefix(prefixes[mode]["tokens"], previous_rows))
                updates += completed
                tokens += int(metrics["tokens"])
                positions += completed * used_batch * length
                elapsed = end_call - started
                row = {**metrics, "update": updates, "total_tokens": tokens, "phase": "fixed_batches",
                    "requested_updates": requested, "mode": mode, "batch": used_batch,
                    "short_batch": 3, "long_batch": 1, "sequence_length": length,
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
            write_json(output / "attempt_status.json", {"version": "V4.2.1", "stage": "training failed",
                "timed_training_started": True, "optimizer_updates": updates, "tokens": tokens,
                "completed_short_rows": rows_used["short"], "completed_long_rows": rows_used["long"]})
            raise
        seconds = time.monotonic() - started
    after = memory_snapshot(jax.devices()[0].memory_stats(), seconds)
    write_json(output / "context_transitions.json", transitions)
    return state, {"rows": log_rows, "checks": [], "transitions": transitions,
        "updates": updates, "tokens": tokens, "positions": positions, "rows_used": rows_used,
        "short_batch": 3, "long_batch": 1, "batch_policy": policy,
        "training_seconds": seconds, "last_start": last_start, "memory_before": memory_before,
        "memory_after": after, "samples": samples, "sampling_seconds": sampling_seconds}
