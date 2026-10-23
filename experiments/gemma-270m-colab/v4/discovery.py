"""Independent early discovery for both contexts, then permanent batch locks."""


def context_phase(tokens, short_batch):
    # Both lengths have been measured before this is used for locked training.
    return "long" if tokens % 4000000 >= 3000000 else "short"


def tokens_to_boundary(tokens, short_batch):
    position = tokens % 4000000
    return (3000000 if position < 3000000 else 4000000) - position


def validate_training_config(config):
    if (config["sequence_length"] != 1024 or config["long_sequence_length"] != 2048
            or config["local_batch_per_chip"] != 1 or config["training_seconds"] != 1200.
            or config["short_phase_tokens"] != 3000000 or config["long_phase_tokens"] != 1000000):
        raise ValueError("V4 requires 1024/2048 contexts, the 3M/1M token cycle and 1200 seconds.")
    window, budget = config["discovery_window_updates"], config["discovery_max_updates"]
    if window != 16 or budget != 256 or config["updates_per_dispatch"] != 16:
        raise ValueError("V4 uses 16-update measurement blocks and at most 256 discovery updates.")
    if not 0 < config["memory_budget_fraction"] < 1:
        raise ValueError("Memory admission must leave device headroom.")
    if not isinstance(config["safety_batch_cap"], int) or not 1 <= config["safety_batch_cap"] <= 32:
        raise ValueError("Safety batch cap must be an integer in 1..32.")


class BatchDiscovery:
    def __init__(self, config):
        self.config = config
        self.updates = 0
        self.turn = "short"
        self.finished_at_update = None
        self.contexts = {mode: {"current_batch": 1, "selected_batch": 1, "selected_tps": None,
            "locked": False, "lock_reason": None, "observations": [],
            "window_updates": 0, "window_tokens": 0, "window_seconds": 0.}
            for mode in ("short", "long")}

    def finished(self):
        return all(row["locked"] for row in self.contexts.values())

    def mode(self):
        if self.finished():
            raise ValueError("The batch controller is locked and must not be called again.")
        if self.contexts[self.turn]["locked"]:
            self.turn = "long" if self.turn == "short" else "short"
        return self.turn

    def remaining_in_window(self, mode):
        return min(self.config["discovery_window_updates"] - self.contexts[mode]["window_updates"],
            self.config["discovery_max_updates"] - self.updates)

    def observe(self, mode, metrics, seconds):
        row = self.contexts[mode]
        row["window_updates"] += int(metrics["updates"])
        row["window_tokens"] += int(metrics["tokens"])
        row["window_seconds"] += seconds
        self.updates += int(metrics["updates"])
        return row["window_updates"] == self.config["discovery_window_updates"]

    def lock(self, mode, reason):
        row = self.contexts[mode]
        row["locked"], row["lock_reason"] = True, reason

    def lock_all(self, reason):
        for mode in self.contexts:
            if not self.contexts[mode]["locked"]:
                self.lock(mode, reason)

    def choose(self, mode, state, analyses, compile_shape, remaining_seconds, output, elapsed):
        row = self.contexts[mode]
        batch = row["current_batch"]
        tps = row["window_tokens"] / row["window_seconds"]
        check = {"at_update": int(state.step), "discovery_updates": self.updates, "mode": mode,
            "batch": batch, "window_updates": row["window_updates"], "window_tokens": row["window_tokens"],
            "dispatch_seconds": row["window_seconds"], "tokens_per_second": tps,
            "previous_selected_tokens_per_second": row["selected_tps"], "elapsed_seconds": elapsed}
        improves = row["selected_tps"] is None or tps >= row["selected_tps"] * 1.02
        if improves:
            row["selected_batch"], row["selected_tps"] = batch, tps
        else:
            self.lock(mode, "Larger batch did not improve useful dispatch tokens/s by at least 2%.")
        row["observations"].append({"batch": batch, "updates": row["window_updates"],
            "tokens": row["window_tokens"], "dispatch_seconds": row["window_seconds"],
            "tokens_per_second": tps, "accepted": improves})
        if self.updates >= self.config["discovery_max_updates"]:
            self.lock_all("The 256-update discovery budget is exhausted; retain measured batches.")
        if not row["locked"]:
            stats = jax.devices()[0].memory_stats() or {}
            limit = stats.get("bytes_limit")
            other = self.config["other_resident_bytes_" + mode]
            current = analyses[mode].get(batch)
            untracked = max(0, stats.get("bytes_in_use", 0)
                - (current["argument_size_in_bytes"] if current else 0) - other)
            projected = projected_candidate_peak(analyses[mode], batch + 1, other + untracked)
            check.update(device_capacity_bytes=limit, projected_next_peak_bytes=projected,
                additional_observed_resident_bytes=untracked,
                allocator_snapshot=memory_snapshot(stats, elapsed))
            if batch >= self.config["safety_batch_cap"]:
                self.lock(mode, "Safety batch cap reached; no claim of a global optimum.")
            elif projected is None or limit is None:
                self.lock(mode, "Memory estimate or device capacity unavailable; retain measured batch.")
            elif remaining_seconds < 60.:
                self.lock_all("Under 60 seconds remain in the training wall budget; end discovery.")
            else:
                # Inspect the next compiled shape instead of declaring a maximum
                # from an extrapolation. Compilation performs no optimizer update.
                candidate = batch + 1
                analysis = compile_shape(mode, candidate, state, "inside_training_budget")
                peak = compiler_peak(analysis)
                check["candidate_compiler_peak_plus_other_resident_bytes"] = (
                    peak + other + untracked if peak is not None else None)
                if peak is None or peak + other + untracked > limit * self.config["memory_budget_fraction"]:
                    self.lock(mode, "Candidate executable exceeds the memory budget or has no estimate; never executed.")
                else:
                    # Compiled is not selected: it must complete its own measured window first.
                    row["current_batch"] = candidate
                    check["next_candidate_batch"] = candidate
        check.update(action="lock" if row["locked"] else "increase", selected_batch=row["selected_batch"],
            reason=row["lock_reason"] or "Measure the next admitted batch using real training updates.")
        row["window_updates"] = row["window_tokens"] = 0
        row["window_seconds"] = 0.
        self.turn = "long" if mode == "short" else "short"
        return check

    def report(self):
        return {"maximum_discovery_updates": self.config["discovery_max_updates"],
            "discovery_updates": self.updates, "window_updates": self.config["discovery_window_updates"],
            "both_locked": self.finished(), "finished_at_update": self.finished_at_update,
            "selected_batches": {mode: row["selected_batch"] for mode, row in self.contexts.items()},
            "contexts": self.contexts,
            "scope": "Largest measured reasonable batch within a bounded real-training search; not a proof of the physical OOM limit."}


def adaptive_train(state, training, prefixes, compile_shape, functions, analyses, config, output):
    finder = BatchDiscovery(config)
    decisions, log_rows, transitions, samples = [], [], [], []
    rows_used = {"short": 0, "long": 0}
    updates = tokens = positions = 0
    maximum_step_seconds = last_progress = last_start = sampling_seconds = 0.
    last_context = None
    memory_before = memory_snapshot(jax.devices()[0].memory_stats(), 0.)
    with (output / "training.jsonl").open("x") as stream, (output / "batch_checks.jsonl").open("x") as checks:
        started = time.monotonic()
        try:
            while True:
                elapsed = time.monotonic() - started
                remaining = config["training_seconds"] - elapsed
                if remaining <= 0:
                    break
                phase = "locked_training" if finder.finished() else "discovery"
                batches = {mode: row["selected_batch"] for mode, row in finder.contexts.items()}
                if phase == "discovery":
                    mode = finder.mode()
                    used_batch = finder.contexts[mode]["current_batch"]
                    requested = min(config["updates_per_dispatch"], finder.remaining_in_window(mode))
                    if requested <= 0:
                        finder.lock_all("Discovery update budget reached.")
                        continue
                else:
                    mode = context_phase(tokens, batches["short"])
                    used_batch = batches[mode]
                    requested = config["updates_per_dispatch"]
                length = 1024 if mode == "short" else 2048
                if (phase, mode) != last_context:
                    transitions.append({"phase": phase, "mode": mode, "batch": used_batch,
                        "tokens": tokens, "update": updates, "elapsed_seconds": elapsed,
                        "short_batch": batches["short"], "long_batch": batches["long"], "cycle": tokens // 4000000})
                    print(f"Context: {phase}, {used_batch} x {length}; tokens={tokens:,}", flush=True)
                    last_context = (phase, mode)
                if updates == 0 or remaining < max(2., 2 * requested * maximum_step_seconds):
                    requested = 1
                if phase == "locked_training":
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
                    write_json(output / "batch_discovery.json", finder.report())
                    print(f"Discovery @{updates:,}: {mode} {check['action']}, selected batch={check['selected_batch']}; {check['reason']}", flush=True)
                if finder.finished() and finder.finished_at_update is None:
                    finder.finished_at_update = updates
                    write_json(output / "batch_discovery.json", finder.report())
                    print("BATCHES LOCKED:", finder.report()["selected_batches"], "No further checks.", flush=True)
                if len(log_rows) == 1 or elapsed - last_progress >= 5.:
                    sample_started = time.monotonic()
                    samples.append(memory_snapshot(jax.devices()[0].memory_stats(), elapsed))
                    sampling_seconds += time.monotonic() - sample_started
                    print(f"updates={updates:,} tokens={tokens:,} batch={used_batch}x{length} loss={row['loss']:.5f} elapsed={elapsed:.2f}s", flush=True)
                    last_progress = elapsed
            jax.block_until_ready(state)
        except Exception:
            stream.flush()
            checks.flush()
            write_json(output / "batch_discovery.json", finder.report())
            write_json(output / "attempt_status.json", {"version": "V4", "stage": "training failed",
                "timed_training_started": True, "optimizer_updates": updates, "tokens": tokens,
                "completed_short_rows": rows_used["short"], "completed_long_rows": rows_used["long"]})
            raise
        seconds = time.monotonic() - started
    if not finder.finished():
        finder.lock_all("Training wall-clock budget ended; retain measured selections.")
        finder.finished_at_update = updates
    after = memory_snapshot(jax.devices()[0].memory_stats(), seconds)
    write_json(output / "context_transitions.json", transitions)
    write_json(output / "batch_discovery.json", finder.report())
    batches = finder.report()["selected_batches"]
    return state, {"rows": log_rows, "checks": decisions, "transitions": transitions,
        "updates": updates, "tokens": tokens, "positions": positions, "rows_used": rows_used,
        "short_batch": batches["short"], "long_batch": batches["long"], "discovery": finder.report(),
        "training_seconds": seconds, "last_start": last_start, "memory_before": memory_before,
        "memory_after": after, "samples": samples, "sampling_seconds": sampling_seconds}
