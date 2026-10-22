"""XProf measured-metric parsing; compiler estimates are reported separately.

Embedded into runner.py. Pure JSON parsers also support local fixture checks.
"""

def _number(value):
    if isinstance(value, dict):
        value = value.get("v")
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(str(value).rstrip("%"))
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _normal_key(key):
    return "".join(c for c in str(key).lower() if c.isalnum())


def _property_values(data):
    """Preserve explicit zero; never turn a missing counter into zero."""
    result = {}
    def visit(node):
        if isinstance(node, dict):
            for key, value in node.items():
                number = _number(value)
                if number is not None:
                    result.setdefault(_normal_key(key), number)
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(data)
    return result


def extract_utilization(overview):
    props = _property_values(overview)
    aliases = {
        "matrix_unit_utilization": ("mxu_utilization_percent", "matrix_unit_utilization_percent"),
        "memory_bandwidth_utilization": ("hbm_bw_utilization_percent", "hbm_utilization_percent"),
        "host_idle_time": ("host_idle_time_percent",),
        "device_idle_time": ("device_idle_time_percent",),
    }
    metrics = {}
    for name, keys in aliases.items():
        found = next(((key, props[_normal_key(key)]) for key in keys
                      if _normal_key(key) in props), None)
        valid = found is not None and 0 <= found[1] <= 100
        metrics[name] = {"value": found[1] if valid else None, "unit": "percent",
            "status": "available" if valid else "unavailable",
            "source": "XProf overview_page",
            "source_field": found[0] if valid else None,
            "reason": None if valid else "XProf did not provide a supported percent counter; no FLOP or occupancy proxy used."}
    return metrics


def _datatable_rows(data):
    rows = []
    def visit(node):
        if isinstance(node, dict):
            if isinstance(node.get("cols"), list) and isinstance(node.get("rows"), list):
                columns = [(col.get("id") or col.get("label") or str(i), col.get("label", ""))
                           for i, col in enumerate(node["cols"])]
                for row in node["rows"]:
                    item = {}
                    for (key, label), cell in zip(columns, row.get("c", [])):
                        value = cell.get("v") if isinstance(cell, dict) else cell
                        item[_normal_key(key)] = value
                        if label:
                            item.setdefault(_normal_key(label), value)
                    rows.append(item)
            for value in node.values():
                if isinstance(value, (dict, list)):
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(data)
    return rows


def operation_category(name):
    text = str(name).lower()
    if "optimizer_update" in text or "adam" in text or "optax" in text:
        return "optimizer_updates"
    if "vocabulary_projection" in text or ("embedder" in text and "dot_general" in text) or "/decode/" in text:
        return "vocabulary_projection"
    if "cross_entropy" in text or "logsumexp" in text:
        return "cross_entropy"
    if any(key in text for key in ("attention", "/attn/", "q_einsum", "kv_einsum", "attn_vec")):
        return "attention"
    if any(key in text for key in ("mlp", "feedforward", "gating_einsum", "linear_einsum")):
        return "mlp"
    return "other_or_fused"


def extract_operation_timings(hlo_stats):
    """Use exclusive HLO self-time only, never sum overlapping parent scopes."""
    time_keys = ("selftimeus", "selftimeinus", "totalselftimeus", "totalselftimeinus")
    name_keys = ("frameworkopname", "frameworkop", "tfopname", "opname", "hloname",
                 "hloopname", "hloexpression", "hloopexpression", "hlooptext", "expression")
    rows = []
    for row in _datatable_rows(hlo_stats):
        duration = next((_number(row[k]) for k in time_keys if k in row and _number(row[k]) is not None), None)
        names = [str(row[k]) for k in name_keys if row.get(k)]
        if duration is None or duration < 0 or not names:
            continue
        name = " | ".join(dict.fromkeys(names))
        rows.append({"name": name, "self_time_us": duration, "category": operation_category(name)})
    categories = {key: 0.0 for key in ("attention", "vocabulary_projection", "cross_entropy",
                                      "optimizer_updates", "mlp", "other_or_fused")}
    for row in rows:
        categories[row["category"]] += row["self_time_us"]
    total = sum(categories.values())
    return {"status": "available" if rows else "unavailable",
        "reason": None if rows else "XProf did not expose supported exclusive HLO timing columns; inspect the saved HLO/trace artifacts.",
        "source": "XProf hlo_stats", "unit": "microseconds",
        "scope": "Summed HLO self-time over profiled device cores; includes forward/backward; excludes idle. Not elapsed wall time.",
        "classification": "Named HLO metadata; fused/unrecognized operations remain other_or_fused.",
        "categories": {key: {"self_time_us": value if any(row["category"] == key for row in rows) else None,
                               "percent_of_accounted_op_time": 100 * value / total
                                   if total > 0 and any(row["category"] == key for row in rows) else None}
                       for key, value in categories.items()},
        "top_operations": sorted(rows, key=lambda row: row["self_time_us"], reverse=True)[:50]}


def _decode_tool_json(payload):
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload) if isinstance(payload, str) else payload


def historical_comparison(config, tokenizer_hash, corpus_hash, throughput, bpb):
    same_data = (tokenizer_hash == "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c"
        and corpus_hash == "74d6d6305b893aa9403e8ac30c10b9d920c23bc7d3c9d020debbd5b3500d707a")
    return {"reference_version": "V3.5", "reference_tokens_per_second": 45734.29605429257,
        "reference_finewiki_macro_bpb": 2.9347027242183685,
        "reference_batch": 2, "reference_training_seconds": 60.014611282999795,
        "current_batch": config["local_batch_per_chip"], "current_budget_seconds": config["training_seconds"],
        "same_tokenizer_and_corpus": same_data,
        "controlled_same_budget_comparison": False,
        "throughput_ratio_to_reference": throughput / 45734.29605429257 if same_data else None,
        "bpb_difference_from_reference": bpb - 2.9347027242183685 if same_data else None,
        "note": "Descriptive historical comparison only. Batch and training duration changed; BPB does not isolate implementation efficiency."}


def extract_roofline_estimates(roofline):
    """Use the aggregate Program row and its own peak units, never sum rates."""
    tables = roofline if isinstance(roofline, list) else [roofline]
    table = next((x for x in tables if isinstance(x, dict) and "cols" in x
                  and "peak_flop_rate" in x.get("p", {})), None)
    missing = {"status": "unavailable", "kind": "compiler_cost_model_estimate",
        "reason": "No supported aggregate Program row and hardware peaks in roofline export."}
    if table is None:
        return missing
    row = next((r for r in _datatable_rows(table) if r.get("category") == "Program"
                and r.get("operation") == "Program" and r.get("step") == "Total"), None)
    if row is None:
        return missing
    flop = _number(row.get("modelfloprate"))
    hbm = _number(row.get("hbmbw"))
    peak_flop = _number(table["p"].get("peak_flop_rate"))
    peak_hbm = _number(table["p"].get("peak_hbm_bw"))
    if any(x is None or x <= 0 for x in (peak_flop, peak_hbm)) or flop is None or hbm is None:
        return missing
    return {"status": "available", "kind": "compiler_cost_model_estimate", "source": "XProf roofline_model aggregate Program row",
        "estimated_compute_tflops_per_second": flop / 1000., "peak_compute_tflops_per_second": peak_flop / 1000.,
        "estimated_compute_percent_of_peak": 100 * flop / peak_flop,
        "estimated_hbm_gib_per_second": hbm, "peak_hbm_gib_per_second": peak_hbm,
        "estimated_hbm_percent_of_peak": 100 * hbm / peak_hbm,
        "estimated_bound_by": row.get("boundby"),
        "note": "Compiler FLOP/byte costs divided by observed duration. Not hardware-counter measurements or MXU busy time. Estimates may exceed physical peak; never clamp them."}


def raw_counter_evidence(profile_data):
    rows = []
    if profile_data is not None:
        for plane in profile_data.planes:
            for line in plane.lines:
                if "counter" in line.name.lower():
                    rows.append({"plane": plane.name, "line": line.name,
                                 "event_count": sum(1 for _ in line.events)})
    return {"tracks": rows, "counter_events": sum(row["event_count"] for row in rows)}


def profiling_requirements(metrics, training_memory):
    def available(name):
        return metrics.get(name, {}).get("status") == "available"
    ops = metrics.get("operation_timings", {})
    categories = ops.get("categories", {})
    checks = {
        "matrix_unit_utilization": available("matrix_unit_utilization"),
        "memory_bandwidth_utilization": available("memory_bandwidth_utilization"),
        "host_and_device_idle_time": available("host_idle_time") and available("device_idle_time"),
        "operation_timings": available("operation_timings") and all(
            categories.get(key, {}).get("self_time_us") is not None
            for key in ("attention", "vocabulary_projection", "optimizer_updates")),
        "training_memory": training_memory.get("status") == "available",
    }
    count = sum(checks.values())
    return {"checks": {k: "PASS" if v else "FAIL" for k, v in checks.items()},
        "requirements_status": "PASS" if all(checks.values()) else "FAIL",
        "telemetry_status": "COMPLETE" if all(checks.values()) else "PARTIAL" if count else "UNAVAILABLE",
        "available_requirement_groups": count, "required_groups": len(checks)}


def summarize_xprof(trace_dir, output, profile_data, completed_updates):
    files = sorted(Path(trace_dir).rglob("*.xplane.pb")) + sorted(Path(trace_dir).rglob("*.xplane.riegeli"))
    raw, errors = {}, {}
    if files:
        try:
            from xprof.convert import raw_to_tool_data
            # Reuse OpStats created by the first conversion; do not rebuild it
            # four times or generate the redundant op_profile/Perfetto exports.
            for index, tool in enumerate(("overview_page", "hlo_stats", "roofline_model")):
                try:
                    payload, _ = raw_to_tool_data.xspace_to_tool_data(
                        [str(p) for p in files], tool, {"use_saved_result": index > 0})
                    if payload is None:
                        raise ValueError("XProf returned no tool data")
                    raw[tool] = _decode_tool_json(payload)
                    write_json(output / "profile_tools" / (tool + ".json"), raw[tool])
                except Exception as error:
                    errors[tool] = type(error).__name__ + ": " + str(error)
        except Exception as error:
            errors["converter"] = type(error).__name__ + ": " + str(error)
    else:
        errors["capture"] = "No raw TPU XPlane files were produced."
    metrics = extract_utilization(raw.get("overview_page", {}))
    evidence = {"hardware_counter_message_present": False, "device_timeline_present": False,
                "device_type": None, "raw_counters": None}
    try:
        if profile_data is None and files and files[0].suffix == ".pb":
            profile_data = jax.profiler.ProfileData.from_file(str(files[0]))
        evidence["raw_counters"] = raw_counter_evidence(profile_data)
    except Exception as error:
        errors["raw_counter_audit"] = type(error).__name__ + ": " + str(error)
    try:
        from xprof.protobuf import op_stats_pb2
        candidates = sorted(Path(trace_dir).rglob("ALL_HOSTS.op_stats*.pb"))
        if candidates:
            stats = op_stats_pb2.OpStats()
            stats.ParseFromString(candidates[-1].read_bytes())
            evidence["device_type"] = stats.run_environment.device_type
            tpu = "TPU" in evidence["device_type"].upper()
            evidence["device_timeline_present"] = tpu and stats.device_op_metrics_db.total_time_ps > 0
            evidence["hardware_counter_message_present"] = tpu and stats.HasField("performance_counter_result")
            if evidence["hardware_counter_message_present"]:
                for name, field in (("matrix_unit_utilization", "matrix_unit_utilization_percent"),
                                    ("memory_bandwidth_utilization", "hbm_utilization_percent")):
                    value = float(getattr(stats.performance_counter_result, field))
                    if math.isfinite(value) and 0 <= value <= 100:
                        metrics[name].update(value=value, status="available", reason=None,
                            source="XProf OpStats.performance_counter_result", source_field=field)
    except Exception as error:
        errors["counter_evidence"] = type(error).__name__ + ": " + str(error)
    for name in ("matrix_unit_utilization", "memory_bandwidth_utilization"):
        if not evidence["hardware_counter_message_present"]:
            metrics[name].update(value=None, status="unavailable", source_field=None,
                reason="No supported TPU utilization-counter message was exported. Default zeros are not measurements; inspect raw-counter audit. Runtime cause is not established.")
    if not evidence["device_timeline_present"]:
        metrics["device_idle_time"].update(value=None, status="unavailable", source_field=None,
            reason="No TPU device timeline supports the overview idle percentage.")
    metrics["device_idle_time"]["scope"] = "XProf device-operation timeline idle; not unused MXU capacity inside active operations."
    reported_host_idle = metrics["host_idle_time"].get("value")
    metrics["host_idle_time"].update(value=None, status="unavailable",
        reason="XProf host-op idle accounting does not establish Python/CPU idle or blocked dispatch time. See separately measured host wait and dispatch gaps.")
    metrics["host_idle_time"]["xprof_host_op_idle_percent_context_only"] = reported_host_idle
    ops = extract_operation_timings(raw.get("hlo_stats", {}))
    for category in ops["categories"].values():
        duration = category["self_time_us"]
        category["milliseconds_per_update"] = duration / (1000 * completed_updates) if duration is not None and completed_updates else None
    metrics["operation_timings"] = ops
    return {"measurements": metrics, "estimated_utilization": extract_roofline_estimates(raw.get("roofline_model", {})),
        "counter_evidence": evidence, "extraction_errors": errors,
        "xplane_files": [str(p.relative_to(trace_dir)) for p in files]}


def record_diagnostic_profile(state, training, group_jit, output, work, dispatches=4, updates_per_dispatch=8):
    """One bounded capture of the existing executable; official weights stay saved."""
    begun = time.monotonic()
    trace_dir = Path(work) / "profile"
    trace_dir.mkdir(exist_ok=False)
    report = {"scope": "Post-checkpoint diagnostic, excluded from official training scores/tokens/time. No diagnostic weights saved.",
        "requested_updates": dispatches * updates_per_dispatch, "completed_updates": 0,
        "capture_status": "FAIL", "capture_error": None, "diagnostic_weights_saved": False,
        "host_dispatch_gaps_ms": [], "host_enqueue_ms": [], "host_sync_wait_ms": [],
        "dispatch_and_device_wait_ms": [], "capture_options": {
            "host_tracer_level": 2, "python_tracer_level": 0, "enable_hlo_proto": True,
            "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC", "tpu_perf_counters": True,
            "create_perfetto_trace": False}}
    tracing, previous_end, profile_data = False, None, None
    try:
        options = jax.profiler.ProfileOptions()
        options.host_tracer_level = 2
        options.python_tracer_level = 0
        options.enable_hlo_proto = True
        options.raise_error_on_start_failure = True
        options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC", "tpu_perf_counters": True}
        jax.profiler.start_trace(str(trace_dir), create_perfetto_trace=False, profiler_options=options)
        tracing = True
        window_start = time.monotonic()
        for index in range(dispatches):
            begin = time.monotonic()
            if previous_end is not None:
                report["host_dispatch_gaps_ms"].append((begin - previous_end) * 1000)
            with jax.profiler.StepTraceAnnotation("diagnostic_train", step_num=index):
                enqueue_start = time.monotonic()
                pending = group_jit(state, training, jnp.array(updates_per_dispatch, jnp.int32))
                enqueue_end = time.monotonic()
                state, values = jax.block_until_ready(pending)
                end = time.monotonic()
            report["host_enqueue_ms"].append((enqueue_end - enqueue_start) * 1000)
            report["host_sync_wait_ms"].append((end - enqueue_end) * 1000)
            report["dispatch_and_device_wait_ms"].append((end - begin) * 1000)
            metrics = jax.device_get(values)
            completed = int(metrics["updates"])
            report["completed_updates"] += completed
            if not bool(metrics["finite"]) or completed != updates_per_dispatch:
                raise FloatingPointError("Nonfinite or incomplete diagnostic dispatch")
            previous_end = end
        report["diagnostic_window_seconds"] = time.monotonic() - window_start
        report["capture_status"] = "PASS"
    except Exception as error:
        report["capture_error"] = type(error).__name__ + ": " + str(error)
    finally:
        if tracing:
            try:
                profile_data = jax.profiler.stop_trace()
            except Exception as error:
                report["stop_error"] = type(error).__name__ + ": " + str(error)
                report["capture_status"] = "FAIL"
        del state
    report["capture_seconds"] = time.monotonic() - begun
    try:
        report.update(summarize_xprof(trace_dir, output, profile_data, report["completed_updates"]))
    except Exception as error:
        report["extraction_errors"] = {"summary": type(error).__name__ + ": " + str(error)}
        report["measurements"] = extract_utilization({})
        report["measurements"]["operation_timings"] = extract_operation_timings({})
        report["estimated_utilization"] = {"status": "unavailable", "reason": "Profile summary failed."}
    window = report.get("diagnostic_window_seconds", 0.)
    report["host_timing"] = {
        "enqueue_wall_ms": sum(report["host_enqueue_ms"]),
        "synchronization_call_wall_ms": sum(report["host_sync_wait_ms"]),
        "between_dispatch_wall_ms": sum(report["host_dispatch_gaps_ms"]),
        "synchronization_call_percent_of_window": 100 * sum(report["host_sync_wait_ms"]) / (1000 * window) if window else None,
        "note": "Host wall-clock durations around enqueue and synchronous completion APIs, including API overhead. These are not OS CPU-idle percentages; synchronization can overlap TPU execution."}
    destination = output / "profile"
    try:
        shutil.copytree(trace_dir, destination)
        report["trace_directory"] = str(destination)
    except Exception as error:
        report["trace_copy_error"] = type(error).__name__ + ": " + str(error)
        report["trace_directory"] = str(trace_dir)
    training_memory = read_json(output / "training_memory.json")
    report.update(profiling_requirements(report["measurements"], training_memory))
    if report["capture_status"] != "PASS":
        report["requirements_status"] = "FAIL"
    report["seconds_including_export_and_analysis"] = time.monotonic() - begun
    write_json(output / "profiling.json", report)
    return report


