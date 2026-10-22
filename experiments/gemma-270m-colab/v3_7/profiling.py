"""XProf capture and conservative extraction; no utilization estimates.

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
    if "vocabulary_projection" in text:
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


def summarize_xprof(trace_dir, output):
    files = sorted(Path(trace_dir).rglob("*.xplane.pb")) + sorted(Path(trace_dir).rglob("*.xplane.riegeli"))
    raw = {}
    errors = {}
    if files:
        try:
            from xprof.convert import raw_to_tool_data
            for tool in ("overview_page", "hlo_stats", "roofline_model", "op_profile"):
                try:
                    payload, _ = raw_to_tool_data.xspace_to_tool_data(
                        [str(path) for path in files], tool, {"use_saved_result": False})
                    if payload is None:
                        raise ValueError("XProf returned no tool data")
                    raw[tool] = _decode_tool_json(payload)
                    write_json(output / "profile_tools" / (tool + ".json"), raw[tool])
                except Exception as error:
                    errors[tool] = type(error).__name__ + ": " + str(error)
        except Exception as error:
            errors["converter"] = type(error).__name__ + ": " + str(error)
    else:
        errors["capture"] = "No TPU XPlane files were produced. Host traces alone cannot establish TPU utilization."
    metrics = extract_utilization(raw.get("overview_page", {}))
    # XProf emits default 0.0% strings even when a trace has no hardware counters.
    # Require the underlying counter message, rather than treating those defaults
    # as measured zero utilization. OpStats is generated by overview conversion.
    evidence = {"device_type": None, "hardware_counter_message_present": False,
                "host_timeline_present": False, "device_timeline_present": False}
    try:
        from xprof.protobuf import op_stats_pb2
        candidates = sorted(Path(trace_dir).rglob("ALL_HOSTS.op_stats*.pb"))
        if candidates:
            stats = op_stats_pb2.OpStats()
            stats.ParseFromString(candidates[-1].read_bytes())
            evidence["device_type"] = stats.run_environment.device_type
            tpu = "TPU" in stats.run_environment.device_type.upper()
            evidence["host_timeline_present"] = stats.host_op_metrics_db.total_time_ps > 0
            evidence["device_timeline_present"] = tpu and stats.device_op_metrics_db.total_time_ps > 0
            evidence["hardware_counter_message_present"] = tpu and stats.HasField("performance_counter_result")
            if evidence["hardware_counter_message_present"]:
                counter = stats.performance_counter_result
                for name, field in (("matrix_unit_utilization", "matrix_unit_utilization_percent"),
                                    ("memory_bandwidth_utilization", "hbm_utilization_percent")):
                    value = float(getattr(counter, field))
                    if math.isfinite(value) and 0 <= value <= 100:
                        metrics[name].update(value=value, status="available", reason=None,
                            source="XProf OpStats.performance_counter_result", source_field=field)
    except Exception as error:
        errors["counter_evidence"] = type(error).__name__ + ": " + str(error)
    for name in ("matrix_unit_utilization", "memory_bandwidth_utilization"):
        if not evidence["hardware_counter_message_present"]:
            metrics[name].update(value=None, status="unavailable", source_field=None,
                reason="The profile has no TPU hardware-counter message; XProf default zeros are not measured utilization.")
    for name, flag in (("host_idle_time", "host_timeline_present"), ("device_idle_time", "device_timeline_present")):
        if not evidence[flag]:
            metrics[name].update(value=None, status="unavailable", source_field=None,
                reason="The profile lacks the required timeline evidence; default idle percentages are not measurements.")
    metrics["operation_timings"] = extract_operation_timings(raw.get("hlo_stats", {}))
    return {"measurements": metrics, "extraction_errors": errors,
        "counter_evidence": evidence,
        "xplane_files": [str(path.relative_to(trace_dir)) for path in files]}


def record_diagnostic_profile(state, training, group_jit, output, work, lr_multiplier, dispatches=4, updates_per_dispatch=16):
    """Consume disposable post-checkpoint state; never save diagnostic weights.

    Official scores, counters and checkpoint have already been recorded. This
    is a bounded profile of 64 updates, not a second training candidate.
    """
    begun = time.monotonic()
    trace_dir = Path(work) / "profile"
    trace_dir.mkdir(exist_ok=False)
    report = {"scope": "Post-checkpoint diagnostic; excluded from official training scores/tokens/time.",
        "requested_updates": dispatches * updates_per_dispatch, "completed_updates": 0,
        "learning_rate_multiplier": float(lr_multiplier), "diagnostic_weights_saved": False,
        "host_dispatch_gaps_ms": [], "dispatch_and_device_wait_ms": [],
        "host_wait_note": "Wall time around synchronized dispatch includes device work; it is not host/device idle time.",
        "status": "unavailable", "capture_error": None}
    tracing = False
    previous_end = None
    try:
        # JAX 0.12.0 exposes profiler options through ProfileOptions.
        options = jax.profiler.ProfileOptions()
        options.host_tracer_level = 2
        options.python_tracer_level = 0
        options.enable_hlo_proto = True
        options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC", "tpu_perf_counters": True}
        jax.profiler.start_trace(str(trace_dir), create_perfetto_trace=True, profiler_options=options)
        tracing = True
        for index in range(dispatches):
            begin = time.monotonic()
            if previous_end is not None:
                report["host_dispatch_gaps_ms"].append((begin - previous_end) * 1000)
            with jax.profiler.StepTraceAnnotation("diagnostic_train", step_num=index):
                state, values = jax.block_until_ready(group_jit(state, training,
                    jnp.array(updates_per_dispatch, jnp.int32), jnp.array(lr_multiplier, jnp.float32)))
            end = time.monotonic()
            metrics = jax.device_get(values)
            completed = int(metrics["updates"])
            report["completed_updates"] += completed
            report["dispatch_and_device_wait_ms"].append((end - begin) * 1000)
            if not bool(metrics["finite"]) or completed != updates_per_dispatch:
                raise FloatingPointError("Nonfinite or incomplete diagnostic dispatch")
            previous_end = end
        report["status"] = "captured"
    except Exception as error:
        report["capture_error"] = type(error).__name__ + ": " + str(error)
    finally:
        if tracing:
            try:
                jax.profiler.stop_trace()
            except Exception as error:
                report["stop_error"] = type(error).__name__ + ": " + str(error)
        del state
    report["capture_seconds"] = time.monotonic() - begun
    try:
        report.update(summarize_xprof(trace_dir, output))
    except Exception as error:
        report["extraction_errors"] = {"summary": type(error).__name__ + ": " + str(error)}
        report["measurements"] = extract_utilization({})
        report["measurements"]["operation_timings"] = extract_operation_timings({})
    destination = output / "profile"
    try:
        shutil.copytree(trace_dir, destination)
        report["trace_directory"] = str(destination)
    except Exception as error:
        report["trace_copy_error"] = type(error).__name__ + ": " + str(error)
        report["trace_directory"] = str(trace_dir)
    report["seconds_including_export_and_analysis"] = time.monotonic() - begun
    report["host_idle_note"] = "Host idle can mean normal waiting for TPU work; idle percentages alone do not establish a dispatch bottleneck."
    write_json(output / "profiling.json", report)
    return report


def parameter_digest(params):
    digest = hashlib.sha256()
    for path, value in sorted(traverse_util.flatten_dict(params).items()):
        array = np.asarray(jax.device_get(value))
        digest.update("/".join(path).encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(memoryview(array).cast("B"))
        del array
    return digest.hexdigest()
