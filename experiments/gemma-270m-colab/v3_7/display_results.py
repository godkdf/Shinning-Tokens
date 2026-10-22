from IPython.display import display, Markdown

result = json.loads((OUTPUT / "result.json").read_text())
profile = result.get("profiling", {})
rows = [
    ("Version / optimizer", result["version"] + " / " + result["optimizer"]["name"]),
    ("Engineering / learning / performance goals", " / ".join(result[key] for key in
        ("engineering_status", "learning_status", "performance_goal_status"))),
    ("Telemetry", result.get("telemetry_status", "INCOMPLETE")),
    ("Report finalized", result.get("report_finalized", False)),
    ("Model parameters", f'{result["model_parameters"]:,}'),
    ("Official training seconds", f'{result["training_seconds"]:.4f}'),
    ("Official updates / scored tokens", f'{result["updates"]:,} / {result["scored_tokens"]:,}'),
    ("Scored tokens/second", f'{result["tokens_per_second"]:,.2f}'),
    ("Attributed original bytes/second", f'{result["original_bytes_per_second_attributed"]:,.2f}'),
    ("Training occupancy", f'{result["training_occupancy"]:.3%}'),
    ("FineWiki macro BPB, initial → final", f'{result["initial_eval"]["wiki"]["language_macro_bpb"]:.7f} → {result["final_eval"]["wiki"]["language_macro_bpb"]:.7f}'),
    ("FineWiki BPB difference from V3.5", f'{result["final_eval"]["wiki"]["language_macro_bpb"] - result["baseline"]["finewiki_macro_bpb"]:+.7f}'),
    ("Head chunk / logical FP32 logits workspace", f'{result["head"]["chunk_tokens"]} tokens / {result["head"]["logical_fp32_logits_workspace_bytes"] / 2**20:.2f} MiB'),
    ("Last dispatch learning-rate multiplier", f'{result["learning_rate_schedule"]["last_dispatch_multiplier"]:.6f}'),
    ("Compilation / head verification seconds", f'{result["compilation_seconds"]:.2f} / {result["head_verification_seconds"]:.2f}'),
    ("Weights save / restore-and-evaluate seconds", f'{result["checkpoint_save_verify_seconds"]:.2f} / {result["checkpoint_restore_seconds"]:.2f}'),
    ("Restored parameters exact / maximum BPB error", f'{result["exact_parameter_restore"]} / {result["reload_max_bpb_error"]:.9f}'),
    ("Diagnostic updates (discarded state)", profile.get("completed_updates", "unavailable")),
    ("Profiling and analysis seconds", f'{profile.get("seconds_including_export_and_analysis", 0):.2f}'),
    ("Total notebook seconds through report", f'{result["notebook_elapsed_seconds"]:.2f}'),
    ("Estimated CU: training / profiling / total", " / ".join(
        f'{result["compute_units_estimate"].get(key, 0):.5f}' for key in ("training", "diagnostic_profile", "through_report"))),
]
display(Markdown("| Measurement | Result |\n|---|---|\n" + "\n".join(f"| {name} | {value} |" for name, value in rows)))

measurements = profile.get("measurements", {})
telemetry_rows = []
for name, label in (("matrix_unit_utilization", "Matrix-unit utilization"),
                    ("memory_bandwidth_utilization", "HBM bandwidth utilization"),
                    ("host_idle_time", "Host idle time"), ("device_idle_time", "Device idle time")):
    metric = measurements.get(name, {})
    value = f'{metric["value"]:.3f}%' if metric.get("status") == "available" else "Unavailable"
    evidence = metric.get("source_field") or metric.get("reason") or "Profile not completed"
    telemetry_rows.append(f"| {label} | {value} | {evidence} |")
display(Markdown("| TPU profile measurement | Result | Source / limitation |\n|---|---|---|\n" + "\n".join(telemetry_rows)))

timings = measurements.get("operation_timings", {})
timing_rows = []
for name, entry in timings.get("categories", {}).items():
    duration, percent = entry.get("self_time_us"), entry.get("percent_of_accounted_op_time")
    duration_text = f"{duration / 1000:.3f} ms" if duration is not None else "Unavailable"
    percent_text = f"{percent:.2f}%" if percent is not None else "Unavailable"
    timing_rows.append(f"| {name} | {duration_text} | {percent_text} |")
if timing_rows:
    display(Markdown("| Operation category | Summed exclusive device time | Share of accounted op time |\n|---|---:|---:|\n" + "\n".join(timing_rows)))
print(timings.get("scope", timings.get("reason", "Operation timings unavailable")))
print("Profiler extraction errors:", json.dumps(profile.get("extraction_errors", {}), indent=2))
print("Capture error:", profile.get("capture_error"))
print("Raw trace:", profile.get("trace_directory"))

language_rows = []
for language, final in result["final_eval"]["wiki"]["languages"].items():
    initial = result["initial_eval"]["wiki"]["languages"][language]["bpb"]
    language_rows.append(f"| {language} | {initial:.6f} | {final['bpb']:.6f} | {final['bpb'] - initial:+.6f} |")
display(Markdown("| FineWiki language | Initial BPB | Final BPB | Change |\n|---|---:|---:|---:|\n" + "\n".join(language_rows)))
print("Timed domain byte shares:", json.dumps(result["training_domain_byte_shares"], indent=2))
print("Timed language byte shares:", json.dumps(result["training_language_byte_shares"], indent=2))
print("Requirements:", json.dumps(result["requirements"], indent=2))
print("Performance goals:", json.dumps(result["performance_goals"], indent=2))
print("Artifacts:", OUTPUT)
