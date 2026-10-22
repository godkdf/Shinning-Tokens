from IPython.display import display, Markdown
result = json.loads((OUTPUT / "result.json").read_text())
def show(rows):
    display(Markdown("| Measurement | Result |\n|---|---|\n" + "\n".join(f"| {name} | {value} |" for name, value in rows)))
def fmt(value, digits=3):
    return f"{value:,.{digits}f}" if isinstance(value, (float, int)) else "UNAVAILABLE"

show([
    ("Version / model parameters", f'{result["version"]} / {result["model_parameters"]:,}'),
    ("Batch × context", f'{result["config"]["local_batch_per_chip"]} × {result["config"]["sequence_length"]:,}'),
    ("Training / learning / performance", f'{result["engineering_status"]} / {result["learning_status"]} / {result["performance_goal_status"]}'),
    ("Profiling requirements / telemetry", f'{result["profiling_status"]} / {result["telemetry_status"]}'),
    ("Actual training seconds", fmt(result["training_seconds"])),
    ("Updates / useful tokens", f'{result["updates"]:,} / {result["scored_tokens"]:,}'),
    ("Useful tokens/s", fmt(result["tokens_per_second"], 1)),
    ("Attributed original bytes/s", fmt(result["original_bytes_per_second_attributed"], 1)),
    ("Useful occupancy", f'{result["training_occupancy"]:.2%}'),
    ("Initial → final FineWiki macro BPB", f'{result["initial_eval"]["wiki"]["language_macro_bpb"]:.6f} → {result["final_eval"]["wiki"]["language_macro_bpb"]:.6f}'),
    ("Initial → final in-mixture pooled BPB", f'{result["initial_eval"]["in_mix"]["pooled_bpb"]:.6f} → {result["final_eval"]["in_mix"]["pooled_bpb"]:.6f}'),
    ("Compilation / initial eval / final eval seconds", f'{result["compilation_seconds"]:.2f} / {result["initial_eval_seconds"]:.2f} / {result["final_eval_seconds"]:.2f}'),
    ("Weights save and file-check seconds", fmt(result["checkpoint_save_verify_seconds"])),
    ("Checkpoint restoration check", "Disabled as requested"),
    ("Whole notebook through report seconds", fmt(result["notebook_elapsed_seconds"])),
    ("Estimated training / total CU at 2.92/hour", f'{result["compute_units_estimate"]["training"]:.5f} / {result["compute_units_estimate"]["through_report"]:.5f}'),
])
memory = result["training_memory"]
show([
    ("Reported device capacity GiB", fmt(memory["device_limit_gib"])),
    ("Sampled maximum active GiB / % capacity", f'{fmt(memory["sampled_peak_active_gib"])} / {fmt(memory["sampled_peak_active_percent"])}%'),
    ("Training allocator peak % / scope", f'{fmt(memory["training_allocator_peak_percent"])}% / {memory["training_allocator_peak_status"]}'),
    ("Compiler buffer estimate % capacity", fmt(memory["compiled_peak_buffer_estimate_percent"]) + "% (estimate)"),
    ("Memory sampling overhead ms", fmt(memory["sampling_seconds_inside_training"] * 1000)),
])
print("Memory limits:", memory["limitations"])
profile = result["profiling"]
rows = []
for name, measurement in profile["measurements"].items():
    if name == "operation_timings":
        continue
    value = fmt(measurement["value"]) + "%" if measurement["status"] == "available" else "UNAVAILABLE: " + str(measurement["reason"])
    rows.append((name, value))
show(rows)
operations = profile["measurements"]["operation_timings"]
display(Markdown("| Operation category | ms/update | Accounted operation time % |\n|---|---:|---:|\n" + "\n".join(
    f'| {name} | {fmt(values.get("milliseconds_per_update"))} | {fmt(values.get("percent_of_accounted_op_time"))} |'
    for name, values in operations["categories"].items())))
print("Operation timing scope:", operations["scope"])
print("Host enqueue/wait/gaps:", json.dumps(profile["host_timing"], indent=2))
estimates = profile["estimated_utilization"]
if estimates["status"] == "available":
    show([
        ("ESTIMATED compute TFLOP/s / % peak", f'{estimates["estimated_compute_tflops_per_second"]:.2f} / {estimates["estimated_compute_percent_of_peak"]:.2f}%'),
        ("ESTIMATED HBM GiB/s / % peak", f'{estimates["estimated_hbm_gib_per_second"]:.2f} / {estimates["estimated_hbm_percent_of_peak"]:.2f}%'),
        ("Diagnostic updates / total diagnostic seconds", f'{profile["completed_updates"]} / {profile["seconds_including_export_and_analysis"]:.2f}'),
    ])
    print(estimates["note"])
else:
    print("Compiler estimates unavailable:", estimates.get("reason"))
language_rows = []
for language, final in result["final_eval"]["wiki"]["languages"].items():
    initial = result["initial_eval"]["wiki"]["languages"][language]["bpb"]
    language_rows.append(f'| {language} | {initial:.6f} | {final["bpb"]:.6f} | {final["bpb"] - initial:+.6f} |')
display(Markdown("| FineWiki language | Initial BPB | Final BPB | Change |\n|---|---:|---:|---:|\n" + "\n".join(language_rows)))
print("Historical comparison:", json.dumps(result["historical_comparison"], indent=2))
print("Required measurements:", json.dumps(profile["checks"], indent=2))
print("Training requirements:", json.dumps(result["requirements"], indent=2))
print("Performance targets:", json.dumps(result["performance_goals"], indent=2))
print("Timed source byte shares:", json.dumps(result["training_domain_byte_shares"], indent=2))
print("Timed language byte shares:", json.dumps(result["training_language_byte_shares"], indent=2))
print("Artifacts:", OUTPUT)
