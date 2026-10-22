"""Training-scoped allocator samples, with explicit peak-counter limitations."""

def memory_snapshot(stats, elapsed):
    stats = stats or {}
    keys = ("bytes_in_use", "bytes_limit", "bytes_reserved", "peak_bytes_in_use", "peak_bytes_reserved")
    sample = {"training_elapsed_seconds": float(elapsed)}
    for key in keys:
        value = stats.get(key)
        sample[key] = int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None
    limit = sample["bytes_limit"]
    for key, result in (("bytes_in_use", "active_percent"), ("bytes_reserved", "reserved_percent")):
        sample[result] = 100. * sample[key] / limit if limit and sample[key] is not None else None
    return sample


def summarize_training_memory(before, samples, after, compiler, sampling_seconds):
    observed = [before, *samples, after]
    limits = [row["bytes_limit"] for row in observed if row.get("bytes_limit")]
    limit = limits[-1] if limits else None
    stable_limit = len(set(limits)) <= 1
    def maximum(key):
        values = [row[key] for row in observed if row.get(key) is not None]
        return max(values) if values else None
    active = maximum("bytes_in_use")
    reserved = maximum("bytes_reserved")
    old_peak, new_peak = before.get("peak_bytes_in_use"), after.get("peak_bytes_in_use")
    # Only a new lifetime high-water mark, strictly reached within the training
    # boundaries, establishes the actual allocator peak for that interval.
    isolated = old_peak is not None and new_peak is not None and new_peak > old_peak
    peak = new_peak if isolated else None
    required = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")
    estimate = (compiler[required[0]] + compiler[required[1]] - compiler[required[2]] + compiler[required[3]]) \
        if compiler and all(key in compiler for key in required) else None
    def percent(value):
        return 100. * value / limit if stable_limit and limit and value is not None else None
    return {
        "status": "available" if active is not None and limit and stable_limit else "unavailable",
        "scope": "JAX allocator counters, synchronized training boundaries and approximately 2-second progress samples; before final evaluation, restore, and profiling.",
        "device_limit_bytes": limit, "device_limit_gib": limit / 2**30 if limit else None,
        "device_limit_stable": stable_limit,
        "sampled_peak_active_bytes": active, "sampled_peak_active_gib": active / 2**30 if active is not None else None,
        "sampled_peak_active_percent": percent(active),
        "sampled_peak_reserved_bytes": reserved, "sampled_peak_reserved_percent": percent(reserved),
        "training_allocator_peak_bytes": peak, "training_allocator_peak_percent": percent(peak),
        "training_allocator_peak_status": "isolated_new_high_water_mark" if isolated else "not_isolatable",
        "lifetime_allocator_peak_before_training_bytes": old_peak,
        "lifetime_allocator_peak_after_training_bytes": new_peak,
        "compiled_peak_buffer_estimate_bytes": estimate,
        "compiled_peak_buffer_estimate_percent": percent(estimate),
        "compiler_estimate_formula": "arguments + outputs - aliased buffers + temporary buffers; an estimate, not measured device usage",
        "sampling_seconds_inside_training": sampling_seconds,
        "sample_count": len(observed), "samples": observed,
        "limitations": "Synchronized samples can miss temporary peaks. Lifetime counters include earlier phases unless a new peak was reached during training. Allocator counters may not expose every runtime-owned buffer. Reserved pool bytes are not active tensor bytes; compiler estimates are separate.",
    }
