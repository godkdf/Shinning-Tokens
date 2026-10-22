"""Bounded CPU correctness checks; no Gemma experiment or TPU claims."""
import ast
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import struct

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("build_v38", HERE / "build_notebook.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
SOURCE = builder.build_source()


def namespace(source, names=None):
    env = {"jax": jax, "jnp": jnp, "np": np, "optax": optax,
           "math": math, "json": json, "Path": Path}
    if names is not None:
        tree = ast.parse(source)
        source = ast.unparse(ast.Module(body=[n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names], type_ignores=[]))
    exec(compile(source, "fixture", "exec"), env)
    return env


PROFILE = namespace((HERE / "profiling.py").read_text())
MEMORY = namespace((HERE / "memory.py").read_text())


def table(columns, rows, props=None):
    return {"cols": [{"id": name, "label": name} for name in columns],
            "rows": [{"c": [{"v": value} for value in row]} for row in rows], "p": props or {}}


def test_embedded_config_and_exclusive_builder():
    notebook = builder.build_notebook(SOURCE)
    cell = next(c for c in notebook["cells"] if "PROGRAM_SOURCE =" in "".join(c["source"]))
    tree = ast.parse("".join(cell["source"]))
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ("CONFIG", "PROGRAM_SOURCE")}
    config = values["CONFIG"]
    assert config["local_batch_per_chip"] == 3
    assert config["sequence_length"] == 1024
    assert config["training_seconds"] == 180
    assert config["profile_dispatches"] * config["profile_updates_per_dispatch"] == 32
    assert values["PROGRAM_SOURCE"] == SOURCE
    assert "restore_archive" not in SOURCE
    assert "reloaded_eval" not in SOURCE
    assert "cooldown_multiplier" not in SOURCE
    assert "exact_chunked_head" not in SOURCE
    assert "updates * 2 * 1024" not in SOURCE
    assert 'config["local_batch_per_chip"] == 2' not in SOURCE
    assert "Starting ONE 60-second" not in SOURCE
    assert 'open("x")' in (HERE / "build_notebook.py").read_text()
    for c in notebook["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))


def test_dependency_lock_is_unchanged():
    def literals(notebook):
        tree = ast.parse("".join(notebook["cells"][3]["source"]))
        return {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ("REQUIREMENTS_INPUT", "RESOLVED_DEPENDENCY_LOCK")}
    previous = json.loads((HERE.parent / "gemma_270m_colab_v5e1_V3_5.ipynb").read_text())
    assert literals(builder.build_notebook(SOURCE)) == literals(previous)


def test_batch_three_preserves_data_and_wrapped_accounting():
    functions = namespace(SOURCE, {"batch_training_arrays", "training_prefix_counts", "completed_prefix"})
    import hashlib
    functions.update(hashlib=hashlib, DOMAINS=list(range(8)), LANGUAGES=list(range(12)))
    arrays = {"tokens": np.arange(20, dtype=np.int32).reshape(5, 4),
        "mask": np.ones((5, 4), dtype=bool), "segment_ids": np.zeros((5, 4), dtype=np.int32),
        "byte_weights": np.full((5, 4), 3, dtype=np.int32),
        "domain_ids": np.zeros((5, 4), dtype=np.int32),
        "language_ids": np.zeros((5, 4), dtype=np.int32)}
    batched, digest = functions["batch_training_arrays"](arrays, 3, 42)
    assert batched["tokens"].shape == (2, 3, 4)
    np.testing.assert_array_equal(np.sort(batched["tokens"][batched["mask"]]), np.arange(20))
    assert int(batched["byte_weights"].sum()) == 60
    assert np.all(batched["segment_ids"][~batched["mask"]] == -1)
    assert int(batched["mask"].sum()) / batched["mask"].size == pytest.approx(20 / 24)
    prefixes = functions["training_prefix_counts"](batched)
    assert int(functions["completed_prefix"](prefixes["tokens"], 5)) == 40 + int(prefixes["tokens"][1])
    assert len(digest) == 64


@struct.dataclass
class State:
    step: object
    params: object
    opt_state: object
    data_position: object


class ToyModel:
    def apply(self, variables, tokens, positions, attention_mask):
        p = variables["params"]
        hidden = p["embed"][tokens]
        return SimpleNamespace(logits=hidden @ p["embed"].T + p["bias"])


def test_native_dense_training_math_and_zero_update_state_match_v35():
    names = {"packed_attention_mask", "token_losses", "make_training_functions"}
    previous = (HERE.parent / "v3_5/runner.py").read_text()
    reference, revised = namespace(previous, names), namespace(SOURCE, names)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(.0003, weight_decay=.01))
    params = {"embed": jnp.arange(17 * 8, dtype=jnp.float32).reshape(17, 8) * .001,
              "bias": jnp.zeros(17, jnp.float32)}
    def initial():
        return State(jnp.array(0), jax.tree.map(jnp.copy, params), optimizer.init(params), jnp.array(0))
    batch = {"tokens": jnp.arange(12).reshape(3, 4), "targets": jnp.arange(1, 13).reshape(3, 4),
        "mask": jnp.ones((3, 4), jnp.bool_), "positions": jnp.tile(jnp.arange(4), (3, 1)),
        "segment_ids": jnp.zeros((3, 4), jnp.int32)}
    training = jax.tree.map(lambda x: x[None], batch)
    a = reference["make_training_functions"](ToyModel(), optimizer)[1]
    b = revised["make_training_functions"](ToyModel(), optimizer)[1]
    untouched, zero = jax.block_until_ready(b(initial(), training, jnp.array(0)))
    assert int(untouched.step) == int(untouched.data_position) == int(zero["updates"]) == 0
    expected, ea = jax.block_until_ready(a(initial(), training, jnp.array(3)))
    actual, eb = jax.block_until_ready(b(initial(), training, jnp.array(3)))
    assert int(actual.step) == int(actual.data_position) == 3
    assert int(eb["tokens"]) == 36
    for left, right in zip(jax.tree.leaves((expected, ea)), jax.tree.leaves((actual, eb))):
        np.testing.assert_allclose(left, right, atol=1e-7, rtol=1e-6)
    assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(actual.params))
    empty = dict(batch, mask=jnp.zeros((3, 4), jnp.bool_))
    rejected, metrics = jax.block_until_ready(b(initial(), jax.tree.map(lambda x: x[None], empty), jnp.array(1)))
    assert not bool(metrics["finite"])
    assert int(rejected.step) == int(rejected.data_position) == int(metrics["updates"]) == 0


def test_roofline_uses_hbm_not_total_memory_bandwidth_and_labels_estimate():
    fixture = table(["step", "category", "operation", "model_flop_rate", "hbm_bw", "measured_memory_bw", "bound_by"],
        [["Total", "Program", "Program", 60000., 500., 900., "HBM"],
         ["Total", "dot", "dot.1", 900000., 10000., 10000., "Compute"]],
        {"peak_flop_rate": "200000", "peak_hbm_bw": "800"})
    result = PROFILE["extract_roofline_estimates"]([fixture])
    assert result["kind"] == "compiler_cost_model_estimate"
    assert result["estimated_compute_percent_of_peak"] == 30.
    assert result["estimated_hbm_percent_of_peak"] == 62.5
    assert result["estimated_hbm_gib_per_second"] == 500.
    assert PROFILE["extract_roofline_estimates"]({})["status"] == "unavailable"


def test_idle_and_counter_gaps_cannot_pass_measurement_requirements():
    measurements = PROFILE["extract_utilization"]({"p": {"mxu_utilization_percent": 0,
        "hbm_bw_utilization_percent": 0, "host_idle_time_percent": 100}})
    # Standalone percentages are not evidence of collected hardware counters.
    for name in ("matrix_unit_utilization", "memory_bandwidth_utilization", "host_idle_time"):
        measurements[name].update(value=None, status="unavailable")
    measurements["operation_timings"] = {"status": "available", "categories": {}}
    result = PROFILE["profiling_requirements"](measurements, {"status": "available"})
    assert result["requirements_status"] == "FAIL"
    assert result["telemetry_status"] == "PARTIAL"
    assert result["checks"]["operation_timings"] == "FAIL"


def test_operation_classification_and_exclusive_durations():
    fixture = table(["framework_op_name", "self_time_us"], [
        ["jit(group)/jvp()/embedder/decode/dot_general", 4000.],
        ["jit(group)/optimizer_update/adam", 1000.],
        ["jit(group)/layer_0/attn/q_einsum", 500.],
        ["unknown_fusion", 500.]])
    result = PROFILE["extract_operation_timings"](fixture)
    assert result["categories"]["vocabulary_projection"]["self_time_us"] == 4000.
    assert result["categories"]["optimizer_updates"]["self_time_us"] == 1000.
    assert result["categories"]["attention"]["self_time_us"] == 500.
    assert result["categories"]["other_or_fused"]["self_time_us"] == 500.
    assert sum(v["percent_of_accounted_op_time"] or 0 for v in result["categories"].values()) == pytest.approx(100.)


def test_memory_samples_do_not_claim_earlier_evaluation_peak():
    snapshot, summarize = MEMORY["memory_snapshot"], MEMORY["summarize_training_memory"]
    before = snapshot({"bytes_limit": 16 * 2**30, "bytes_in_use": 4 * 2**30, "peak_bytes_in_use": 10 * 2**30}, 0)
    during = snapshot({"bytes_limit": 16 * 2**30, "bytes_in_use": 5 * 2**30, "peak_bytes_in_use": 10 * 2**30}, 2)
    after = snapshot({"bytes_limit": 16 * 2**30, "bytes_in_use": 4 * 2**30, "peak_bytes_in_use": 10 * 2**30}, 180)
    result = summarize(before, [during], after, None, .001)
    assert result["sampled_peak_active_percent"] == 31.25
    assert result["training_allocator_peak_bytes"] is None
    assert result["training_allocator_peak_status"] == "not_isolatable"


def test_changed_budget_comparison_cannot_be_controlled():
    result = PROFILE["historical_comparison"]({"local_batch_per_chip": 3, "training_seconds": 180},
        "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c",
        "74d6d6305b893aa9403e8ac30c10b9d920c23bc7d3c9d020debbd5b3500d707a", 45734.29605429257, 2.5)
    assert result["same_tokenizer_and_corpus"]
    assert not result["controlled_same_budget_comparison"]
    assert result["throughput_ratio_to_reference"] == 1.
    assert result["current_batch"] == 3


@pytest.mark.parametrize("have_counters", [False, True])
def test_actual_opstats_evidence_overrides_default_zero_percentages(tmp_path, monkeypatch, have_counters):
    from xprof.convert import raw_to_tool_data
    from xprof.protobuf import op_stats_pb2
    trace = tmp_path / "profile"
    trace.mkdir()
    (trace / "host.xplane.pb").write_bytes(b"fixture")
    stats = op_stats_pb2.OpStats()
    stats.run_environment.device_type = "TPU v5 Lite"
    stats.device_op_metrics_db.total_time_ps = 1000000
    if have_counters:
        stats.performance_counter_result.matrix_unit_utilization_percent = 0.
        stats.performance_counter_result.hbm_utilization_percent = 64.5
    (trace / "ALL_HOSTS.op_stats_v2.pb").write_bytes(stats.SerializeToString())
    def converter(paths, tool, params):
        return ({"p": {"mxu_utilization_percent": 0., "hbm_bw_utilization_percent": 0.,
                       "device_idle_time_percent": .3, "host_idle_time_percent": 100.}}, "application/json")
    monkeypatch.setattr(raw_to_tool_data, "xspace_to_tool_data", converter)
    monkeypatch.setitem(PROFILE, "write_json", lambda path, value: None)
    result = PROFILE["summarize_xprof"](trace, tmp_path, SimpleNamespace(planes=[]), 32)
    mxu = result["measurements"]["matrix_unit_utilization"]
    hbm = result["measurements"]["memory_bandwidth_utilization"]
    assert mxu["value"] == (0. if have_counters else None)
    assert hbm["value"] == (64.5 if have_counters else None)
    assert mxu["status"] == ("available" if have_counters else "unavailable")
    assert result["measurements"]["host_idle_time"]["status"] == "unavailable"
    assert result["measurements"]["device_idle_time"]["value"] == .3


def test_failed_profile_preserves_saved_result_and_reports_failure(tmp_path, monkeypatch):
    import time
    import shutil
    output, work = tmp_path / "output", tmp_path / "work"
    output.mkdir()
    work.mkdir()
    (output / "training_memory.json").write_text('{"status": "available"}')
    (output / "result.json").write_text('{"official_updates": 123}')
    def denied(*args, **kwargs):
        raise RuntimeError("fixture: counter capture denied")
    monkeypatch.setattr(jax.profiler, "start_trace", denied)
    monkeypatch.setitem(PROFILE, "time", time)
    monkeypatch.setitem(PROFILE, "shutil", shutil)
    monkeypatch.setitem(PROFILE, "read_json", lambda p: json.loads(p.read_text()))
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    monkeypatch.setitem(PROFILE, "write_json", save)
    result = PROFILE["record_diagnostic_profile"](object(), {}, None, output, work)
    assert result["capture_status"] == "FAIL"
    assert result["requirements_status"] == "FAIL"
    assert result["completed_updates"] == 0
    assert json.loads((output / "result.json").read_text()) == {"official_updates": 123}
    assert "capture denied" in result["capture_error"]


def test_diagnostic_is_bounded_and_does_not_save_advanced_weights(tmp_path, monkeypatch):
    import time
    import shutil
    output, work = tmp_path / "output", tmp_path / "work"
    output.mkdir()
    work.mkdir()
    (output / "training_memory.json").write_text('{"status": "available"}')
    (output / "weights.npz").write_bytes(b"official-weights")
    monkeypatch.setattr(jax.profiler, "start_trace", lambda *a, **k: None)
    monkeypatch.setattr(jax.profiler, "stop_trace", lambda: None)
    measurements = PROFILE["extract_utilization"]({})
    measurements["operation_timings"] = PROFILE["extract_operation_timings"]({})
    monkeypatch.setitem(PROFILE, "summarize_xprof", lambda *a: {
        "measurements": measurements, "estimated_utilization": {"status": "unavailable"}})
    monkeypatch.setitem(PROFILE, "time", time)
    monkeypatch.setitem(PROFILE, "shutil", shutil)
    monkeypatch.setitem(PROFILE, "read_json", lambda p: json.loads(p.read_text()))
    monkeypatch.setitem(PROFILE, "write_json", lambda p, v: p.write_text(json.dumps(v)))
    requests = []
    def group(state, training, requested):
        requests.append(int(requested))
        return state + requested, {"updates": requested, "finite": jnp.array(True)}
    result = PROFILE["record_diagnostic_profile"](jnp.array(100), {}, group, output, work)
    assert requests == [8, 8, 8, 8]
    assert result["completed_updates"] == 32
    assert result["capture_status"] == "PASS"
    assert result["requirements_status"] == "FAIL"
    assert not result["diagnostic_weights_saved"]
    assert (output / "weights.npz").read_bytes() == b"official-weights"
    assert len(result["host_dispatch_gaps_ms"]) == 3
    assert len(result["host_sync_wait_ms"]) == 4
