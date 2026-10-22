"""Bounded correctness checks; these do not run a Gemma TPU experiment."""
import ast
import importlib.util
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import struct
from flax import linen as nn

HERE = Path(__file__).resolve().parent


def helpers(filename, **extra):
    namespace = {"jax": jax, "jnp": jnp, "np": np, "math": math, "json": json, **extra}
    exec(compile((HERE / filename).read_text(), filename, "exec"), namespace)
    return namespace


HEAD = helpers("head.py")
PROFILE = helpers("profiling.py")
MEMORY = helpers("memory.py")


def test_cooldown_boundaries():
    schedule = HEAD["cooldown_multiplier"]
    np.testing.assert_allclose([schedule(t, 60) for t in (-1, 0, 30, 45, 60, 70)],
                              [1, 1, 1, .65, .3, .3])
    assert schedule(900, 1800) == 1
    assert schedule(1800, 1800) == pytest.approx(.3)
    with pytest.raises(ValueError):
        schedule(0, 0)


def test_training_memory_percentages_and_peak_scope():
    snapshot = MEMORY["memory_snapshot"]
    summarize = MEMORY["summarize_training_memory"]
    gib = 2**30
    start = snapshot({"bytes_limit": 16*gib, "bytes_in_use": 4*gib,
                      "bytes_reserved": 6*gib, "peak_bytes_in_use": 10*gib}, 0)
    sample = snapshot({"bytes_limit": 16*gib, "bytes_in_use": 5*gib,
                       "bytes_reserved": 7*gib, "peak_bytes_in_use": 10*gib}, 30)
    stop = snapshot({"bytes_limit": 16*gib, "bytes_in_use": 4*gib,
                     "bytes_reserved": 7*gib, "peak_bytes_in_use": 10*gib}, 60)
    compiler = {"argument_size_in_bytes": 4*gib, "output_size_in_bytes": 3*gib,
                "alias_size_in_bytes": 3*gib, "temp_size_in_bytes": 2*gib}
    result = summarize(start, [sample], stop, compiler, .001)
    assert result["sampled_peak_active_gib"] == 5
    assert result["sampled_peak_active_percent"] == 31.25
    assert result["sampled_peak_reserved_percent"] == 43.75
    assert result["compiled_peak_buffer_estimate_percent"] == 37.5
    # An earlier evaluation's larger peak cannot be relabeled a training peak.
    assert result["training_allocator_peak_bytes"] is None
    stop["peak_bytes_in_use"] = 12*gib
    result = summarize(start, [sample], stop, compiler, .001)
    assert result["training_allocator_peak_percent"] == 75
    missing = snapshot(None, 0)
    result = summarize(missing, [], missing, None, 0)
    assert result["status"] == "unavailable"
    assert result["sampled_peak_active_percent"] is None


@pytest.mark.parametrize("count,chunk", [(1, 128), (128, 128), (137, 128), (257, 64)])
@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 2e-5), (jnp.bfloat16, .03)])
def test_exact_loss_and_both_gradients(count, chunk, dtype, tolerance):
    # Non-multiple token counts exercise padding removal and zero pad gradients.
    a, b = jax.random.split(jax.random.key(count))
    x = (jax.random.normal(a, (count, 16)) * .1).astype(dtype)
    w = (jax.random.normal(b, (513, 16)) * .1).astype(dtype)
    ids = jnp.arange(count, dtype=jnp.int32) % w.shape[0]
    mask = (jnp.arange(count) % 3 != 0).astype(jnp.float32)
    mask = mask.at[0].set(1)
    def dense(xx, ww):
        logits = (xx @ ww.T).astype(jnp.float32)
        return jax.scipy.special.logsumexp(logits, axis=-1) - logits[jnp.arange(count), ids]
    def chunked(xx, ww):
        return HEAD["exact_chunked_head"](xx, ww, ids, chunk)
    np.testing.assert_allclose(jax.jit(chunked)(x, w), jax.jit(dense)(x, w), rtol=2e-5, atol=2e-5)
    # Check gradients with nonuniform masking, as used by the real training loss.
    def objective(fn, xx, ww):
        return jnp.sum(fn(xx, ww) * mask) / mask.sum()
    ga = jax.jit(jax.grad(lambda xx, ww: objective(dense, xx, ww), (0, 1)))(x, w)
    gb = jax.jit(jax.grad(lambda xx, ww: objective(chunked, xx, ww), (0, 1)))(x, w)
    for expected, actual in zip(ga, gb):
        error = np.linalg.norm(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
        norm = np.linalg.norm(np.asarray(expected, dtype=np.float32))
        assert error / max(norm, 1e-12) < tolerance
        assert actual.dtype == expected.dtype == dtype


def test_matrix_percent_is_not_replaced_by_flop_proxy():
    result = PROFILE["extract_utilization"]({"p": {
        "flop_rate_utilization_relative_to_roofline": .7,
        "memory_bw_utilization_relative_to_hw_limit": .8}})
    assert all(value["value"] is None for value in result.values())
    result = PROFILE["extract_utilization"]([{"p": {
        "mxu_utilization_percent": 0, "hbm_bw_utilization_percent": 64.5,
        "device_idle_time_percent": 4.2, "host_idle_time_percent": "93%"}}])
    assert result["matrix_unit_utilization"]["value"] == 0
    assert result["matrix_unit_utilization"]["status"] == "available"
    assert result["memory_bandwidth_utilization"]["value"] == 64.5
    assert result["host_idle_time"]["value"] == 93
    assert result["device_idle_time"]["value"] == 4.2
    for invalid in (float("nan"), -1, 101, True):
        result = PROFILE["extract_utilization"]({"mxu_utilization_percent": invalid})
        assert result["matrix_unit_utilization"]["value"] is None


def test_operation_timings_use_exclusive_time_and_keep_unknowns():
    data = {"cols": [{"id": "op_name"}, {"id": "self_time_us"}, {"id": "total_time_us"}],
        "rows": [{"c": [{"v": name}, {"v": own}, {"v": inclusive}]} for name, own, inclusive in
            [("vocabulary_projection/dot", 40, 100), ("optimizer_update/adam", 20, 100),
             ("layer/attention/dot", 30, 100), ("unknown_fusion", 10, 100)]]}
    result = PROFILE["extract_operation_timings"](data)
    categories = result["categories"]
    assert sum(value["self_time_us"] or 0 for value in categories.values()) == 100
    assert categories["vocabulary_projection"]["percent_of_accounted_op_time"] == 40
    assert categories["other_or_fused"]["self_time_us"] == 10
    assert categories["mlp"]["self_time_us"] is None
    # Inclusive timing alone cannot establish the exclusive operation breakdown.
    data["cols"][1]["id"] = "unrecognized_duration"
    assert PROFILE["extract_operation_timings"](data)["status"] == "unavailable"


def test_generated_notebook_source_and_code_cells():
    path = HERE.parent / "gemma_270m_colab_v5e1_V3_7.ipynb"
    notebook = json.loads(path.read_text())
    embedded = None
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert not cell.get("outputs")
        text = "".join(cell["source"])
        tree = ast.parse(text, filename=f"cell_{index}")
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PROGRAM_SOURCE" for t in node.targets):
                embedded = ast.literal_eval(node.value)
    assert embedded == (HERE / "runner.py").read_text()
    ast.parse(embedded)
    assert '"version": "V3.5"' in embedded  # Preserve the measured comparator.
    assert embedded.index("exact_restore =") < embedded.index("profile = record_diagnostic_profile(")
    assert '"report_finalized": False' in embedded
    assert 'result["report_finalized"] = True' in embedded


def test_grouped_updates_zero_warmup_and_dynamic_lr_without_recompile():
    source = (HERE / "runner.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_training_functions")
    namespace = {"jax": jax, "jnp": jnp, "optax": optax}
    def token_losses(model, params, batch):
        return (params["w"] * batch["tokens"] - 1) ** 2
    namespace["token_losses"] = token_losses
    exec(compile(ast.Module(body=[node], type_ignores=[]), "grouped_updates", "exec"), namespace)
    @struct.dataclass
    class State:
        step: object
        params: object
        opt_state: object
        data_position: object
    optimizer = optax.adamw(learning_rate=.01)
    params = {"w": jnp.array(0.5)}
    state = State(jnp.array(0), params, optimizer.init(params), jnp.array(0))
    _, group = namespace["make_training_functions"](None, optimizer)
    training = {"tokens": jnp.ones((4, 1, 2)), "mask": jnp.ones((4, 1, 2), dtype=bool)}
    state, values = group(state, training, jnp.array(0), jnp.array(1.0))
    assert int(state.step) == int(state.data_position) == int(values["updates"]) == 0
    original = float(state.params["w"])
    state, values = group(state, training, jnp.array(3), jnp.array(.3))
    assert int(state.step) == int(state.data_position) == int(values["updates"]) == 3
    assert float(state.params["w"]) > original
    size = group._cache_size()
    state, values = group(state, training, jnp.array(1), jnp.array(1.0))
    assert group._cache_size() == size
    assert int(state.step) == 4


def test_native_entry_and_tied_embedding_gradients():
    class Embedder(nn.Module):
        def setup(self):
            self.table = self.param("input_embedding", nn.initializers.normal(), (513, 16))
        def encode(self, tokens):
            return self.table[tokens] * jnp.sqrt(16).astype(self.table.dtype)
        def decode(self, hidden):
            return hidden @ self.table.T
    class Block(nn.Module):
        @nn.compact
        def __call__(self, hidden, positions, cache, attention_mask):
            return None, hidden + nn.Dense(16, use_bias=False)(hidden)
    class TinyNative(nn.Module):
        def setup(self):
            self.embedder = Embedder()
            self.blocks = [Block(name="layer_0")]
            self.final_norm = nn.RMSNorm()
    # A subclass shares the exact parameter paths and implements the reference.
    @struct.dataclass
    class Output:
        logits: object
    class Model(TinyNative):
        def __call__(self, tokens, positions, attention_mask):
            x = self.embedder.encode(tokens)
            for block in self.blocks:
                _, x = block(x, positions, None, attention_mask)
            return Output(self.embedder.decode(self.final_norm(x)))
    model = Model()
    tokens = jnp.array([[1, 2, 3, 4, 5]], jnp.int32)
    batch = {"tokens": tokens, "targets": tokens + 1, "positions": jnp.arange(5)[None],
             "segment_ids": jnp.zeros_like(tokens)}
    mask = lambda segments: jnp.ones((*segments.shape, segments.shape[-1]), bool)
    namespace = helpers("head.py", packed_attention_mask=mask)
    params = model.init(jax.random.key(1), tokens, batch["positions"], mask(tokens))["params"]
    a = lambda p: jnp.mean(namespace["token_losses"](model, p, batch))
    b = lambda p: jnp.mean(namespace["reference_token_losses"](model, p, batch))
    av, ag = jax.jit(jax.value_and_grad(a))(params)
    bv, bg = jax.jit(jax.value_and_grad(b))(params)
    np.testing.assert_allclose(av, bv, atol=1e-4)
    for x, y in zip(jax.tree.leaves(ag), jax.tree.leaves(bg)):
        assert np.linalg.norm(np.asarray(x - y)) / max(np.linalg.norm(np.asarray(y)), 1e-12) < .03


def test_profiler_options_and_capture_on_cpu(tmp_path):
    import shutil
    import time
    from flax import traverse_util
    # Exercise the actual pinned JAX capture API, without claiming TPU counters.
    def write_json(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, allow_nan=False))
    namespace = helpers("profiling.py", Path=Path, time=time, shutil=shutil, write_json=write_json)
    output = tmp_path / "output"
    work = tmp_path / "work"
    output.mkdir()
    work.mkdir()
    def step(value, training, requested, scale):
        return value + requested * scale, {"updates": requested, "finite": jnp.array(True)}
    step = jax.jit(step, donate_argnums=(0,))
    value = jnp.array(0.)
    value, _ = step(value, {}, jnp.array(0), jnp.array(1.))
    profile = namespace["record_diagnostic_profile"](value, {}, step, output, work, .3, 2, 2)
    assert profile["capture_error"] is None
    assert profile["completed_updates"] == 4
    assert (output / "profiling.json").is_file()
    assert list((output / "profile").rglob("*.xplane.pb"))
    assert not profile["counter_evidence"]["hardware_counter_message_present"]
    assert profile["measurements"]["matrix_unit_utilization"]["value"] is None
    assert profile["measurements"]["memory_bandwidth_utilization"]["value"] is None


def test_weights_archive_restore_and_digest(tmp_path):
    import hashlib
    import zipfile
    from flax import serialization, traverse_util
    tree = ast.parse((HERE / "runner.py").read_text())
    names = {"numpy_leaf", "archive_tree", "save_archive", "restore_archive", "parameter_digest"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"jax": jax, "np": np, "Path": Path, "json": json, "hashlib": hashlib,
                 "zipfile": zipfile, "serialization": serialization, "traverse_util": traverse_util}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "checkpoint", "exec"), namespace)
    params = {"embedder": {"input_embedding": jnp.arange(20, dtype=jnp.float32).reshape(5, 4)},
              "layer_0": {"norm": jnp.array([.1, .2, .3, .4])}}
    original = namespace["parameter_digest"](params)
    archive = tmp_path / "weights.npz"
    count = namespace["save_archive"](archive, {"params": params}, {"kind": "weights_only"})
    assert count == 2
    restored = namespace["restore_archive"](archive, {"params": params})
    assert namespace["parameter_digest"](restored["params"]) == original
    modified = jax.tree.map(lambda x: x + 1, restored["params"])
    assert namespace["parameter_digest"](modified) != original
    # Disposable diagnostic updates do not change the saved checkpoint.
    assert namespace["parameter_digest"](namespace["restore_archive"](archive, {"params": params})["params"]) == original
