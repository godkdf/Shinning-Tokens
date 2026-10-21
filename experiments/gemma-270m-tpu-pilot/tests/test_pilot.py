"""Scientific correctness checks on 16 virtual CPU devices; no cloud charges."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=16")

from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from launch_pilot import resource_is_v5e16
from pilot_common import chunk_document, load_checkpoint, save_checkpoint
from pilot_core import (PilotState, build_steps, check_global_loss, host_value,
                        loss_parts, make_batch, make_mesh)


class ToyModel:
    def apply(self, variables, tokens):
        return SimpleNamespace(logits=variables["params"]["weights"][tokens])


def test_chunks_score_every_target_once_and_mask_tail():
    rows = list(chunk_document([3, 4, 5, 6, 7, 8, 9], 4, 2, 1))
    targets = np.concatenate([row[1][row[2]] for row in rows])
    np.testing.assert_array_equal(targets, [3, 4, 5, 6, 7, 8, 9, 1])
    tail = list(chunk_document([3], 4, 2, 1))[0]
    assert tail[2].tolist() == [True, True, False, False]
    assert rows[1][0][0] == rows[0][1][-1]


def test_padding_token_cannot_silently_remove_text():
    import pytest
    with pytest.raises(ValueError, match="padding"):
        list(chunk_document([0, 3], 4, 2, 1))


def test_masked_targets_do_not_change_loss():
    params = {"weights": jnp.arange(64, dtype=jnp.float32).reshape(8, 8) / 100}
    batch = {"tokens": jnp.array([[1, 2, 0, 0]]), "targets": jnp.array([[2, 3, 0, 0]]),
             "mask": jnp.array([[True, True, False, False]])}
    first = loss_parts(ToyModel(), params, batch)
    batch["targets"] = jnp.array([[2, 3, 6, 7]])
    second = loss_parts(ToyModel(), params, batch)
    np.testing.assert_allclose(first, second)


def training_fixture():
    mesh, replicated, data, local_indices = make_mesh()
    assert mesh.size == 16
    params = {"weights": jnp.arange(64, dtype=jnp.float32).reshape(8, 8) / 100}
    optimizer = optax.sgd(0.1)
    state = PilotState(jnp.array(0), params, optimizer.init(params), jax.random.PRNGKey(42), jnp.array(0))
    state = jax.device_put(state, replicated)
    arrays = {"tokens": np.tile([1, 2, 3, 4], (16, 1)).astype(np.int32),
              "targets": np.tile([2, 3, 4, 5], (16, 1)).astype(np.int32),
              "mask": np.ones((16, 4), bool)}
    arrays["mask"][0, 1:] = False
    arrays["targets"][0, 0] = 7
    batch = make_batch(arrays, np.arange(16), local_indices, data)
    train, _ = build_steps(ToyModel(), optimizer, optax.constant_schedule(0.1), replicated, data, 16)
    return params, state, arrays, batch, train, mesh, replicated


def test_distributed_gradient_matches_global_token_weighted_reference():
    params, state, arrays, batch, train, mesh, replicated = training_fixture()
    # Training donates its state buffers. Keep an independent host reference.
    params = jax.tree.map(lambda value: np.array(value), params)
    assert check_global_loss(mesh, replicated)
    # Match each chip's BF16 arithmetic, then sum its FP32 master-weight gradient.
    # A single 16-row BF16 reduction has different rounding from 16 one-row reductions.
    global_tokens = int(arrays["mask"].sum())
    reference_grad = jnp.zeros_like(params["weights"])
    for row in range(16):
        local = {key: jnp.array(value[row:row + 1]) for key, value in arrays.items()}
        def objective(weights):
            total, _ = loss_parts(ToyModel(), weights, local)
            return total / global_tokens
        reference_grad += jax.grad(objective)(params)["weights"]
    expected = params["weights"] - 0.1 * reference_grad
    updated, metrics = jax.block_until_ready(train(state, batch))
    actual = host_value(updated.params["weights"])
    # BF16 collective reduction rounds once more than the FP32 reference sum.
    # Bound that rounding by one BF16 spacing at the largest SGD update.
    rounding_bound = float(jnp.finfo(jnp.bfloat16).eps) * float(jnp.max(jnp.abs(0.1 * reference_grad)))
    np.testing.assert_allclose(actual, expected, atol=rounding_bound, rtol=0)
    incorrect_grad = jnp.zeros_like(reference_grad)
    for row in range(16):
        local = {key: jnp.array(value[row:row + 1]) for key, value in arrays.items()}
        def incorrect_objective(weights):
            numerator, denominator = loss_parts(ToyModel(), weights, local)
            return numerator / denominator / 16
        incorrect_grad += jax.grad(incorrect_objective)(params)["weights"]
    incorrectly_weighted = params["weights"] - 0.1 * incorrect_grad
    assert float(jnp.max(jnp.abs(actual - incorrectly_weighted))) > 10 * rounding_bound
    assert int(host_value(metrics["tokens"])) == 61
    assert int(host_value(updated.step)) == 1
    assert int(host_value(updated.data_position)) == 16


def test_nonfinite_gradient_does_not_apply_update():
    _, state, _, batch, train, _, replicated = training_fixture()
    bad = jnp.full_like(state.params["weights"], jnp.nan)
    state = state.replace(params=jax.device_put({"weights": bad}, replicated))
    updated, metrics = jax.block_until_ready(train(state, batch))
    assert not bool(host_value(metrics["finite"]))
    assert int(host_value(updated.step)) == 0
    assert int(host_value(updated.data_position)) == 0


def test_checkpoint_preserves_optimizer_rng_and_position(tmp_path):
    original = {"step": np.array(7, np.int32), "params": {"matrix": np.arange(12, dtype=np.float32).reshape(3, 4)},
                "opt_state": {"0": {}, "1": {"count": np.array(7), "mu": np.ones(4, np.float32)}},
                "rng": np.array([0, 42], np.uint32), "data_position": np.array(112)}
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, original)
    loaded = load_checkpoint(path)
    np.testing.assert_array_equal(loaded["params"]["matrix"], original["params"]["matrix"])
    np.testing.assert_array_equal(loaded["opt_state"]["1"]["mu"], original["opt_state"]["1"]["mu"])
    np.testing.assert_array_equal(loaded["rng"], original["rng"])
    assert loaded["data_position"] == 112


def test_launcher_accepts_only_v5e_16():
    assert resource_is_v5e16({"acceleratorType": "v5litepod-16"})
    assert resource_is_v5e16({"acceleratorConfig": {"type": "V5LITEPOD", "topology": "4x4"}})
    assert not resource_is_v5e16({"acceleratorType": "v5litepod-8"})
    assert not resource_is_v5e16({"acceleratorType": "v6e-16"})


def test_native_model_parameter_shapes_and_tied_embedding():
    from flax import traverse_util
    from pilot_core import make_model
    # Set the macOS-only import alias, then inspect the unreduced native model.
    make_model(cpu_fixture=True)
    model = make_model(cpu_fixture=False)
    shapes = jax.eval_shape(lambda key: model.init(key, tokens=jnp.ones((1, 1), jnp.int32)),
                            jax.random.PRNGKey(42))["params"]
    flat = traverse_util.flatten_dict(shapes)
    assert sum(value.size for value in flat.values()) == 268_098_176
    assert flat[("embedder", "input_embedding")].shape == (262144, 640)
    assert sum(path[0].startswith("layer_") for path in flat) > 18
    assert all(value.dtype == jnp.float32 for value in flat.values())
