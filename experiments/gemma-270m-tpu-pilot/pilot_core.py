"""JAX model, globally weighted loss, and replicated-state training primitives."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct, traverse_util
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


@struct.dataclass
class PilotState:
    step: jax.Array
    params: object
    opt_state: object
    rng: jax.Array
    data_position: jax.Array


def make_model(cpu_fixture=False):
    # Kauldron's dtype registry assumes Linux's extended-precision NumPy alias.
    # This compatibility alias is used only by the explicitly marked Mac CPU fixture.
    if cpu_fixture and not hasattr(np, "float128"):
        np.float128 = np.longdouble
    from gemma import gm
    model = gm.nn.Gemma3_270M(dtype=jnp.float32)
    if cpu_fixture:
        config = dataclasses.replace(model.config, num_embed=128, embed_dim=32,
            hidden_dim=64, num_heads=2, head_dim=16, num_kv_heads=1,
            attention_types=model.config.attention_types[:2], sliding_window_size=8)
        model = model.clone(config=config)
    return model


def host_value(array):
    if isinstance(array, jax.Array):
        return np.asarray(array.addressable_shards[0].data)
    return np.asarray(array)


def make_mesh():
    devices = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
    mesh = Mesh(np.array(devices), ("data",))
    replicated = NamedSharding(mesh, P())
    data = NamedSharding(mesh, P("data", None))
    local_indices = [i for i, device in enumerate(devices) if device.process_index == jax.process_index()]
    return mesh, replicated, data, local_indices


def make_optimizer(config, params):
    schedule = optax.join_schedules([
        optax.linear_schedule(0.0, config["learning_rate"], config["warmup_updates"]),
        optax.constant_schedule(config["learning_rate"]),
    ], [config["warmup_updates"]])
    mask = jax.tree.map(lambda value: value.ndim > 1, params)
    optimizer = optax.chain(optax.clip_by_global_norm(config["gradient_clip"]),
        optax.adamw(schedule, b1=config["adam_b1"], b2=config["adam_b2"],
                    eps=config["adam_eps"], weight_decay=config["weight_decay"], mask=mask))
    return optimizer, schedule


def loss_parts(model, params, batch):
    # Master weights remain FP32. Only the forward computation sees BF16 copies.
    compute_params = jax.tree.map(lambda value: value.astype(jnp.bfloat16), params)
    logits = model.apply({"params": compute_params}, tokens=batch["tokens"]).logits
    logits = logits.astype(jnp.float32)
    target_logits = jnp.take_along_axis(logits, batch["targets"][..., None], axis=-1)[..., 0]
    losses = jax.scipy.special.logsumexp(logits, axis=-1) - target_logits
    weights = batch["mask"].astype(jnp.float32)
    return jnp.sum(jnp.where(weights > 0, losses, 0.0)), jnp.sum(weights)


def build_steps(model, optimizer, schedule, replicated, data, global_batch):
    def train(state, batch):
        def objective(params):
            numerator, denominator = loss_parts(model, params, batch)
            return numerator / jnp.maximum(denominator, 1.0), denominator
        (loss, tokens), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        grad_norm = optax.global_norm(grads)
        finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm) & (tokens > 0)

        def apply_update(_):
            updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
            params = optax.apply_updates(state.params, updates)
            return state.replace(step=state.step + 1, params=params, opt_state=opt_state,
                                 data_position=state.data_position + global_batch)

        # A nonfinite gradient is reported and never applied to the weights.
        new_state = jax.lax.cond(finite, apply_update, lambda _: state, operand=None)
        return new_state, {"loss": loss, "tokens": tokens, "grad_norm": grad_norm,
                           "finite": finite, "learning_rate": schedule(state.step)}

    def evaluate(params, batch):
        return loss_parts(model, params, batch)

    train_jit = jax.jit(train, in_shardings=(replicated, data),
                        out_shardings=(replicated, replicated), donate_argnums=(0,))
    eval_jit = jax.jit(evaluate, in_shardings=(replicated, data), out_shardings=replicated)
    return train_jit, eval_jit


def initial_state(model, config, replicated):
    def initialize(key):
        variables = model.init(key, tokens=jnp.ones((1, 1), jnp.int32))
        return variables["params"]
    params = jax.jit(initialize, out_shardings=replicated)(jax.random.PRNGKey(config["seed"]))
    jax.block_until_ready(params)
    if not all(value.dtype == jnp.float32 for value in jax.tree.leaves(params)):
        raise TypeError("Master parameters must all be FP32.")
    optimizer, schedule = make_optimizer(config, params)
    state = PilotState(step=jnp.array(0, jnp.int32), params=params,
                       opt_state=optimizer.init(params), rng=jax.random.PRNGKey(config["seed"]),
                       data_position=jnp.array(0, jnp.int32))
    state = jax.device_put(state, replicated)
    jax.block_until_ready(state)
    return state, optimizer, schedule


def state_parameter_info(state, model):
    flat = traverse_util.flatten_dict(state.params)
    parameter_count = sum(value.size for value in flat.values())
    embedding = flat[("embedder", "input_embedding")]
    tied = (embedding.shape == (model.config.num_embed, model.config.embed_dim)
            and sum(value.shape == embedding.shape for value in flat.values()) == 1)
    layer_path = next(path for path, value in sorted(flat.items())
                      if path[0].startswith("layer_") and value.ndim > 1)
    return parameter_count, tied, layer_path


def transformer_probe(state, path):
    leaf = traverse_util.flatten_dict(state.params)[path]
    return np.array(host_value(leaf).reshape(-1)[:64])


def make_batch(arrays, row_indices, local_indices, data, valid_rows=None):
    local_rows = np.asarray(row_indices)[local_indices]
    batch = {key: np.asarray(arrays[key][local_rows]) for key in ("tokens", "targets", "mask")}
    if valid_rows is not None:
        batch["mask"] = batch["mask"] & np.asarray(valid_rows)[local_indices, None]
    return {key: jax.make_array_from_process_local_data(data, value) for key, value in batch.items()}


def evaluate_all(params, arrays, executable, local_indices, data, global_batch, manifest_hash):
    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(arrays["tokens"]), global_batch):
        indices = np.arange(start, start + global_batch)
        valid = indices < len(arrays["tokens"])
        batch = make_batch(arrays, np.minimum(indices, len(arrays["tokens"]) - 1),
                           local_indices, data, valid)
        numerator, denominator = jax.block_until_ready(executable(params, batch))
        total_nll += float(host_value(numerator))
        total_tokens += int(host_value(denominator))
    if total_tokens == 0:
        raise ValueError("Validation contains zero scored tokens.")
    return {"loss": total_nll / total_tokens, "nll_sum": total_nll,
            "scored_tokens": total_tokens, "manifest_hash": manifest_hash}


def check_global_loss(mesh, replicated):
    sharding = NamedSharding(mesh, P("data"))
    device_indices = [i for i, d in enumerate(mesh.devices.flat) if d.process_index == jax.process_index()]
    weights = np.arange(1, mesh.size + 1, dtype=np.float32)
    values = weights ** 2
    numerator = jax.make_array_from_process_local_data(sharding, values[device_indices])
    denominator = jax.make_array_from_process_local_data(sharding, weights[device_indices])
    fn = jax.jit(lambda n, d: n.sum() / d.sum(), in_shardings=(sharding, sharding), out_shardings=replicated)
    actual = float(host_value(jax.block_until_ready(fn(numerator, denominator))))
    expected = float(values.sum() / weights.sum())
    return bool(np.isclose(actual, expected, rtol=1e-6, atol=1e-6))
