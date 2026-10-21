def make_optimizer(config, params):
    """All-parameter Muon, including a declared one-row extension for vectors.

    Use the public low-level Muon transform: the high-level muon() wrapper has
    an AdamW branch. No AdamW transform is constructed here. RMS shape scaling
    is explicit, so the very wide tied table and one-row normalization vectors
    do not inherit unrelated width-dependent update magnitudes.
    """
    flat = traverse_util.flatten_dict(params)
    specs, scales, routing = {}, {}, []
    for path, value in sorted(flat.items()):
        name = "/".join(path)
        shape = tuple(value.shape)
        lifted_shape = (1, value.size) if value.ndim < 2 else shape
        if value.ndim < 2:
            reduction, output = (0,), (1,)
        elif path == ("embedder", "input_embedding"):
            # Native decoding multiplies hidden states by this table's transpose.
            reduction, output = (1,), (0,)
        elif "q_einsum" in name:
            reduction, output = (1,), (0, 2)
        elif "kv_einsum" in name:
            reduction, output = (2,), (1, 3)
        elif "attn_vec_einsum" in name:
            reduction, output = (0, 1), (2,)
        elif path[-1] == "gating_einsum":
            reduction, output = (2,), (1,)
        elif value.ndim == 2:
            reduction, output = (0,), (1,)
        else:
            raise ValueError(f"Unclassified all-Muon parameter: {name} {shape}")
        specs[path] = optax.contrib.MuonDimensionNumbers(
            reduction_axis=reduction, output_axis=output)
        fan_in = math.prod(lifted_shape[axis] for axis in reduction)
        fan_out = math.prod(lifted_shape[axis] for axis in output)
        scales[path] = config["muon_consistent_rms"] * math.sqrt(max(fan_in, fan_out))
        routing.append({"path": name, "shape": list(shape),
            "lifted_shape": list(lifted_shape), "parameters": value.size,
            "optimizer": "Muon", "reduction_axes": list(reduction),
            "output_axes": list(output), "shape_scale": scales[path],
            "vector_extension": value.ndim < 2, "weight_decay": value.ndim > 1})
    spec_tree = traverse_util.unflatten_dict(specs)
    scale_tree = traverse_util.unflatten_dict(scales)
    decay = jax.tree.map(lambda value: value.ndim > 1, params)
    def lift(tree):
        return jax.tree.map(lambda value: value.reshape((1, value.size))
                            if value.ndim < 2 else value, tree)
    def shape_update(updates, state, params=None):
        del params
        return jax.tree.map(lambda value, scale: value * scale, updates, scale_tree), state
    schedule = optax.join_schedules([
        optax.linear_schedule(0., config["learning_rate"], config["warmup_updates"]),
        optax.constant_schedule(config["learning_rate"]),
    ], [config["warmup_updates"]])
    inner = optax.chain(
        optax.clip_by_global_norm(config["gradient_clip"]),
        optax.contrib.scale_by_muon(ns_steps=config["muon_ns_steps"],
            beta=config["muon_beta"], eps=1e-8, mu_dtype=jnp.float32,
            nesterov=True, weight_dimension_numbers=spec_tree),
        optax.GradientTransformation(lambda _: optax.EmptyState(), shape_update),
        optax.add_decayed_weights(config["weight_decay"], mask=decay),
        optax.scale_by_learning_rate(schedule))
    def init_fn(parameters):
        return inner.init(lift(parameters))
    def update_fn(gradients, state, parameters=None):
        if parameters is None:
            raise ValueError("All-Muon requires parameters for decoupled weight decay.")
        updates, new_state = inner.update(lift(gradients), state, lift(parameters))
        updates = jax.tree.map(lambda update, parameter: update.reshape(parameter.shape),
                               updates, parameters)
        return updates, new_state
    count = sum(value.size for value in flat.values())
    vector_count = sum(value.size for value in flat.values() if value.ndim < 2)
    return optax.GradientTransformation(init_fn, update_fn), {
        "name": "Muon-all-with-vector-extension", "muon_parameters": count,
        "adamw_parameters": 0, "learning_rate": config["learning_rate"],
        "beta": config["muon_beta"], "newton_schulz_steps": config["muon_ns_steps"],
        "nesterov": True, "momentum_dtype": "float32",
        "shape_scaling": "consistent RMS", "consistent_rms": config["muon_consistent_rms"],
        "nominal_lr_times_rms": config["learning_rate"] * config["muon_consistent_rms"],
        "embedding_parameters": flat[("embedder", "input_embedding")].size,
        "vector_extension_parameters": vector_count,
        "vector_extension": "rank-0/1 parameters reshaped to 1 x N for Muon, then restored",
        "warmup_updates": config["warmup_updates"], "routing": routing}
