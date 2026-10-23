"""Adaptive controls. Only completed, official training updates inform timing."""


def validate_training_config(config):
    if (config["sequence_length"] != 1024 or config["long_sequence_length"] != 2048
            or config["local_batch_per_chip"] != 1 or config["training_seconds"] != 1200.
            or config["batch_check_updates"] != 100 or config["short_phase_tokens"] != 3000000
            or config["long_phase_tokens"] != 1000000):
        raise ValueError("V3.9 requires initial 1x1024, long 1x2048, 100-update checks and 1200 seconds.")
    if not 0 < config["memory_budget_fraction"] < 1:
        raise ValueError("Memory budget must leave device headroom.")
    if not isinstance(config["safety_batch_cap"], int) or config["safety_batch_cap"] < 1:
        raise ValueError("Safety batch cap must be a positive integer.")


def context_phase(tokens, short_batch):
    return "long" if short_batch >= 3 and tokens % 4000000 >= 3000000 else "short"


def tokens_to_boundary(tokens, short_batch):
    if short_batch < 3:
        return None
    position = tokens % 4000000
    return (3000000 if position < 3000000 else 4000000) - position


def bounded_dispatch(prefix, rows, batch, proposed, tokens, short_batch):
    """Switch on the first complete optimizer update crossing a token boundary."""
    remaining = tokens_to_boundary(tokens, short_batch)
    if remaining is None:
        return proposed
    start = int(completed_prefix(prefix, rows))
    for count in range(1, proposed + 1):
        if int(completed_prefix(prefix, rows + count * batch)) - start >= remaining:
            return count
    return proposed


def compiler_memory(executable):
    try:
        analysis = executable.memory_analysis()
        values = {key: int(getattr(analysis, key)) for key in
            ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")}
        if any(value < 0 for value in values.values()):
            return None
        return values
    except (AttributeError, TypeError, NotImplementedError):
        return None


def compiler_peak(values):
    if values is None:
        return None
    return (values["argument_size_in_bytes"] + values["output_size_in_bytes"]
        - values["alias_size_in_bytes"] + values["temp_size_in_bytes"])


def projected_candidate_peak(analyses, candidate, other_resident_bytes):
    """Conservative compilation gate, distinct from the next executable's estimate."""
    known = sorted((b, compiler_peak(v)) for b, v in analyses.items() if v is not None)
    if not known:
        return None
    b, peak = known[-1]
    if len(known) > 1:
        earlier_b, earlier_peak = known[-2]
        slope = max(0., (peak - earlier_peak) / (b - earlier_b))
        # A tiny/negative adjacent slope must not justify an unlimited jump.
        slope = max(slope, analyses[b]["temp_size_in_bytes"] / b)
        projected = peak + (candidate - b) * slope * 1.10
    else:
        projected = peak + analyses[b]["temp_size_in_bytes"] * (candidate - b) / b * 1.10
    return int(projected) + other_resident_bytes


def batch_decision(current, throughput, previous_throughput, projected, limit, fraction, cap):
    if previous_throughput is not None and throughput < previous_throughput * 1.02:
        return {"action": "rollback_and_hold", "batch": current - 1,
            "reason": "The increased short batch failed to improve useful dispatch tokens/s by at least 2%."}
    if current >= cap:
        return {"action": "hold", "batch": current, "reason": "Safety batch cap reached; no claim of a global optimum."}
    if projected is None or limit is None:
        return {"action": "hold", "batch": current, "reason": "Compiler memory estimate or device capacity unavailable; increase cannot be justified."}
    if projected > limit * fraction:
        return {"action": "hold", "batch": current, "reason": "Conservative next-batch projection exceeds the device memory budget."}
    return {"action": "compile_candidate", "batch": current + 1,
        "reason": "Measured current window and projected memory permit checking the next executable."}


def make_training_functions(model, optimizer, batch_size=1, mode="short"):
    cursor_field = "short_rows" if mode == "short" else "long_rows"
    def one_step(state, batch):
        def objective(params):
            losses = token_losses(model, params, batch)
            count = jnp.sum(batch["mask"], dtype=jnp.float32)
            nll = jnp.sum(jnp.where(batch["mask"], losses, 0.))
            return nll / jnp.maximum(count, 1.), count
        (loss, count), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        norm = optax.global_norm(grads)
        finite = jnp.isfinite(loss) & jnp.isfinite(norm) & (count > 0)
        def apply(_):
            with jax.named_scope("optimizer_update"):
                delta, new_opt = optimizer.update(grads, state.opt_state, state.params)
                next_params = optax.apply_updates(state.params, delta)
            return state.replace(step=state.step + 1, params=next_params, opt_state=new_opt,
                data_position=state.data_position + 1,
                **{cursor_field: getattr(state, cursor_field) + batch_size})
        return jax.lax.cond(finite, apply, lambda _: state, operand=None), (loss, count, norm, finite)
    def group(state, training, requested):
        initial = (state, jnp.array(0, jnp.int32), jnp.array(0., jnp.float32),
                   jnp.array(0, jnp.int32), jnp.array(0., jnp.float32), jnp.array(True))
        def condition(carry):
            return (carry[1] < requested) & carry[5]
        def body(carry):
            state, steps, nll, count, max_norm, finite = carry
            indices = (getattr(state, cursor_field) + jnp.arange(batch_size)) % training["tokens"].shape[0]
            batch = jax.tree.map(lambda values: values[indices], training)
            state, (loss, tokens, norm, good) = one_step(state, batch)
            return (state, steps + good.astype(jnp.int32), nll + loss * tokens,
                count + jnp.where(good, tokens, 0).astype(jnp.int32),
                jnp.maximum(max_norm, norm), finite & good)
        state, steps, nll, count, norm, finite = jax.lax.while_loop(condition, body, initial)
        return state, {"updates": steps, "nll_sum": nll, "tokens": count,
                       "max_gradient_norm": norm, "finite": finite}
    return one_step, jax.jit(group, donate_argnums=(0,))


def row_training_arrays(arrays, seed):
    order = np.random.default_rng(seed).permutation(arrays["tokens"].shape[0])
    return {key: value[order] for key, value in arrays.items()}, hashlib.sha256(order.tobytes()).hexdigest()


def row_prefix_counts(arrays):
    return training_prefix_counts({key: value[:, None, :] for key, value in arrays.items()})


def training_specification_matches(config, training):
    return (config["local_batch_per_chip"] == 1 and config["training_seconds"] == 1200.
        and training["short"]["tokens"].shape[1:] == (1024,)
        and training["long"]["tokens"].shape[1:] == (2048,))
