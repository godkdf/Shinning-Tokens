"""Artifact text transformation for the V3.2 compiled-call repair."""


def apply_invocation_fix(source):
    source = source.replace(
        '    compilation_seconds = time.monotonic() - phase\n    if int(state.step) != 0:',
        '''    # Execute through jax.jit so JAX supplies the closed-over kernel constants.
    # A dynamic zero request warms the SAME training signature without an update.
    state, warmup_metrics = jax.block_until_ready(group_jit(state, training, jnp.array(0, jnp.int32)))
    if int(warmup_metrics["updates"]) != 0 or int(state.step) != 0 or int(state.data_position) != 0:
        raise ValueError("The zero-update JIT warmup changed the optimizer/data cursor.")
    compilation_seconds = time.monotonic() - phase
    write_json(output / "dispatch_preparation.json", {
        "repair": "V3.2-jit-dispatch-1", "dispatch": "jax.jit wrapper",
        "training_warmup_requested_updates": 0, "training_warmup_completed_updates": int(warmup_metrics["updates"]),
        "pretraining_optimizer_step": int(state.step), "pretraining_data_position": int(state.data_position),
        "compiled_objects_used_for": "precompilation and memory inspection only",
        "initial_evaluation": "normal JIT dispatch before the timer; includes first-call dispatch overhead"})
    if int(state.step) != 0:''', 1)
    source = source.replace('evaluators[split](parameters, device_data[split])',
                            'evaluation_jit(parameters, device_data[split])')
    source = source.replace('executable(state, training, jnp.array(requested, jnp.int32))',
                            'group_jit(state, training, jnp.array(requested, jnp.int32))')
    source = source.replace('"kernel_verification_seconds": kernel_report["seconds"], "compilation_seconds": compilation_seconds,',
        '''"kernel_verification_seconds": kernel_report["seconds"], "compilation_seconds": compilation_seconds,
        "dispatch_preparation": read_json(output / "dispatch_preparation.json"),
        "compilation_timing_scope": "Precompilation plus zero-update training JIT warmup; first evaluator JIT dispatch is in initial_eval_seconds",''')
    return source
