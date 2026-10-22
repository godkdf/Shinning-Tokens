"""Exact full-vocabulary loss with bounded token-chunk workspace.

Embedded into runner.py by build_notebook.py; JAX imports come from the runner.
"""

HEAD_CHUNK_TOKENS = 128


def cooldown_multiplier(elapsed, budget, start_fraction=0.5, floor=0.3):
    if budget <= 0 or not 0 <= start_fraction < 1 or not 0 < floor <= 1:
        raise ValueError("Invalid elapsed-time learning-rate schedule.")
    progress = max(0.0, min(1.0, float(elapsed) / budget))
    fraction = max(0.0, (progress - start_fraction) / (1.0 - start_fraction))
    return 1.0 - (1.0 - floor) * fraction


def exact_chunked_head(hidden, table, targets, chunk_tokens=HEAD_CHUNK_TOKENS):
    """Same BF16 projection / FP32 logsumexp as native Gemma; no sampling.

    Rematerialize each chunk in backward so scan does not retain every logit
    tensor. The tied table receives both encoding and decoding gradients.
    """
    shape = targets.shape
    hidden = hidden.reshape((-1, hidden.shape[-1]))
    labels = targets.reshape(-1)
    count = labels.shape[0]
    padding = (-count) % chunk_tokens
    x = jnp.pad(hidden, ((0, padding), (0, 0))).reshape(
        (-1, chunk_tokens, hidden.shape[-1]))
    y = jnp.pad(labels, ((0, padding),)).reshape((-1, chunk_tokens))

    @jax.checkpoint
    def body(carry, inputs):
        features, ids = inputs
        with jax.named_scope("vocabulary_projection"):
            logits = jnp.dot(features, table.T).astype(jnp.float32)
        with jax.named_scope("cross_entropy"):
            selected = jnp.take_along_axis(logits, ids[:, None], axis=-1)[:, 0]
            losses = jax.scipy.special.logsumexp(logits, axis=-1) - selected
        return carry, losses

    _, losses = jax.lax.scan(body, None, (x, y))
    return losses.reshape(-1)[:count].reshape(shape)


def native_hidden(model, tokens, positions, attention_mask):
    """Native cache-free text-only Gemma blocks, with unchanged parameter paths."""
    with jax.named_scope("input_embedding"):
        x = model.embedder.encode(tokens)
    for index, block in enumerate(model.blocks):
        with jax.named_scope("transformer_layer_" + str(index)):
            _, x = block(x, positions, None, attention_mask)
    return model.final_norm(x)


def token_losses(model, params, batch):
    compute = jax.tree.map(lambda value: value.astype(jnp.bfloat16), params)
    hidden = model.apply({"params": compute}, batch["tokens"], batch["positions"],
        packed_attention_mask(batch["segment_ids"]), method=native_hidden)
    return exact_chunked_head(hidden, compute["embedder"]["input_embedding"], batch["targets"])


def reference_token_losses(model, params, batch):
    compute = jax.tree.map(lambda value: value.astype(jnp.bfloat16), params)
    logits = model.apply({"params": compute}, tokens=batch["tokens"],
        positions=batch["positions"],
        attention_mask=packed_attention_mask(batch["segment_ids"])).logits.astype(jnp.float32)
    selected = jnp.take_along_axis(logits, batch["targets"][..., None], axis=-1)[..., 0]
    return jax.scipy.special.logsumexp(logits, axis=-1) - selected


def verify_exact_head(model, params, training):
    """Bounded correctness checks, no optimizer updates and no parameter search."""
    begun = time.monotonic()
    key_x, key_w = jax.random.split(jax.random.PRNGKey(370))
    x = jax.random.normal(key_x, (137, 32), dtype=jnp.float32) * .1
    w = jax.random.normal(key_w, (2051, 32), dtype=jnp.float32) * .1
    labels = jnp.arange(137, dtype=jnp.int32) % 2051
    checks = []
    for dtype, tolerance in ((jnp.float32, 2e-5), (jnp.bfloat16, .03)):
        xx, ww = x.astype(dtype), w.astype(dtype)
        def dense(a, b):
            logits = jnp.dot(a, b.T).astype(jnp.float32)
            return jnp.mean(jax.scipy.special.logsumexp(logits, -1)
                - jnp.take_along_axis(logits, labels[:, None], -1)[:, 0])
        def chunked(a, b):
            return jnp.mean(exact_chunked_head(a, b, labels))
        av, ag = jax.block_until_ready(jax.jit(jax.value_and_grad(dense, (0, 1)))(xx, ww))
        bv, bg = jax.block_until_ready(jax.jit(jax.value_and_grad(chunked, (0, 1)))(xx, ww))
        errors = [float(jnp.linalg.norm(a.astype(jnp.float32) - b.astype(jnp.float32))
                        / jnp.maximum(jnp.linalg.norm(a.astype(jnp.float32)), 1e-12))
                  for a, b in zip(ag, bg)]
        passed = abs(float(av) - float(bv)) <= 2e-4 and max(errors) <= tolerance
        checks.append({"dtype": str(dtype), "loss_error": abs(float(av) - float(bv)),
            "gradient_relative_errors": errors, "gradient_tolerance": tolerance, "passed": passed})
    # Verify our native-block entry point against Gemma's ordinary __call__.
    # The real vocabulary is retained; only the probe's token sequence is short.
    batch = {name: values[0, :1, :16] for name, values in training.items()}
    actual = jax.block_until_ready(jax.jit(lambda p: token_losses(model, p, batch))(params))
    expected = jax.block_until_ready(jax.jit(lambda p: reference_token_losses(model, p, batch))(params))
    model_error = float(jnp.max(jnp.abs(actual - expected)))
    passed = all(row["passed"] for row in checks) and model_error <= .005
    report = {"passed": passed, "head_checks": checks, "native_model_loss_max_abs_error": model_error,
        "native_model_tolerance": .005, "chunk_tokens": HEAD_CHUNK_TOKENS,
        "vocabulary": "all rows, exact normalizer", "optimizer_updates": 0,
        "seconds": time.monotonic() - begun}
    if not passed:
        raise ValueError("Exact chunked head failed native loss/gradient verification: " + json.dumps(report))
    return report
