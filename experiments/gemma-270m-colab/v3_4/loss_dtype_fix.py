"""Repair the Tokamax loss gradient boundary without changing its TPU math."""


def install_tokamax_loss_dtype_fix():
    """Keep FP32 kernel accumulation; cast returned cotangents to input dtypes."""
    from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu as loss_impl
    cls = loss_impl.PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp
    original = cls._fwd
    if getattr(original, "_gemma_v31_dtype_fix", False):
        return

    @functools.wraps(original)
    def fixed(self, residuals, out, dout, x, labels, w, *, reduction="sum", config, return_residuals):
        (dx, dlabels, dw), saved = original(self, residuals, out, dout, x, labels, w,
            reduction=reduction, config=config, return_residuals=return_residuals)
        # The TPU kernel accumulates gradients in FP32 even for BF16 inputs.
        # Native Gemma BF16 multiply pullbacks require BF16 cotangents.
        return (dx.astype(x.dtype), dlabels, dw.astype(w.dtype)), saved

    fixed._gemma_v31_dtype_fix = True
    cls._fwd = fixed


def fused_loss_probe(x, w, labels):
    """Exercise unreduced loss through BF16 intermediate multiply operations."""
    import tokamax
    xx = x * jnp.asarray(.5, dtype=x.dtype)
    ww = w * jnp.asarray(.75, dtype=w.dtype)
    losses = tokamax.linear_softmax_cross_entropy_loss(xx, labels, ww,
        reduction="none", implementation="mosaic_tpu")
    return jnp.mean(losses.astype(jnp.float32))


def reference_loss_probe(x, w, labels):
    xx = x * jnp.asarray(.5, dtype=x.dtype)
    ww = w * jnp.asarray(.75, dtype=w.dtype)
    logits = jnp.matmul(xx.astype(jnp.float32), ww.astype(jnp.float32))
    return jnp.mean(jax.scipy.special.logsumexp(logits, axis=-1) - logits[jnp.arange(x.shape[0]), labels])
