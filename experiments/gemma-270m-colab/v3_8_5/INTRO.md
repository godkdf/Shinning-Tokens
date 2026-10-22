# Gemma 270M — V3.8.5 — batch 2 × 2,048

One **180-second** run on **TPU v5e-1**. The default batch is **2 sequences ×
2,048 tokens = 4,096 positions per update**. Both training and evaluation use
2,048-token chunks. The longer context changes the task relative to earlier
1,024-token reports; better BPB would not isolate a batch-size improvement.

Keep the V3.8 model and training math: native dense exact full-vocabulary loss,
AdamW, **learning rate 0.0003**, **5 warmup updates then constant**, BF16 compute,
FP32 master weights, and weight decay **0.01**. The default architecture has
**268,100,736 randomly initialized parameters** and tied embeddings.

Reuse the frozen source corpus, tokenizer and pinned dependency environment.
Rebuild chunks and packing for 2,048 tokens. The new cache key includes sequence
length; old 1,024-token caches remain untouched. Reuse matching 2,048-token
caches on later runs, validating metadata, checksums and array widths.

Save one FP32 weights-only artifact. **No checkpoint restoration or reload
evaluation.** Keep training memory reporting and the separate **32-update**
post-save diagnostic; diagnostic weights are discarded. Missing hardware
counters remain explicit and fail profiling completeness, not saved training.
The operation classifier distinguishes native MLP work from its parent
`_apply_attention` scope.

Run cells top to bottom. Reports and weights go to a unique **V3.8.5** Drive
directory. Existing notebooks and caches are preserved. No automatic rerun,
batch search, fallback or learning-rate change. The configured batch is used
consistently in batching, occupancy and requirement checks.
