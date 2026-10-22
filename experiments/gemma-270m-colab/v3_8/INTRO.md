# Gemma 270M — Colab TPU v5e-1 — V3.8

One **180-second** training run, **batch 3 × 1,024 = 3,072 positions/update**.
V3.8 restores V3.5's native dense vocabulary head and exact full softmax.
AdamW uses learning rate **0.0003**, **5 warmup updates**, then a constant rate;
weight decay **0.01**. No chunked head, cooldown, optimizer sweep or batch fallback.
Default model: **268,100,736 randomly initialized parameters**, BF16 computation,
FP32 master weights, tied embeddings, native Gemma attention and frozen byte corpus.

Final evaluation and one FP32 weights-only save follow training. **No checkpoint
restoration, parameter reload comparison or reload evaluation is performed.**
The normal file checksum and artifact-schema checks remain part of saving.

Training memory is recorded in GiB and percent of reported device capacity.
After saving, **4 × 8 = 32 diagnostic updates** profile disposable in-memory
state; their weights are discarded. Record operation times, device idle, host
enqueue/wait/gap durations, raw hardware-counter evidence and compiler-based
compute/HBM estimates. Counter requests do not guarantee hardware support.
Missing required measurements produce **profiling FAIL**, independently of
training correctness. No missing measurement becomes a default 0%.

V3.5's **45,734 tokens/s**, **2.9347 FineWiki macro BPB** came from batch 2 × 1,024
and approximately 60 seconds. V3.8 changes both batch and duration; historical
differences are descriptive, not a controlled implementation comparison.

Exactly 180 seconds at the supplied **2.92 CU/hour** is **0.146 CU**. Preparation,
compilation, evaluation, saving and diagnostics add connected runtime. Reuse the
existing pinned environment and verified data caches. Run cells from top to bottom;
reports, weights and tokenizer files are saved to a new V3.8 Drive run directory.
