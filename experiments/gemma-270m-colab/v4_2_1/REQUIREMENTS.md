# V4.2.1 engineering contract

- Use native tiktoken 0.14.0 `o200k_base`, with regex 2026.9.29. Verify ranks
  SHA256 `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`
  and full exported asset SHA256
  `97973057009b00298ee5c575a23b895c912518ded47f9dc1559032e9f2532187`.
- Densely map the 200000 defined IDs. Exclude the 19 unused native IDs. Add
  protocol PAD/BOS/EOS separately; tied table 200003 ×640, total 228327936.
- Preserve Gemma3_270M transformer layers, native loss, optimizer math, shared
  initialization fingerprint and all prior training package pins.
- Raw text uses encode_ordinary with no automatic special tokens. Verify
  native token-byte concatenation equals original UTF-8. Assign exact native
  token byte lengths; never infer byte counts from character offsets.
- Preserve raw/source identities and encode-once sequential packing. Namespace
  the changed native byte-accounting caches so Gemma token arrays cannot match.
- Fixed short batch 3 at length 1024 and long batch 1 at length 2048, from the
  beginning. Remove batch discovery, ramps and candidate compilation entirely.
- Compile the two exact shapes and the fixed evaluator before training.
  Apply 85% memory admission to those shapes. Never silently substitute batches.
- Train 1200 seconds; repeat 3M-short/1M-long cumulative scored-token intervals.
  Preserve Adam/model state, update counters and independent context row cursors.
- Keep initial/final evaluation, source-byte exposure, memory and profile
  reporting. Profiling uses only the same two shapes and disposable weights.
- Save weights only; no checkpoint restoration. Preserve failed earlier runs.
- Create a new notebook exclusively. CPU tokenizer/data checks and static
  notebook checks are permitted; no accelerator/device tests or model training.
