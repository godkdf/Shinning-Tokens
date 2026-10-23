# V3.9 engineering contract

- One chip, one initialization, one attempt, one 1,200-second interval.
- First 100 updates: 1 × 1,024; checks land exactly on multiples of 100 updates.
- At most 16 official updates per dispatch. Never discard a completed update.
- Increase short batch by one after memory admission, using real training
  windows for timing. A <2% improvement rolls the shape back one and ends the
  ramp. No weights, Adam moments, step count or learning-rate reset.
- Memory gate: compiler arguments + outputs − aliases + temporaries, plus
  inactive context and evaluation inputs, ≤85% of device capacity. Gate the
  next compilation with a conservative extrapolation first. Missing estimates
  or capacity stop increases. Do not intentionally provoke OOM or retry training.
- Context schedule follows cumulative actual scored tokens, including EOS,
  excluding padding. Enable long context only while accepted short batch ≥3.
  Complete the crossing update and record the exact transition token count.
- Separate row cursors preserve next-row continuity across shapes and modes.
  Gather cyclically without batch-size padding or repeated first rows on switches.
- Expanded source quotas sum to 10,000 basis points of original UTF-8 bytes.
  Freeze revisions and raw hashes on first preparation; share the new corpus
  across later tokenizers. Repository-based code splitting prevents files from
  the same identified repository crossing train/evaluation boundaries.
- Prepare both context caches with distinct keys and checksums. Fixed scoring
  uses only the 1,024-context holdout, 64 KiB per language plus 1 MiB in-mixture.
- CPU prefix accounting must match device token totals after every dispatch,
  and final byte totals across both contexts. Accounting arrays stay on CPU.
- Finite checks reject a bad update; no automatic attempt or batch retry.
- Preserve model, exact dense loss, optimizer, precision and initialization.
- Save official evaluation and FP32 weights before diagnostic updates. Verify
  file hashes/schema without loading weights back into a model.
- Record allocator memory samples, reported capacity, compiler estimates by
  shape, total memory estimates including both resident contexts, batch checks,
  context switches, compilation timing, useful tokens/s, bytes/s and CU estimates.
- Request matrix-unit and HBM counters, host/device idle and operation timings.
  Unsupported/absent measurements are unavailable and fail profiling
  completeness; cost-model estimates must never fill measured-counter fields.
- Profiling: 16 final-short updates +16 long updates; mixed operation averages
  explicitly labelled. Diagnostic tokens and weights excluded from official
  results, but diagnostic time included in total runtime/cost.
- Separate engineering, learning, performance and profiling statuses. Throughput
  ≥45,734 tokens/s, pretraining compilation ≤180 s, total report ≤1,800 s and
  useful occupancy ≥95% are targets, not promises. Preparation may exceed them.
- Builders exclusively create V3.9 files and refuse to overwrite delivered files.
