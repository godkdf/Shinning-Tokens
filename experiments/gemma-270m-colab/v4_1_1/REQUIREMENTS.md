# V4.1.1 engineering contract

- Preserve V4's Gemma3_270M transformer, optimizer, dependency lock, frozen
  raw corpus, holdouts, encode-once preparation and context packing algorithm.
- Use only `google/gemma-4-E4B` tokenizer revision
  `411aa17b749aa952df1359d2dcea73917a544d9a`; verify tokenizer.json SHA256
  `12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6`.
- Expect 262,144 contiguous native IDs. Retain the 3 virtual protocol tokens:
  PAD=0, BOS=262145, EOS=262146; native IDs map to 1..262144.
- Resize the existing tied embedding/output table to 262147 × 640 through
  `config.num_embed`. Expect exactly 268100096 FP32 master parameters.
  Retain the non-embedding initialization fingerprint and under-300M check.
- Disable native automatic special-token insertion. Preserve native tokenizer
  normalization, vocabulary and decoder; byte-exact corpus checks still apply.
- Key document encoding and both packed caches by the Gemma 4 tokenizer hash.
  Reuse shared raw bytes/source locks; never reinterpret Gemma 3 token arrays.
- Keep 1200 seconds, at most 256 real discovery updates, 16-update windows,
  85% memory admission, +2% throughput acceptance and safety cap 32. Lock both
  context batches once; keep the repeating 3M-short/1M-long token schedule.
- Preserve initial/final 1024-context evaluation, the mixed post-save profile,
  hardware-counter availability reporting, memory accounting and operation times.
- Save/hash/schema-check fresh FP32 weights only. No checkpoint restoration.
- Include tokenizer identity and vocabulary adaptation in the results and
  report archive. Retain separate engineering/learning/performance/profile status.
- Exclusively create a new notebook and runner. No on-device tests or training
  during authoring; use static syntax, notebook and preservation checks.
