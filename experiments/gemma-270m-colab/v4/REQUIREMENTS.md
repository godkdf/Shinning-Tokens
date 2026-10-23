# V4 engineering contract

- Preserve V3.9.1's source recipe/fingerprint, source commits, quota-filtered
  prefix cache identity, exact loss, model/optimizer math and dependency lock.
- Encode and verify each document once per cold encoding cache. Include dense
  token IDs, exact monotonic-frontier byte weights, document boundaries, source
  hashes and language/domain IDs in the encoding cache. No context length in
  encoding identity; tokenizer/raw hashes and encoding schema must be present.
- Checksum binary encodings and metadata before reuse. Validate contiguous
  offsets, target counts, binary lengths, and original byte totals. Old completed
  1,024/2,048 packed caches remain independently reusable without encoding again.
- Pack sequentially with the existing chunk/window/segment semantics. Preserve
  all original text targets and one virtual EOS per document in both contexts.
- Discover independently at 1,024 and 2,048. Use at most 256 official updates,
  16-update windows, one sequence per increase, and a 32-sequence safety cap.
- Apply 85% device capacity admission with the actual candidate compiler buffer
  estimate, other resident inputs and observed excess allocations. Record the
  inherited projection as diagnostic context; do not infer the maximum from
  extrapolation alone. A +2% throughput
  improvement is required for accepting a larger batch of the same context.
- Selected shapes must have completed measurement windows. Stop an individual
  search on a memory/throughput/cap boundary. Stop both when the discovery budget
  is exhausted. Never execute a candidate rejected by its compiler estimate.
- All discovery updates count; preserve model parameters, Adam moments, global
  optimizer step and both row cursors across all shape changes. No reset, resume
  restore, uncounted update or retry. Training fails on nonfinite updates.
- After both lock, perform no discovery checks, window accounting, shape search
  or 100-update dispatch truncation. Keep dispatches at most 16 updates and the
  ordinary deadline/token-boundary reductions.
- Discovery is a declared exception to the normal context cycle. After locking,
  use cumulative official tokens (including discovery): repeat 3M short/1M long,
  with both contexts already measured. Both use their selected batch even if
  the short batch is below 3.
- Keep 1,200-second wall budget, including discovery compilation. Initial
  baseline compilation stays outside as before; compilation cache logic unchanged.
- Keep initial/final combined 1,024 evaluator, fixed held-out bytes, 16-language
  and original 12-language scores. No intermediate held-out evaluation.
- Save/hash/schema-check one FP32 weights artifact; no model restoration.
- Keep one mixed 32-update diagnostic, 16 at each selected shape. No separate
  context profiles. Missing hardware counters remain explicit, not substituted
  by compiler estimates or host synchronization-call timing.
- Record `encoding_preparation.json`, `encoding_manifest.json` when used,
  `batch_discovery.json`, all discovery decisions, context/phase transitions,
  real per-shape compiler estimates, memory samples and source-byte exposure.
- Preserve separate engineering, learning, performance and profiling status.
- Build exclusively under a new V4 filename. Static validation only during
  authoring; device testing and live performance validation belong to the user.
