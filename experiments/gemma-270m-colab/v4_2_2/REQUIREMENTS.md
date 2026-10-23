# V4.2.2 engineering contract

- Preserve V4.2.1's native o200k implementation, tokenizer/rank hashes, exact
  byte accounting, encoding and packing cache identities, raw data, backbone,
  optimizer, dependency pins, checkpoint policy and measurement code.
- Start at 3 ×1024 and 1 ×2048; use those batches for exactly 1000 successful
  updates unless a recoverable initial memory failure triggers discovery earlier.
- Truncate dispatches exactly at global update 1000. No earlier candidate
  compilation unless the explicit memory-fallback path has been activated.
- Use the previous bounded controller with initial batches 3/1, a 256-update
  discovery budget, 16-update windows, +1 candidates, 85% memory admission,
  +2% throughput acceptance and cap 32. After locking, stop discovery entirely.
- Discovery windows temporarily override context exposure. All successful
  updates count; preserve model/Adam state and both data cursors. Keep the
  cumulative 3M-short/1M-long cycle outside discovery and the 1200-second budget.
- A memory-rejected initial batch may fall back to 1; exclude failed batch
  sizes from later discovery. Retry no committed training update. Check original
  buffer validity and counters before recovering an actual execution error.
- Stop if one sequence cannot fit, buffers were invalidated, runtime died or
  memory recovery cannot preserve the last committed state. No model reset,
  checkpoint restoration, error swallowing or non-memory-error retries.
- A candidate compilation OOM keeps the previous measured batch. Compilation
  performs no optimizer updates. Failures during an active discovery execution
  or locked training still stop; they are not silently restarted.
- Record policy, actual trigger/start/finish, early-memory events, decisions,
  compilation costs and context transitions. Initial/final evaluation and
  mixed post-save profiling stay unchanged apart from using selected batches.
- Exclusively create a new version. Static and CPU host-control tests only;
  no TPU/device tests or actual model training during authoring.
