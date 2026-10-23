# V4.3.1 engineering contract

- Preserve V4.2.2's transformer, model initialization fingerprint, loss,
  optimizer, dependency pins, 1200-second budget, data recipe, evaluation,
  profiling and delayed discovery/memory-recovery behavior.
- Use official bigscience/bloom tokenizer.json at revision
  `7f10a99ce7c08f03c7719a586cb2cbda1433ac05`, SHA256
  `3fa39cd4b1500feb205bcce3b9703a4373414cafe4970e0657b413f7ddd2a9d3`.
- Expect 250680 contiguous native IDs. Densely map to 1..250680; reserve
  PAD=0, BOS=250681 and EOS=250682. Tie input embedding and vocabulary output
  at shape 250683 ×640; expect exactly 260763136 master parameters.
- Use the official ByteLevel BPE and native added-token matching with
  add_special_tokens=False. No normalization changes or chat templates.
- Reconstruct per-token bytes using the inverse ByteLevel alphabet, with literal
  UTF-8 for native special tokens. Verify every complete document against the
  frozen original bytes. No character-offset approximation.
- Reject U+000B explicitly because the pinned BLOOM vocabulary cannot represent
  it. Do not add a token, normalize the text, skip a document or count lost bytes.
- Encode once and pack both contexts sequentially. Preserve raw/source caches;
  use BLOOM-specific encoded and packed cache schemas plus tokenizer SHA256.
- Keep 3 ×1024 /1 ×2048 for 1000 completed updates unless an explicit memory
  fallback triggers discovery earlier. Keep exactly the prior 256-update
  discovery limit, 16-update windows, 85% admission, +2% acceptance and cap 32.
- Retain the cumulative 3M/1M schedule outside discovery, independent row
  cursors and continuous Adam/model state. No repeated discovery after locking.
- Retain safe-stop behavior for invalid buffers, batch-one OOM or dead runtime.
  Save FP32 weights only; no restoration check or silent restart.
- Exclusively create a new notebook/version; no prior artifact overwrite.
  Static and CPU tokenizer/data checks only; no device tests or training.
