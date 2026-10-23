# V4.5.1 engineering contract

- Preserve the previous transformer, loss, optimizer, package pins, original
  frozen source corpus, discovery controller, memory recovery, timer and profile.
- Use official Qwen/Qwen3-0.6B tokenizer.json at revision
  c1899de289a04d12100db370d81485cdf75e47ca, SHA256
  aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4.
- Preserve official NFC, BPE, regex and all 26 added-token definitions. Disable
  only automatic special-token insertion through add_special_tokens=False.
  No chat template or remote Python. No model weights are downloaded.
- 151669 defined IDs plus 3 protocol tokens; tied table 151672 ×640, total
  197396096 parameters. Same non-embedding initialization fingerprint.
- Keep original raw files unchanged. Build a separate official-NFC view with
  identical document counts, order, splits and provenance. Record original
  span hashes/byte lengths plus normalized hashes/counts. Do not refill quotas.
- Verify normalized-view identity and checksums before reuse. Reject new exact
  train/held-out span collisions introduced by normalization.
- Encode once and pack both contexts sequentially. Exact reconstruction and
  byte counts refer to NFC text. The BPB denominator is normalized UTF-8 bytes.
- Explicitly label metrics, reports and UI as normalized-text scoring. Export
  both source and scored manifests plus raw/NFC totals and changed-document
  counts. Describe compatibility fields accurately; never call them raw BPB.
- Keep initial 3 ×1024 /1 ×2048, discovery after 1000 successful updates
  (earlier on recoverable memory failure), 256 discovery updates, 16-update
  windows, 85% admission, +2% acceptance, cap 32, then permanent locks.
- Retain 1200 total training seconds, 3M/1M cycle outside discovery, continuous
  Adam/model state, initial/final evaluation and weights-only saving. No restore.
- Exclusively create a new artifact. Static and CPU data/scoring checks only;
  no TPU/device tests or actual model training during authoring.
