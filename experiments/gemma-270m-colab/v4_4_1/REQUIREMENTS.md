# V4.4.1 engineering contract

- Preserve V4.3.1's transformer, initialization fingerprint, loss, optimizer,
  package pins, data recipe, budget, evaluation, profile and batch-discovery
  controller/memory recovery. Only tokenization and vocabulary dimensions change.
- Use original Llama 3 from pinned NousResearch/Meta-Llama-3-8B revision
  315b20096dc791d381d514deb5f8bd9c8d6d3061, tokenizer.json SHA256
  e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7.
- Reconstruct all 128000 native byte ranks and require Meta original rank-file
  SHA256 82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55.
  Validate the 256 original special IDs. Retain pinned license/use-policy files.
- Use native tiktoken 0.14.0, Meta's regex, 400000-character slices and
  25000-character run limits. Ordinary-text policy; no automatic BOS/EOS,
  normalization, chat template or Hugging Face encoder execution.
- Densely map 128256 IDs to 1..128256; protocol PAD=0, BOS=128257, EOS=128258.
  Tie input/output table 128259 ×640; exactly 182411776 FP32 master parameters.
- Verify exact native token bytes against every frozen raw document; assign
  actual byte lengths even for UTF-8 fragments. Never drop or normalize bytes.
- Encode once and pack both contexts sequentially. Preserve raw/source caches;
  use separate original-Llama-3 encoded/packed cache schemas and tokenizer hash.
- Preserve initial 3 ×1024 /1 ×2048, discovery after 1000 completed updates
  (earlier on recoverable memory failure), 256 discovery updates, 16-update
  windows, 85% admission, +2% acceptance, cap 32 and no checks after locking.
- Preserve 1200 total training seconds and the cumulative 3M/1M schedule outside
  discovery. No checkpoint restoration, model reset or hidden training retry.
- Create a new artifact exclusively. Static and CPU tokenizer/data checks only;
  no accelerator initialization, device tests or model training during authoring.
