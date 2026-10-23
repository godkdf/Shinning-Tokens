# V4.6.1 engineering contract

- Preserve the established model, optimizer, data recipe, dependency pins,
  discovery controller, memory recovery, evaluation and measurements.
- Use official deepseek-ai/DeepSeek-V3-Base tokenizer.json at revision
  afb92e1fa402c2be2a9eb085312bb02e0384d6c7, SHA256
  621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41.
- Preserve the official identity normalization, BPE/pretokenizer and native
  added-token matching. No automatic native BOS/EOS or chat template.
- Require 128815 contiguous native IDs. Map to 1..128815; packing PAD=0,
  BOS=128816, EOS=128817. Tied table 128818 ×640; 182769536 parameters.
- Keep frozen original source bytes. Encode once; verify exact byte
  reconstruction for every document; pack both contexts sequentially.
  Added tokens and partial UTF-8 pieces retain exact byte attribution.
- Separate DeepSeek encoded/packed caches; reuse raw/source and runtime caches.
- Start 3 ×1024 /1 ×2048. Discover after 1000 successful updates, earlier
  after recoverable memory failure. At most 256 discovery updates, 16-update
  windows, 85% admission, +2% throughput acceptance, cap32. Lock once.
- Keep 1200 total training seconds, cumulative 3M/1M schedule outside
  discovery, continuous model/Adam state and no checkpoint restoration.
- Exclusively create a new notebook; preserve previously delivered artifacts.
  Static and CPU tokenizer/data checks only during authoring; no device tests.
