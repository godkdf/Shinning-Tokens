# Gemma backbone — V4.6.1 / DeepSeek-V3-Base tokenizer

One **20-minute / 1,200-second** training run on **one TPU v5e-1**.
Keep the Gemma3_270M transformer and use the official
[DeepSeek-V3-Base tokenizer](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base).
Only the vocabulary dimensions change. No pretrained model weights are loaded.
Run this self-contained notebook top to bottom.

| Item | V4.6.1 |
|---|---:|
| Base vocabulary entries | 128,000 |
| Added-token declarations, including 3 overlapping base IDs | 818 |
| Total defined native IDs | **128,815** |
| Native IDs marked special | 804 |
| Additional packing PAD/BOS/EOS | 3 |
| Tied embedding/output table | **128,818 × 640** |
| Vocabulary parameters | **82,443,520** |
| Transformer/norm parameters | **100,326,016** |
| Total parameters | **182,769,536** |
| Initial batches | **3 × 1,024 / 1 × 2,048** |
| Discovery trigger | **1,000 completed updates**, earlier on recoverable memory failure |

Download only official `tokenizer.json`, pinned to revision
`afb92e1fa402c2be2a9eb085312bb02e0384d6c7` and require SHA256
`621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41`.
Execute the official BPE/pretokenizer with the already pinned Hugging Face
tokenizers package. Preserve native added-token matching. Set
`add_special_tokens=False`; no native BOS/EOS injection or chat template.
The official empty normalizer sequence leaves original text unchanged.

Map native IDs 0..128814 to model IDs 1..128815. Protocol PAD=0,
BOS=128816 and EOS=128817. Native ID 0 remains a real scored token after
remapping, so it cannot be confused with padding. Fresh random initialization,
seed and shared backbone fingerprint are retained. This is the Gemma model
with DeepSeek tokenization, not the DeepSeek transformer architecture.

Reuse the frozen **256 MiB** raw corpus, source caches and holdouts. Encode
each document once, reconstruct its exact original UTF-8 bytes, assign exact
token-byte lengths, then pack 1,024 and 2,048 sequentially. Added-token
spellings use literal UTF-8 bytes; ordinary pieces can contain partial UTF-8
characters. Reject any byte mismatch. DeepSeek encoding/packing namespaces
and tokenizer hashes preserve earlier tokenizer caches. **BPB uses original
source bytes**, with no NFC text view. HF_TOKEN remains for corpus sources.

Start at **3 × 1,024 / 1 × 2,048** for **1,000 successful updates**. Then
discover both batches in at most **256 additional real updates**, using
**16-update windows**, **+1 sequence** per candidate, **85% memory admission**,
**2% throughput improvement** and cap **32**. Lock once; no later rechecks.
No candidate compile starts with under **60 seconds** left. Discovery updates
and candidate compilation consume the same **1,200-second** wall-clock budget.
Model/Adam state and context cursors continue without resets.

Keep the repeating **3M short / 1M long processed-token schedule** outside
discovery. Discovery briefly measures both contexts before returning to that
cumulative schedule. Preserve early memory fallback: initial short-batch
rejection can use 1 × 1,024 and trigger discovery early; execution recovery
requires surviving valid buffers and matching counters. Batch-one OOM,
invalid donated state or a killed runtime stops the run. No checkpoint
restoration or silent restart.

Retain BF16 computation, FP32 master weights, tied vocabulary head, native
dense vocabulary loss, AdamW **0.0003**, **5** warmup updates, clipping
**1.0**, decay **0.01**, seed **42**, initial/final held-out BPB, memory
accounting, weights-only saving and the existing post-save hardware profile.
Report missing hardware counters as unavailable; label estimates separately.
Compare original-byte BPB and bytes/s across original-byte tokenizers;
token loss and tokens/s use different token units. Qwen3's NFC-scored report
uses a different text basis and needs that distinction retained.

Authoring validation is static and CPU tokenizer/data checks only.
No accelerator initialization, device tests or model training.
