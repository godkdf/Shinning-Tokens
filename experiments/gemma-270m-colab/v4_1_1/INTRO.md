# Gemma 270M — V4.1.1 / Gemma 4 tokenizer

One **1,200-second / 20-minute** training run on **one TPU v5e-1**, starting
from random weights. Built on V4's encode-once preparation and independent
batch discovery for 1,024 and 2,048 contexts. Run all cells top to bottom.

| Item | V4 | V4.1.1 |
|---|---:|---:|
| Transformer backbone | Gemma3_270M | Same |
| Native tokenizer IDs | 262,145 | 262,144 |
| Additional protocol PAD/BOS/EOS | 3 | 3 |
| Tied embedding/output rows | 262,148 | 262,147 |
| Embedding width | 640 | 640 |
| Backbone parameters | 100,326,016 | 100,326,016 |
| Tied vocabulary parameters | 167,774,720 | 167,774,080 |
| Total parameters | 268,100,736 | **268,100,096** |

Use the official [Gemma 4 E4B tokenizer](https://huggingface.co/google/gemma-4-E4B),
pinned to revision `411aa17b749aa952df1359d2dcea73917a544d9a` and verified against
SHA256 `12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6`.
Only tokenizer assets are downloaded. The transformer remains the same
18-layer, width-640 Gemma3_270M architecture used in V4.

All 262,144 native IDs, including its 24 declared special tokens, map to model
IDs 1–262,144. The model uses protocol PAD=0, BOS=262,145 and EOS=262,146.
The tokenizer's automatic BOS is disabled when encoding raw documents. The
input embedding and output projection share one **262,147 × 640** table.
AdamW state is initialized for this table along with the same backbone.
The seed and backbone initialization fingerprint check remain unchanged.

Reuse the same **256 MiB frozen raw corpus and holdouts** from Drive. The first
Gemma 4 preparation builds its own checksum-protected encoding and both context
packings. It cannot reuse Gemma 3 token IDs. Verify byte-exact decoding and byte
attribution once per document, then pack 1,024 and 2,048 sequentially. Later
runs can reuse matching Gemma 4 caches. Source downloads still use 6 workers.

Keep V4's settings: BF16 computation, FP32 master weights, AdamW learning rate
**0.0003**, **5** warmup updates, clipping **1.0**, decay **0.01**. Use up to
**256 real updates** for independent batch discovery, **16-update** windows,
**85%** memory admission, **2%** throughput acceptance and cap **32**. Lock
both batches and stop rechecking. All discovery updates and later candidate
compilation count inside the 20-minute training budget.

After discovery, repeat **3M short / 1M long** scored-token intervals using
the selected batches. Evaluate the same 1,024-context holdout initially and
finally. Keep all memory, timing and profiling reports, with missing counters
explicitly marked unavailable. BPB and original bytes/s are the useful tokenizer
comparisons; token loss and tokens/s depend on the tokenizer.

Save one FP32 weights-only artifact and one mixed post-save diagnostic.
Checkpoint restoration remains disabled. Native vocabulary-loss layout,
compilation reuse and evaluation/profile frequency retain V4's implementation.
This notebook trains fresh weights; it does not continue a V4 checkpoint.
Static authoring checks only; device testing belongs to the user.
