# Gemma backbone — V4.3.1 / BLOOM tokenizer

One **20-minute /1,200-second** run on **one TPU v5e-1**. Based on V4.2.2,
using the official **bigscience/bloom tokenizer** with the same Gemma3_270M
transformer backbone. Run all cells top to bottom.

| Item | V4.3.1 |
|---|---:|
| Native BLOOM token IDs, including its 4 special tokens | **250,680** |
| Additional protocol PAD/BOS/EOS | **3** |
| Tied embedding/output table | **250,683 ×640** |
| Vocabulary parameters | **160,437,120** |
| Transformer/norm parameters | **100,326,016** |
| Total parameters | **260,763,136** |
| Initial short batch | **3 ×1,024** |
| Initial long batch | **1 ×2,048** |
| Batch discovery trigger | After **1,000 completed updates**, or an earlier recoverable memory failure |

The official [BLOOM tokenizer](https://huggingface.co/bigscience/bloom) is pinned
to revision `7f10a99ce7c08f03c7719a586cb2cbda1433ac05`; tokenizer.json must have
SHA256 `3fa39cd4b1500feb205bcce3b9703a4373414cafe4970e0657b413f7ddd2a9d3`.
Load its exact ByteLevel BPE pipeline using the existing pinned
**tokenizers 0.23.3** package. No new package versions or model weights are
downloaded. This trains fresh random weights with the same seed/backbone
initialization check as V4.2.2.

Map native IDs 0..250679 to model IDs 1..250680. Model PAD=0, BOS=250681 and
EOS=250682. The original 4 BLOOM special-token IDs stay native vocabulary
entries. The 3 protocol packing tokens remain separate. The tied table has
**32,435,200 more parameters than the o200k version**, and **7,337,600 fewer
than V4**. Transformer layers, width, attention and optimizer math are unchanged.

Raw documents use the official tokenizer with **add_special_tokens=False**,
retaining its native added-token matching and no chat template. The tokenizer
has no normalizer. Reconstruct exact UTF-8 bytes by inverting the ByteLevel
alphabet; native special spellings contribute their literal UTF-8 bytes.
Check every document against the original bytes. This correctly accounts for
tokens that split a Unicode character and avoids character-offset estimates.

**BLOOM vocabulary limitation:** the pinned official vocabulary has no token
containing vertical tab **U+000B**. The official encoder silently drops that
character. This notebook instead stops with an explicit error if it occurs in
the frozen corpus. It never removes the character, skips the document, changes
the corpus or adds an unofficial vocabulary token. Other byte mismatches also
stop preparation before training.

Reuse the frozen **256 MiB** raw corpus and holdouts. Encode once and pack both
1,024 and 2,048 contexts sequentially. BLOOM's tokenizer hash and byte-accounting
schemas give it separate encoding/packing caches. Previous o200k/Gemma caches
are preserved. The public BLOOM tokenizer cache persists in Drive. HF_TOKEN
continues to supply corpus access, including Stack-smol.

Use **3 ×1,024 /1 ×2,048** through completed update **1,000**. Dispatches stop
exactly at that boundary. Then run at most **256 additional real discovery
updates**, in **16-update windows**, independently for both contexts. Start
from 3/1; consider **+1 sequence** at a time, require the existing **85% memory
admission** and **2% throughput improvement**, cap at **32**, then lock once.
All successful updates and discovery compilation count inside the same
**1,200-second training budget**. No candidate compile starts with under
**60 seconds** remaining.

Outside discovery, retain the repeating **3M-short /1M-long scored-token
schedule**. During discovery, temporarily measure both contexts. Preserve the
same model, Adam state, update counter and independent context row cursors.

Memory handling is unchanged from V4.2.2. An initial short-batch memory
rejection falls back to **1 ×1,024**, starts discovery early and excludes the
failed batch. An actual execution error can recover only if the runtime and
original training buffers survive and their counters match the last committed
dispatch. A candidate compilation OOM retains the previous measured batch.
An OOM at batch 1, invalid donated buffers, or a killed runtime stops the run.
There is no checkpoint restoration, model reset or silent restart.

Keep BF16 computation, FP32 master weights, native dense vocabulary loss,
AdamW **0.0003**, **5** warmup updates, clipping **1.0**, weight decay **0.01**
and seed **42**. Keep initial/final held-out BPB at context 1,024, memory
accounting, weights-only saving and the mixed post-save diagnostic at final
batches. Missing hardware counters remain explicitly unavailable. Compare
tokenizers primarily through held-out BPB and original bytes/s; tokens/s and
token loss use different token units.

Static and CPU tokenizer/data checks during authoring only. No TPU/device
tests or model training are run here.
