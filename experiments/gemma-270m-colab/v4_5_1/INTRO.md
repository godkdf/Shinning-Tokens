# Gemma backbone — V4.5.1 / official Qwen3 NFC tokenizer

One **20-minute /1,200-second** run on **one TPU v5e-1**. Keep the same
Gemma3_270M transformer backbone and use the official **Qwen3 tokenizer with
NFC normalization enabled**, as requested. Run top to bottom.

| Item | V4.5.1 |
|---|---:|
| Base vocabulary IDs | **151,643** |
| Added IDs, including 14 marked special | **26** |
| Total defined native IDs | **151,669** |
| Additional protocol PAD/BOS/EOS | **3** |
| Tied embedding/output table | **151,672 ×640** |
| Vocabulary parameters | **97,070,080** |
| Transformer/norm parameters | **100,326,016** |
| Total parameters | **197,396,096** |
| Initial batches | **3 ×1,024 /1 ×2,048** |
| Batch discovery | After **1,000 completed updates**, or earlier on recoverable memory failure |

Use [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) tokenizer.json,
revision `c1899de289a04d12100db370d81485cdf75e47ca`, SHA256
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
The existing pinned **tokenizers 0.23.3** executes its exact NFC normalizer,
regex, BPE and added-token matching. No remote Python or model weights are
downloaded. Use **add_special_tokens=False**, no chat template, and the same
separate packing tokens. Map native IDs to 1..151669; model PAD=0, BOS=151670,
EOS=151671. Fresh random initialization retains the shared backbone check.

## Scores are explicitly on NFC-normalized text

NFC can change the source bytes: `e` followed by U+0301 uses **3 UTF-8 bytes**,
while the resulting `é` uses **2**. The model predicts the normalized text,
so **BPB divides by normalized bytes**, not by the original 3 bytes. For a
10-bit loss in that example, this is **5 BPB**, rather than incorrectly
reporting 10/3 BPB for a different text representation.

The frozen **256 MiB source corpus stays untouched**. Create a separate,
checksum-protected NFC text view with the **same documents, order, split
assignments and source provenance**. Original source coordinates and span
hashes remain recorded alongside each normalized span. No documents are
dropped, added, resampled or refilled. The normalized byte total can be larger
or smaller than the frozen source total; the normalization report records it.
If normalization creates an identical training/held-out span, stop before
training instead of introducing an undisclosed evaluation collision.

Record `raw_data_manifest.json`, `data_manifest.json` for the NFC view, and
`normalization_report.json`: raw/NFC byte totals, deltas, changed-document counts,
source/view hashes and cache reuse. The primary metric and display explicitly
say **NFC-normalized BPB** and **NFC scoring bytes/s**. This is a different
scoring basis from earlier original-byte BPB. Legacy result fields named
`original_*` remain for compatibility and explicitly contain **NFC scoring
bytes** in this version; new `processed_scoring_bytes`/`scoring_bytes_per_second`
fields state their meaning directly.

Encode the NFC view once and create both 1,024 and 2,048 packings sequentially.
Verify exact normalized-text reconstruction and per-token byte lengths,
including UTF-8 fragments and added-token spellings. Keep official NFC active
during encoding; it is idempotent on the scoring view. Separate view, encoding
and packing cache identities preserve all previous raw/tokenizer caches.

## Training policy stays the same

Start at **3 ×1,024 /1 ×2,048** for exactly **1,000 successful updates**, then
discover both batches in at most **256 additional real updates**. Keep
**16-update windows**, **+1 sequence** candidates, **85% memory admission**,
**2% throughput improvement**, cap **32**, and no checks after locking.
No candidate compile starts with under **60 seconds** left. Discovery updates
and candidate compilation count inside the same **1,200-second** budget.

Keep the repeating **3M-short /1M-long scored-token cycle** outside discovery;
temporarily measure both contexts during discovery. Preserve model/Adam state
and independent context row cursors. Recoverable initial memory failures may
start discovery early. Invalid donated state, batch-one OOM or a killed runtime
stops the run. No checkpoint restoration, model reset or silent restart.

Keep BF16 computation, FP32 master weights, native dense vocabulary loss,
AdamW **0.0003**, **5** warmup updates, clipping **1.0**, decay **0.01**, seed
**42**, initial/final held-out evaluation at context 1,024, memory reporting,
weights-only saving and the mixed post-save diagnostic. Missing counters remain
explicitly unavailable. Training package pins are unchanged. HF_TOKEN is for
the same corpus sources, including Stack-smol; Qwen3's tokenizer is public.

Static and CPU tokenizer/data/scoring checks only during authoring. No TPU
tests, accelerator initialization or model training are run here.
