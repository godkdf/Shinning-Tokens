# Gemma backbone — V4.4.1 / original Llama 3 tokenizer

One **20-minute /1,200-second** training run on **one TPU v5e-1**. Based on
V4.3.1, retaining the same Gemma3_270M transformer backbone and using the
**original Llama 3 tokenizer through native tiktoken**. Run top to bottom.

| Item | V4.4.1 |
|---|---:|
| Ordinary Llama 3 tokens | **128,000** |
| Native special tokens | **256** |
| Total defined native IDs | **128,256** |
| Additional packing PAD/BOS/EOS | **3** |
| Tied embedding/output table | **128,259 ×640** |
| Vocabulary parameters | **82,085,760** |
| Transformer/norm parameters | **100,326,016** |
| Total parameters | **182,411,776** |
| Initial batches | **3 ×1,024 /1 ×2,048** |
| Discovery trigger | **1,000 completed updates**, or an earlier recoverable memory failure |

Use original Llama 3, not Llama 3.1/3.2. The tokenizer comes from the pinned
public [NousResearch redistribution](https://huggingface.co/NousResearch/Meta-Llama-3-8B)
already audited in this project's registry. Its tokenizer.json SHA256 must be
`e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7` at revision
`315b20096dc791d381d514deb5f8bd9c8d6d3061`. Reconstruct the original native
rank file and require Meta's SHA256
`82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55` before use.
Copy the pinned Llama 3 license/use-policy files into the report bundle.

Use the same native **tiktoken 0.14.0** package already pinned in the environment,
with Meta's original regex and [reference chunking](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py).
Retain 400,000-character slices and 25,000-character consecutive
whitespace/non-whitespace limits. The Hugging Face tokenizer JSON is only a
verified data source; its automatic BOS postprocessor is not used. Raw documents
use ordinary-text encoding: native special-token spellings remain ordinary text,
with no automatic BOS/EOS, normalization or chat template.

Map native IDs 0..128255 to model IDs 1..128256. Protocol PAD=0, BOS=128257 and
EOS=128258. Keep the tied vocabulary head and fresh random initialization with
the same seed/backbone fingerprint. The vocabulary change gives **78,351,360
fewer parameters than the BLOOM version**; transformer layers, width, attention
and optimizer math are unchanged. No pretrained model weights are downloaded.

Reuse the frozen **256 MiB** raw corpus and holdouts. Encode each document
once, reconstruct its original UTF-8 bytes from native token pieces, and assign
exact per-token byte lengths. Then create both context packings sequentially.
Native Llama 3 encoding/packing cache schemas and tokenizer hash separate these
arrays from all earlier tokenizer versions. Shared raw/source caches and pinned
training package versions remain unchanged. HF_TOKEN is still for corpus
access, including Stack-smol.

Start at **3 ×1,024 /1 ×2,048** for exactly **1,000 successful updates**.
Then discover both batches in at most **256 additional real updates**, using
**16-update windows**, **+1 sequence** per candidate, **85% memory admission**,
**2% throughput improvement** and cap **32**. Lock once and stop rechecking.
No candidate compile starts with under **60 seconds** left. Every successful
discovery update and its compilation count inside the same **1,200-second**
budget, using continuous model/Adam state and independent context cursors.

Keep the repeating **3M-short /1M-long scored-token schedule** outside discovery.
Discovery temporarily measures both contexts, then the cumulative schedule
continues with the selected batches. Memory handling is inherited unchanged:
an initial short-batch rejection can fall back to 1 ×1,024 and start discovery
early; actual execution recovery requires valid original buffers/counters.
A batch-one OOM, invalid donated state or killed runtime stops the run.
No checkpoint restoration, model reset or silent restart is performed.

Keep BF16 computation, FP32 master weights, native dense vocabulary loss,
AdamW **0.0003**, **5** warmup updates, clipping **1.0**, decay **0.01** and
seed **42**. Keep initial/final held-out evaluation at context 1,024, memory
reports, weights-only saving and the mixed post-save diagnostic at final
batches. Missing hardware counters remain explicitly unavailable. Compare BPB
and original bytes/s across tokenizers; token loss and tokens/s use different
token units.

Static and CPU tokenizer/data checks only during authoring. No TPU/device
tests or model training are run here.
