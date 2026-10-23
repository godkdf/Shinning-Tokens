# Gemma backbone — V4.2.1 / native o200k

One **20-minute / 1,200-second** run on **one TPU v5e-1**. Use the same
Gemma3_270M transformer backbone with native **tiktoken `o200k_base`**.
Run the notebook top to bottom. No earlier notebook is replaced.

| Setting | Value |
|---|---:|
| Short training batch | **3 ×1,024 =3,072 positions/update** |
| Long training batch | **1 ×2,048 =2,048 positions/update** |
| Batch discovery / ramp / later rechecks | **Disabled** |
| Training budget | **1,200 seconds** |
| Context cycle | **3M short /1M long scored tokens**, repeating |
| Frozen training corpus | **256 MiB** |
| Defined native vocabulary IDs | **200,000** |
| Additional packing PAD/BOS/EOS | **3** |
| Tied embedding/output table | **200,003 ×640** |
| Vocabulary parameters | **128,001,920** |
| Transformer/norm parameters | **100,326,016** |
| Total parameters | **228,327,936** |

Native o200k has **199,998 mergeable tokens**, **2 special tokens** and **19
unused IDs** in its 200,019-ID native address range. Densely map only the
200,000 defined IDs to 1..200000. Model PAD=0, BOS=200001 and EOS=200002.
Use the existing virtual packing tokens and tied output head. The smaller
vocabulary reduces the total by **39,772,800 parameters versus V4**; all
transformer layers and their initialization checks stay the same.

Use [OpenAI's native tiktoken](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py),
pinned to **0.14.0**, with **regex 2026.9.29**. Verify the official o200k rank
artifact and the project's full pattern/ranks/special-token snapshot by SHA256.
This is native byte-level BPE. Raw documents use `encode_ordinary`, so literal
special-token spellings remain ordinary text and no chat template is applied.
The exported `tokenizer/o200k_base.json` contains the native encoding arguments;
it is data for `tiktoken.Encoding`, rather than a Hugging Face tokenizer file.

Reuse the frozen raw corpus and holdouts from Drive. Encode each document
once; reconstruct its exact UTF-8 bytes from native token byte pieces. Attribute
each token its actual byte length, including tokens that split a Unicode
character. Then build the 1,024/2,048 packings sequentially. Native encoding
and packing cache identities are separate from all Gemma caches. Matching
completed o200k caches can be reused on subsequent runs.

Compile **only 3 ×1,024 and 1 ×2,048**, plus the held-out evaluator, before
training. Check the two requested shapes against the existing **85% compiler
memory budget**; a rejected shape stops the run instead of changing the batch.
There are no candidate shapes or training-time compilations. The first update
uses 3 ×1,024. Repeat short-token intervals 0–3M, 4–7M, 8–11M, and long intervals
3–4M, 7–8M, 11–12M, continuing until the timer ends. Switch after the complete
update crossing each boundary, preserving model/Adam state and context cursors.

Keep V4's BF16 computation, FP32 master weights, native dense vocabulary loss,
AdamW **0.0003**, **5** warmup updates, clipping **1.0** and decay **0.01**.
Initialize fresh random weights with the same seed/backbone fingerprint.
Keep initial/final held-out BPB at context 1,024, 16-language and legacy
12-language scores, memory accounting and one mixed post-save profile at the
same two fixed shapes. Missing hardware counters remain explicitly unavailable.
BPB and original bytes/s are useful comparisons across tokenizers; token loss,
token counts and tokens/s reflect different token units.

The environment retains the existing training package pins and adds the two
native tokenizer dependencies in a separate versioned environment. HF_TOKEN
is still used for source data including Stack-smol; no Gemma tokenizer/model
download is required. Save one FP32 weights-only artifact; no restoration check.
No device tests or training benchmarks are run during authoring.
