# Gemma 270M — V3.9

One **1,200-second (20-minute)** training attempt on **one TPU v5e** with a
**256 MiB original UTF-8 corpus**, **16 human languages**, and six code/markup
buckets. Run cells top to bottom. Earlier notebooks and corpus caches stay intact.

Start at **1 × 1,024 for the first 100 updates**. At every **100 official
updates**, inspect timing and memory. Increase short batch by **one sequence**
when the next executable fits **85% of reported device capacity**, including
the inactive context's resident inputs and fixed evaluation buffers. The
remaining **15%** is headroom, not a guarantee against every runtime allocation.

Each increased batch must deliver at least **2% more useful tokens/s** over
its next **100 short-context training updates**. Otherwise return to the prior
batch and hold. Completed updates, weights and optimizer moments are retained.
There are **zero throwaway batch-probe training updates**. The numeric safety
cap is **32 sequences**; reaching it does not establish a global optimum.
Checks continue every 100 updates after holding. Long-context checks defer
short-batch decisions until enough short-context updates are available.

Once short batch reaches **3**, use cumulative **scored training tokens**:

| Repeating interval | Training shape |
|---|---|
| 0–3 million tokens | Current accepted short batch × 1,024 |
| 3–4 million tokens | 1 × 2,048 |
| 4–7 million tokens | Current accepted short batch × 1,024 |
| 7–8 million tokens | 1 × 2,048 |

Repeat every **4 million tokens**. Switch immediately after the first complete
optimizer update crossing a boundary. Overshoot is less than one update's
scored tokens. Do not cut gradients, discard targets or reset the data cursor.
The two packings have independent persistent row cursors over the same text;
switching back resumes that packing's next row. This intentionally revisits
the same corpus under different contexts; reported exposure includes repeats.

Initial short/long compilation and evaluation occur before the training timer.
**Every later candidate compilation counts inside the 1,200-second wall-clock
budget**, and its time is also reported separately. Do not initiate a candidate
compile with less than **60 seconds** remaining. An in-flight compile can
overrun the deadline; record it and start no further optimizer update.

Preserve the V3.8 native dense exact full-vocabulary loss, AdamW, learning rate
**0.0003**, **5 warmup updates then constant**, BF16 compute, FP32 master
weights, gradient clipping **1.0**, decay **0.01**, and tied embeddings. Default
model: **268,100,736 random parameters**. Batch changes do not rescale the LR.

Initial/final evaluation stays at **1,024 tokens**. Report the **16-language
FineWiki macro BPB**, the original **12-language subset macro**, all languages,
and code/markup domain BPB. Expanded text and training duration make historical
scores descriptive comparisons.

Save one FP32 weights-only artifact. **No checkpoint restoration or reload
evaluation.** After saving, capture **32 disposable diagnostic updates**:
16 at the final short batch and 16 at 1 × 2,048. Operation categories average
these shapes and are labelled accordingly. Hardware counters, roofline
estimates and host API timings remain separate. Missing matrix-unit or HBM
counters are **UNAVAILABLE**, and profiling completeness **FAILS**; this does
not invalidate saved training weights.

**Before running:** the Hugging Face account used by HF_TOKEN must have access
to [Gemma](https://huggingface.co/google/gemma-3-270m) and
[Stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol).
The latter requires accepting its dataset access conditions. Access is checked
before large corpus preparation; no substitute data is silently used.
