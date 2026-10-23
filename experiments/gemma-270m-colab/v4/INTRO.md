# Gemma 270M — V4

One **20-minute /1,200-second** training attempt on one **TPU v5e-1**. Keep
V3.9.1's **256 MiB expanded corpus**, **268,100,736-parameter** default Gemma,
native dense full-vocabulary loss, BF16 computation, FP32 master weights,
AdamW at **0.0003**, **5 warmup updates**, clipping **1.0**, decay **0.01**.
Run all cells top to bottom. Earlier notebooks and cached data are preserved.

## Encode once; pack twice

On a cold preparation, encode each train/evaluation document, verify byte-exact
decoding, densely map its IDs and calculate original-byte attribution **once**.
Save a checksum-protected, context-independent encoding. Build the **1,024**
and **2,048** context packings sequentially from it; never encode the training
corpus a second time just to change context length. Preallocate source chunks
and read encoded token arrays through memory maps to limit CPU RAM overhead.
The frozen document order, virtual BOS/EOS, masks, byte denominators and packing
algorithm remain the same. Final context caches stay compatible with V3.9.1.

If both matching packed caches already exist, reuse them and do **zero**
encoding. If one is missing, reuse/build the independent document encoding
and create the missing packing. Cold raw text uses the same **6 source workers**,
the **514 MiB English JSON cache ceiling**, per-stratum quota filtering and
deduplication-refill fix from V3.9.1. Data/cache checksums remain mandatory.

## Discover both batch sizes, then stop checking

Use at most the first **256 official optimizer updates** for discovery, in
**16-update measurement blocks**, alternating short and long contexts while
their controllers remain active. Each context starts at batch **1** and can
increase by **one sequence** per admitted candidate. Discovery may end earlier
when both contexts encounter their memory/throughput limits.

For each context independently:

- Keep compiler-estimated peak memory, inactive context inputs, evaluation
  inputs and additional observed allocations within **85% of device capacity**.
- Record the inherited workspace projection, then inspect the next candidate's
  actual compiler buffer estimate before executing it. A projection alone does
  not declare the maximum. Missing memory information stops increases.
- A larger batch must improve useful dispatch tokens/s by at least **2%**
  over that context's previous accepted batch. Otherwise keep the earlier batch.
- Only a candidate that completes a real measurement block can be selected.
- Safety cap: **32 sequences**. At the 256-update boundary, retain each context's
  largest accepted, measured batch. No claim of finding the physical OOM limit.

All discovery updates train the **same model and Adam state**, count toward
the official score/token total, and preserve independent context row cursors.
There are no throwaway optimizer probes, model resets or checkpoint restores.
Candidate compilation remains **inside the 1,200-second wall-clock budget**.
No candidate compile starts with less than **60 seconds** remaining.

Once both batches are selected, print **BATCHES LOCKED** and use those values
for the remainder of the run. **No later ramp checks, measurement windows,
candidate compilations, or 100-update dispatch boundaries.** Keep ordinary
progress, finite-value checks and memory sampling.

## Preserve the original context cycle

Discovery deliberately uses both contexts before the normal cycle begins.
Its tokens still count toward the cumulative token counter. After locking,
use the existing token intervals with both selected batches:

| Cumulative scored tokens | Shape |
|---|---|
| 0–3M, 4–7M, 8–11M, … | Selected short batch ×1,024 |
| 3–4M, 7–8M, 11–12M, … | Selected long batch ×2,048 |

Switch after the first complete optimizer update crossing a boundary. No
partial gradients or dropped targets. Finish at 20 minutes, not a token limit.
Both contexts have already been measured, so the normal cycle uses both even
if the selected short batch is below 3.
The same fixed 1,024-context holdout is scored initially and finally; report
the 16-language and original 12-language FineWiki macro BPB.

Save FP32 weights only; **no restoration check**. Keep the single mixed
post-save diagnostic: **16 updates at each selected context/batch**, 32 total,
with disposable weights. Hardware-counter availability and cost-model estimates
remain separate. Missing required counters still fail profiling completeness.

V4 does **not** add a custom vocabulary-loss backward/layout, change compilation
reuse, adopt a 9M/1M schedule, add intermediate BPB, or create separate profiles.
HF_TOKEN still requires Gemma and [Stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol)
access. TPU execution remains for the user; no device tests were run here.
