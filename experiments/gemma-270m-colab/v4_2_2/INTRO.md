# Gemma backbone — V4.2.2 / native o200k, delayed batch discovery

One **1,200-second /20-minute** run on **one TPU v5e-1**. Keep V4.2.1's
native **tiktoken 0.14.0 `o200k_base`**, frozen **256 MiB** corpus, same
Gemma3_270M backbone and **228,327,936** total parameters. Run top to bottom.

| Stage | Behavior |
|---|---|
| Updates **1–1,000** | **3 ×1,024** short / **1 ×2,048** long |
| After completing update **1,000** | Start independent batch discovery for both contexts |
| Discovery | At most **256 additional real updates**, in **16-update windows** |
| After both batches lock | Use the selected batches for the rest of the run; no rechecks |
| Earlier memory failure | Start discovery early if the runtime and training state remain usable |

Dispatches are shortened to end **exactly at update 1,000**. Discovery starts
with the existing batches, measures their throughput, then considers **+1
sequence** per candidate. A larger batch must fit the **85%** compiler-memory
budget, including other resident inputs/observed allocations, and improve useful
tokens/s by **at least 2%**. Safety cap: **32 sequences** per context. Stop each
search at its memory/throughput/cap boundary. Only measured accepted increases
are selected. No new candidate compile starts with under **60 seconds** left.

Keep the repeating **3M-short /1M-long cumulative scored-token schedule**
before and after discovery. During discovery, temporarily alternate measurement
windows between the two contexts so both batches can be selected promptly.
Every successful update counts, using the same model, Adam state and independent
row cursors. Discovery compilation and measurement are **inside the same
20-minute budget**; the run does not gain extra time.

If initial 3 ×1,024 compilation fails for memory or exceeds the admission budget,
fall back to **1 ×1,024**, start discovery at update **0**, and exclude the
failed batch from subsequent increases. A memory error during the first 1,000
updates can trigger the same fallback only if all original training buffers
remain valid and their counters still match the last completed dispatch.
No failed dispatch is counted and no model/optimizer reset is performed.
Candidate compilation memory errors retain the previous measured batch.

An error at **1 sequence** cannot be fixed by reducing the batch further.
A killed/disconnected runtime or invalidated donated buffers also stops the
run. These cases are reported as failures; no checkpoint restoration or silent
restart is attempted. TPU memory execution is not tested during authoring.

All other V4.2.1 behavior stays: native byte-exact encoding once, two sequential
context packings, SHA256-verified tokenizer identity, **200,003 ×640** tied
vocabulary table, shared raw/source/native-tokenizer caches, BF16 computation,
FP32 master weights, AdamW **0.0003**, **5** warmup updates, clipping **1.0**,
decay **0.01**, seed **42**, initial/final 1,024-context held-out evaluation,
weights-only artifact and one mixed post-save profile at the final batches.
The same pinned native-tokenizer environment is reused.

Record discovery start/finish updates, memory-fallback events, decisions,
candidate compiler estimates, source-byte exposure, memory samples and context
transitions. Missing hardware counters remain explicitly unavailable. Static
and CPU host-control checks only; no accelerator/model training tests.
