# V3.9.1: expanded-corpus preparation fix

Open `../gemma_270m_colab_v5e1_V3_9_1.ipynb` in Colab and run top to bottom.
Everything is embedded. Use the same Drive account and ShiningGem root.
Do not delete existing caches or the failed V3.9 run directory.

The supplied V3.9 attempt failed during source preparation, before model
training. Its executed source matches the delivered notebook. The final corpus
target was 256 MiB, but the temporary cache for each source job was capped at
256 MiB of serialized JSON. The combined English job also stored candidates
from already-satisfied buckets while waiting for its slowest bucket/split.
The report's last progress sample showed 200.1 MiB of candidate text; JSON
also includes escaping and provenance. That sample precedes the actual limit
exception and does not establish the exact text total at failure.

Changes:

- Stop retaining candidates for each bucket/split after its 125% text-byte
  target and four-document minimum. Continue scanning other upstream rows.
- Stage selection for a complete input row before touching counters, and keep
  the existing commit-after-row cursor discipline.
- Give each job an independent JSON disk limit of
  `max(256 MiB, 4 × sum(final text quotas))`. The English job receives **514 MiB**:
  128 MiB training +0.5 MiB in-mixture evaluation, multiplied by four.
- Refill only deficient bucket/split strata after global deduplication, rather
  than accumulating new candidates for completed strata as well.
- Report actual cached JSON bytes, per-stratum text bytes and each job's limit.
  Limit exceptions include numerical diagnostics and preserve committed parts.

Reuse: keep the raw recipe fingerprint, pinned source lock, prefix cache
version, job hashes, checksums and part format. Existing verified completed
jobs are reused, and partial streams resume their committed cursor. Remote
`skip()` can still reread source data where the remote format cannot seek.
The quota filter changes surplus-candidate retention, not final byte quotas.
An already-finalized verified raw corpus can also be reused unchanged.

Training/model math, 1,200-second wall budget, adaptive batch decisions,
3M-short/1M-long token schedule, evaluation, weights-only artifact and profiling
remain unchanged. A new V3.9.1 output folder preserves failed V3.9 logs. The old
per-run attempt marker is neither deleted nor reused. No checkpoint restoration.

No device tests or training runs were performed for this correction, per user
instruction. Only source syntax, notebook format and embedding are checked.
Dataset access, quota fulfilment and TPU execution remain for the Colab run.
