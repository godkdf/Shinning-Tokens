# V3.7 engineering contract

Preserve all earlier notebook versions. Use the V3.5 pinned dependency lock,
including JAX 0.12.0 and XProf 2.24.0. Reuse verified corpus, tokenization,
environment, and compilation caches. No package search, batch search, or
automatic repeat of a failed training attempt.

The inherited 28 V3.5 requirements continue to cover device identity, source
pins, fixed byte corpus, split separation, lossless tokenization, intact
packing, attention masks, resident inputs, Gemma architecture, AdamW routing,
timed dispatch, finite updates, fixed evaluation, artifact schema, byte
accounting, and bounded source downloads.

V3.7 interprets the official-update requirement as applying to the ranked
training state through checkpoint persistence. A separately labeled diagnostic
may consume disposable post-checkpoint state, with all its updates excluded
from the official score/counts and its time included in total runtime cost.

| Additional check | Required behavior |
|---|---|
| V37-029 | Chunked-head values and input/table gradients agree with the dense reference on bounded synthetic fixtures. The native-model entry point agrees with native Gemma loss on a short real-vocabulary probe. No optimizer update is performed by these checks. |
| V37-030 | Restore the persisted weights; verify full parameter SHA256 equality and maximum per-language BPB error at most 0.000001. |
| V37-031 | Record an elapsed-time cooldown multiplier between 0.30 and 1.00 for every official dispatch; retain the 5-update warmup. |

## Profiling measurements

| Measurement | Evidence and meaning |
|---|---|
| Matrix-unit utilization | XProf's explicit MXU percentage; no token-rate or FLOP proxy. |
| Memory-bandwidth utilization | XProf's explicit HBM bandwidth utilization percentage, distinct from memory capacity usage. |
| Host/device idle time | Separate XProf percentages. Host waiting for TPU work is not automatically a problem. Record dispatch gaps and synchronized-call durations separately. |
| Operation timings | Exclusive HLO self-time, grouped using attention, vocabulary projection, loss, and optimizer metadata. Fused/unclassified work remains explicit; do not sum overlapping parent timings. |
| Training memory | Active and reserved allocator bytes/GiB and percentage of the reported device limit, sampled at synchronized training boundaries and progress points. Record sampling overhead inside the training clock. Separate lifetime peaks and compiled buffer estimates. |

Capture 4 diagnostic dispatches of 16 updates after checkpoint verification.
Use named scopes, device/host tracing, and TPU performance counters. Save raw
XPlane/Perfetto data plus XProf overview, HLO stats, roofline, and op-profile
exports. Every extracted metric has a status, unit, source, and missing-data
reason. Telemetry completeness is reported independently of training success.
No missing measurement may silently appear as zero or be labeled measured.

The 60-second interval excludes preparation, correctness checks, compilation,
evaluation, saving, restoration, and profiling. All those stages consume real
connected runtime. Estimate CU at the supplied 2.92 CU/hour, with no dollar
conversion. Total runtime includes diagnostic capture and analysis.

Performance goals are comparisons, not engineering guarantees: throughput at
least the historical V2 45,559.448 tokens/s, compilation at most 60 seconds,
whole-notebook runtime at most 300 seconds, occupancy at least 95%, and FineWiki
macro BPB at most the measured V3.5 2.9347027242183685 when its corpus/tokenizer
hashes match. Report each goal separately. Restored weights remain weights-only.
