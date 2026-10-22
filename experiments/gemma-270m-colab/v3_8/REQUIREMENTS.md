# V3.8 engineering contract

Preserve every previously delivered notebook and source version. The notebook is
self-contained and embeds V3.5's successful dependency lock (JAX 0.12.0,
XProf 2.24.0); no dependency search. Reuse matching environments, corpus,
tokenization and JAX caches. Every run has a unique directory and one attempt
marker. Failed attempts are never silently retried.

## Fixed training settings

| Setting | V3.8 |
|---|---|
| TPU | Exactly one v5e chip |
| Official training budget | 180 seconds; no dispatch begins after the deadline |
| Batch × context | 3 × 1,024, 3,072 positions/update |
| Default model | Native Gemma3 270M, 268,100,736 random parameters |
| Precision | BF16 computation, FP32 master weights |
| Vocabulary loss | Native dense exact full softmax; no chunking or sampling |
| Optimizer | AdamW, LR 0.0003, 5 warmup updates then constant, decay 0.01 |
| Dispatch bound | At most 16 updates per official host dispatch |
| Save | One FP32 weights-only artifact, local/Drive checksum and schema checks |
| Restoration check | Disabled by the user; no restored arrays or reload evaluation |

The inherited V3.5 requirements retain source pins, fixed byte corpus, split
separation, lossless tokenization, intact packing, causal/segment masks,
resident inputs, native Gemma, exact optimizer routing, finite updates,
fixed evaluation bytes, weights schema, CPU byte accounting, and bounded
source downloads. V38-025 enforces batch 3 × 1,024 and 180 seconds.
V38-029 confirms that restoration was not executed. All occupancy calculations
use the configured batch and context; padding contributes zero tokens/bytes.

## Measurements

Record active/reserved allocator bytes, GiB and capacity percentages before,
during (approximately two-second boundaries), and immediately after training.
Distinguish sampled maximum, lifetime allocator high-water mark and compiler
buffer estimate. Do not add overlapping memory counters. Include sampling
overhead inside the official clock. Samples can miss temporary peaks.

After final evaluation and saving, profile exactly 4 dispatches × 8 updates of
disposable state with the existing JIT executable. Exclude these updates from
official training counts, scores and saved weights. Request host/device tracing
and TPU performance counters. Save XPlane and three XProf exports: overview,
HLO statistics, roofline. Reuse generated OpStats between conversions. Do not
request duplicate Perfetto or redundant op-profile exports.

| Required measurement | Acceptance evidence |
|---|---|
| Matrix-unit utilization | Explicit supported TPU hardware-counter percentage |
| HBM bandwidth utilization | Explicit supported TPU hardware-counter percentage |
| Host/device idle | Valid idle measurements; host-op idle alone is insufficient |
| Operation timings | Exclusive HLO self-time, ms/update and percent; attention, vocabulary projection and optimizer identified |
| Training memory | Training-scoped allocator samples and reported capacity |

Capture status, measurement status and training status are independent. Missing
any required group produces profiling FAIL and PARTIAL/UNAVAILABLE telemetry;
the completed model and report remain saved. Record raw counter-track event
counts. Do not assert a runtime failure cause without evidence. XProf default
zero percentages are not counter measurements. Host enqueue, synchronization
and dispatch-gap wall times are reported separately from CPU/host idle.

Compiler FLOP/byte costs divided by observed time are useful roofline estimates,
explicitly labelled as estimates. They do not satisfy measured-MXU/HBM checks.
Use the aggregate Program row and the same export's peaks/units; do not sum
per-operation rates or substitute total memory bandwidth for HBM bandwidth.

## Reporting and cost

Engineering targets: at least the historical V3.5 throughput of 45,734.296
useful tokens/s, compilation at most 60 seconds, complete notebook through
report at most 480 seconds, useful occupancy at least 95%. These are targets,
not guarantees. BPB improvement relative to the run's own initial evaluation is
the learning check. Historical V3.5 BPB is context, not a matched-budget goal:
batch and duration changed. Preserve all per-language scores and realized
byte shares; throughput comparison requires matching corpus/tokenizer hashes.

The timer excludes compilation, evaluation, saving and profiling. Report their
cost in total elapsed time, including trace export/copy and analysis. At the
supplied 2.92 CU/hour, 180 seconds is 0.146 CU; 480 seconds is 0.389333 CU.
Local checks cannot establish TPU speed, hardware counters or actual memory fit.
