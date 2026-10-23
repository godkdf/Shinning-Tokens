# V4

Notebook: `../gemma_270m_colab_v5e1_V4.ipynb`. Upload only this notebook to
Colab, choose TPU v5e-1, use the same Drive account, and run top to bottom.

Based on V3.9.1. Two requested changes:

1. Encode/verify byte attribution once; pack the same encoded documents at
   1,024 and 2,048 sequentially. A new encoding cache stores little-endian int32
   IDs/byte weights plus document boundaries and IDs, with full checksums.
   Existing raw/prefix/packed caches remain compatible and are never deleted.
2. Independent early batch discovery for both contexts: at most 256 real updates,
   16 updates per measurement block, +1 candidate sequence, 85% memory ceiling,
   +2% throughput acceptance, safety cap 32. Lock each at its accepted batch;
   once discovery ends, no rechecking for the rest of the run. This is bounded
   discovery of reasonable batches, not an intentional OOM search.
   Compile the next candidate and inspect its buffer estimate before executing
   it; retain projections as diagnostics rather than treating them as proof
   that a larger batch cannot fit. Rejected shapes perform zero updates.

The early discovery blocks override the context cycle so both lengths can be
measured promptly. Their tokens count toward the global counter. After locking,
retain the original cumulative 3M/1M intervals with both selected batches.
Both contexts are already measured, so neither depends on a short batch ≥3.
Maintain one model, Adam state, update counter and independent context cursors.
No dummy optimizer probes or discarded training. All discovery compilation
counts inside the unchanged 20-minute wall budget.

Native loss/layout, compilation reuse, 3M/1M exposure, initial/final-only BPB,
and the mixed post-save profile are preserved. No custom loss gradient,
9M/1M schedule, intermediate evaluation or split profiling is included.

Source layout: `encoding.py` supplies context-independent encoding/packing,
`discovery.py` supplies bounded per-context batch selection and timed dispatch,
and `build_notebook.py` derives and exclusively writes the self-contained
`runner.py`/notebook from the preserved V3.9.1 artifact. All helper code is
embedded; this folder is not uploaded separately.

Validation during authoring is static syntax, notebook structure, source
embedding, and preservation of explicitly unchanged functions/dependency pins.
No device tests or training benchmarks are run. The first Colab execution
establishes live source access, memory admission, actual batches and speed.
