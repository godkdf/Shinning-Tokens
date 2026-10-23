# V4.2.2

Upload `../gemma_270m_colab_v5e1_V4_2_2.ipynb` to Colab and run top to bottom
on TPU v5e-1. Only the notebook is required.

Preserves V4.2.1 and changes batch policy: use **3 ×1024 /1 ×2048 for the
first 1000 completed updates**, then discover both batches once, in at most
256 real updates. Keep 16-update windows, +1 candidates, 85% memory admission,
+2% throughput acceptance and cap 32. Lock both and stop rechecking.

Memory failure can trigger discovery earlier when recovery preserves state.
An initial short-batch memory rejection falls back to 1 ×1024 and excludes the
failed batch. Candidate compilation OOM retains the previous measured batch.
Invalid donated buffers, an OOM at batch 1, or a killed runtime stop the run.
There is no checkpoint restore or hidden model reset.

Keep native o200k, 228327936 parameters, 256 MiB raw corpus, pinned environment,
3M/1M schedule outside discovery, and the 1200-second training wall budget.
Both context cursors and Adam state continue throughout. All successful
discovery updates and candidate compilation count inside the same budget.

`delayed_training.py` supplies embedded host control. `build_notebook.py`
derives the notebook/runner from preserved V4.2.1 and reuses the earlier V4
discovery controller. Provenance records base/controller/output hashes.
Static and CPU simulation checks only; no accelerator tests or model training.
