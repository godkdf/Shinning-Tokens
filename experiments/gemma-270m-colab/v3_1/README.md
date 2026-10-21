# Gemma Colab V3.1

Open `../gemma_270m_colab_v5e1_V3_1.ipynb` in Colab Pro, select one v5e-1 TPU and Run all. The notebook is self-contained.

Changes from V3:
- Include the working upstream Tokamax dependency pin and visible setup command logs.
- Six concurrent source workers; thirty source jobs; deterministic final merge.
- Read only seven required FineTranslations columns.
- Persist source candidate batches and row cursors; reuse completed source caches.
- Reuse completed V3 raw/token caches and the shared installed environment.
- Keep one 60-second training attempt, the model, optimizer and byte-mixture settings.
- Match fused-loss gradient output dtypes to BF16 inputs before native Gemma backpropagation; preserve FP32 kernel accumulation.

`parallel_data.py` implements persistent concurrent fetching. `build_notebook.py` assembles the notebook and `runner.py` from V3 plus this implementation. Assembly does not execute notebook code or run checks.

The supplied V3 notebook recorded patched setup in 43.2 seconds, one v5e TPU, and sequential preparation through Wiki/English/FineWeb2/math/code. It contained no training measurements. Original unfinished V3 source samples were not persisted individually and cannot be recovered from logs.

No local checks or tests were run. V3.1 performance and compatibility remain pending the user's Colab run.
