# V4.6.1

Upload `../gemma_270m_colab_v5e1_V4_6_1.ipynb` to Colab, select TPU v5e-1
and run top to bottom. Only the notebook is required.

Official DeepSeek-V3-Base tokenizer with the same Gemma transformer:
128815 native IDs, tied table 128818 ×640 and **182769536 parameters**.
Preserve official added-token matching and identity normalization; no
automatic native BOS/EOS or chat template. Original UTF-8 byte scoring.

Keep initial **3 ×1024 /1 ×2048**, discovery after **1000 completed
updates** (earlier on recoverable memory failure), cumulative **3M/1M**
schedule outside discovery and **1200-second** budget. Established controller,
optimizer, memory recovery and measurements remain unchanged.

Reuse frozen raw data and package pins. Encode once, verify all original
document bytes and pack both contexts. Separate DeepSeek cache namespaces.
No pretrained weights or remote Python code are loaded.

`deepseek_tokenizer.py` supplies embedded tokenizer code. `build_notebook.py`
creates notebook, runner and hash provenance exclusively from the preserved
original-byte V4.4.1 baseline. This avoids inheriting Qwen3's NFC scoring view;
operational settings match the later tokenizer variants.

Validation is static and CPU tokenizer/data checks. No TPU/device tests,
accelerator initialization, training or performance benchmarks.
