# V4.4.1

Upload `../gemma_270m_colab_v5e1_V4_4_1.ipynb` to Colab, select TPU v5e-1,
use the same Drive account and run top to bottom. Only the notebook is required.

Based on V4.3.1 with original Llama 3 tokenization through native tiktoken.
The pinned public redistribution is checked against both its JSON hash and
Meta's original tokenizer.model hash. Pattern, special-ID table and reference
chunking match original Llama 3. Special spellings are ordinary text; no native
BOS/EOS or chat template is added. No model weights are downloaded.

Same Gemma3_270M transformer, resized tied vocabulary table 128259 ×640,
**182411776 parameters**. Keep initial **3 ×1024 /1 ×2048**, delayed discovery
after **1000 completed updates** (earlier on recoverable memory failure),
3M/1M schedule outside discovery and **1200-second** total training budget.
Controller, memory recovery, evaluation, optimizer and measurements are retained.

Reuse frozen raw data and package pins. Encode once and pack both contexts
with exact native byte lengths. Llama 3 cache identities preserve prior arrays.
The output tokenizer directory contains source tokenizer.json, reconstructed
tokenizer.model, native constructor configuration, ID mapping, identity, license
and use policy. Source JSON is data; encoding executes native tiktoken.

`llama3_tokenizer.py` supplies embedded tokenizer code. `build_notebook.py`
exclusively creates the notebook and runner from preserved V4.3.1.
Provenance records base/output hashes and tokenizer pins. Documentation and
all runtime helpers are embedded; no supporting files need uploading.

Verification is static and CPU tokenizer/data checks. No TPU/device tests,
accelerator initialization, model training or performance benchmarks.
