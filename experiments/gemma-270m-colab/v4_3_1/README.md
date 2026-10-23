# V4.3.1

Upload `../gemma_270m_colab_v5e1_V4_3_1.ipynb` to Colab, select TPU v5e-1,
use the same Drive account and run top to bottom. Only the notebook is required.

Based on V4.2.2, changing the tokenizer to the pinned official bigscience/bloom
ByteLevel BPE. Same Gemma3_270M transformer, with a 250683 ×640 tied vocabulary
table and **260763136 total parameters**. No pretrained weights are loaded.

Keep **3 ×1024 /1 ×2048** initially, discovery after **1000 completed updates**
(earlier on a recoverable memory failure), the 256-update discovery budget,
3M/1M schedule outside discovery and the **1200-second** total training budget.
Controller, memory recovery, evaluation and profiling functions are preserved.

`bloom_tokenizer.py` supplies the official tokenizer loader and exact token-byte
adapter. BLOOM normalizer, pretokenizer and added-token matching are retained.
Encode once, then build both context packings; BLOOM hash/schema identities
keep its token arrays separate from previous versions. Raw corpus and training
environment pins remain unchanged. Previously installed tiktoken dependencies
may remain in the shared environment; BLOOM encoding does not use them.

The pinned official BLOOM vocabulary cannot represent vertical tab U+000B.
Preparation explicitly rejects it instead of silently dropping bytes. Other
round-trip failures also stop; the corpus and official vocabulary stay intact.

`build_notebook.py` exclusively creates the embedded runner and notebook from
preserved V4.2.2. `build_provenance.json` records their hashes and tokenizer pin.
Documentation is embedded; no extra source files need uploading.

Authoring validation: static notebook/source checks and CPU tokenizer/data
checks. No TPU/device initialization, model training or performance benchmarks.
