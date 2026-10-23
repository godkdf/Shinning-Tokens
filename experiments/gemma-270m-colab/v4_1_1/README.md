# V4.1.1

Upload `../gemma_270m_colab_v5e1_V4_1_1.ipynb` to Colab, select TPU v5e-1,
use the same Drive account and run top to bottom. Only the notebook is needed.

This version derives from V4 and changes the tokenizer to the official pinned
Gemma 4 tokenizer already registered in `configs/registry.json`. The existing
Gemma3_270M transformer stays intact. Its tied embedding/output table changes
from 262148 × 640 to 262147 × 640: **268100096 parameters**, 640 fewer than V4.
No pretrained or previous-run weights are loaded.

Raw corpus/source/dependency caches remain shared. Gemma 4 encoded and packed
caches have different tokenizer hashes. Training stays at 20 minutes with V4's
bounded dual-context batch discovery, 3M/1M schedule and existing measurements.
No new loss layout, compilation strategy, intermediate evaluation, split
profile or checkpoint restoration check is introduced.

`build_notebook.py` derives the self-contained runner and notebook exclusively
from V4. `build_provenance.json` records base/output hashes and tokenizer pins.
`INTRO.md` and `REQUIREMENTS.md` are embedded in the notebook. No auxiliary
Python files or tokenizer assets need uploading.

Authoring verification is static only. No accelerator initialization, device
tests, training or performance benchmarks are run locally.
