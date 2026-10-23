# V4.5.1

Upload `../gemma_270m_colab_v5e1_V4_5_1.ipynb` to Colab, select TPU v5e-1,
use the same Drive account and run top to bottom. Only the notebook is needed.

Official Qwen3 tokenizer with NFC **enabled**, following the user's explicit
choice. Same Gemma3_270M transformer; tied vocabulary table 151672 ×640,
**197396096 parameters**. The tokenizer SHA256 and revision are pinned.

Keep initial **3 ×1024 /1 ×2048**, discovery after **1000 completed updates**
(earlier on recoverable memory failure), 3M/1M context cycle outside discovery,
and **1200-second** total training budget. Controller, memory recovery,
optimizer and profile code remain unchanged.

The 256 MiB frozen source corpus stays untouched. A separate checksum-protected
NFC view preserves document order/splits/provenance. Normalize with the official
tokenizer normalizer, encode once, then pack both contexts. New exact normalized
train/held-out collisions stop preparation. Cache identities are separate.

BPB and bytes/s use **NFC-normalized UTF-8 bytes**, explicitly labelled. These
scores have a different text basis from earlier raw-byte reports. Original
source and scoring manifests plus changed-document/raw/NFC counts are exported.
Legacy `original_*` byte fields contain normalized scoring bytes for compatibility;
explicit scoring-byte fields and scope metadata accompany them.

`qwen3_tokenizer.py` supplies the official pipeline/byte adapter. `nfc_view.py`
builds the immutable scoring copy. `build_notebook.py` exclusively creates the
self-contained runner/notebook from preserved V4.4.1. Provenance records their
hashes and tokenizer pin; all documentation/helpers are embedded.

CPU tokenizer, normalized-view, packing and scoring checks plus static source
checks during authoring. No TPU tests, accelerator initialization or training.
