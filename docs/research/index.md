# Tokenizer transfer and adaptation

These **9 references** answer different questions: can we reuse a row, initialize a new vocabulary, map vector coordinates, or train a model to imitate another tokenizer’s model? None is a tokenizer-only benchmark.

**Evidence status: source reviewed; no local reproduction.** Published results and upstream code are not Lab measurements. [Evidence policy](../../research/README.md).

| Method | What transfers | Preparation / training |
|---|---|---|
| [Exact-row copy baseline](exact-row-copy.md) | Existing matching rows | Copy only |
| [Subtoken averaging / Fast Vocabulary Transfer](fvt.md) | Mean of source subpiece rows | Arithmetic initializer |
| [WECHSEL](wechsel.md) | Bilingual-neighbor row combinations | Bilingual resources; later adaptation |
| [FOCUS](focus.md) | Shared-anchor row combinations | Auxiliary embeddings; later adaptation |
| [Constrained Word2Vec (CW2V)](cw2v.md) | Context-trained added rows | Initializer training; later adaptation |
| [Embedding-space mapping](embedding-mapping.md) | Vector coordinates | Fit on paired anchors |
| [Zero-Shot Tokenizer Transfer (ZeTT)](zett.md) | Hypernetwork-predicted rows | Compatible hypernetwork must already exist |
| [Universal Logit Distillation (ULD)](uld.md) | Teacher behavior | Teacher inference + student training |
| [Approximate Likelihood Matching (ALM)](alm.md) | Teacher behavior | Teacher inference + student training |

## Read results correctly

Lower token counts show more compact segmentation on that text. They do not establish better language modeling. High embedding cosine similarity does not establish useful generation. Compare adapted models on identical held-out original UTF-8 text, report per-language results and normalize likelihood by bytes when tokenizer segmentations differ.

For controlled studies, preserve the same parent model and adaptation budget, record all vocabulary changes, and include simple exact-copy, random and subtoken-mean controls. Keep teacher inference, auxiliary training and adaptation costs separate.

Each page’s “Lab evaluation” section is proposed evaluation guidance, not a reported paper result. Complete requirements and source metadata are in [the catalog](../../research/methods.json).
