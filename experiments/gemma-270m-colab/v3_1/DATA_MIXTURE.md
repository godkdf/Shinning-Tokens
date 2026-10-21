# V3.1 data preparation

## Fixed data mixture and scientific limits

| Training source | Original UTF-8 byte share |
|---|---:|
| FinePDFs-Edu via Smol-Data | 27.5% |
| DCLM via Smol-Data | 16.5% |
| FineWeb-Edu via Smol-Data | 11.0% |
| FineWeb2, 11 non-English languages | 30.0% |
| FineMath `finemath-3plus` | 5.0% |
| CodeParrot clean Python | 5.0% |
| FineTranslations, original text + English translation | 5.0% |
| **Total** | **100.0%** |

The first three sources use [Smol-Data's globally shuffled English mixture](https://huggingface.co/datasets/HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled). Its upstream 50/30/20 recipe uses reference-token counts; V3 reads its source labels and enforces **27.5/16.5/11.0% original-byte quotas itself**. It does not assume token shares equal byte shares.

FineWeb2's 30% is split: French **3%**, Spanish **3%**, Portuguese **2.4%**, Chinese **4.5%**, Arabic **4.5%**, Hindi **4.5%**, Swahili **2.4%**, Zulu **1.5%**, Xhosa **1.5%**, Setswana **1.5%**, Sesotho **1.2%** of total training bytes. These are declared screening priorities, including Southern African languages, not measured optimal weights.

Translations allocate **1.25%** of total bytes to each French/Arabic/Hindi/Swahili source bucket, including its English side. Sides are separate documents with one shared source-group split; neither can attend to the other. Only educational score **≥2** and non-early-stopped rows are accepted. These English sides are synthetic translations, unlike the extracted PDF source. Actual language exposure is reported rather than assumed to be 50/50.

FineWiki reserves **64 KiB per language** across English plus the 11 languages above, approximately **768 KiB total**. It is a separate domain holdout. The in-mixture set contains approximately **256 KiB**, allocated by the training source weights. UTF-8 boundaries allow up to **3 bytes shortfall per bucket**. Selected source spans are at most **32 KiB**; clipping preserves the exact source prefix, with no normalization.

Canonical URL/source groups use a deterministic hash split: **10% of groups eligible for evaluation, 90% for training**; the evaluation byte quota is much smaller than 10% of the training quota. Code groups are whole repositories. Global exact source/span hashes remove exact repeats across splits, and Wikipedia URLs are excluded from non-Wiki inputs. This does not prove absence of near duplicates or web copies without identifiable URLs.

Dataset commits are resolved **once on first preparation**, saved in the shared Drive source lock, and reused with frozen raw hashes for all tokenizer runs. Sampling uses bounded deterministic stream prefixes, followed by seeded packing/order; it is not a uniform sample of each full corpus. Realized 60-second source shares can differ from pool weights and are recorded.

The [OLMo 3 paper](https://arxiv.org/abs/2512.13961) motivates high-quality web/PDF/code/math coverage and byte-based evaluation. The [dataset-mixing article](https://huggingface.co/blog/codelion/optimal-dataset-mixing) and [Smol-Data collection](https://huggingface.co/collections/HuggingFaceFW/smol-data) inform the static English mixture. **The final multilingual weights are an engineering choice, not a reproduced optimal-mixture result.** V3 performs no mixture search, optimizer sweep or multiple pilot runs.

Primary BPB is `sum(original-text token negative log probabilities) / (original UTF-8 bytes × ln(2))`. Virtual EOS loss is reported separately. For example **69,314.7 nats over 100,000 bytes → 1 BPB**. Language macro BPB gives each of the 12 languages equal weight; pooled BPB weights them by bytes. Accumulation is FP32. All candidates score the same text, but a 1,024-token context covers different byte lengths; this is a fixed-token-context, fixed-time screen.

Vocabulary IDs are densely remapped with separate PAD/BOS/EOS protocol IDs. Total parameters change by **640 per added vocabulary row**; the backbone stays identical, and its initial parameter SHA256 must match the shared baseline. The notebook fails if a tokenizer changes selected text on decode or exceeds the 300M limit. It supports Hugging Face `tokenizer.json` artifacts without remote tokenizer code.

The timed throughput report includes tokens/s and **attributed original bytes/s**. Byte attribution uses the monotonic original-character offset frontier; each source byte is counted once per full pass. It is not a claim that every tokenizer token has an independent byte boundary. Repeated passes count as repeated processed bytes. This one-seed, one-minute result can reject obvious failures; it cannot establish a final scientific tokenizer ranking.

V2's recorded **45,559 tokens/s** used different data, batch, optimizer and vocabulary handling. V3 reports that historical number only as context. It does not claim a controlled speed ratio or compare old token loss with new BPB.

V3.1 fetches source prefixes concurrently, then merges in the same canonical source order. Its initial cache target is **125% of each byte quota**, with at least **4 candidate spans per split/source bucket**, to allow global deduplication without reopening most streams. Only the original byte quotas enter the final corpus. If the buffer is insufficient, the same pinned stream continues from its committed cursor, within the original row limit and **256 MiB per-source cached JSONL cap**. The cap bounds cached candidate bytes, not HTTP traffic or Parquet row-group read sizes.

Source batches target **4 MiB** and commit after complete input rows, so a batch can exceed that target by one row. Commit attempts also occur after **5 seconds of row processing**; a blocked network read cannot be checkpointed mid-read. An interrupted job resumes at its saved row cursor; `skip()` can reread earlier source data when the remote format cannot seek directly. Completed caches avoid refetching those jobs. The original V3 implementation did not persist individual source samples before finalizing the corpus, so its unfinished sequential progress cannot be recovered from console logs. A complete, verified V3 raw corpus remains reusable.

