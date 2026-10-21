# Kimi tokenizer inventory

Checked **2026-10-21** against the official [Moonshot AI repositories](https://huggingface.co/moonshotai/models). The pinned inventory contains **19 repositories**, **16 included text-capable checkpoints**, and **3 excluded non-text pipelines**. Fifteen checkpoints form **8 new profiles**; **Kimi Dev 72B** is an exact alias of the existing **Qwen 2.5** profile.

| Profile | Included checkpoints | Ordinary pieces | Special IDs | Total IDs | Literal special markers |
|---|---|---:|---:|---:|---|
| `moonlight` | Moonlight 16B-A3B; Instruct | 163,584 | 258 | 163,842 | Recognized |
| `kimi_vl` | Kimi VL Instruct; Thinking; Thinking 2506 | 163,584 | 258 | 163,842 | Recognized |
| `kimi_k2_base` | Kimi K2 Base | 163,584 | 258 | 163,842 | Ordinary text |
| `kimi_k2` | Kimi K2 Instruct; Instruct 0905 | 163,584 | 256 | 163,840 | Recognized |
| `kimi_k2_thinking` | Kimi K2 Thinking | 163,584 | 256 | 163,840 | Recognized |
| `kimi_linear` | Kimi Linear Base; Instruct | 163,584 | 258 | 163,842 | Recognized |
| `kimi_k25` | Kimi K2.5; K2.6; K2.7 Code | 163,584 | 256 | 163,840 | Recognized |
| `kimi_k3` | Kimi K3 | 163,584 | 256 | 163,840 | Recognized |
| Existing `qwen25` alias | Kimi Dev 72B | 151,651 | 14 | 151,665 | Native JSON added-token policy |

All eight use the same ordinary rank file, **2,795,286 bytes**, SHA-256 `b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103`. Identical ordinary pieces do not make the complete tokenizers identical: their configured special strings, reserved ranges or default marker recognition differ. The 28 new Kimi-to-Kimi ordinary-vocabulary pairs consequently have **Jaccard = 1**, while contextual segmentation can still differ for literal special spellings.

For example, K2 Instruct encodes `[BOS]` as **[163584]**, while K2 Base encodes the same literal text as **[72706, 4110, 60]**. Neither benchmark adapter automatically inserts BOS/EOS. K3 replaces several K2 chat-control names at existing IDs. K2.5/K2.6/K2.7 have identical inspected plain-text tokenizer settings; chat formatting and model generation changes do not create separate profiles here.

Four upstream wrappers construct **258 reserved IDs**, even though their model configurations declare only **163,840 vocabulary rows**. Their tokenizer inventory therefore reaches ID **163841**. The lab reports these two extra reserved IDs rather than silently deleting them. This count does not establish that such IDs can be passed to a checkpoint's embedding matrix.

Kimi Dev's `tokenizer.json` is byte-identical to the existing Qwen 2.5 artifact, SHA-256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`. Native serialization also yields the same complete pipeline identity. Its alias, pinned revision and license notice are added without duplicating its frozen corpus measurements.

That existing profile applies **NFC normalization**: **11,521 / 431,570 inputs (2.6696%)** change bytes, all through canonical Unicode normalization, with **0 execution errors**. These remain byte-exact fidelity failures. The eight new Kimi/Moonlight profiles do not normalize and reconstruct all sampled inputs exactly.

## Scope and reproduction

The new profiles run **431,570 original inputs / 867 partitions each**, with the same UTF-8, UTF-16LE/BE, vocabulary, throughput and memory protocol as v0.2. See [v0.4 results](../reports/v0.4.md). Each input is Unicode text; multimedia tokenizers are not evaluated on pixels, media frames, audio codes or chat templates.

The adapter uses native tiktoken and records the inspected direct encode method's regex, complete special map, **400,000-character slices** and **25,000-character consecutive whitespace/non-whitespace safeguards**. Full Transformers wrapper parity is **not independently established**. No repository code executes through `trust_remote_code`.

Use `configs/kimi-selection.json` for these eight profiles, or select a smaller tokenizer/language/dataset scope in the local launcher. Omitted download selections mean zero files. Native text data requires no model weights. Explicitly download the desired assets while online, then cached runs work offline.

After caching those eight profiles and the full pinned corpus, this command replays, audits and assembles the extension locally:

```sh
uv run --frozen python -m tokenizer_lab.publish_kimi --run
```

The two Kimi Audio speech pipelines and the MoonViT vision encoder are excluded. API-only Kimi models without an inspected public tokenizer artifact are not represented. This inventory describes the pinned official repositories inspected, not every API tokenizer or community conversion.

Exact revisions, rank/config/source hashes, aliases and exclusions: [configs/kimi-candidates.json](../configs/kimi-candidates.json). Per-profile runtime configuration and download sizes: [configs/registry.json](../configs/registry.json). Available upstream licenses and model-card notices: [licenses/kimi/](../licenses/kimi/).
