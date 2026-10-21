# Tokenizer families and pinned generations

The complete registry now has **54 profiles**. The **2026-10-21** extensions add native **GPT-2 (50,257 IDs)**, original **Llama 3 (128,256 IDs)** and **8 Kimi/Moonlight profiles (163,840 or 163,842 IDs)** to the 44-profile v0.2 lab. The Kimi survey includes 16 text-capable checkpoints: 15 grouped into 8 new profiles and Kimi Dev 72B as an exact Qwen 2.5 alias. The 38-profile generation survey below remains separately scoped. Llama 3 uses a byte-identical public redistribution verified against Meta's original rank-file SHA-256. See [v0.3 results](../reports/v0.3.md), [Kimi identities](KIMI.md), [v0.4 results](../reports/v0.4.md) and [the registry](../configs/registry.json).

Verified **2026-10-19**: **105 official repository representatives**, **104 text repositories included**, **38 distinct recorded tokenizer profiles**: **29 native Rust JSON profiles** and **9 declared legacy native profiles**. **0 repository inspections failed**. MiniMax H3 is excluded because it is audio/video generation rather than a text LLM generation.

The machine-readable source is [`configs/tokenizer-candidates.json`](../configs/tokenizer-candidates.json). Each representative has an immutable model revision, tokenizer asset SHA-256, canonical structural SHA-256, local tokenizer-only path, and a list of verified aliases. The catalog records the inspected checkpoint set; it does not claim that every model-size, quantization, or instruction-tuning checkpoint ever published was inspected.

## What is a different tokenizer?

A new model name is not sufficient. JSON pipelines are loaded by the native `tokenizers` library and serialized into a consistent representation. We hash the model (including token IDs and merge order), normalizer, pre-tokenizer, decoder, post-processor and added-token inventory. Padding and truncation are disabled. JSON formatting and object-key order do not create a new profile. Different special-token IDs, added-token flags or text processing do create a profile.

`ordinary_pipeline_sha256` deliberately excludes added-token matching and post-processing. Matching that field alone does **not** establish equality: a literal marker may encode differently, and special-token handling may differ. Chat-template hashes and wrapper-config hashes are recorded on each checkpoint alias, but a chat-template-only change does not create a new plain-text benchmark. Benchmarks disable automatic BOS/EOS insertion.

Legacy native profiles hash their verified assets and the declared audited wrapper settings. A matching `.model` file alone is not enough to merge ChatGLM2 and ChatGLM3. Source Python is inspected as text and is **never executed with `trust_remote_code`**. Model weights are not downloaded.

## Distinct profiles

Vocabulary counts below are tokenizer ID entries, not padded model embedding rows. Native legacy counts describe supplied assets and declared settings; the exact benchmark adapter must preserve these declarations. The full alias revisions and hashes are in the JSON catalog.

| ID | Families | Vocabulary entries | Adapter | Verified checkpoint aliases | Pinned representative |
| --- | --- | ---: | --- | ---: | --- |
| `glm_original_en` | glm | 50,265 | `legacy_gpt2` | 2 | [zai-org/glm-10b](https://huggingface.co/zai-org/glm-10b/tree/696788d4f82ac96b90823555f547d1e754839ff4) |
| `glm_original_zh` | glm | 50,010 | `sentencepiece` | 2 | [zai-org/glm-10b-chinese](https://huggingface.co/zai-org/glm-10b-chinese/tree/01717dd11fb4d3a80c465eee9eb4789bc954046d) |
| `chatglm1` | glm | 130,344 | `sentencepiece` | 1 | [zai-org/chatglm-6b](https://huggingface.co/zai-org/chatglm-6b/tree/bf0f5cfb575eebebf9b655c5861177acfee03f16) |
| `chatglm2` | glm | 64,794 | `sentencepiece` | 1 | [zai-org/chatglm2-6b](https://huggingface.co/zai-org/chatglm2-6b/tree/d2e2d91789248536a747d9ce60642a336444186c) |
| `chatglm3_base` | glm | 64,798 | `sentencepiece` | 1 | [zai-org/chatglm3-6b-base](https://huggingface.co/zai-org/chatglm3-6b-base/tree/9680e216e62c3a69bc9eec480c35d59c325079bf) |
| `chatglm3` | glm | 64,798 | `sentencepiece` | 1 | [zai-org/chatglm3-6b](https://huggingface.co/zai-org/chatglm3-6b/tree/e9e0406d062cdb887444fe5bd546833920abd4ac) |
| `glm4_native` | glm | 151,343 | `tiktoken_asset` | 1 | [zai-org/glm-4-9b](https://huggingface.co/zai-org/glm-4-9b/tree/8cd2b585357ba9e702647ac4e6fa4fafe5cc7bee) |
| `glm4` | glm | 151,343 | `huggingface` | 2 | [zai-org/glm-4-9b-chat-hf](https://huggingface.co/zai-org/glm-4-9b-chat-hf/tree/8599336fc6c125203efb2360bfaf4c80eef1d1bf) |
| `glm_edge` | glm | 59,260 | `huggingface` | 1 | [zai-org/glm-edge-1.5b-chat](https://huggingface.co/zai-org/glm-edge-1.5b-chat/tree/7b201d3c160c25beda4cf0d107617ad975cd1ca8) |
| `glm4_0414` | glm | 151,343 | `huggingface` | 2 | [zai-org/GLM-4-9B-0414](https://huggingface.co/zai-org/GLM-4-9B-0414/tree/645b8482494e31b6b752272bf7f7f273ef0f3caf) |
| `glm45` | glm | 151,365 | `huggingface` | 4 | [zai-org/GLM-4.5](https://huggingface.co/zai-org/GLM-4.5/tree/cbb2c7cfb52fa128a9660cb1a7a78e017899e115) |
| `glm47_flash` | glm | 154,856 | `huggingface` | 6 | [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash/tree/7dd20894a642a0aa287e9827cb1a1f7f91386b67) |
| `glm_roberta_large` | glm | 50,268 | `legacy_gpt2` | 1 | [zai-org/glm-roberta-large](https://huggingface.co/zai-org/glm-roberta-large/tree/89621ea13317286dcd459c0206b86ebc1a1d84b9) |
| `qwen1_native` | qwen | 151,851 | `tiktoken_asset` | 1 | [Qwen/Qwen-1_8B](https://huggingface.co/Qwen/Qwen-1_8B/tree/fa6e214ccbbc6a55235c26ef406355b6bfdf5eed) |
| `qwen1_hf` | qwen | 151,646 | `huggingface` | 6 | [Qwen/Qwen-tokenizer](https://huggingface.co/Qwen/Qwen-tokenizer/tree/5909c8222473b2c73b0b73fb054552cd4ef6a8eb) |
| `codeqwen15` | qwen | 92,302 | `huggingface` | 1 | [Qwen/CodeQwen1.5-7B](https://huggingface.co/Qwen/CodeQwen1.5-7B/tree/5ce5a1554e50a9e3bb236de7c0b8a2a1746186e4) |
| `qwen25` | mimo, qwen | 151,665 | `huggingface` | 10 | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/060db6499f32faf8b98477b0a26969ef7d8b9987) |
| `qwen3` | mimo, qwen | 151,669 | `huggingface` | 9 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) |
| `qwen35` | qwen | 248,070 | `huggingface` | 2 | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) |
| `qwen35_base` | qwen | 248,066 | `huggingface` | 1 | [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68) |
| `qwen38` | qwen | 248,077 | `huggingface` | 2 | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) |
| `deepseek_coder1` | deepseek | 32,022 | `huggingface` | 2 | [deepseek-ai/deepseek-coder-1.3b-base](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base/tree/c919139c3a9b4070729c8b2cca4847ab29ca8d94) |
| `deepseek_coder15` | deepseek | 100,016 | `huggingface` | 1 | [deepseek-ai/deepseek-coder-7b-base-v1.5](https://huggingface.co/deepseek-ai/deepseek-coder-7b-base-v1.5/tree/98f0904cee2237e235f10408ae12292037b21dac) |
| `deepseek_llm` | deepseek | 100,015 | `huggingface` | 3 | [deepseek-ai/deepseek-llm-7b-base](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base/tree/7683fea62db869066ddaff6a41d032262c490d4f) |
| `deepseek_math_7b_base` | deepseek | 100,002 | `huggingface` | 3 | [deepseek-ai/deepseek-math-7b-base](https://huggingface.co/deepseek-ai/deepseek-math-7b-base/tree/036a8c6189aac6e2fc4e07b46e1e57c6b647bca5) |
| `deepseek_coder_v2` | deepseek | 100,018 | `huggingface` | 4 | [deepseek-ai/DeepSeek-Coder-V2-Lite-Base](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base/tree/ea9b066cee82f82906fdd58898cb3788b1c5d770) |
| `deepseek_v3_base` | deepseek | 128,815 | `huggingface` | 3 | [deepseek-ai/DeepSeek-V3-Base](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/tree/afb92e1fa402c2be2a9eb085312bb02e0384d6c7) |
| `deepseek_r1` | deepseek | 128,815 | `huggingface` | 2 | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad) |
| `deepseek_v31_base` | deepseek | 128,815 | `huggingface` | 4 | [deepseek-ai/DeepSeek-V3.1-Base](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Base/tree/d3d4eafdc470de44bbf6f0a74f852eb522357be8) |
| `deepseek_v32` | deepseek | 128,815 | `huggingface` | 1 | [deepseek-ai/DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6) |
| `deepseek_v4_base` | deepseek | 129,280 | `huggingface` | 5 | [deepseek-ai/DeepSeek-V4-Flash-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base/tree/8855555deef230a27a21a8d6f294b7b7497759b6) |
| `deepseek_v41` | deepseek | 129,280 | `huggingface` | 2 | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6) |
| `mimo_v25_base` | mimo | 151,675 | `huggingface` | 6 | [XiaomiMiMo/MiMo-V2.5-Base](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Base/tree/c9d050ce275c75318392dfbc11843516d41916c8) |
| `mimo_v26_distill_qwen` | mimo | 248,077 | `huggingface` | 1 | [XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/tree/2367e865d009c13ac81713a2878291d33ab28177) |
| `minimax_text01` | minimax | 200,026 | `huggingface` | 2 | [MiniMaxAI/MiniMax-Text-01](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/tree/a7351bf2bee0e1253919d349f1ad304e6dac13e9) |
| `minimax_m1` | minimax | 200,036 | `huggingface` | 3 | [MiniMaxAI/MiniMax-M1-40k](https://huggingface.co/MiniMaxAI/MiniMax-M1-40k/tree/2d1d1c2f00c97fc1245bfce7648649b76e0a8e6e) |
| `minimax_m2` | minimax | 200,054 | `huggingface` | 4 | [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2/tree/757303d492a50514c312788b5247a4f696a4c6a3) |
| `minimax_m3` | minimax | 200,061 | `huggingface` | 1 | [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3/tree/f0e1c1e04d40177e4673a22097036854f536e9c0) |

## Shared generations

- **glm_original_en**: glm-10b, glm-2b. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **glm_original_zh**: glm-10b-chinese, glm-large-chinese. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **glm4**: glm-4-9b-chat-hf, glm-4-9b-chat-1m-hf. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **glm4_0414**: GLM-4-9B-0414, GLM-4-32B-Base-0414. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **glm45**: GLM-4.5, GLM-4.5-Air, GLM-4.6, GLM-4.7. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **glm47_flash**: GLM-4.7-Flash, GLM-5, GLM-5.1, GLM-5.2, GLM-5.3, GLM-5.3-Flash. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **qwen1_hf**: Qwen-tokenizer, Qwen1.5-0.5B, Qwen1.5-0.5B-Chat, Qwen2-0.5B, Qwen2-0.5B-Instruct, Qwen2-Math-1.5B. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **qwen25**: Qwen2.5-0.5B, Qwen2.5-0.5B-Instruct, Qwen2.5-Coder-0.5B, Qwen2.5-Math-1.5B, Qwen3-0.6B-Base, MiMo-7B-Base, MiMo-7B-SFT, MiMo-7B-RL, MiMo-7B-RL-0530, MiMo-V2-Flash-Base. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **qwen3**: QwQ-32B, Qwen3-0.6B, Qwen3-4B-Instruct-2507, Qwen3-Coder-30B-A3B-Instruct, Qwen3-Next-80B-A3B-Instruct, Qwen3-Coder-Next, MiMo-V2-Flash, MiMo-V2.5-Pro-Base, MiMo-V2.5-Pro. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **qwen35**: Qwen3.5-0.8B, Qwen3.6-27B. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **qwen38**: Qwen3.8-27B, Qwen3.8-Flash-Next. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **deepseek_coder1**: deepseek-coder-1.3b-base, deepseek-coder-1.3b-instruct. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **deepseek_llm**: deepseek-llm-7b-base, deepseek-llm-7b-chat, deepseek-moe-16b-base. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **deepseek_math_7b_base**: deepseek-math-7b-base, DeepSeek-V2-Lite, DeepSeek-V2-Lite-Chat. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **deepseek_coder_v2**: DeepSeek-Coder-V2-Lite-Base, DeepSeek-Coder-V2-Lite-Instruct, DeepSeek-V2.5, DeepSeek-V2.5-1210. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **deepseek_v3_base**: DeepSeek-V3-Base, DeepSeek-V3, DeepSeek-V3-0324. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **deepseek_r1**: DeepSeek-R1, DeepSeek-R1-0528. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **deepseek_v31_base**: DeepSeek-V3.1-Base, DeepSeek-V3.1, DeepSeek-V3.1-Terminus, DeepSeek-V3.2-Exp. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **deepseek_v4_base**: DeepSeek-V4-Flash-Base, DeepSeek-V4-Flash, DeepSeek-V4-Pro, DeepSeek-V4-Flash-0731, DeepSeek-V4-Pro-0813. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **deepseek_v41**: DeepSeek-V4.1-Flash, DeepSeek-V4.1-Flash-1017. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **mimo_v25_base**: MiMo-V2.5-Base, MiMo-V2.5, MiMo-V2.6-Pro-RL, MiMo-V2.6-Pro-MOPD, MiMo-V2.6-Flash-RL, MiMo-V2.6-Flash-MOPD. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **minimax_text01**: MiniMax-Text-01, MiniMax-Text-01-hf. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **minimax_m1**: MiniMax-M1-40k, MiniMax-M1-80k, MiniMax-M1-80k-hf. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.
- **minimax_m2**: MiniMax-M2, MiniMax-M2.1, MiniMax-M2.5, MiniMax-M2.7. These have the same recorded canonical profile; alias model quality and runtime wrappers are not being equated.

## Native adapter requirements

| Profile | Required tokenizer assets | Declared text behavior |
| --- | --- | --- |
| `glm_original_en` | `added_tokens.json`, `merges.txt`, `tokenizer_config.json`, `vocab.json` | GPT-2 ByteLevel BPE plus 8 supplied added-token IDs; config declares missing `[UNK]` surface, so wrapper parity requires the reference-version check below. |
| `glm_original_zh` | `added_tokens.json`, `cog-pretrain.model`, `tokenizer_config.json` | SentencePiece `cog-pretrain.model` with explicit 10 added IDs; GLM-large-Chinese is verified as an alias. |
| `chatglm1` | `ice_text.model`, `tokenizer_config.json` | SentencePiece; pinned config explicitly sets image-token offset to 0 (source default is 20,000); newline/tab/2–80-space transformations and inverse decoding. |
| `chatglm2` | `tokenizer.model`, `tokenizer_config.json` | SentencePiece plus 5 appended command tokens; same underlying SP file as ChatGLM3; no automatic BOS/EOS in benchmark. |
| `chatglm3_base` | `tokenizer.model`, `tokenizer_config.json` | SentencePiece plus 9 appended command/role tokens; base config does not declare the chat checkpoint’s 4 added-token decoder records. |
| `chatglm3` | `special_tokens_map.json`, `tokenizer.model`, `tokenizer_config.json` | SentencePiece plus 9 commands; config additionally matches `[gMASK]`, `sop`, `<|user|>` and `<|assistant|>` as 4 normalized added tokens marked special:false. |
| `glm4_native` | `tokenizer.model`, `tokenizer_config.json` | Native tiktoken rank file despite `.model` suffix; 1–3 Unicode digit chunks; 14 added special IDs 151329–151342; no NFC normalization. |
| `glm_roberta_large` | `added_tokens.json`, `merges.txt`, `tokenizer_config.json`, `vocab.json` | ByteLevel BPE with declared wrapper configuration and supplied added-token IDs; inspect native_spec for exact prefix-space and special-token settings. |
| `qwen1_native` | `qwen.tiktoken`, `tokenizer_config.json` | Native tiktoken; NFC normalization; single Unicode digit split; 208 declared special IDs 151643–151850; literals are allowed as special tokens. |

Exact regex strings, special-token maps, SentencePiece offsets, embedded normalizer metadata, config added-token records and source-file hashes are stored in each `native_spec`. The initial JSON smoke probe is stored as data with input, IDs, decoded output and byte-exact result. It verifies loading and an example, not benchmark coverage or full behavioral equivalence.

## Concrete distinctions

- **Original Qwen normalizes NFC.** Input `e` + U+0301 can decode as U+00E9. Those are visually similar but have different UTF-8 and UTF-16 representations. The fidelity benchmark must retain the mismatch.
- **Qwen1 native has 208 special entries; the verified Qwen HF / Qwen1.5 / Qwen2 JSON group has 3 added entries.** Shared ordinary vocabulary does not remove that difference.
- **GLM4 HF and GLM4-0414 share the ordinary core but differ in post-processing.** Their full profiles remain separate even though add_special_tokens=False suppresses inserted prefix commands in this intrinsic protocol.
- **ChatGLM2 and ChatGLM3 share the same SentencePiece bytes but append 5 and 9 commands.** ChatGLM3 chat also has literal matching records that the base config lacks.
- **A filename is not a format.** GLM4’s `tokenizer.model` is a base64/rank tiktoken file. Earlier ChatGLM `tokenizer.model` files are binary SentencePiece models.

## Limitations and explicit scope

Original GLM English declares `[UNK]` in tokenizer_config.json, but the supplied vocab.json and added_tokens.json do not contain it. Some Transformers versions can add configured missing special strings automatically. The catalog gives the supplied ID inventory and flags this case; it does not invent an official extra ID. The benchmark adapter must document its chosen behavior and compare reference behavior before claiming complete legacy-wrapper parity.

Multimodal LLM checkpoints in the selected text-generation series are treated here as their text tokenizer pipelines. Images, audio codecs, video encoders, OCR-only processors, quantization forks and unrelated distillation checkpoints are outside this catalog’s core scope. MiniMax H3 is listed under excluded with its official revision and reason. The older GLM-130B standalone ecosystem is not included in the inspected Hugging Face representative set; this is an explicit inventory boundary, not a claim of tokenizer equivalence or unavailability.

Tokenizer terms remain those of the original publishers. Asset hashes establish which files were evaluated; they do not grant redistribution rights. The offline preparation/bundle manifest must include the chosen tokenizer assets and terms. Nothing in this catalog downloads pretrained model weights.

## Primary sources

Every checkpoint in the catalog links directly to its official pinned Hugging Face revision. Declared legacy processing is read from the pinned `tokenization_*.py` source and `tokenizer_config.json` assets stored with that checkpoint. SentencePiece field interpretation follows the [official model schema](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto).
