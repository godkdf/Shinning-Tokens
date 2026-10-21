# Upstream attribution and terms

Lab source code, original diagnostics and documentation use Apache-2.0. Upstream artifacts and data are not relicensed by this repository.

Expanded tokenizer generations are pinned individually in `configs/tokenizer-candidates.json` and `configs/registry.json`, including exact official source repositories, revisions and upstream terms links for GLM, Qwen, DeepSeek, Xiaomi MiMo and MiniMax. A family's checkpoints can have different licenses; the catalog preserves per-checkpoint notices rather than assigning one family-wide license.

Kimi/Moonlight sources are pinned in `configs/kimi-candidates.json`. Native text adapters reproduce inspected tokenizer ranks, regex, configured special IDs, reserved-ID ranges, encode policy and chunk limits; they do not execute model repository Python or load weights. Model cards and available license/third-party notices for all included checkpoints are retained under `licenses/kimi/`. K2/K2.5/K2.6/K2.7 use their recorded modified MIT terms; K3 has its separate Kimi K3 License. Moonlight, Kimi VL, Kimi Linear and Kimi Dev model cards declare MIT. These licenses and any incorporated notices remain upstream terms, not the Lab's Apache license. Chat, image and audio behavior is outside the text benchmark.

FLORES-200 text comes from Meta's official release under CC-BY-SA-4.0; original sentence IDs, split and language codes are retained. See `docs/EXPANDED.md` for attribution and the archive hash. MADLAD-400 uses the pinned AllenAI clean shards under ODC-BY-1.0; per-configuration source paths and revisions are recorded. The dirty/noise configuration is excluded.

Language geography metadata includes Unicode CLDR territory records under the Unicode license and Glottolog country records under CC-BY-4.0. Their original files and SHA-256 values are retained in `benchmarks/metadata` and `benchmarks/languages.json`. License notices are in `licenses/`. These browsing groups are broad and overlap; they do not assign languages exclusively to a continent.

| Source | Reference and terms |
|---|---|
| Gemma 3 tokenizer and derivatives | [Gemma terms](https://ai.google.dev/gemma/terms); the agreement, use restrictions and modification notice are included in `licenses/`. V3.1 is a modified Gemma 3 tokenizer; use and redistribution must comply with those terms and incorporated restrictions. Google does not endorse this experiment. |
| Gemma4 | [Gemma4 Apache-2.0 license](https://ai.google.dev/gemma/apache_2), separate from the Gemma 3 terms. |
| BLOOM | [Pinned repository](https://huggingface.co/bigscience/bloom); follow its model/tokenizer notices. |
| JAIS Family 256M | [Repository](https://huggingface.co/inception42/jais-family-256m); follow upstream license and notices. |
| Qwen3 | [Repository](https://huggingface.co/Qwen/Qwen3-0.6B); follow its license and notices. |
| tiktoken / o200k | [OpenAI tiktoken](https://github.com/openai/tiktoken), MIT. Native encoding identity is hashed by this Lab. |
| GPT-2 / tiktoken | [OpenAI GPT-2](https://github.com/openai/gpt-2) and [tiktoken](https://github.com/openai/tiktoken), MIT. 50,256 ordinary byte pieces plus one reserved special ID. Original encoder and merge hashes are recorded in the registry. |
| Original Llama 3 tokenizer | [Meta Llama 3 Community License](https://github.com/meta-llama/llama3/blob/main/LICENSE) and incorporated acceptable use policy; copies are in `licenses/Llama-3-Community-License.txt` and `licenses/Llama-3-Acceptable-Use-Policy.md`. Built with Meta Llama 3. Llama 3 is licensed under the Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. Acquired from the pinned [NousResearch redistribution](https://huggingface.co/NousResearch/Meta-Llama-3-8B/tree/315b20096dc791d381d514deb5f8bd9c8d6d3061), with identical official Git blobs and reconstructed original rank SHA-256. Adapter reference splitting follows Meta's licensed tokenizer wrapper. Upstream terms continue to apply. |
| FineWeb2 | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), ODC-BY-1.0 dataset attribution. |
| FineWeb English | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), ODC-BY-1.0 dataset attribution. |
| Unicode script data | [Unicode 15.1.0](https://www.unicode.org/Public/15.1.0/ucd/Scripts.txt); see the copyright/license notice in the pinned file and [Unicode data license](https://www.unicode.org/license.txt). |

FineWeb texts originate from third-party web pages. Dataset licensing does not erase the rights or attribution of those sources; original URLs/IDs remain in the sampled records. Inspect source metadata before republishing text outside this research bundle. Research method pages cite the original authors and primary sources separately.
