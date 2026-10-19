# Upstream attribution and terms

Lab source code, original diagnostics and documentation use Apache-2.0. Upstream artifacts and data are not relicensed by this repository.

Expanded tokenizer generations are pinned individually in `configs/tokenizer-candidates.json` and `configs/registry.json`, including exact official source repositories, revisions and upstream terms links for GLM, Qwen, DeepSeek, Xiaomi MiMo and MiniMax. A family's checkpoints can have different licenses; the catalog preserves per-checkpoint notices rather than assigning one family-wide license.

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
| FineWeb2 | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), ODC-BY-1.0 dataset attribution. |
| FineWeb English | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), ODC-BY-1.0 dataset attribution. |
| Unicode script data | [Unicode 15.1.0](https://www.unicode.org/Public/15.1.0/ucd/Scripts.txt); see the copyright/license notice in the pinned file and [Unicode data license](https://www.unicode.org/license.txt). |

FineWeb texts originate from third-party web pages. Dataset licensing does not erase the rights or attribution of those sources; original URLs/IDs remain in the sampled records. Inspect source metadata before republishing text outside this research bundle. Research method pages cite the original authors and primary sources separately.
