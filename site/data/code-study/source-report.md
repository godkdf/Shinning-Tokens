# Real-source code tokenizer results

Corpus: 32,000 files; 32 languages. Completed profiles: 54/54.

Held-out evaluation only. Compression values do not establish code-generation quality. Ratios with fidelity failures are diagnostic.

| Tokenizer | Files | Tokens | Bytes/token | Exact | Canonical-only | Other mismatch | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama3 | 25,600 | 93,510,896 | 3.7423 | 25,600 | 0 | 0 | 0 |
| kimi_k2_thinking | 25,600 | 93,708,759 | 3.7344 | 25,600 | 0 | 0 | 0 |
| kimi_k25 | 25,600 | 93,708,759 | 3.7344 | 25,600 | 0 | 0 | 0 |
| kimi_k2 | 25,600 | 93,709,171 | 3.7344 | 25,600 | 0 | 0 | 0 |
| kimi_linear | 25,600 | 93,709,171 | 3.7344 | 25,600 | 0 | 0 | 0 |
| moonlight | 25,600 | 93,709,174 | 3.7344 | 25,600 | 0 | 0 | 0 |
| kimi_vl | 25,600 | 93,709,174 | 3.7344 | 25,600 | 0 | 0 | 0 |
| kimi_k3 | 25,600 | 93,709,240 | 3.7344 | 25,600 | 0 | 0 | 0 |
| kimi_k2_base | 25,600 | 93,709,369 | 3.7344 | 25,600 | 0 | 0 | 0 |
| o200k | 25,600 | 93,879,802 | 3.7276 | 25,600 | 0 | 0 | 0 |
| glm47_flash | 25,600 | 94,277,878 | 3.7119 | 25,600 | 0 | 0 | 0 |
| glm45 | 25,600 | 94,294,876 | 3.7112 | 25,600 | 0 | 0 | 0 |
| glm4_native | 25,600 | 94,296,038 | 3.7112 | 25,600 | 0 | 0 | 0 |
| glm4 | 25,600 | 94,296,038 | 3.7112 | 25,600 | 0 | 0 | 0 |
| glm4_0414 | 25,600 | 94,296,038 | 3.7112 | 25,600 | 0 | 0 | 0 |
| minimax_m3 | 25,600 | 95,112,382 | 3.6793 | 25,583 | 17 | 0 | 0 |
| minimax_m2 | 25,600 | 95,112,740 | 3.6793 | 25,583 | 17 | 0 | 0 |
| minimax_m1 | 25,600 | 95,113,190 | 3.6793 | 25,583 | 17 | 0 | 0 |
| minimax_text01 | 25,600 | 95,113,191 | 3.6793 | 25,583 | 17 | 0 | 0 |
| bloom | 25,600 | 96,634,449 | 3.6214 | 25,599 | 0 | 1 | 0 |
| qwen3 | 25,600 | 98,768,002 | 3.5432 | 25,583 | 17 | 0 | 0 |
| mimo_v25_base | 25,600 | 98,768,002 | 3.5432 | 25,583 | 17 | 0 | 0 |
| qwen25 | 25,600 | 98,768,462 | 3.5431 | 25,583 | 17 | 0 | 0 |
| qwen1_native | 25,600 | 98,768,812 | 3.5431 | 25,583 | 17 | 0 | 0 |
| qwen1_hf | 25,600 | 98,768,812 | 3.5431 | 25,583 | 17 | 0 | 0 |
| deepseek_v41 | 25,600 | 100,059,406 | 3.4974 | 25,600 | 0 | 0 | 0 |
| deepseek_v32 | 25,600 | 100,059,415 | 3.4974 | 25,600 | 0 | 0 | 0 |
| deepseek_v4_base | 25,600 | 100,059,415 | 3.4974 | 25,600 | 0 | 0 | 0 |
| deepseek_r1 | 25,600 | 100,060,107 | 3.4974 | 25,600 | 0 | 0 | 0 |
| deepseek_v31_base | 25,600 | 100,060,107 | 3.4974 | 25,600 | 0 | 0 | 0 |
| deepseek_v3_base | 25,600 | 100,060,661 | 3.4974 | 25,600 | 0 | 0 | 0 |
| qwen38 | 25,600 | 104,303,329 | 3.3551 | 25,583 | 17 | 0 | 0 |
| qwen35 | 25,600 | 104,303,335 | 3.3551 | 25,583 | 17 | 0 | 0 |
| qwen35_base | 25,600 | 104,303,863 | 3.3551 | 25,583 | 17 | 0 | 0 |
| mimo_v26_distill_qwen | 25,600 | 104,307,820 | 3.3550 | 25,583 | 17 | 0 | 0 |
| glm_edge | 25,600 | 107,626,218 | 3.2515 | 25,600 | 0 | 0 | 0 |
| jais | 25,600 | 107,720,652 | 3.2487 | 25,600 | 0 | 0 | 0 |
| gemma_v31 | 25,600 | 110,956,187 | 3.1539 | 25,583 | 0 | 17 | 0 |
| gemma3_parent | 25,600 | 112,575,375 | 3.1086 | 25,583 | 0 | 17 | 0 |
| gemma4 | 25,600 | 112,618,060 | 3.1074 | 25,583 | 0 | 17 | 0 |
| deepseek_coder_v2 | 25,600 | 115,437,429 | 3.0315 | 25,600 | 0 | 0 | 0 |
| deepseek_math_7b_base | 25,600 | 115,437,800 | 3.0315 | 25,600 | 0 | 0 | 0 |
| deepseek_coder15 | 25,600 | 115,438,243 | 3.0315 | 25,341 | 0 | 259 | 0 |
| deepseek_llm | 25,600 | 115,438,243 | 3.0315 | 25,341 | 0 | 259 | 0 |
| chatglm1 | 25,600 | 118,814,922 | 2.9453 | 23,295 | 8 | 2,297 | 0 |
| codeqwen15 | 25,600 | 119,099,072 | 2.9383 | 25,566 | 17 | 17 | 0 |
| deepseek_coder1 | 25,600 | 123,703,053 | 2.8290 | 25,341 | 0 | 259 | 0 |
| chatglm2 | 25,600 | 126,322,014 | 2.7703 | 25,583 | 0 | 17 | 0 |
| chatglm3_base | 25,600 | 126,322,014 | 2.7703 | 25,583 | 0 | 17 | 0 |
| chatglm3 | 25,600 | 126,322,090 | 2.7703 | 25,493 | 0 | 107 | 0 |
| glm_original_zh | 25,600 | 133,678,666 | 2.6178 | 11 | 0 | 25,589 | 0 |
| glm_original_en | 25,600 | 164,346,956 | 2.1293 | 25,600 | 0 | 0 | 0 |
| glm_roberta_large | 25,600 | 164,347,035 | 2.1293 | 25,600 | 0 | 0 | 0 |
| gpt2 | 25,600 | 164,347,087 | 2.1293 | 25,600 | 0 | 0 | 0 |

## Per-language observed token counts

Minimum measured held-out token count among profiles reconstructing every file exactly. These are descriptive results for this frozen corpus, not significance tests or model-quality rankings. Equal-token profiles are ties; complete per-profile language results are in by-language.csv.

| Language | Files | Lowest-count exact profile(s) | Tokens | Bytes/token |
|---|---:|---|---:|---:|
| Python | 800 | llama3 | 4,653,540 | 4.3577 |
| JavaScript | 800 | llama3 | 2,040,368 | 3.9447 |
| TypeScript | 800 | glm47_flash | 1,828,854 | 4.1723 |
| Ruby | 800 | llama3 | 2,066,138 | 3.8395 |
| PHP | 800 | bloom | 1,807,957 | 4.4813 |
| Perl | 800 | glm47_flash | 2,898,271 | 3.4903 |
| Lua | 800 | glm47_flash | 2,951,248 | 3.7705 |
| C | 800 | llama3 | 5,827,584 | 3.5254 |
| C++ | 800 | llama3 | 4,694,102 | 3.9482 |
| Rust | 800 | llama3 | 3,525,104 | 4.3569 |
| Go | 800 | llama3 | 3,469,674 | 3.6235 |
| Zig | 800 | llama3 | 3,889,475 | 4.0473 |
| Assembly | 800 | bloom | 5,636,697 | 2.8753 |
| Java | 800 | glm4, glm45, glm47_flash, glm4_0414, glm4_native | 1,577,134 | 4.2260 |
| Kotlin | 800 | llama3 | 1,233,642 | 4.9843 |
| Scala | 800 | llama3 | 2,159,362 | 3.8175 |
| C# | 800 | llama3 | 1,634,816 | 4.9246 |
| Swift | 800 | llama3 | 1,120,087 | 4.5810 |
| Dart | 800 | llama3 | 1,838,893 | 4.2350 |
| Haskell | 800 | o200k | 2,695,894 | 3.7858 |
| OCaml | 800 | llama3 | 2,298,809 | 3.5688 |
| Elixir | 800 | kimi_k2, kimi_k25, kimi_k2_base, kimi_k2_thinking, kimi_k3, kimi_linear, kimi_vl, moonlight | 1,961,869 | 3.6164 |
| Erlang | 800 | kimi_k2, kimi_k25, kimi_k2_base, kimi_k2_thinking, kimi_k3, kimi_linear, kimi_vl, moonlight | 3,142,301 | 3.7492 |
| R | 800 | kimi_k2, kimi_k25, kimi_k2_base, kimi_k2_thinking, kimi_k3, kimi_linear, kimi_vl, moonlight | 1,366,319 | 3.6166 |
| Julia | 800 | kimi_k2, kimi_k25, kimi_k2_base, kimi_k2_thinking, kimi_k3, kimi_linear, kimi_vl, moonlight | 2,635,602 | 3.4871 |
| MATLAB | 800 | glm47_flash | 1,153,728 | 4.2109 |
| Fortran | 800 | llama3 | 3,127,348 | 3.3994 |
| SQL | 800 | bloom | 4,194,505 | 2.9644 |
| Bash | 800 | kimi_k2, kimi_k25, kimi_k2_base, kimi_k2_thinking, kimi_k3, kimi_linear, kimi_vl, moonlight | 4,315,005 | 3.5972 |
| PowerShell | 800 | kimi_k2, kimi_k25, kimi_k2_base, kimi_k2_thinking, kimi_k3, kimi_linear, kimi_vl, moonlight | 3,227,855 | 3.0813 |
| Solidity | 800 | o200k | 1,685,777 | 4.0820 |
| COBOL | 800 | kimi_k2, kimi_k25, kimi_k2_base, kimi_k2_thinking, kimi_k3, kimi_linear, kimi_vl, moonlight | 6,039,453 | 3.9801 |

## Collection limitations

- Curated popular GitHub repositories, not a random sample of developer traffic.
- Only valid UTF-8 files of 80 bytes to 1 MiB are sampled; other encodings, binary and oversized files are excluded.
- Repository license metadata and notices retained; file-specific licensing exceptions require review before redistribution.
- Language attribution uses repository metadata and suffixes; dialects not independently validated.
- Exact and whitespace-equivalent duplicates removed globally; near-duplicate family audit pending.
- Repository names are provisional families; related non-fork repositories can remain.
- Editing, context/tool and adversarial companion suites are not part of this source-file corpus.
