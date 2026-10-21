# Real-source code corpus v0.1

Locally collected research corpus. Upstream source licenses apply; this is not an Apache-2.0 relicensing of the collected files.

32,000 files across 32 languages and 608 repositories.

| Language | Files | Development | Evaluation | Repositories | Original bytes |
|---|---:|---:|---:|---:|---:|
| Python | 1,000 | 200 | 800 | 18 | 21,983,697 |
| JavaScript | 1,000 | 200 | 800 | 16 | 9,347,633 |
| TypeScript | 1,000 | 200 | 800 | 12 | 8,895,942 |
| Ruby | 1,000 | 200 | 800 | 13 | 8,940,697 |
| PHP | 1,000 | 200 | 800 | 14 | 10,048,609 |
| Perl | 1,000 | 200 | 800 | 28 | 12,273,014 |
| Lua | 1,000 | 200 | 800 | 17 | 12,303,793 |
| C | 1,000 | 200 | 800 | 16 | 23,773,912 |
| C++ | 1,000 | 200 | 800 | 14 | 27,315,397 |
| Rust | 1,000 | 200 | 800 | 13 | 23,218,593 |
| Go | 1,000 | 200 | 800 | 13 | 14,518,430 |
| Zig | 1,000 | 200 | 800 | 15 | 26,593,907 |
| Assembly | 1,000 | 200 | 800 | 36 | 24,057,323 |
| Java | 1,000 | 200 | 800 | 11 | 7,286,908 |
| Kotlin | 1,000 | 200 | 800 | 12 | 7,335,496 |
| Scala | 1,000 | 200 | 800 | 13 | 10,176,898 |
| C# | 1,000 | 200 | 800 | 13 | 9,938,733 |
| Swift | 1,000 | 200 | 800 | 12 | 7,815,678 |
| Dart | 1,000 | 200 | 800 | 10 | 9,124,517 |
| Haskell | 1,000 | 200 | 800 | 16 | 12,257,433 |
| OCaml | 1,000 | 200 | 800 | 12 | 10,836,044 |
| Elixir | 1,000 | 200 | 800 | 16 | 9,740,648 |
| Erlang | 1,000 | 200 | 800 | 14 | 14,764,754 |
| R | 1,000 | 200 | 800 | 23 | 5,913,936 |
| Julia | 1,000 | 200 | 800 | 19 | 10,794,512 |
| MATLAB | 1,000 | 200 | 800 | 24 | 5,124,565 |
| Fortran | 1,000 | 200 | 800 | 16 | 14,629,696 |
| SQL | 1,000 | 200 | 800 | 31 | 14,260,018 |
| Bash | 1,000 | 200 | 800 | 68 | 17,044,048 |
| PowerShell | 1,000 | 200 | 800 | 26 | 11,674,340 |
| Solidity | 1,000 | 200 | 800 | 21 | 8,767,623 |
| COBOL | 1,000 | 200 | 800 | 26 | 27,143,364 |

## Integrity and reuse

Each compressed language partition preserves original valid UTF-8 bytes, including BOMs and line endings. The manifest records exact commits, archive hashes, repository license metadata, collected notices, source-file hashes and exclusions. The text-free file index enables replay at frozen commits. Raw text partitions and per-file result checkpoints are excluded from ordinary Git commits.

From the repository root, restore the frozen corpus:

```sh
uv run --frozen python -m tokenizer_lab.codebench restore
uv run --frozen python -m tokenizer_lab.codebench audit
```

With all registered tokenizer assets already cached, evaluate every profile:

```sh
uv run --frozen python -m tokenizer_lab.codebench run
```

The run writes results/code-realistic-v0.1/REPORT.md, summary.csv and per-tokenizer JSON aggregates. Correctness workers block network connections. End-to-end collection times are not isolated tokenizer speed rankings.

## Limitations

- Curated popular GitHub repositories, not a random sample of developer traffic.
- Only valid UTF-8 files of 80 bytes to 1 MiB are sampled; other encodings, binary and oversized files are excluded.
- Repository license metadata and notices retained; file-specific licensing exceptions require review before redistribution.
- Language attribution uses repository metadata and suffixes; dialects not independently validated.
- Exact and whitespace-equivalent duplicates removed globally; near-duplicate family audit pending.
- Repository names are provisional families; related non-fork repositories can remain.
- Editing, context/tool and adversarial companion suites are not part of this source-file corpus.

Size-band shortfalls and download exclusions are explicit in manifest.json. This first source-file run does not establish all acceptance criteria in docs/BENCHMARK_V03_PLAN.md.
