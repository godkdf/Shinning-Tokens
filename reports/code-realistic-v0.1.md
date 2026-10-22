# Realistic code benchmark v0.1

**54 tokenizer profiles; 2,419,200 measurements; 0 execution errors.**

| Workload | Cases | Tokenizer inputs per profile |
|---|---:|---:|
| Original real source files | 32,000 | 32,000 |
| Real modified-file pairs: before, after and Git diff | 3,200 | 9,600 |
| Constructed repository contexts | 1,600 | 1,600 |
| Controlled stress cases | 1,600 | 1,600 |
| **Total** | **38,400 cases** | **44,800 inputs** |

The source corpus spans **32 programming languages and 608 repositories**, with 1,000 files per language. All suites use 20% development / 80% evaluation splits. The real editing suite has 100 pairs per language; each pair uses actual pinned Git commits and a modified existing file. All 6,400 before/after versions were verified against Git blob IDs. Contexts serialize complete real files; stress cases are labelled correlated transformations.

**28 / 54 profiles reconstructed all 44,800 inputs byte-for-byte.** Canonical-only normalization, other mismatches, unknown emissions and execution errors are distinct outcomes. An execution completing without errors does not imply byte-exact reconstruction.

## Results and evidence

- [Combined per-profile CSV](code-realistic-v0.1-summary.csv)
- [Original source results and per-language table](../results/code-realistic-v0.1/REPORT.md)
- [Original source results by language](../results/code-realistic-v0.1/by-language.csv)
- [Edit/context/stress results by scenario and language](../results/code-scenarios-v0.1/by-scenario.csv)
- [Independent validation ledger](code-realistic-v0.1-validation.json)

Validation reconciles all 2,419,200 raw measurement rows with frozen input IDs, SHA-256 values, byte lengths, language, dataset, split and repository. It verifies all 10,368 checkpoint hashes, consistent code/dependency fingerprints, UTF-8/UTF-16 fidelity flags and exact agreement between raw rows and published aggregates. The software test suite passed 89 tests.

## Scope and limits

These measurements describe token counts and reconstruction fidelity. Collection times include correctness checks, reporting and concurrent processes; they are not isolated encode/decode speed rankings. Model generation quality is not measured.

All 32 language file quotas are met. Of 128 language/size groups, 102 meet the proposed minimum of 100 files; 26 remain below that target. Per-language counts and shortfalls are recorded in the source manifest.

The corpus is provisional: exact and whitespace-equivalent source duplicates were removed, but near-duplicate repository families and language dialects are not independently audited. Popular public repositories and balanced language quotas do not establish production traffic weights. Historical file-specific licensing exceptions are not independently audited; upstream licenses and notices remain applicable.

This implements the dataset and tokenization/fidelity portion of the NASA-inspired plan. Isolated latency/throughput trials, complete requirements traceability, near-duplicate/dialect review, broader operational validation and native platform acceptance remain separate work. No NASA compliance or certification is claimed.

Original results under results/v0.1 through v0.4 remain unchanged.
