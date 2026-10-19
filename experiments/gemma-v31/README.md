# Gemma V3.1 expansion

Hypothesis: appending **2,000 ordinary AddedTokens** can reduce token counts on multilingual text while preserving the inherited vocabulary IDs. This is an exploratory construction, not a claim of better model learning.

The control contains **262,145 IDs**; the candidate contains **264,145 IDs**. The inherited model vocabulary, merges, normalizer and decoder are unchanged. AddedToken matching can alter segmentation of neighboring text and may create regressions. The ordered additions and their flags are in `additions.jsonl`; never recover them through a spreadsheet that can interpret text as formulas.

The parent was reconstructed by removing IDs 262145–264144 from the supplied candidate. Its official gated replay is unverified. `identity.json` records parent/child structural hashes and the candidate file hash. `build-expansion` checks both structures and reproduces the supplied file exactly.

Historical MADLAD selection result: **25,485 documents**, **418 configurations**, **33,149,918 → 32,220,477 tokens**, saving **929,441 tokens (2.8037505%)**. **382 configurations improved, 36 regressed**; zero recorded round-trip mismatches. These are selection-data results, not held-out results. The fresh FineWeb and diagnostic measurements are generated into `results/v0.1/hub.json` with per-language paired intervals. No downstream training or generation evaluation has been performed here.
