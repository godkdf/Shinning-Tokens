# Prepare and publish release 0.2.3

This guide covers the **GitHub source/release assets and static explorer**. Application **0.2.3** contains **54 profiles**; latest result cohort **v0.4** uses protocol **intrinsic-v0.2**. Publication, repository visibility and the hosted Site audience are separate owner actions.

## Release contents

The prepared directory is `artifacts/releases/0.2.3/`.

| Artifact | What it contains |
|---|---|
| `tokenizer-lab-0.2.3-source.tar.gz` | Code, documentation, configs, input manifests, notices, aggregate results and saved measuring sources. No original-text payloads, tokenizer assets, model weights or per-document checkpoints. |
| Four `tokenizer-lab-0.2.3-starter-<target>.zip` files | One pinned runtime, locked native dependencies, current 54-profile catalog, local UI and saved aggregates. **0 tokenizer / 0 corpus-language payloads**. |
| `tokenizer-lab-0.2.3-explorer.zip` | The complete prepared static `site/` tree: saved inventories, maps, compressed text, aggregate results and multipart starter downloads. No model weights. |
| `release-manifest.json`, `SHA256SUMS` | Artifact byte sizes, SHA-256 values, scope and validation status. |
| `RELEASE_NOTES.md` | Ready-to-use release description with measured results and limits. |
| `validation.json` | Publication checks, platform verification scope and local evidence. |

Generated distributions are release attachments, not regular Git files. GitHub blocks regular repository files above **100 MiB**; keep the large archives in Releases. [GitHub's size guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## Build from the prepared checkout

```sh
uv sync --frozen --extra test
uv run --frozen pytest -q
uv run --frozen python -m tokenizer_lab.publication starters
uv run --frozen python -m tokenizer_lab.publication source
uv run --frozen python -m tokenizer_lab.publication audit
uv run --frozen python -m tokenizer_lab.publication package
uv run --frozen python -m tokenizer_lab.publication verify
```

The starter builder uses the pinned runtime archives and `uv.lock` wheels, reuses matching cached inputs, installs zero selected tokenizer/corpus payloads, and writes checksum-verified multipart downloads. It never substitutes a tokenizer or rewrites a frozen measurement cohort. A matching already-built starter is reused; stale source inputs trigger a fresh staged build rather than relabelling an old ZIP.

`source` is curated: only declared project directories and two established experiment folders are included. It excludes local custom tokenizer registrations, caches, credentials, unrelated `experiments/gemma-270m-tpu-pilot/`, model weights and raw observation checkpoints. Every member has a recorded size/hash. Source-generation does not depend on committing unrelated local work.

`audit --source-only` checks versions, inventories, aggregate counts, audit-chain hashes, local Markdown links, public-source secret patterns and regular Git file sizes without requiring binary starter downloads. The full audit additionally checks release ZIPs, multipart reconstruction, static-asset sizes, source members and every packaged starter member against its manifest. It does not claim to re-run all historical tokenizer measurements.

When files change after preparing artifacts, rebuild the affected artifacts and repeat the audit. A source checksum and a dataset checksum identify different bytes. The source archive's manifest deliberately excludes its own generated checksum/index to avoid circular hashes.

## Verification scope

The current source test result and exact release validations are in `validation.json`. Asset-dependent integration tests can skip on a fresh source-only checkout. Local tokenizer study audits remain in `results/v0.2`, `results/v0.3` and `results/v0.4`; the combined chain retains **23,304,780 observations / 46,818 checkpoints** without claiming one simultaneous re-audit.

The Apple Silicon starter is extracted into a new path, its members checked, and its bundled interpreter exercised with external networking blocked. Windows, Linux and Intel Mac packages are built and hash-checked but their native execution remains **deferred**. `.github/workflows/portability.yml` is manual-only so publishing this candidate does not automatically run those deferred jobs. Do not promote package checks to native-execution claims.

## Publish the prepared candidate

1. Review `validation.json`, `SHA256SUMS`, this release's diff and the upstream notices. The known wrapper, sampling and normalization limits are in the release notes.
2. Sign in to the correct GitHub account and verify access to `godkdf/Shinning-Tokens`. The repository was private during preparation. A read-only check from this environment returned **Repository not found**; that is not evidence the confirmed private repository should be renamed or recreated.
3. Push the reviewed source commit to the existing repository. Create tag **v0.2.3** from that exact commit and a draft GitHub release using `RELEASE_NOTES.md`.
4. Attach all files listed by `release-manifest.json`, plus its manifest/checksums/validation/notes. Verify the uploaded hashes before publishing the draft. The builder does not create a remote release or change visibility.
5. If the repository is intended to become public, its owner changes visibility explicitly. Preserve private access until that decision is made.
6. Publish the static explorer separately when requested. Use the existing Site and preserve its audience. The deployable content is the prepared `site/` directory; it contains compressed data and multipart downloads. **Do not expose the loopback Python API.**

The source repo, static hosted explorer, standalone starter packages and original tokenizer assets are separate distributions. A successful Git push does not upload ignored release assets or update the hosted Site. A prepared local package is not a deployed URL.
