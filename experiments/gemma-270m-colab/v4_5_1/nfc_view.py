"""Immutable NFC scoring view of the unchanged frozen source corpus."""
NFC_VIEW_SCHEMA = "qwen3-official-nfc-view-1"


def prepare_nfc_view(shared, source_raw, tokenizer, identity, output):
    source_manifest = verify_raw(source_raw)
    source_digest = sha256_file(source_raw / "manifest.json")
    key = hashlib.sha256(json.dumps({"schema": NFC_VIEW_SCHEMA,
        "source_manifest_sha256": source_digest, "tokenizer_sha256": identity["tokenizer_sha256"],
        "tokenizers_version": identity["tokenizers_version"]}, sort_keys=True).encode()).hexdigest()[:24]
    cache = shared / "text_views" / key
    reused = cache.exists()
    if reused:
        manifest = verify_raw(cache)
        report = manifest["text_normalization"]
        if (report["schema"] != NFC_VIEW_SCHEMA or report["source_manifest_sha256"] != source_digest
                or report["tokenizer_sha256"] != identity["tokenizer_sha256"]
                or report["tokenizers_version"] != identity["tokenizers_version"]):
            raise ValueError("NFC scoring-view identity differs; cache preserved.")
    else:
        temporary = cache.with_name(key + ".partial-" + str(time.time_ns()))
        temporary.mkdir(parents=True, exist_ok=False)
        counts, totals, languages, changed, hashes = {}, {}, {}, {}, {}
        started = time.monotonic()
        for split in ("train", "in_mix", "wiki"):
            counts[split], totals[split], changed[split] = 0, 0, 0
            languages[split] = {language: 0 for language in LANGUAGES}
            hashes[split] = set()
            with (source_raw / (split + ".jsonl")).open(encoding="utf-8") as source, \
                    (temporary / (split + ".jsonl")).open("w", encoding="utf-8") as destination:
                for line in source:
                    original = json.loads(line)
                    before = original["text"]
                    before_bytes = before.encode("utf-8")
                    if len(before_bytes) != original["utf8_bytes"] or hashlib.sha256(before_bytes).hexdigest() != original["sha256"]:
                        raise ValueError("Frozen source document bytes/hash differ before NFC.")
                    normalized = tokenizer.normalize(before)
                    if tokenizer.normalize(normalized) != normalized:
                        raise ValueError("Official NFC normalization is not idempotent.")
                    after = normalized.encode("utf-8")
                    digest = hashlib.sha256(after).hexdigest()
                    row = {**original, "text": normalized, "sha256": digest, "utf8_bytes": len(after),
                        "pre_nfc_span_sha256": original["sha256"], "pre_nfc_utf8_bytes": original["utf8_bytes"],
                        "text_transformation": "official Qwen3 NFC; source span coordinates refer to pre-NFC text"}
                    destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[split] += 1
                    totals[split] += len(after)
                    languages[split][row["language"]] += len(after)
                    changed[split] += int(normalized != before)
                    hashes[split].add(digest)
            if counts[split] != source_manifest["document_counts"][split]:
                raise ValueError("NFC view changed the frozen document count or split.")
        overlap = len(hashes["train"] & (hashes["in_mix"] | hashes["wiki"]))
        if overlap:
            raise ValueError(f"NFC creates {overlap} identical train/held-out spans. Source corpus preserved; scoring view rejected.")
        report = {"schema": NFC_VIEW_SCHEMA, "source_manifest_sha256": source_digest,
            "tokenizer_sha256": identity["tokenizer_sha256"], "tokenizers_version": identity["tokenizers_version"],
            "scoring_text_form": "NFC-normalized UTF-8", "source_raw_bytes": source_manifest["original_bytes"],
            "scored_nfc_bytes": totals, "byte_change": {split: totals[split] - source_manifest["original_bytes"][split] for split in totals},
            "changed_documents": changed, "document_counts": counts, "normalized_train_eval_span_overlap": overlap,
            "normalization_seconds": time.monotonic() - started,
            "source_corpus_modified": False, "documents_dropped": 0,
            "quota_scope": "Frozen source-byte quotas retained; normalized byte totals are measured, not refilled.",
            "comparison_note": "NFC BPB uses normalized bytes and is not a controlled comparison against earlier original-byte BPB."}
        manifest = {**source_manifest, "document_counts": counts, "original_bytes": totals,
            "language_bytes": languages, "byte_accounting_scope": "NFC-normalized scoring text",
            "text_normalization": report,
            "deduplication_scope": source_manifest.get("deduplication_scope", "") + "; NFC train/held-out exact-span separation verified",
            "files": {split + ".jsonl": sha256_file(temporary / (split + ".jsonl")) for split in counts}}
        write_json(temporary / "manifest.json", manifest)
        temporary.rename(cache)
    shutil.copyfile(source_raw / "manifest.json", output / "raw_data_manifest.json")
    write_json(output / "normalization_report.json", {**report, "cache_reused": reused})
    return cache, manifest, report
