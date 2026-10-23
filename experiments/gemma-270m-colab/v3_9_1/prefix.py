"""Bound useful candidate storage independently of the upstream source mix."""


def prefix_cache_limit(job):
    # The 256 MiB corpus is final text. Cached JSON also contains eval, spare
    # candidates, escaping and provenance; it needs an independent disk budget.
    return max(PREFIX_JOB_LIMIT_BYTES, 4 * sum(job["targets"].values()))


def select_prefix_records(records, state, desired):
    """Stage an entire input row without modifying the committed cursor/counts."""
    counts = dict(state["candidate_bytes"])
    documents = dict(state["candidate_documents"])
    selected = []
    for record in records:
        stratum = record["split"] + ":" + record["bucket"]
        if stratum not in desired:
            continue
        if counts.get(stratum, 0) >= desired[stratum] and documents.get(stratum, 0) >= 4:
            continue
        selected.append(record)
        counts[stratum] = counts.get(stratum, 0) + record["utf8_bytes"]
        documents[stratum] = documents.get(stratum, 0) + 1
    return selected
