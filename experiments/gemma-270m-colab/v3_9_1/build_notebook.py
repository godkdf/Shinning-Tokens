"""Preserve V3.9; create a new self-contained V3.9.1 notebook exclusively."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "gemma_270m_colab_v5e1_V3_9.ipynb"
TARGET = HERE.parent / "gemma_270m_colab_v5e1_V3_9_1.ipynb"


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one anchor; got {source.count(old)}: {old[:100]}")
    return source.replace(old, new, 1)


def embedded_literals(notebook):
    cell = next(c for c in notebook["cells"] if "PROGRAM_SOURCE =" in "".join(c["source"]))
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse("".join(cell["source"])).body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("PROGRAM_SOURCE", "CONFIG")}


def build_source(original):
    source = original.replace("V3.9", "V3.9.1")
    # Keep protocol fingerprint, pinned sources, shared path, cache schema and
    # job hashes unchanged so the interrupted V3.9 prefixes can be continued.
    source = once(source, "def materialize_prefix(job, root, factor, report_status, stop):",
        "def materialize_prefix(job, root, factor, report_status, stop, required_strata=None):")
    source = once(source,
        '    desired = {key: math.ceil(value * factor) for key, value in job["targets"].items()}',
        '    active = set(job["targets"]) if required_strata is None else set(required_strata)\n'
        '    if not active <= set(job["targets"]):\n'
        '        raise ValueError("Unknown source-cache stratum requested.")\n'
        '    desired = {key: math.ceil(value * factor) for key, value in job["targets"].items() if key in active}\n'
        '    cache_limit = prefix_cache_limit(job)')
    source = once(source,
        '            "candidate_bytes": sum(state["candidate_bytes"].values()), "prefix_cache_reused": original_parts > 0,',
        '            "candidate_bytes": sum(state["candidate_bytes"].values()), "prefix_cache_reused": original_parts > 0,\n'
        '            "cache_limit_bytes": cache_limit, "cached_json_bytes": sum(part["bytes"] for part in state["parts"]),\n'
        '            "candidate_bytes_by_stratum": dict(state["candidate_bytes"]),\n'
        '            "selection_policy": "v391-quota-filtered-prefix", "requested_strata": sorted(active),')
    source = once(source,
        '            records = candidate_records(job, row)\n'
        '            encoded_records = [(json.dumps(record, ensure_ascii=False) + "\\n").encode("utf-8") for record in records]\n'
        '            if committed_bytes + len(buffer) + sum(map(len, encoded_records)) > PREFIX_JOB_LIMIT_BYTES:\n'
        '                raise RuntimeError("Source-prefix cache would exceed its 256 MiB bound: " + job["key"])',
        '            records = select_prefix_records(candidate_records(job, row), state, desired)\n'
        '            encoded_records = [(json.dumps(record, ensure_ascii=False) + "\\n").encode("utf-8") for record in records]\n'
        '            projected_json_bytes = committed_bytes + len(buffer) + sum(map(len, encoded_records))\n'
        '            if projected_json_bytes > cache_limit:\n'
        '                raise RuntimeError(f"Source-prefix disk cache limit reached: {job[\'key\']}; "\n'
        '                    f"projected={projected_json_bytes:,}, limit={cache_limit:,} bytes; "\n'
        '                    f"candidate UTF-8 bytes by stratum={state[\'candidate_bytes\']}. "\n'
        '                    "Completed source batches preserved; no training started.")')
    source = once(source,
        '            report = materialize_prefix(job, cache_root, factor, status, refill_stop)',
        '            missing = [key for key in targets if targets[key] - actual[key] > rules["byte_quota_tolerance"]]\n'
        '            report = materialize_prefix(job, cache_root, factor, status, refill_stop, required_strata=missing)')
    source = once(source,
        '        "max_cached_bytes_per_job": PREFIX_JOB_LIMIT_BYTES, "persistent_batch_target_bytes": PREFIX_PART_BYTES,',
        '        "max_cached_bytes_per_job": max(prefix_cache_limit(job) for job in jobs),\n'
        '        "cached_bytes_limit_by_job": {job["key"]: prefix_cache_limit(job) for job in jobs},\n'
        '        "prefix_selection": "stop caching each satisfied stratum; refill deficient strata only",\n'
        '        "persistent_batch_target_bytes": PREFIX_PART_BYTES,')
    source = once(source, "\ndef main():", "\n" + (HERE / "prefix.py").read_text() + "\n\ndef main():")
    ast.parse(source)
    return source


def markdown(text):
    return {"cell_type": "markdown", "id": hashlib.sha256(text.encode()).hexdigest()[:8],
        "metadata": {}, "source": text.splitlines(True)}


def build_notebook():
    notebook = json.loads(BASE.read_text())
    values = embedded_literals(notebook)
    source = build_source(values["PROGRAM_SOURCE"])
    config = values["CONFIG"]
    config.update(version="V3.9.1", experiment="GEM-COLAB-V3.9.1",
        implementation_revision="V3.9.1-quota-bounded-source-cache-1")
    # Requirement IDs continue the V39 training contract; only data preparation changes.
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "PROGRAM_SOURCE =" in text:
            tail = text[text.index('CONFIG["tokenizer_reference"]'):]
            tail = tail.replace("V3.9", "V3.9.1").replace("V3_9", "V3_9_1")
            text = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + "\n" + tail
        else:
            text = text.replace("V3.9", "V3.9.1").replace("V3_9", "V3_9_1")
        cell["source"] = text.splitlines(True)
        if cell["cell_type"] == "code":
            ast.parse(text)
            cell.update(outputs=[], execution_count=None)
    notebook["metadata"]["colab"]["name"] = TARGET.name
    intro = "".join(notebook["cells"][0]["source"])
    intro += "\n## Data preparation fix\n\nStop caching a source bucket/split once its 125% candidate target and four-span minimum are met. This prevents completed English buckets from filling the cache while another bucket catches up. JSON disk limits scale with the source job's quotas: **514 MiB for the shared English source**, at least **256 MiB for other jobs**. These are temporary disk-cache limits, not TPU HBM limits or final corpus sizes. The final training corpus remains **256 MiB**. Refill only deficient strata after deduplication.\n\nUse the same Drive account and top-level ShiningGem folder. Existing V3.9 source locks and verified cache parts are reused; interrupted streams continue from their committed row cursor. A remote stream's `skip()` may reread earlier remote data. Run this new notebook top to bottom; do not delete the V3.9 cache or old run folder. Training still runs for **1,200 seconds**, with the same adaptive batch/context schedule.\n"
    notebook["cells"][0] = markdown(intro)
    data = "".join(notebook["cells"][16]["source"])
    data = once(data, 'candidate JSONL cap remains 256 MiB.',
        'candidate JSONL cap is `max(256 MiB, 4 × total job text quotas)`: 514 MiB for the combined English job. Satisfied bucket/split strata are no longer cached while waiting for another stratum. Deduplication refills extend only deficient strata. Existing verified V3.9 prefixes remain reusable.')
    notebook["cells"][16] = markdown(data)
    return notebook, source


def main():
    if TARGET.exists() or (HERE / "runner.py").exists():
        raise FileExistsError("Preserving existing artifacts; create a new version.")
    notebook, source = build_notebook()
    with (HERE / "runner.py").open("x") as stream:
        stream.write(source)
    with TARGET.open("x") as stream:
        json.dump(notebook, stream, indent=1, ensure_ascii=False)
        stream.write("\n")
    print(TARGET)


if __name__ == "__main__":
    main()
