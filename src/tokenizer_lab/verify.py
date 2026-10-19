"""Independent consistency audit of the generated release bundle."""
from collections import Counter
from .io import ROOT,read_json,read_jsonl,validate_text_record,file_hash,write_json

def verify():
    hub=read_json(ROOT/"results/v0.1/hub.json")
    documents=[]
    for path in ["benchmarks/fineweb-v0.1/corpus.jsonl.gz","benchmarks/diagnostics/corpus.jsonl.gz"]:
        documents.extend(read_jsonl(ROOT/path))
    assert len({d["id"] for d in documents})==len(documents),"Duplicate document IDs"
    for d in documents:validate_text_record(d)
    docs={d["id"]:d for d in documents};groups={}
    corpus=read_json(ROOT/"benchmarks/fineweb-v0.1/manifest.json")
    assert corpus["covered_languages"]==24 and all(x["status"] in ["complete","short"] for x in corpus["languages"]),"Incomplete language coverage"
    assert corpus["actual_utf8_bytes"]==sum(d["utf8_bytes"] for d in documents if d["dataset"]=="fineweb-mixture-v0.1")
    observed=set()
    for row in read_jsonl(ROOT/"results/raw/observations.jsonl.gz"):
        assert row["id"] in docs and row["sha256"]==docs[row["id"]]["sha256"],"Changed input identity"
        identity=(row["tokenizer"],row["id"]);assert identity not in observed,"Duplicate observation";observed.add(identity)
        groups.setdefault((row["dataset"],row["language"],row["tokenizer"]),[]).append(row)
    assert len(observed)==len(documents)*len(hub["profiles"]),"Incomplete tokenizer/document matrix"
    for b in hub["benchmarks"]:
        rows=groups[(b["dataset"],b["language"],b["tokenizer"])]
        for name,expected in [("documents",len(rows)),("tokens",sum(r["tokens"] for r in rows)),("utf8_bytes",sum(docs[r["id"]]["utf8_bytes"] for r in rows)),("exact",sum(r["exact"] for r in rows))]:
            assert b[name]==expected,f"Aggregate mismatch: {name}"
        assert b["exact"]+b["canonical_only"]+b["mismatches"]+b["errors"]==b["documents"],"Fidelity partitions differ"
        if b["bytes_per_token"] is not None:assert abs(b["bytes_per_token"]-b["utf8_bytes"]/b["tokens"])<1e-10
    vocabularies={};profiles={p["id"]:p for p in hub["profiles"]}
    for p in hub["profiles"]:
        v=read_json(ROOT/f"site/data/vocab/{p['id']}.json.gz")
        assert len(v["rows"])==p["vocabulary"]["total"]
        assert len({r[0] for r in v["rows"]})==len(v["rows"]),"Duplicate vocabulary IDs"
        assert v["artifact_sha256"]==p["artifact_sha256"]
        vocabularies[p["id"]]={r[0] for r in v["rows"]}
    assert len(hub["pairs"])==15 and len(hub["mappings"])==30,"Incomplete comparison graph"
    assert len({(m["source"],m["target"]) for m in hub["mappings"]})==30,"Duplicate mapping direction"
    for m in hub["mappings"]:
        data=read_json(ROOT/"site"/m["path"]);source=m["source"];target=m["target"]
        assert data["source_sha256"]==profiles[source]["artifact_sha256"] and data["target_sha256"]==profiles[target]["artifact_sha256"]
        assert {r["source_id"] for r in data["rows"]}==vocabularies[source] and len(data["rows"])==len(vocabularies[source]),"Incomplete mapping rows"
        assert dict(Counter(r["status"] for r in data["rows"]))==m["counts"],"Mapping summary differs"
        assert all(set(r.get("target_ids") or [])<=vocabularies[target] and set(r.get("matching_piece_ids") or [])<=vocabularies[target] for r in data["rows"]),"Mapping contains invalid target IDs"
    assert not hub["checks"]["failures"],"Recorded pipeline failures"
    assert len(hub["runtime"])==len(hub["profiles"]),"Runtime missing"
    files={}
    for base in ["benchmarks","configs","experiments","research","src","tests","docs","reports","results/v0.1","results/raw","site/data"]:
        for path in sorted((ROOT/base).rglob("*")):
            if path.is_file() and path.name!="bundle-manifest.json" and "__pycache__" not in path.parts:files[str(path.relative_to(ROOT))]={"sha256":file_hash(path),"bytes":path.stat().st_size}
    for name in ["pyproject.toml","uv.lock","README.md","BENCHMARK.md","THIRD_PARTY.md","LICENSE"]:
        path=ROOT/name;files[name]={"sha256":file_hash(path),"bytes":path.stat().st_size}
    result={"schema_version":"1.0","status":"passed","documents":len(documents),"observations":len(observed),"tokenizers":len(hub["profiles"]),"vocabulary_pairs":15,"directional_maps":30,"files":files}
    write_json(ROOT/"results/bundle-manifest.json",result)
    return {k:v for k,v in result.items() if k!="files"}|{"hashed_files":len(files)}
