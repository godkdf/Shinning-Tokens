"""Explicit selections, verified payloads, cancellable network workers."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid
from .io import ROOT,read_json,write_json,file_hash
from .engine import selection,selected_partitions
from .adapters import registry,asset_path

_jobs={};_lock=threading.RLock()

def _safe_path(relative):
    path=(ROOT/relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):raise ValueError("Payload path escapes Lab directory")
    return path

def _cached(path,expected):
    return path.is_file() and (not expected or file_hash(path)==expected)

def catalog():
    entries=[]
    for spec in registry():
        digest=spec.get("offline_sha256") if spec["kind"]=="tiktoken" else spec.get("sha256")
        valid=_cached(asset_path(spec),digest)
        if valid:
            valid=all(_cached(asset_path(spec).parent/a["file"],a.get("sha256")) for a in spec.get("companions",[]))
        entries.append(spec|{"cached":valid})
    langs=ROOT/"benchmarks/languages.json"
    corpus=read_json(ROOT/"benchmarks/catalog.json") if (ROOT/"benchmarks/catalog.json").exists() else {"datasets":[]}
    return {"tokenizers":entries,"languages":read_json(langs)["languages"] if langs.exists() else [],"datasets":corpus["datasets"],"default_selection":{"tokenizers":[],"datasets":[],"languages":[]},"offline":os.environ.get("TOKENIZER_LAB_OFFLINE")=="1"}

def plan(value):
    value=selection(value);specs={s["id"]:s for s in registry()}
    unknown=set(value["tokenizers"])-specs.keys()
    if unknown:raise ValueError("Unknown tokenizers: "+", ".join(sorted(unknown)))
    files=[];missing=[]
    for id in value["tokenizers"]:
        spec=specs[id];path=asset_path(spec);digest=spec.get("offline_sha256") if spec["kind"]=="tiktoken" else spec.get("sha256")
        records=[{"path":str(path.relative_to(ROOT)),"sha256":digest,"bytes":spec.get("bytes"),"source_file":spec.get("file")}]
        records += [{"path":str((path.parent/a["file"]).relative_to(ROOT)),"sha256":a["sha256"],"bytes":a.get("bytes"),"source_file":a["file"]} for a in spec.get("companions",[]) if a["file"]!=path.name]
        for item in records:
            target=_safe_path(item["path"]);cached=_cached(target,item["sha256"])
            if cached or item.get("bytes") is None:
                if target.exists():item["bytes"]=target.stat().st_size
            row=item|{"id":id+":"+Path(item["path"]).name,"tokenizer":id,"kind":"tokenizer","cached":cached}
            if spec.get("repo") and spec.get("revision"):row["url"]=f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{item['source_file']}"
            elif spec["kind"]=="tiktoken":row["native_encoding"]=spec["encoding"]
            elif not cached:missing.append({"id":id,"reason":"Local artifact: reinstall its experiment bundle or register your local tokenizer file"})
            files.append(row)
        for optional in [ROOT/f"site/data/vocab/{id}.json.gz",path.parent/"ported-tokenizer.json"]:
            if optional.is_file():files.append({"id":id+":"+optional.name,"kind":"cached derived data","path":str(optional.relative_to(ROOT)),"bytes":optional.stat().st_size,"sha256":file_hash(optional),"cached":True})
    index_path=ROOT/"site/data/text-index.json"
    texts=read_json(index_path)["partitions"] if index_path.exists() else []
    for part in selected_partitions(value):
        item=next((x for x in texts if x["dataset"]==part["dataset"] and x["language"]==part["language"]),None)
        if item is None:missing.append({"id":part["language"],"reason":"No frozen compressed text manifest"});continue
        relative="site/"+item["path"];target=_safe_path(relative)
        files.append({"id":part["dataset"]+":"+part["language"],"kind":"corpus","dataset":part["dataset"],"language":part["language"],"path":relative,"bytes":item["bytes"],"sha256":item["sha256"],"cached":_cached(target,item["sha256"]),"source":part.get("source",{})})
        if item.get("examples_path"):
            optional=_safe_path("site/"+item["examples_path"])
            if optional.is_file():files.append({"id":part["language"]+":saved encodings","kind":"cached derived data","path":str(optional.relative_to(ROOT)),"bytes":optional.stat().st_size,"sha256":file_hash(optional),"cached":True})
    needed=[f for f in files if not f["cached"]];cached=[f for f in files if f["cached"]]
    unknown_network=any(f["kind"]=="corpus" or f.get("native_encoding") for f in needed)
    total=lambda rows:sum(f["bytes"] for f in rows) if all(isinstance(f.get("bytes"),int) for f in rows) else None
    flores=any(f.get("dataset","").startswith("flores200-") for f in needed)
    return {"selection":value,"files":files,"missing":missing,"download_bytes":None if unknown_network else total(needed),"cached_bytes":total(cached),"total_bytes":total(files),"transient_download_bytes":25_585_843 if flores else None,
            "network_note":("FLORES source replay temporarily reads the 25,585,843-byte official archive once; only selected language/split payloads are installed, then the temporary archive is deleted. " if flores else "")+("Corpus source transfers differ from installed compressed payload size. MADLAD uses bounded gzip streams; FineWeb uses Parquet byte ranges. Exact network total is unknown until transfer completes." if unknown_network else "Only the listed uncached files will be transferred.")}

def _snapshot(job):
    return {k:v for k,v in job.items() if not k.startswith("_")}

def get(id):
    with _lock:
        if id not in _jobs:raise KeyError("Unknown download")
        job=_jobs[id];status=ROOT/job["_status"]
        if status.exists():
            try:job.update(read_json(status))
            except (OSError,json.JSONDecodeError):pass
        code=job["_process"].poll()
        if code is not None and job["status"] in ["queued","running"]:
            job.update(status="failed",error="Download worker stopped before completion")
        return _snapshot(job)

def start(value):
    proposal=plan(value)
    if proposal["missing"]:raise ValueError("Selected local inputs are unavailable; inspect the plan")
    with _lock:
        if any(get(id)["status"] in ["queued","running"] for id in _jobs):raise ValueError("A download is already active")
        id=uuid.uuid4().hex;folder=ROOT/f".cache/download-jobs/{id}";folder.mkdir(parents=True)
        write_json(folder/"plan.json",proposal)
        status=str((folder/"status.json").relative_to(ROOT))
        # Explicit network action lives in its own process. Offline run workers
        # retain their network guard and never inherit this exception.
        process=subprocess.Popen([sys.executable,"-m","tokenizer_lab.downloads",str(folder/"plan.json"),str(folder/"status.json")],cwd=ROOT,env=os.environ|{"TOKENIZER_LAB_OFFLINE":"0"},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        _jobs[id]={"id":id,"status":"queued","progress":{"completed":0,"total":len(proposal["files"])},"_status":status,"_process":process}
        return _snapshot(_jobs[id])

def cancel(id):
    with _lock:
        get(id);job=_jobs[id]
        if job["_process"].poll() is None:job["_process"].terminate();job["_process"].wait(timeout=10)
        job.update(status="cancelled");write_json(ROOT/job["_status"],_snapshot(job))
        for path in (ROOT/".cache/download-jobs"/id).glob("*.part"):path.unlink(missing_ok=True)
        return _snapshot(job)

def remove(value):
    with _lock:
        if any(get(id)["status"] in ["queued","running"] for id in _jobs):raise ValueError("Cancel the active download before removing cached files")
        proposal=plan(value);paths={f["path"] for f in proposal["files"]}
        for p in selected_partitions(value):paths.add(p["path"])
        freed=0;removed=[]
        for relative in sorted(paths):
            path=_safe_path(relative)
            if path.is_file():freed+=path.stat().st_size;path.unlink();removed.append(relative)
        return {"removed_files":removed,"freed_bytes":freed}

def _replay_corpus(item,work,session):
    """Restore exactly the frozen recipe; reject changed data by final hash."""
    import gzip,tarfile,random
    from .corpora import document
    dataset=item["dataset"];language=item["language"];source=item["source"]
    if dataset.startswith("flores200-"):
        archive=work/"flores-source.part"
        if not archive.exists():
            with session.get("https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",stream=True,timeout=(30,90)) as response:
                response.raise_for_status()
                with archive.open("wb") as output:
                    for chunk in response.iter_content(1<<20):output.write(chunk)
            if file_hash(archive)!=source["archive_sha256"]:raise ValueError("Official FLORES archive hash mismatch")
        split=dataset.removeprefix("flores200-")
        with tarfile.open(archive,"r:gz") as tar:
            member=next(m for m in tar.getmembers() if m.name.endswith(f"/{split}/{language}.{split}"))
            texts=tar.extractfile(member).read().decode("utf-8").splitlines()
        return [document(t,dataset=dataset,language=language,index=i+1,source_id=f"flores200:{split}:{i+1}",domain="parallel translation evaluation",split=split,source_line=i+1) for i,t in enumerate(texts)]
    if dataset=="madlad400-sample-v0.2":
        from .io import sha256
        code=source["source_config"];rows=[];seen=set();size=0
        url=f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{source['source_path']}"
        with session.get(url,stream=True,timeout=(30,90)) as response:
            response.raise_for_status()
            with gzip.GzipFile(fileobj=response.raw) as stream:
                for line_no in range(10000):
                    line=stream.readline(2_000_001)
                    if not line:break
                    if len(line)>2_000_000:raise ValueError("Source line exceeds bounded parser limit")
                    entry=json.loads(line);text=entry.get("text")
                    if not isinstance(text,str) or not text.strip():continue
                    raw=text.encode("utf-8");h=sha256(raw)
                    if len(raw)>100000 or h in seen:continue
                    seen.add(h);base=language.split("_madlad-")[0]
                    row=document(text,dataset=dataset,language=base,index=f"{code}:{len(rows)}",source_id=str(entry.get("url") or entry.get("id") or h),domain="web text sample",source_config=code,source_line=line_no+1,source_shard=source["source_path"],source_revision=source["revision"]);row["language"]=language;rows.append(row);size+=len(raw)
                    if size>=65536 or len(rows)>=32:break
        return rows
    if dataset=="fineweb-mixture-v0.1":
        import pyarrow.parquet as pq
        from .corpora import HTTPRangeFile
        from .io import sha256
        manifest=read_json(ROOT/"benchmarks/fineweb-v0.1/manifest.json");info=next(x for x in manifest["languages"] if x["language"]==language)
        repo=info["repo"];rev=info["revision"];sub=str(Path(info["source_path"]).parent)
        response=session.get(f"https://huggingface.co/api/datasets/{repo}/tree/{rev}/{sub}",params={"limit":1000},timeout=60);response.raise_for_status()
        files=sorted([x for x in response.json() if x.get("type")=="file" and x["path"].endswith(".parquet")],key=lambda x:x["path"])
        rng=random.Random(int(sha256(f"20261019:{language}".encode())[:16],16));entry=rng.choice(files)
        if entry["path"]!=info["source_path"]:raise ValueError("Pinned source selection changed")
        remote=HTTPRangeFile(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{entry['path']}",entry["size"],session);parquet=pq.ParquetFile(remote)
        order=list(range(parquet.num_row_groups));rng.shuffle(order);rows=[];seen=set();size=0
        for group in order:
            table=parquet.read_row_group(group,columns=[c for c in ["text","id","url","dump"] if c in parquet.schema.names]).to_pylist();rng.shuffle(table)
            for entryrow in table:
                text=entryrow.get("text")
                if not isinstance(text,str) or not text.strip():continue
                raw=text.encode("utf-8");h=sha256(raw)
                if h in seen or len(raw)>100000:continue
                seen.add(h);rows.append(document(text,dataset=dataset,language=language,index=len(rows),source_id=entryrow.get("url") or entryrow.get("id") or h,source_document_id=entryrow.get("id"),source_shard=entry["path"],source_row_group=group,source_revision=rev));size+=len(raw)
                if size>=manifest["target_bytes_per_language"]:return rows
        return rows
    from . import corpora
    from .io import read_jsonl
    # Construct diagnostics in scratch space so unselected categories are not
    # installed as side effects of a selected local payload restoration.
    previous=corpora.ROOT
    try:
        corpora.ROOT=work;corpora.build_diagnostics()
        return [r for r in read_jsonl(work/"benchmarks/diagnostics/corpus.jsonl.gz") if r["dataset"]==dataset and r["language"]==language]
    finally:corpora.ROOT=previous

def worker(plan_path,status_path):
    import requests
    proposal=read_json(plan_path);work=Path(plan_path).parent;done=0;transferred=0
    def status(state,**extra):write_json(status_path,{"status":state,"progress":{"completed":done,"total":len(proposal["files"]),"network_bytes":transferred},**extra})
    status("running")
    try:
        session=requests.Session()
        counter=[0]
        def count_body(response,*args,**kwargs):
            read=response.raw.read;stream=response.raw.stream;streaming=[False]
            def counted(*a,**kw):
                data=read(*a,**kw)
                if not streaming[0]:counter[0]+=len(data)
                return data
            def counted_stream(*a,**kw):
                streaming[0]=True
                try:
                    for data in stream(*a,**kw):counter[0]+=len(data);yield data
                finally:streaming[0]=False
            response.raw.read=counted
            response.raw.stream=counted_stream
            return response
        session.hooks["response"]=[count_body]
        for item in proposal["files"]:
            target=_safe_path(item["path"])
            if not _cached(target,item.get("sha256")):
                temp=work/(uuid.uuid4().hex+".part")
                if item["kind"]=="corpus":
                    rows=_replay_corpus(item,work,session);write_json(temp,{"schema_version":"2.0","documents":rows,"examples":[]},compact=True)
                    # write_json keys compression off .part; create the exact
                    # deterministic gzip representation expected by text index.
                    import gzip
                    temp.write_bytes(gzip.compress(temp.read_bytes(),mtime=0))
                elif item.get("native_encoding"):
                    from .adapters import TikAdapter
                    spec=next(s for s in registry() if s["id"]==item["tokenizer"]);TikAdapter(spec,download=True);done+=1;status("running");continue
                else:
                    with session.get(item["url"],stream=True,timeout=(30,90)) as response:
                        response.raise_for_status()
                        with temp.open("wb") as output:
                            for chunk in response.iter_content(1<<20):output.write(chunk);transferred=counter[0];status("running")
                if item.get("sha256") and file_hash(temp)!=item["sha256"]:raise ValueError("Restored payload differs from pinned manifest: "+item["id"])
                target.parent.mkdir(parents=True,exist_ok=True);temp.replace(target)
            done+=1;transferred=counter[0];status("running")
        status("complete",result={"installed_files":len(proposal["files"]),"http_response_body_bytes":transferred,"network_bytes_note":"HTTP response body bytes observed by this worker; excludes protocol/TLS headers and native tiktoken internal fetches"})
    except Exception as exc:status("failed",error=f"{type(exc).__name__}: {exc}")
    finally:
        for file in work.glob("*.part"):file.unlink(missing_ok=True)
        import shutil
        shutil.rmtree(work/"benchmarks",ignore_errors=True)

if __name__=="__main__":worker(*sys.argv[1:])
