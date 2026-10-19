"""Selected, cached-only measurements with one corpus partition in memory."""
from __future__ import annotations
import datetime
import importlib.metadata
import os
import threading
import uuid
from .io import ROOT,read_json,write_json,read_jsonl,write_jsonl,file_hash,sha256

def selection(value):
    if not isinstance(value,dict):raise ValueError("Selection must be an object")
    out={}
    for key in ("tokenizers","datasets","languages","custom_tokenizers"):
        items=value.get(key,[])
        if not isinstance(items,list) or any(not isinstance(x,str) for x in items):raise ValueError(f"{key} must be an explicit list of strings")
        out[key]=list(dict.fromkeys(items))
    return out

def selected_partitions(value):
    value=selection(value);catalog=read_json(ROOT/"benchmarks/catalog.json")
    valid={p["dataset"] for p in catalog["partitions"]};langs={p["language"] for p in catalog["partitions"]}
    for key,known in [("datasets",valid),("languages",langs)]:
        unknown=set(value[key])-known
        if unknown:raise ValueError(f"Unknown {key}: {', '.join(sorted(unknown))}")
    return [p for p in catalog["partitions"] if p["dataset"] in value["datasets"] and p["language"] in value["languages"]]

def read_partition(part):
    source=ROOT/part["path"]
    if source.is_file():
        if file_hash(source)!=part["sha256"]:raise ValueError("Corpus hash mismatch: "+part["path"])
        docs=list(read_jsonl(source))
    else:
        index=read_json(ROOT/"site/data/text-index.json")
        item=next(p for p in index["partitions"] if p["dataset"]==part["dataset"] and p["language"]==part["language"])
        source=ROOT/"site"/item["path"]
        if file_hash(source)!=item["sha256"]:raise ValueError("Compressed text hash mismatch")
        docs=read_json(source)["documents"]
    if len(docs)!=part["documents"]:raise ValueError("Corpus document count mismatch")
    return docs

def run_selection(value,*,progress=None,cancel=None,output=None,vocabulary=False,runtime=False):
    from .adapters import load_adapter,registry,register_custom
    from .metrics import measure,aggregate
    from .transport import probe
    value=selection(value);cancel=cancel or threading.Event();progress=progress or (lambda p:None)
    for path in value["custom_tokenizers"]:
        from pathlib import Path
        spec=register_custom(path,Path(path).stem);value["tokenizers"].append(spec["id"])
    value["tokenizers"]=list(dict.fromkeys(value["tokenizers"]))
    known={s["id"] for s in registry()}
    if set(value["tokenizers"])-known:raise ValueError("Unknown tokenizer selection")
    if any(not value[k] for k in ("tokenizers","languages","datasets")):raise ValueError("Select at least one tokenizer, language and benchmark")
    parts=selected_partitions(value)
    if not parts:raise ValueError("Selected languages have no documents in the selected benchmarks")
    out=ROOT/(output or f"results/local/{uuid.uuid4().hex}");out.mkdir(parents=True,exist_ok=True)
    code={p.name:file_hash(p) for p in (ROOT/"src/tokenizer_lab").glob("*.py") if p.name in ["engine.py","adapters.py","legacy.py","metrics.py","transport.py","io.py"]}
    packages={p:importlib.metadata.version(p) for p in ["tokenizers","tiktoken","sentencepiece"]}
    result={"schema_version":"2.0","generated_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"selection":value,"benchmarks":[],"profiles":[],"runtime":[],"transport_probes":{},"checkpoints":[],"failures":[],"measurement_code":code,"packages":packages,"network_downloads":0}
    total=len(parts)*len(value["tokenizers"]);done=0
    for id in value["tokenizers"]:
        if cancel.is_set():raise InterruptedError("Benchmark cancelled")
        adapter=load_adapter(id,download=False)
        if vocabulary:
            from .vocabulary import inspect_vocabulary
            profile,_=inspect_vocabulary(adapter,output=ROOT/f"site/data/vocab/{id}.json.gz");result["profiles"].append(profile)
        result["transport_probes"][id]=probe(adapter)
        for part in parts:
            if cancel.is_set():raise InterruptedError("Benchmark cancelled")
            checkpoint=out/f"checkpoints/{id}/{part['dataset']}/{part['language']}.jsonl.gz";meta=checkpoint.with_suffix(".meta.json")
            key={"protocol":"intrinsic-v0.2","artifact_sha256":adapter.asset_sha,"input_sha256":part["sha256"],"measurement_code":code,"packages":packages}
            metadata=read_json(meta) if meta.exists() else {}
            if checkpoint.exists() and all(metadata.get(k)==v for k,v in key.items()) and metadata.get("sha256")==file_hash(checkpoint):
                rows=list(read_jsonl(checkpoint))
                if len(rows)!=part["documents"]:raise ValueError("Checkpoint count mismatch")
            else:
                rows=measure(adapter,read_partition(part));write_jsonl(checkpoint,rows);write_json(meta,key|{"sha256":file_hash(checkpoint)})
            result["benchmarks"].extend(aggregate(rows));result["checkpoints"].append({"tokenizer":id,"dataset":part["dataset"],"language":part["language"],"path":str(checkpoint.relative_to(ROOT)),"sha256":file_hash(checkpoint),"documents":len(rows)})
            done+=1;progress({"completed":done,"total":total,"message":f"{id}: {part['dataset']} / {part['language']} ({done}/{total})"})
        if runtime:
            import subprocess,sys,json,itertools
            groups=[read_partition(p) for p in parts if p["dataset"]=="fineweb-mixture-v0.1"]
            if not groups:groups=[read_partition(p)[:4] for p in parts]
            texts=[];size=0
            for batch in itertools.zip_longest(*groups):
                for row in batch:
                    if row is not None:texts.append(row["text"]);size+=row["utf8_bytes"]
                if size>=1_000_000:break
            input_path=out/"runtime-input.json";write_json(input_path,texts)
            process=subprocess.run([sys.executable,"-m","tokenizer_lab.cli","_runtime",id,str(input_path),"5"],capture_output=True,text=True,check=True,env=os.environ|{"TOKENIZER_LAB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"})
            item=json.loads(process.stdout);item["input_sha256"]=file_hash(input_path);result["runtime"].append(item)
        write_json(out/"result.json",result)
        del adapter
    result["checks"]={"tokenizers":len(value["tokenizers"]),"partitions":len(parts),"documents_per_tokenizer":sum(p["documents"] for p in parts),"observations":sum(x["documents"] for x in result["benchmarks"]),"errors":sum(x["errors"] for x in result["benchmarks"])}
    result["output"]=str(out/"result.json");write_json(out/"result.json",result)
    return result
