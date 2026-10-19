"""Run, checkpoint, and export measured evidence into the explorer."""
from __future__ import annotations
import datetime
import importlib.metadata
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from .io import ROOT,read_json,write_json,read_jsonl,write_jsonl,file_hash,sha256
from .adapters import load_adapter,registry
from .vocabulary import inspect_vocabulary,compare_vocabularies
from .metrics import measure,aggregate,paired_bootstrap
from .conversion import map_vocabulary

def run(config_path="configs/launch.json",*,skip_maps=False,skip_runtime=False):
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    config=read_json(ROOT/config_path);documents=[];corpora=[]
    for corpus,manifest in [(config["corpus"],config["manifest"]),("benchmarks/diagnostics/corpus.jsonl.gz","benchmarks/diagnostics/manifest.json")]:
        if not (ROOT/corpus).exists():raise FileNotFoundError(f"Missing {corpus}. Run tokenizer-lab prepare first.")
        rows=list(read_jsonl(ROOT/corpus));documents.extend(rows)
        meta=read_json(ROOT/manifest);meta["corpus_file_sha256"]=file_hash(ROOT/corpus);corpora.append(meta)
    document_hash=sha256(json.dumps([(r["id"],r["sha256"]) for r in documents],separators=(",",":")).encode())
    out=ROOT/"results/v0.1";out.mkdir(parents=True,exist_ok=True)
    site_data=ROOT/"site/data";site_data.mkdir(parents=True,exist_ok=True)
    profiles=[];payloads={};adapters={};all_rows=[];runtime=[];failures=[]
    for id in config["tokenizers"]:
        print("TOKENIZER",id,flush=True)
        try:
            adapter=load_adapter(id);adapters[id]=adapter
            profile,payload=inspect_vocabulary(adapter,output=site_data/f"vocab/{id}.json.gz");profiles.append(profile);payloads[id]=payload
            checkpoint=out/f"checkpoints/{id}.jsonl.gz";meta=checkpoint.with_suffix(".meta.json")
            key={"protocol":config["protocol"],"artifact_sha256":adapter.asset_sha,"input_manifest_hash":document_hash,
                 "measurement_code":{name:file_hash(ROOT/"src/tokenizer_lab"/name) for name in ["adapters.py","metrics.py","io.py"]},
                 "packages":{x:importlib.metadata.version(x) for x in ["tokenizers","tiktoken"]}}
            metadata=read_json(meta) if meta.exists() else {}
            if checkpoint.exists() and {k:metadata.get(k) for k in key}==key and metadata.get("checkpoint_sha256")==file_hash(checkpoint):
                rows=list(read_jsonl(checkpoint))
                if len(rows)!=len(documents) or [(x["id"],x["sha256"]) for x in rows]!=[(x["id"],x["sha256"]) for x in documents]:
                    raise ValueError("Checkpoint row identities do not match the frozen corpus")
                print("  restored",len(rows),"verified rows",flush=True)
            else:
                rows=measure(adapter,documents);write_jsonl(checkpoint,rows);write_json(meta,key|{"checkpoint_sha256":file_hash(checkpoint)})
                print("  measured",len(rows),"documents",flush=True)
            all_rows.extend(rows)
            if not skip_runtime:
                selected=[];size=0
                # Interleave languages so the runtime sample is not all one language.
                groups={}
                for d in documents:
                    if d["dataset"]=="fineweb-mixture-v0.1":groups.setdefault(d["language"],[]).append(d["text"])
                for batch in itertools.zip_longest(*groups.values()):
                    for text in batch:
                        if text is not None:selected.append(text);size+=len(text.encode())
                    if size>=config["runtime_bytes"]:break
                input_path=out/"runtime-input.json";write_json(input_path,selected)
                proc=subprocess.run([sys.executable,"-m","tokenizer_lab.cli","_runtime",id,str(input_path),str(config["runtime_repeats"])],capture_output=True,text=True,check=True,env=os.environ|{"TOKENIZERS_PARALLELISM":"false"})
                result=json.loads(proc.stdout);result["input_sha256"]=file_hash(input_path);runtime.append(result)
        except Exception as exc:
            failures.append({"tokenizer":id,"error":f"{type(exc).__name__}: {exc}"[:1200]});print("  FAILED",str(exc),flush=True)
    primary=[p for p in profiles if p["role"]=="primary"]
    pairs=[compare_vocabularies(payloads[a["id"]],payloads[b["id"]]) for a,b in itertools.combinations(primary,2)]
    mappings=[]
    if not skip_maps:
        for source,target in itertools.permutations(primary,2):
            sid,tid=source["id"],target["id"];path=site_data/f"mappings/{sid}--{tid}.json.gz"
            if path.exists():
                existing=read_json(path)
                if existing.get("source_sha256")==adapters[sid].asset_sha and existing.get("target_sha256")==adapters[tid].asset_sha:
                    mappings.append({"source":sid,"target":tid,"path":f"data/mappings/{sid}--{tid}.json.gz","rows":len(existing["rows"]),"counts":dict(Counter(x["status"] for x in existing["rows"]))});continue
            print("MAP",sid,"→",tid,flush=True);mapping=map_vocabulary(adapters[sid],adapters[tid],payloads[sid]);write_json(path,mapping,compact=True)
            mappings.append({"source":sid,"target":tid,"path":f"data/mappings/{sid}--{tid}.json.gz","rows":len(mapping["rows"]),"counts":dict(Counter(x["status"] for x in mapping["rows"]))})
    benchmarks=aggregate(all_rows)
    hub={"schema_version":"1.0","generated_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"profiles":profiles,
         "benchmarks":benchmarks,"runtime":runtime,"pairs":pairs,"mappings":mappings,"corpora":corpora,"experiments":[],
         "checks":{"configured_tokenizers":len(config["tokenizers"]),"loaded_tokenizers":len(profiles),"documents_per_tokenizer":len(documents),
                   "expected_observations":len(documents)*len(config["tokenizers"]),"observations":len(all_rows),"failures":failures,
                   "input_manifest_hash":document_hash,"pair_count":len(pairs),"directional_mapping_count":len(mappings)},
         "protocol":{"id":config["protocol"],"title":"Intrinsic tokenizer analysis","description":"Original UTF-8 text, native tokenizers, no automatic special tokens, no implicit normalization or truncation.","weighting":"Per-language metrics first; pooled counts separately labelled; no combined quality score.","docs_path":"https://github.com/godkdf/Shinning-Tokens/blob/main/BENCHMARK.md"},
         "reproduction":{"commands":[{"label":"Install","command":"uv sync --extra test"},{"label":"Prepare corpus","command":"uv run tokenizer-lab prepare"},{"label":"Run measurements","command":"uv run tokenizer-lab run --config configs/launch.json"},{"label":"Open explorer","command":"uv run tokenizer-lab serve"}],"notes":["No model weights are needed for intrinsic analysis.","Gemma assets may require accepting their upstream terms. Artifacts and corpus revisions are fixed by manifests."]}}
    if "gemma3_parent" in adapters and "gemma_v31" in adapters:
        parent=[r for r in all_rows if r["tokenizer"]=="gemma3_parent"];child=[r for r in all_rows if r["tokenizer"]=="gemma_v31"]
        hub["experiments"].append({"id":"gemma-v31","title":"Gemma V3.1: 2,000 appended tokens","parent":"gemma3_parent","child":"gemma_v31",
          "status":"fresh intrinsic measurements","description":"Compare against the exact Gemma 3 parent control. Gemma4 is a general comparator, not the causal parent.",
          "paired_results":paired_bootstrap(parent,child,repeats=config["bootstrap_replicates"],seed=config["seed"]),
          "historical_note":"Earlier 2.8038% reduction was measured on the same MADLAD selection sample, not a held-out evaluation.","model_quality_evaluated":False})
    enrich(hub,all_rows)
    write_json(out/"hub.json",hub);write_json(site_data/"hub.json",hub)
    write_jsonl(ROOT/"results/raw/observations.jsonl.gz",all_rows)
    shutil.copyfile(ROOT/"results/raw/observations.jsonl.gz",site_data/"observations.jsonl.gz")
    write_examples(adapters,documents,site_data)
    write_report(hub)
    if (ROOT/"research/methods.json").exists():shutil.copyfile(ROOT/"research/methods.json",site_data/"methods.json")
    return hub

def write_examples(adapters,documents,site_data):
    selected=[];seen=set()
    for d in documents:
        if d["dataset"]=="fineweb-mixture-v0.1" and d["language"] not in seen and len(d["text"])<3000:
            selected.append(d);seen.add(d["language"])
    unicode=[d for d in documents if d["dataset"]=="unicode-diagnostics-v0.1"]
    selected.extend([d for d in unicode if d.get("transformation")=="raw"])
    selected.extend([d for d in unicode if d.get("transformation")=="NFD"][:5])
    selected.extend([d for d in documents if d["dataset"]=="numbers-v0.1"][:30])
    seen_code=set()
    for d in documents:
        if d["dataset"]=="code-diagnostics-v0.1" and d["source_id"] not in seen_code:
            selected.append(d);seen_code.add(d["source_id"])
    examples=[]
    for d in selected:
        encodings={}
        for id,adapter in adapters.items():
            try:
                ids=adapter.encode(d["text"]);decoded=adapter.decode_bytes(ids)
                encodings[id]={"ids":ids,"pieces":adapter.pieces(ids),"decoded":decoded.decode("utf-8",errors="backslashreplace"),"exact":decoded==d["text"].encode(),"tokens":len(ids),"bytes_per_token":len(d["text"].encode())/len(ids) if ids else None}
            except Exception as exc:encodings[id]={"error":str(exc)}
        preview=d["text"].replace("\n"," ↵ ").replace("\t"," ⇥ ")[:65]
        examples.append({"id":d["id"],"label":f"{d['language']} · {d.get('transformation',d['domain'])} · {preview}","language":d["language"],"text":d["text"],"sha256":d["sha256"],"encodings":encodings})
    write_json(site_data/"examples.json",{"schema_version":"1.0","examples":examples},compact=True)

def enrich(hub,rows):
    hub["environment"]={"python":platform.python_version(),"packages":{p:importlib.metadata.version(p) for p in ["tokenizers","tiktoken","numpy","regex","pyarrow","requests","psutil","huggingface-hub"]}}
    for exp in hub["experiments"]:
        if exp["id"]!="gemma-v31":continue
        parent=sum(r["tokens"] for r in rows if r["tokenizer"]=="gemma3_parent" and r["dataset"]=="fineweb-mixture-v0.1")
        child=sum(r["tokens"] for r in rows if r["tokenizer"]=="gemma_v31" and r["dataset"]=="fineweb-mixture-v0.1")
        exp.update(docs_path="experiments/gemma-v31/README.md",selection_data="Historical MADLAD selection sample; 25,485 documents, 418 configurations",evaluation_data="Fresh pinned FineWeb mixture; exact original bytes; near-duplicate decontamination against selection data not established",
          metrics=[{"label":"Parent tokens · pooled FineWeb","value":parent},{"label":"Candidate tokens · pooled FineWeb","value":child},{"label":"Tokens saved","value":parent-child},{"label":"Reduction · %","value":100*(1-child/parent)}],
          limitations=["Reconstructed parent control; official gated replay not verified.","Intrinsic compression does not establish generation quality.","Some languages regress; inspect the paired table."])
    if not any(e["id"]=="embedding-mapping" for e in hub["experiments"]):
        report=read_json(ROOT/"experiments/embedding-mapping/historical-report.json")
        hub["experiments"].append({"id":"embedding-mapping","title":"GPT-OSS → BLOOM input-vector mapping","status":"historical local report; not rerun","description":"A separate exploratory mapping from 2,880-dimensional source vectors to 1,024-dimensional target vectors. Core CLI baselines are executable; the original model experiment is historical.",
          "metrics":[{"label":"Fit / validation / test anchors","value":"32,077 / 3,828 / 4,095"},{"label":"Test MSE","value":report["test_map"]["coordinate_mse"]},{"label":"Training-mean baseline MSE","value":report["test_training_mean_baseline"]["coordinate_mse"]},{"label":"Mean cosine","value":report["test_map"]["mean_cosine"]}],"docs_path":"experiments/embedding-mapping/README.md","limitations":["Input initialization only; no downstream generation evaluation.","Original split construction has not been independently audited here."]})

def write_report(hub):
    lines=["# Tokenizer Lab — measured results","",f"Protocol: **{hub['protocol']['id']}**. Generated: {hub['generated_at']}.","",
           "All counts below derive from saved original-text observations. Diagnostic compression is not model quality.","",
           "| Tokenizer | Vocabulary IDs | Special IDs | ID space | Unique ordinary byte pieces |","|---|---:|---:|---:|---:|"]
    for p in hub["profiles"]:
        v=p["vocabulary"];lines.append(f"| {p['name']} | {v['total']:,} | {v['special']:,} | {v['id_space']:,} | {v['unique_byte_pieces']:,} |")
    lines.extend(["","## Language measurements","","| Dataset | Language | Tokenizer | Documents | Tokens | Bytes/token | Exact | Errors |","|---|---|---|---:|---:|---:|---:|---:|"])
    for b in hub["benchmarks"]:
        val=f"{b['bytes_per_token']:.4f}" if b["bytes_per_token"] is not None else "unavailable"
        lines.append(f"| {b['dataset']} | {b['language']} | {b['tokenizer']} | {b['documents']} | {b['tokens']:,} | {val} | {b['exact']}/{b['documents']} | {b['errors']} |")
    path=ROOT/"reports/v0.1.md";path.parent.mkdir(exist_ok=True);path.write_text("\n".join(lines)+"\n")
