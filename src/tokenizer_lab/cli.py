"""Public command line. Commands print structured evidence, not estimated scores."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
from .io import ROOT, read_json, write_json, file_hash

def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))

def main(argv=None):
    parser=argparse.ArgumentParser(prog="tokenizer-lab")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("doctor")
    prep=sub.add_parser("prepare");prep.add_argument("--diagnostics-only",action="store_true");prep.add_argument("--bytes-per-language",type=int,default=1_000_000)
    run=sub.add_parser("run");run.add_argument("--config",default="configs/launch.json");run.add_argument("--skip-maps",action="store_true");run.add_argument("--skip-runtime",action="store_true")
    sub.add_parser("build-site")
    serve=sub.add_parser("serve");serve.add_argument("--port",type=int,default=8765)
    launch=sub.add_parser("launch");launch.add_argument("--port",type=int,default=8765);launch.add_argument("--no-browser",action="store_true")
    selected=sub.add_parser("run-selected")
    for key in ["tokenizers","languages","datasets"]:selected.add_argument("--"+key,nargs="*",default=[])
    selected.add_argument("--custom-tokenizers",nargs="*",default=[]);selected.add_argument("--selection");selected.add_argument("--output");selected.add_argument("--vocabulary",action="store_true");selected.add_argument("--runtime",action="store_true")
    expanded=sub.add_parser("prepare-expanded");expanded.add_argument("--flores",action="store_true");expanded.add_argument("--madlad",action="store_true")
    cv=sub.add_parser("convert");cv.add_argument("--source",required=True);cv.add_argument("--target",required=True);input_group=cv.add_mutually_exclusive_group(required=True);input_group.add_argument("--text");input_group.add_argument("--ids",help="JSON array of source IDs")
    vocab=sub.add_parser("vocab");vocab.add_argument("tokenizer");vocab.add_argument("--output")
    compare=sub.add_parser("compare");compare.add_argument("source");compare.add_argument("target")
    sub.add_parser("verify")
    expansion=sub.add_parser("build-expansion");expansion.add_argument("--parent",required=True);expansion.add_argument("--output",default=str(ROOT/"artifacts/gemma_v31/tokenizer.json"))
    install=sub.add_parser("install-experiment");install.add_argument("--bundle",required=True)
    rt=sub.add_parser("_runtime");rt.add_argument("tokenizer");rt.add_argument("input");rt.add_argument("repeats",type=int)
    avg=sub.add_parser("transfer-average");avg.add_argument("--embeddings",required=True);avg.add_argument("--sequences",required=True);avg.add_argument("--output",required=True)
    mapping=sub.add_parser("transfer-fit");mapping.add_argument("--source",required=True);mapping.add_argument("--target",required=True);mapping.add_argument("--output",required=True);mapping.add_argument("--regularization",type=float,default=.01)
    apply=sub.add_parser("transfer-apply");apply.add_argument("--source",required=True);apply.add_argument("--mapping",required=True);apply.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    if args.command=="doctor":
        import importlib.metadata
        from .adapters import registry,resolve_asset
        assets=[]
        for spec in registry():
            try:
                path=resolve_asset(spec,download=False) if spec["kind"]!="tiktoken" else None
                assets.append({"id":spec["id"],"status":"cached" if path else "native runtime","sha256":file_hash(path) if path else None})
            except FileNotFoundError as exc:assets.append({"id":spec["id"],"status":"missing","detail":str(exc)})
        emit({"root":str(ROOT),"python":sys.version.split()[0],"packages":{p:importlib.metadata.version(p) for p in ["tokenizers","tiktoken","numpy","pyarrow"]},"assets":assets})
    elif args.command=="prepare":
        from .corpora import build_diagnostics,build_fineweb
        emit(build_diagnostics())
        if not args.diagnostics_only:
            if args.bytes_per_language<1:parser.error("Byte target must be positive")
            emit(build_fineweb(bytes_per_language=args.bytes_per_language))
    elif args.command=="run":
        from .pipeline import run
        hub=run(args.config,skip_maps=args.skip_maps,skip_runtime=args.skip_runtime);emit(hub["checks"])
        if hub["checks"]["failures"]:sys.exit(1)
    elif args.command=="build-site":
        source=ROOT/"results/v0.1/hub.json"
        if not source.exists():parser.error("Run measurements before building the site")
        shutil.copyfile(source,ROOT/"site/data/hub.json")
        shutil.copyfile(ROOT/"research/methods.json",ROOT/"site/data/methods.json")
        emit({"site":str(ROOT/"site"),"checks":read_json(source)["checks"]})
    elif args.command=="serve":
        from .server import serve
        serve(args.port)
    elif args.command=="launch":
        from .server import serve
        serve(args.port,launch=not args.no_browser)
    elif args.command=="run-selected":
        from .engine import run_selection
        value={key:getattr(args,key) for key in ["tokenizers","languages","datasets","custom_tokenizers"]}
        if args.selection:
            if any(value.values()):parser.error("Use a selection file or explicit selection lists, not both")
            value=read_json(Path(args.selection).expanduser())
        result=run_selection(value,output=args.output,vocabulary=args.vocabulary,runtime=args.runtime)
        emit(result["checks"])
    elif args.command=="prepare-expanded":
        from .corpus_catalog import import_existing,prepare_flores,prepare_madlad
        emit(import_existing())
        if args.flores:emit(prepare_flores())
        if args.madlad:
            result=prepare_madlad();emit({k:v for k,v in result.items() if k!="configs"})
    elif args.command in ("convert","vocab","compare"):
        from .adapters import load_adapter
        if args.command=="convert":
            from .conversion import convert
            emit(convert(load_adapter(args.source),load_adapter(args.target),text=args.text,ids=json.loads(args.ids) if args.ids is not None else None))
        elif args.command=="vocab":
            from .vocabulary import inspect_vocabulary
            profile,_=inspect_vocabulary(load_adapter(args.tokenizer),output=Path(args.output) if args.output else None);emit(profile)
        else:
            from .vocabulary import inspect_vocabulary,compare_vocabularies
            _,a=inspect_vocabulary(load_adapter(args.source));_,b=inspect_vocabulary(load_adapter(args.target));emit(compare_vocabularies(a,b))
    elif args.command=="verify":
        from .verify import verify
        emit(verify())
    elif args.command=="build-expansion":
        from .expansion import build
        emit(build(args.parent,args.output))
    elif args.command=="install-experiment":
        from .release import install_inputs
        emit(install_inputs(args.bundle))
    elif args.command=="_runtime":
        from .runtime import measure_runtime
        emit(measure_runtime(args.tokenizer,read_json(args.input),repeats=args.repeats))
    elif args.command.startswith("transfer-"):
        import numpy as np
        from .transfer import subtoken_average,ridge_mapping,apply_mapping
        if args.command=="transfer-average":np.save(args.output,subtoken_average(np.load(args.embeddings,allow_pickle=False),read_json(args.sequences)))
        elif args.command=="transfer-fit":np.savez(args.output,**ridge_mapping(np.load(args.source,allow_pickle=False),np.load(args.target,allow_pickle=False),regularization=args.regularization))
        else:
            with np.load(args.mapping,allow_pickle=False) as mapping:np.save(args.output,apply_mapping(np.load(args.source,allow_pickle=False),mapping))
        emit({"output":args.output,"scope":"Embedding initialization only; input/output heads and subsequent training must be evaluated separately."})

if __name__=="__main__":main()
