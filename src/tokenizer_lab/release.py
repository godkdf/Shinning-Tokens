"""Package the exact expansion inputs with their upstream notices."""
from pathlib import Path
import gzip
import io
import tarfile
from .io import ROOT,read_json,write_json,file_hash,sha256

INPUTS=["artifacts/gemma_v31/tokenizer.json",".cache/assets/gemma3_parent/tokenizer.json"]

def package_inputs():
    path=ROOT/"site/data/gemma-v31-inputs.tar.gz"
    members=INPUTS+["licenses/NOTICE-GEMMA.txt","licenses/Gemma-3-Terms.html","licenses/Gemma-3-Prohibited-Use.html","experiments/gemma-v31/README.md","experiments/gemma-v31/identity.json","experiments/gemma-v31/additions.jsonl"]
    with path.open("wb") as output:
        with gzip.GzipFile(fileobj=output,filename="",mode="wb",mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode="w") as archive:
                for name in members:
                    raw=(ROOT/name).read_bytes();info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644
                    archive.addfile(info,io.BytesIO(raw))
    manifest={"schema_version":"1.0","file":"gemma-v31-inputs.tar.gz","sha256":file_hash(path),"bytes":path.stat().st_size,
              "inputs":{name:{"sha256":file_hash(ROOT/name),"bytes":(ROOT/name).stat().st_size} for name in INPUTS},
              "terms":"Gemma 3 Terms of Use and incorporated use restrictions; agreement and modification notice included.","provenance":"Supplied experimental candidate and reconstructed parent control; no official parent replay verification."}
    write_json(ROOT/"experiments/gemma-v31/bundle.json",manifest)
    return manifest

def install_inputs(path):
    manifest=read_json(ROOT/"experiments/gemma-v31/bundle.json");path=Path(path)
    if file_hash(path)!=manifest["sha256"]:raise ValueError("Experiment bundle hash mismatch")
    staged={}
    with tarfile.open(path,"r:gz") as archive:
        for name in INPUTS:
            member=archive.getmember(name)
            if not member.isfile() or member.size!=manifest["inputs"][name]["bytes"]:raise ValueError("Invalid experiment member")
            raw=archive.extractfile(member).read()
            if sha256(raw)!=manifest["inputs"][name]["sha256"]:raise ValueError("Experiment member hash mismatch")
            staged[name]=raw
    for name,raw in staged.items():
        target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    write_json(ROOT/".cache/assets/gemma3_parent/provenance.json",{"source":"reconstructed parent control","operation":"Remove precisely the 2,000 appended IDs >= 262145; preserve all inherited components.","official_asset_replay_verified":False})
    return {"installed":list(staged),"verified_bundle_sha256":manifest["sha256"]}
