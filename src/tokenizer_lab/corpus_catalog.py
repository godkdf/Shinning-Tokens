"""Partitioned corpus preparation and small searchable language metadata."""
from __future__ import annotations
import concurrent.futures
from collections import Counter,defaultdict
import csv
import gzip
import json
import re
import tarfile
from pathlib import Path
from .io import ROOT,read_json,write_json,read_jsonl,write_jsonl,file_hash,validate_text_record,sha256
from .corpora import document

DATASETS={"fineweb-mixture-v0.1":"FineWeb pinned mixture","unicode-diagnostics-v0.1":"Unicode diagnostics","numbers-v0.1":"Number diagnostics","code-diagnostics-v0.1":"Code diagnostics","flores200-dev":"FLORES-200 dev","flores200-devtest":"FLORES-200 devtest","madlad400-sample-v0.2":"MADLAD-400 bounded sample"}
SCRIPT_CODES={"Latin":"Latn","Arabic":"Arab","Cyrillic":"Cyrl","Devanagari":"Deva","Han":"Hani","Ethiopic":"Ethi","Bengali":"Beng","Tamil":"Taml","Telugu":"Telu","Thai":"Thai","Hebrew":"Hebr","Hangul":"Hang","Hiragana":"Jpan","Katakana":"Jpan","Myanmar":"Mymr","Georgian":"Geor","Armenian":"Armn","Greek":"Grek"}

def safe(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+",value):raise ValueError("Invalid corpus identifier")
    return value

def save_partition(dataset,language,rows,source=None):
    safe(dataset);safe(language)
    path=ROOT/f"benchmarks/partitions/{dataset}/{language}.jsonl.gz"
    for r in rows:validate_text_record(r)
    if len({r["id"] for r in rows})!=len(rows):raise ValueError("Duplicate document IDs")
    write_jsonl(path,rows)
    info={"dataset":dataset,"language":language,"path":str(path.relative_to(ROOT)),"documents":len(rows),"utf8_bytes":sum(len(r["text"].encode()) for r in rows),"sha256":file_hash(path),"source":source or {}}
    write_json(path.with_suffix(".meta.json"),info)
    return info

def import_existing():
    for path in [ROOT/"benchmarks/fineweb-v0.1/corpus.jsonl.gz",ROOT/"benchmarks/diagnostics/corpus.jsonl.gz"]:
        groups=defaultdict(list)
        for row in read_jsonl(path):groups[(row["dataset"],row["language"])].append(row)
        for (dataset,language),rows in groups.items():save_partition(dataset,language,rows,{"legacy_path":str(path.relative_to(ROOT))})
    return rebuild_catalog()

def prepare_flores(archive=None):
    path=Path(archive or ROOT/".cache/discovery/flores200_dataset.tar.gz")
    if not path.exists():
        import requests
        path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_suffix('.part')
        try:
            with requests.get('https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz',stream=True,timeout=(30,90)) as response:
                response.raise_for_status()
                with partial.open('wb') as output:
                    for chunk in response.iter_content(1<<20):output.write(chunk)
            if file_hash(partial)!='b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6':raise ValueError('FLORES archive differs from pinned release')
            partial.replace(path)
        finally:partial.unlink(missing_ok=True)
    archive_hash=file_hash(path);expected={"dev":997,"devtest":1012};count=0
    with tarfile.open(path,"r:gz") as source:
        for member in source.getmembers():
            parts=Path(member.name).parts
            if not member.isfile() or len(parts)<2 or parts[-2] not in expected:continue
            split=parts[-2];filename=parts[-1]
            if not filename.endswith("."+split):continue
            language=filename[:-(len(split)+1)]
            texts=source.extractfile(member).read().decode("utf-8",errors="strict").splitlines()
            if len(texts)!=expected[split]:raise ValueError(f"Unexpected FLORES sentence count: {language}/{split}: {len(texts)}")
            rows=[document(text,dataset="flores200-"+split,language=language,index=i+1,source_id=f"flores200:{split}:{i+1}",domain="parallel translation evaluation",split=split,source_line=i+1) for i,text in enumerate(texts)]
            save_partition("flores200-"+split,language,rows,{"url":"https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz","archive_sha256":archive_hash,"license":"CC-BY-SA-4.0","selection":"All published sentences; no normalization or filtering"});count+=1
    return {"partitions":count,"archive_sha256":archive_hash,**rebuild_catalog()}

def madlad_identity(code,rows):
    import langcodes,pycountry
    from .vocabulary import Scripts
    bits=re.split(r"[_-]",code);base=bits[0]
    lang=langcodes.Language.get(base)
    found=pycountry.languages.get(alpha_2=lang.language) or pycountry.languages.get(alpha_3=lang.language)
    iso=getattr(found,"alpha_3",base)
    explicit=next((x for x in bits[1:] if len(x)==4 and x[0].isupper()),None)
    scripts=Scripts();sample="".join(r["text"] for r in rows)[:8192]
    counts=Counter(scripts.char(c) for c in sample);strong={k:v for k,v in counts.items() if k not in ["Common","Inherited","Unknown"]}
    script=explicit or (SCRIPT_CODES.get(max(strong,key=strong.get)) if strong else None) or lang.maximize().script or "Zyyy"
    return safe(iso+"_"+script+("_"+bits[-1] if len(bits)>1 and not explicit and len(bits[-1])==2 else ""))

def prepare_madlad(*,bytes_per_language=65536,max_documents=32,workers=6):
    import requests
    discovery=ROOT/".cache/discovery";metadata=ROOT/'benchmarks/metadata'
    meta=read_json(discovery/"MADLAD-400.json" if (discovery/"MADLAD-400.json").exists() else metadata/'madlad-source.json');revision=meta["sha"]
    folders=read_json(discovery/"madlad-languages.json" if (discovery/"madlad-languages.json").exists() else metadata/'madlad-languages.json')
    codes=[x["path"].split("/")[-1] for x in folders if x["type"]=="directory" and x["path"].split("/")[-1] not in ["noise","unknown","zxx_xx_dtynoise"]]
    out=ROOT/"benchmarks/madlad-v0.2";out.mkdir(parents=True,exist_ok=True)
    def one(code):
        cache=out/f"{safe(code)}.json"
        if cache.exists():
            saved=read_json(cache)
            if saved.get("revision")==revision and saved.get("target_bytes")==bytes_per_language and saved.get("max_documents")==max_documents and saved.get("partition") and (ROOT/saved["partition"]["path"]).exists() and file_hash(ROOT/saved["partition"]["path"])==saved["partition"]["sha256"]:return saved
        result={"config":code,"revision":revision,"target_bytes":bytes_per_language,"max_documents":max_documents,"status":"pending"}
        try:
            session=requests.Session();url=f"https://huggingface.co/api/datasets/allenai/MADLAD-400/tree/{revision}/data/{code}"
            r=session.get(url,params={"limit":1000},timeout=60);r.raise_for_status()
            files=sorted((x for x in r.json() if x["type"]=="file" and "clean" in x["path"] and "noisy" not in x["path"] and x["path"].endswith(".jsonl.gz")),key=lambda x:x["path"])
            if not files:raise ValueError("No clean JSONL gzip source")
            entry=files[0];url=f"https://huggingface.co/datasets/allenai/MADLAD-400/resolve/{revision}/{entry['path']}"
            raw_rows=[];seen=set();size=0;excluded=Counter()
            with session.get(url,stream=True,timeout=(30,90)) as response:
                response.raise_for_status()
                with gzip.GzipFile(fileobj=response.raw) as stream:
                    for line_no in range(10000):
                        line=stream.readline(2_000_001)
                        if not line:break
                        if len(line)>2_000_000:raise ValueError("Source line exceeds bounded parser limit")
                        item=json.loads(line);text=item.get("text")
                        if not isinstance(text,str) or not text.strip():excluded["empty_or_nontext"]+=1;continue
                        raw=text.encode("utf-8",errors="strict");h=sha256(raw)
                        if len(raw)>100000:excluded["over_100000_bytes"]+=1;continue
                        if h in seen:excluded["duplicate_text"]+=1;continue
                        seen.add(h);raw_rows.append((text,item.get("url") or item.get("id") or h,line_no+1));size+=len(raw)
                        if size>=bytes_per_language or len(raw_rows)>=max_documents:break
            if not raw_rows:raise ValueError("No eligible documents")
            temp=[{"text":r[0]} for r in raw_rows];language=madlad_identity(code,temp)
            rows=[document(text,dataset="madlad400-sample-v0.2",language=language,index=f"{code}:{i}",source_id=str(source_id),domain="web text sample",source_config=code,source_line=line,source_shard=entry["path"],source_revision=revision) for i,(text,source_id,line) in enumerate(raw_rows)]
            # A separate config suffix prevents collisions between source varieties sharing ISO/script.
            language=language+"_madlad-"+code
            for row in rows:row["language"]=language
            partition=save_partition("madlad400-sample-v0.2",language,rows,{"repo":"allenai/MADLAD-400","revision":revision,"source_config":code,"source_path":entry["path"],"license":"ODC-BY-1.0","selection":"First unique whole documents in first pinned clean shard, bounded by bytes/documents; not uniform and not held out from earlier MADLAD selection studies","excluded":dict(excluded)})
            result.update(status="complete",partition=partition)
        except Exception as exc:result.update(status="unavailable",error=f"{type(exc).__name__}: {exc}"[:600])
        write_json(cache,result);print("MADLAD",code,result["status"],flush=True);return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:results=list(pool.map(one,codes))
    manifest={"schema_version":"2.0","dataset":"madlad400-sample-v0.2","repo":"allenai/MADLAD-400","revision":revision,"target_bytes":bytes_per_language,"max_documents":max_documents,"configs":results,"requested_configs":len(codes),"complete_configs":sum(r["status"]=="complete" for r in results)}
    write_json(out/"manifest.json",manifest);rebuild_catalog();return manifest

def rebuild_catalog():
    partitions=[]
    for path in sorted((ROOT/"benchmarks/partitions").rglob("*.meta.json")):
        info=read_json(path)
        if (ROOT/info["path"]).exists():partitions.append(info)
    catalog={"schema_version":"2.0","datasets":[{"id":k,"name":v} for k,v in DATASETS.items() if any(p["dataset"]==k for p in partitions)],"partitions":partitions}
    write_json(ROOT/"benchmarks/catalog.json",catalog)
    export_texts(catalog)
    return {"partitions":len(partitions),"documents":sum(p["documents"] for p in partitions),"language_varieties":len({p["language"] for p in partitions})}

def export_texts(catalog):
    index=[]
    for part in catalog["partitions"]:
        path=ROOT/f"site/data/text/{part['dataset']}/{part['language']}.json.gz"
        rows=list(read_jsonl(ROOT/part["path"]))
        write_json(path,{"schema_version":"2.0","documents":rows,"examples":[]},compact=True)
        index.append({**{k:part[k] for k in ["dataset","language","documents","utf8_bytes"]},"path":str(path.relative_to(ROOT/"site")),"sha256":file_hash(path),"bytes":path.stat().st_size})
    write_json(ROOT/"site/data/text-index.json",{"schema_version":"2.0","partitions":index})
    build_languages(catalog)

def build_languages(catalog):
    import langcodes,pycountry
    from langcodes.data_dicts import LIKELY_SUBTAGS
    by_id={};discovery=ROOT/"benchmarks/metadata"
    if not discovery.exists():discovery=ROOT/".cache/discovery"
    names={}
    readme=discovery/"flores-readme.md"
    if readme.exists():
        for line in readme.read_text().splitlines():
            match=re.fullmatch(r"([^|]+)\s*\|\s*([a-z]{3}_[A-Za-z]{4})\s*",line)
            if match:names[match[2]]=match[1].strip()
    region_map={};language_regions=defaultdict(set)
    if (discovery/"cldr-territory-containment.json").exists():
        containment=read_json(discovery/"cldr-territory-containment.json")["supplemental"]["territoryContainment"]
        def expand(code,continent):
            for child in containment.get(code,{}).get("_contains",[]):
                if len(child)==2:region_map[child]=continent
                else:expand(child,continent)
        for code,continent in [("002","Africa"),("150","Europe"),("142","Asia"),("009","Oceania"),("021","North America"),("013","North America"),("029","North America"),("005","South America")]:expand(code,continent)
        for region,info in read_json(discovery/"cldr-territory-info.json")["supplemental"]["territoryInfo"].items():
            for language in info.get("languagePopulation",{}):
                base=langcodes.Language.get(language.replace("_","-")).language
                if region in region_map:language_regions[base].add(region_map[region])
    glottolog=discovery/"glottolog-languages.csv";glotto_regions={}
    if glottolog.exists():
        with glottolog.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                regions={region_map[c] for c in re.split(r"[;, ]+",row.get("Countries", "")) if c in region_map}
                if row.get("ISO639P3code") and regions:glotto_regions[row["ISO639P3code"]]=regions
    autonyms={"zul":["isiZulu","Zulu"],"xho":["isiXhosa","Xhosa"],"sot":["Sesotho","Southern Sotho"],"tsn":["Setswana","Tswana"],"swh":["Swahili","Kiswahili"],"zho":["Chinese","Mandarin Chinese"],"cmn":["Mandarin","Chinese"],"ori":["Odia","Oriya"],"ory":["Odia","Oriya"]}
    for p in catalog["partitions"]:
        id=p["language"];bits=id.split("_");base=bits[0];script=bits[1] if len(bits)>1 else None
        try:
            lang=langcodes.Language.get(base);name=names.get(id) or lang.display_name("en")
            if name in ["Unknown language",base]:name=getattr(pycountry.languages.get(alpha_3=base),"name",base)
            continents=set(language_regions.get(lang.language,[]))|glotto_regions.get(base,set())
            if not continents and lang.language in LIKELY_SUBTAGS:
                region=langcodes.Language.get(LIKELY_SUBTAGS[lang.language]).territory
                if region in region_map:continents.add(region_map[region])
            aliases=autonyms.get(base,[])+[lang.display_name(lang.language),base,id]
        except Exception:name=names.get(id,id);aliases=[id];continents=set()
        if p["dataset"] in ["unicode-diagnostics-v0.1","numbers-v0.1","code-diagnostics-v0.1"]:name=id;continents=set()
        label=name+(" · "+script if script and "script" not in name.lower() else "")
        if "_madlad-" in id:label+=" · MADLAD "+id.split("_madlad-")[-1]
        entry=by_id.setdefault(id,{"id":id,"name":label,"aliases":sorted(set(aliases)),"continents":sorted(continents) or ["Unspecified"],"datasets":[],"documents":{},"script":script})
        if p["dataset"] not in entry["datasets"]:entry["datasets"].append(p["dataset"])
        entry["documents"][p["dataset"]]=p["documents"]
    result={"schema_version":"2.0","languages":sorted(by_id.values(),key=lambda x:x["name"].casefold()),"continent_note":"Overlapping broad geography from CLDR territory language records, Glottolog countries, and explicit likely-subtag territories. A language can belong to multiple groups; Unspecified means insufficient location metadata, not absence from a continent.","metadata_sources":["Unicode CLDR territoryInfo/territoryContainment","Glottolog country records","langcodes language-data","FLORES-200 official language names"],"metadata_file_hashes":{p.name:file_hash(p) for p in [discovery/"cldr-territory-info.json",discovery/"cldr-territory-containment.json",glottolog,readme] if p.exists()}}
    write_json(ROOT/"site/data/languages.json",result);write_json(ROOT/"benchmarks/languages.json",result)
    return result
