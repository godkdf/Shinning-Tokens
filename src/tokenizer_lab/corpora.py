"""Small frozen corpora with original UTF-8 bytes and inspectable selection."""
from __future__ import annotations
import io
import json
import random
import unicodedata as ud
from pathlib import Path
import urllib.parse
from .io import ROOT, sha256, file_hash, read_json, write_json, write_jsonl, read_jsonl

LANGUAGES = {
 "eng_Latn":"English","afr_Latn":"Afrikaans","zul_Latn":"isiZulu","xho_Latn":"isiXhosa",
 "sot_Latn":"Sesotho","tsn_Latn":"Setswana","swh_Latn":"Swahili","yor_Latn":"Yoruba","ibo_Latn":"Igbo",
 "amh_Ethi":"Amharic","arb_Arab":"Arabic","hin_Deva":"Hindi","tam_Taml":"Tamil","tel_Telu":"Telugu",
 "ben_Beng":"Bengali","cmn_Hani":"Mandarin","jpn_Jpan":"Japanese","tha_Thai":"Thai","rus_Cyrl":"Russian",
 "fin_Latn":"Finnish","tur_Latn":"Turkish","fra_Latn":"French","spa_Latn":"Spanish","por_Latn":"Portuguese"}

def document(text,*,dataset,language,index,source_id=None,domain="natural language",**metadata):
    raw=text.encode("utf-8",errors="strict")
    return {"id":f"{dataset}:{language}:{index}","text":text,"dataset":dataset,"language":language,
            "domain":domain,"source_id":source_id or f"{dataset}:{language}:{index}","utf8_bytes":len(raw),"sha256":sha256(raw),**metadata}

def build_diagnostics():
    seed=20261019;rng=random.Random(seed);rows=[]
    bases=["Hello, world!", "Nwaanyị ahụ zụtara ọka.", "Ẹ káàárọ̀, ayé.", "Sawubona mhlaba.","Habari za asubuhi.",
           "السَّلَامُ عَلَيْكُمْ", "नमस्ते दुनिया", "தமிழ் மொழி", "తెలుగు భాష", "বাংলা ভাষা", "你好，世界", "こんにちは世界", "สวัสดีชาวโลก",
           "እንደምን አለህ", "é à ö ǹ", "a\u0301 e\u0323\u0301", "👩🏽‍💻 🏳️‍🌈 👨‍👩‍👧‍👦", "a\u200db\u200cc", "▁literal Ġliteral", "<bos> <eos> <unk>",
           "=function =True =False =None", "a\x00b\x01c", "one\r\ntwo\nthree\tfour", "\u00a0\u2009\u202fspaces", "Arabic العربية English 中文"]
    transforms=[("raw",lambda x:x),("NFC",lambda x:ud.normalize("NFC",x)),("NFD",lambda x:ud.normalize("NFD",x)),
       ("space",lambda x:" "+x),("two_spaces",lambda x:"  "+x),("trailing",lambda x:x+" "),("newline",lambda x:x+"\n"),
       ("tab",lambda x:"\t"+x),("quote",lambda x:'"'+x+'"'),("brackets",lambda x:"["+x+"]"),
       ("repeat",lambda x:x+" "+x),("case",lambda x:x.upper()),("joiner",lambda x:"\u200d"+x),
       ("nonjoiner",lambda x:x+"\u200c"),("nbsp",lambda x:x.replace(" ","\u00a0")),("number",lambda x:x+" 0123456789"),
       ("mixed",lambda x:x+" বাংলা العربية"),("equals",lambda x:"="+x),("emoji",lambda x:x+"🧑🏿‍🚀"),("punctuation",lambda x:"("+x+")!")]
    for i,text in enumerate(bases):
        for kind,fn in transforms:
            rows.append(document(fn(text),dataset="unicode-diagnostics-v0.1",language="mixed",index=f"{i}-{kind}",source_id=f"unicode-family:{i}",domain="controlled Unicode diagnostic",transformation=kind))
    formats=[("integer",lambda n:str(n)),("signed",lambda n:"-"+str(n)),("zeros",lambda n:f"{n:012d}"),
             ("decimal",lambda n:f"{n//100}.{n%100:02d}"),("scientific",lambda n:f"{n}e-07"),
             ("hex",lambda n:hex(n)),("comma",lambda n:f"{n:,}"),("identifier",lambda n:f"item_{n}"),
             ("Arabic-Indic",lambda n:str(n).translate(str.maketrans("0123456789","٠١٢٣٤٥٦٧٨٩"))),
             ("expression",lambda n:f"({n}+3)/2 = {n/2+1.5}")]
    for j in range(1000):
        n=rng.randrange(10**(j%12),10**(j%12+1))
        for name,fn in formats:
            rows.append(document(fn(n),dataset="numbers-v0.1",language=name,index=j,source_id=f"numeric-family:{j}",domain="generated number diagnostic",digits=len(str(n))))
    code={"Python":["def f(x):\n    return x * 2\n", "class Example:\n    value = None\n", "import json\nprint(json.dumps({'a': 1}))\n"],
          "JavaScript":["export async function load(url) {\n  return await fetch(url);\n}\n", "const item = {id: 42, active: true};\n", "console.log(items.map(x => x.value));\n"],
          "C":["#include <stdio.h>\nint main(void) { printf(\"Hello\\n\"); return 0; }\n", "for (int i = 0; i < 10; ++i) { total += values[i]; }\n", "struct node { int value; struct node *next; };\n"],
          "SQL":["SELECT id, COUNT(*) FROM events GROUP BY id ORDER BY id;\n", "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);\n", "SELECT * FROM users WHERE score >= 3.14 AND active = TRUE;\n"],
          "JSON":[json.dumps({"id":42,"active":True,"name":"Ẹ káàárọ̀","items":[1,2,3]},ensure_ascii=False)],
          "HTML":["<div class=\"example\"><p>Hello &amp; welcome</p></div>\n"]}
    for language,examples in code.items():
        for i,text in enumerate(examples):
            for indent in ["", " ", "    ", "\t"]:
                rows.append(document(indent+text,dataset="code-diagnostics-v0.1",language=language,index=f"{i}-{len(rows)}",source_id=f"code-family:{language}:{i}",domain="handwritten code diagnostic"))
    path=ROOT/"benchmarks/diagnostics/corpus.jsonl.gz";write_jsonl(path,rows)
    manifest={"schema_version":"1.0","id":"diagnostics-v0.1","seed":seed,"documents":len(rows),"sha256":file_hash(path),
              "unicode_cases":500,"number_cases":10000,"code_cases":len(rows)-10500,"license":"Apache-2.0",
              "scope":"Constructed diagnostics, not representative language, code, or arithmetic-ability evaluation.","normalization":"Only explicitly labelled paired Unicode transformations; no implicit preprocessing."}
    write_json(path.parent/"manifest.json",manifest)
    return manifest

class HTTPRangeFile(io.RawIOBase):
    """Read only needed Parquet row groups; never silently fetch a whole huge shard."""
    def __init__(self,url,size,session,block_size=1<<20):
        self.url=url;self.size=size;self.session=session;self.pos=0;self.blocks={};self.block_size=block_size;self.downloaded=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError("Negative seek")
        return self.pos
    def read(self,size=-1):
        if size<0:size=self.size-self.pos
        size=min(size,self.size-self.pos);out=bytearray()
        while size>0:
            block=self.pos//self.block_size
            if block not in self.blocks:
                start=block*self.block_size;end=min(self.size,start+self.block_size)-1
                r=self.session.get(self.url,headers={"Range":f"bytes={start}-{end}"},timeout=(20,60),stream=True)
                r.raise_for_status()
                content=r.raw.read(end-start+2)
                if len(content)!=end-start+1:
                    r.close();raise IOError("Server did not honor bounded byte-range request")
                if r.status_code!=206 and not(start==0 and self.size<=self.block_size):
                    r.close();raise IOError("Server ignored Range; refusing full shard download")
                self.blocks[block]=content;self.downloaded+=len(content);r.close()
            data=self.blocks[block];offset=self.pos%self.block_size;n=min(size,len(data)-offset)
            if n<=0:break
            out.extend(data[offset:offset+n]);self.pos+=n;size-=n
        return bytes(out)

def build_fineweb(*,bytes_per_language=1_000_000,seed=20261019):
    import requests
    import pyarrow.parquet as pq
    session=requests.Session();session.headers["User-Agent"]="TokenizerLab/0.1 reproducible-research"
    out=ROOT/"benchmarks/fineweb-v0.1";out.mkdir(parents=True,exist_ok=True)
    previous=read_json(out/"manifest.json") if (out/"manifest.json").exists() else {}
    lock_path=out/"source-lock.json"
    if lock_path.exists():lock=read_json(lock_path)
    else:
        lock={}
        for repo in ["HuggingFaceFW/fineweb-2","HuggingFaceFW/fineweb"]:
            response=session.get("https://huggingface.co/api/datasets/"+repo,timeout=45);response.raise_for_status()
            lock[repo]=response.json()["sha"]
        write_json(lock_path,lock)
    manifest={"schema_version":"1.0","id":"fineweb-mixture-v0.1","source_revisions":lock,"seed":seed,
              "target_bytes_per_language":bytes_per_language,"normalization":"none","languages":[],
              "selection":"Seeded selection of one pinned test shard and row-group order, then first whole nonduplicate documents to byte target. English uses original FineWeb sample shard. This is bounded sampling, not a uniform sample of all web text.",
              "weighting":"Equal language summaries; raw pooled totals separately labelled. No combined tokenizer-quality score.",
              "attribution":[{"dataset":"FineWeb2","url":"https://huggingface.co/datasets/HuggingFaceFW/fineweb-2","license":"ODC-BY-1.0"},{"dataset":"FineWeb","url":"https://huggingface.co/datasets/HuggingFaceFW/fineweb","license":"ODC-BY-1.0"}]}
    all_rows=[]
    for language,name in LANGUAGES.items():
        saved=out/"samples"/(language+".jsonl.gz");info_file=saved.with_name(language+".manifest.json")
        if saved.exists() and info_file.exists():
            rows=list(read_jsonl(saved));info=read_json(info_file)
            if info.get("sample_sha256")==file_hash(saved) and previous.get("seed")==seed and previous.get("target_bytes_per_language")==bytes_per_language and info.get("revision") in lock.values():
                all_rows.extend(rows);manifest["languages"].append(info);print(language,"cached",len(rows),flush=True);continue
        info={"language":language,"name":name,"status":"pending","documents":0,"utf8_bytes":0}
        try:
            repo="HuggingFaceFW/fineweb" if language=="eng_Latn" else "HuggingFaceFW/fineweb-2";rev=lock[repo]
            sub="sample/10BT" if language=="eng_Latn" else f"data/{language}/test"
            api=f"https://huggingface.co/api/datasets/{repo}/tree/{rev}/{sub}"
            response=session.get(api,params={"limit":1000},timeout=45);response.raise_for_status()
            files=[x for x in response.json() if x.get("type")=="file" and x["path"].endswith(".parquet")]
            if not files:raise ValueError("No test Parquet shard at pinned source path")
            files.sort(key=lambda x:x["path"]);rng=random.Random(int(sha256(f"{seed}:{language}".encode())[:16],16));entry=rng.choice(files)
            url=f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{entry['path']}"
            remote=HTTPRangeFile(url,entry["size"],session);parquet=pq.ParquetFile(remote)
            order=list(range(parquet.num_row_groups));rng.shuffle(order);rows=[];seen=set();size=0
            for group in order:
                columns=[c for c in ["text","id","url","dump"] if c in parquet.schema.names]
                table=parquet.read_row_group(group,columns=columns).to_pylist()
                rng.shuffle(table)
                for source in table:
                    text=source.get("text")
                    if not isinstance(text,str) or not text.strip():continue
                    raw=text.encode("utf-8",errors="strict");h=sha256(raw)
                    if h in seen or len(raw)>100_000:continue
                    seen.add(h);rows.append(document(text,dataset="fineweb-mixture-v0.1",language=language,index=len(rows),source_id=source.get("url") or source.get("id") or h,source_document_id=source.get("id"),source_shard=entry["path"],source_row_group=group,source_revision=rev))
                    size+=len(raw)
                    if size>=bytes_per_language:break
                if size>=bytes_per_language:break
            if not rows:raise ValueError("No eligible UTF-8 documents")
            write_jsonl(saved,rows);info.update(status="complete" if size>=bytes_per_language else "short",documents=len(rows),utf8_bytes=size,repo=repo,revision=rev,source_path=entry["path"],source_bytes_downloaded=remote.downloaded,sample_sha256=file_hash(saved))
            write_json(info_file,info);all_rows.extend(rows)
        except Exception as exc:info.update(status="unavailable",reason=f"{type(exc).__name__}: {exc}"[:500])
        manifest["languages"].append(info);write_json(out/"manifest.json",manifest);print(language,info["status"],info.get("utf8_bytes"),flush=True)
    write_jsonl(out/"corpus.jsonl.gz",all_rows);manifest.update(documents=len(all_rows),actual_utf8_bytes=sum(x["utf8_bytes"] for x in all_rows),corpus_sha256=file_hash(out/"corpus.jsonl.gz"),covered_languages=sum(x["status"] in ["complete","short"] for x in manifest["languages"]))
    write_json(out/"manifest.json",manifest)
    return manifest
