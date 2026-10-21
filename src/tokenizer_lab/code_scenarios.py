"""Real Git edits, frozen repository contexts and labelled synthetic stresses."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from threading import Condition
import time

from . import codebench
from .io import ROOT, file_hash, read_json, read_jsonl, sha256, write_json, write_jsonl

BASE = ROOT / "benchmarks/code-realistic-v0.1"
CORPUS = ROOT / "benchmarks/code-scenarios-v0.1"
RESULTS = ROOT / "results/code-scenarios-v0.1"
CACHE = ROOT / ".cache/code-scenarios-v0.1"


def git(folder, *args, timeout=120):
    return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null", "-c", "core.quotePath=false", "-C", str(folder), *args],
        stderr=subprocess.PIPE, timeout=timeout, env=os.environ | {"GIT_TERMINAL_PROMPT": "0", "GIT_LITERAL_PATHSPECS": "1"})


def edits_for_repo(spec, language):
    """Read real modified-file pairs. No checkout, hooks, tests or source execution."""
    key = spec["repo"].replace("/", "--")
    output = CACHE / "edits" / f"{key}.json.gz"
    if output.exists():
        return read_json(output)
    folder = CACHE / "git" / (key+".git")
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        git(folder, "init", "--bare", "--quiet")
        git(folder, "remote", "add", "origin", spec["url"]+".git")
        git(folder, "config", "remote.origin.promisor", "true")
        git(folder, "config", "remote.origin.partialclonefilter", "blob:none")
        git(folder, "fetch", "--quiet", "--no-tags", "--filter=blob:none", "--depth=256", "origin", spec["commit"], timeout=180)
        commits = git(folder, "rev-list", "--parents", "--max-count=200", "FETCH_HEAD").decode().splitlines()
        candidates = []
        seen_paths = Counter()
        for line in commits:
            parts = line.split()
            if len(parts) < 2:
                continue
            commit, parent = parts[:2]
            try:
                diff = git(folder, "diff-tree", "--no-commit-id", "--raw", "-r", "-z", "--diff-filter=M", parent, commit)
            except subprocess.CalledProcessError:
                continue
            fields = diff.split(b"\x00")
            count = 0
            for index in range(0, len(fields)-1, 2):
                header, raw_path = fields[index:index+2]
                try:
                    path = raw_path.decode("utf-8", "strict")
                except UnicodeDecodeError:
                    continue
                if Path(path).suffix not in language["extensions"] or any(p in codebench.EXCLUDED or p.startswith('.') for p in Path(path).parts):
                    continue
                values = header.decode("ascii").split()
                if len(values) != 5 or values[-1] != "M" or seen_paths[path] >= 3:
                    continue
                candidates.append({"path": path, "before_blob": values[2], "after_blob": values[3], "before_commit": parent, "after_commit": commit})
                seen_paths[path] += 1
                count += 1
                if count >= 5:
                    break
            if len(candidates) >= 60:
                break
        if candidates:
            objects = sorted({r[k] for r in candidates for k in ["before_blob", "after_blob"]})
            # Fetch requested blobs in one protocol exchange, not one HTTP call
            # per source file. All objects remain in this disposable bare repo.
            git(folder, "fetch", "--quiet", "--no-tags", "origin", *objects, timeout=180)
            if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) > 192 << 20:
                raise ValueError("History working set exceeds 192 MiB cap")
            for item in candidates:
                sizes = [int(git(folder, "cat-file", "-s", item[k])) for k in ["before_blob", "after_blob"]]
                if any(n < 80 or n > 65536 for n in sizes):
                    continue
                try:
                    before = git(folder, "cat-file", "blob", item["before_blob"]).decode("utf-8", "strict")
                    after = git(folder, "cat-file", "blob", item["after_blob"]).decode("utf-8", "strict")
                except UnicodeDecodeError:
                    continue
                if before == after or '\x00' in before or '\x00' in after:
                    continue
                if re.search(r"(?i)(auto.?generated|DO NOT EDIT|@generated)", before[:2000]+after[:2000]):
                    continue
                if language["id"] == "bash" and not (item["path"].endswith('.bash') or re.search(r"^#![^\n]*\bbash\b", before)):
                    continue
                patch = git(folder, "diff", "--no-ext-diff", "--no-textconv", "--unified=3", item["before_commit"], item["after_commit"], "--", item["path"]).decode("utf-8", "strict")
                if not patch or len(patch.encode()) > 131072:
                    continue
                records.append(item | {"repository": spec["repo"], "split": spec["split"], "license": spec["license"], "language": language["id"],
                    "before": before, "after": after, "diff": patch, "change_type": "modified existing file",
                    "id": language["id"]+":"+codebench.stable(spec["repo"]+item["after_commit"]+item["path"])[:24]})
                if len(records) >= 25:
                    break
        write_json(output, records, compact=True)
        return records
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def row(text, *, identifier, language, split, dataset, source_id, **extra):
    raw = text.encode("utf-8", "strict")
    return {"id": identifier, "text": text, "utf8_bytes": len(raw), "sha256": sha256(raw), "language": language,
        "split": split, "dataset": dataset, "source_id": source_id, "repository": source_id,
        "domain": dataset, "size_band": codebench.band(len(raw)), **extra}


def make_contexts(docs, language):
    groups = defaultdict(list)
    for doc in docs:
        if doc['utf8_bytes'] <= 8192:
            groups[(doc['split'], doc['repository'])].append(doc)
    result = []
    rng = random.Random(20261021)
    for split, target in [('development',10),('evaluation',40)]:
        families = sorted(k for k,v in groups.items() if k[0] == split and len(v) >= 2)
        if not families:
            raise ValueError('No multi-file context families: '+language+':'+split)
        for i in range(target):
            key = families[i % len(families)]
            selected = rng.sample(groups[key], min(4,len(groups[key])))
            text = 'Repository context\n' + ''.join(f"\n=== FILE: {d['path']} ===\n{d['text']}\n=== END FILE ===\n" for d in selected)
            result.append(row(text, identifier=f'{language}:context:{split}:{i}', language=language, split=split,
                dataset='repository-context-v0.1', source_id=key[1], source_files=[{k:d[k] for k in ['path','commit','sha256','source_url']} for d in selected],
                construction='Frozen serialization of real source files; not captured user traffic or fabricated tool output'))
    return result


def make_stresses(docs, language):
    import unicodedata
    rng = random.Random(20261021)
    selected = []
    for split,n in [('development',1),('evaluation',4)]:
        eligible=[d for d in docs if d['split']==split and d['utf8_bytes'] <= 2048]
        selected.extend(rng.sample(eligible,n))
    transforms = [
        ('original',lambda t:t),('CRLF',lambda t:t.replace('\r\n','\n').replace('\n','\r\n')),
        ('tab_indent',lambda t:'\t'+t.replace('\n','\n\t')),
        ('NFC',lambda t:unicodedata.normalize('NFC',t)),('NFD',lambda t:unicodedata.normalize('NFD',t)),
        ('mixed_unicode',lambda t:t+'\nα变量_é_e\u0301_👩🏽‍💻\n'),
        ('special_literal',lambda t:t+'\n<bos> <eos> <unk> <|endoftext|> [MASK]\n'),
        ('NUL_control',lambda t:t+'\x00\x01\u200d\u200c\u00a0\n'),
        ('long_line',lambda t:t+'\n'+'identifier_0123456789'*800+'\n'),
        ('whitespace',lambda t:t+'\n'+' '*8192+'\t'*256+'\n')]
    return [row(fn(d['text']),identifier=f'{language}:stress:{i}:{name}',language=language,split=d['split'],
        dataset='code-stress-v0.1',source_id=d['repository'],transformation=name,base_source_sha256=d['sha256'],
        controlled=True,syntactic_validity='Not asserted; controlled tokenizer diagnostic')
        for i,d in enumerate(selected) for name,fn in transforms]


def build():
    cfg=read_json(codebench.CONFIG)
    base=read_json(BASE/'manifest.json')
    CORPUS.mkdir(parents=True,exist_ok=True)
    all_partitions=[]
    statistics={}
    pair_index=[]
    failures=[]
    for language in cfg['languages']:
        lid=language['id']
        part=next(p for p in base['partitions'] if p['language']==lid)
        docs=list(read_jsonl(ROOT/part['path']))
        pairs=[]
        seen=set()
        for split,target in [('development',20),('evaluation',80)]:
            specs=[s for s in base['repositories'] if s['language']==lid and s['split']==split]
            chosen=[]
            for offset in range(0,len(specs),4):
                if len(chosen)>=target:break
                outcomes={}
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures={pool.submit(edits_for_repo,s,language):s for s in specs[offset:offset+4]}
                    for future in as_completed(futures):
                        spec=futures[future]
                        try:outcomes[spec['repo']]=future.result()
                        except Exception as exc:
                            failures.append({'language':lid,'repository':spec['repo'],'error':str(exc)[:300]})
                            codebench.event('edit_repository_failed',language=lid,repo=spec['repo'],error=str(exc)[:160])
                for spec in specs[offset:offset+4]:
                    for value in outcomes.get(spec['repo'],[]):
                        signature=sha256((value['before']+'\x00'+value['after']).encode())
                        if signature in seen:continue
                        seen.add(signature);chosen.append(value)
                        if len(chosen)>=target:break
                    if len(chosen)>=target:break
                codebench.event('edit_progress',language=lid,split=split,pairs=len(chosen),target=target)
            pairs.extend(chosen)
        write_json(CACHE/'selected'/f'{lid}.json.gz',pairs,compact=True)
        pair_index.extend({k:v for k,v in p.items() if k not in ['before','after','diff']} |
            {name+'_sha256':sha256(p[name].encode()) for name in ['before','after','diff']} for p in pairs)
        variants={}
        for variant in ['before','after','diff']:
            variants['edit-'+variant]=[row(p[variant],identifier=p['id']+':'+variant,language=lid,split=p['split'],
                dataset='real-edit-'+variant+'-v0.1',source_id=p['repository'],pair_id=p['id'],path=p['path'],
                before_commit=p['before_commit'],after_commit=p['after_commit'],license=p['license']) for p in pairs]
        variants['context']=make_contexts(docs,lid)
        variants['stress']=make_stresses(docs,lid)
        for variant,rows in variants.items():
            dest=CORPUS/f'{lid}__{variant}.jsonl.gz';write_jsonl(dest,rows)
            all_partitions.append({'language':lid+'__'+variant,'programming_language':lid,'variant':variant,
                'path':str(dest.relative_to(ROOT)),'sha256':file_hash(dest),'documents':len(rows),'utf8_bytes':sum(r['utf8_bytes'] for r in rows)})
        statistics[lid]={'edit_pairs':len(pairs),'edit_splits':dict(Counter(p['split'] for p in pairs)),'contexts':50,'stresses':50}
        write_json(CACHE/'build-progress.json',statistics)
    write_jsonl(CORPUS/'edit-index.jsonl.gz',pair_index)
    manifest={'protocol':'code-scenarios-v0.1','created_at':codebench.now(),'base_manifest_sha256':file_hash(BASE/'manifest.json'),
        'edit_index_sha256':file_hash(CORPUS/'edit-index.jsonl.gz'),
        'partitions':all_partitions,'languages':statistics,'edit_pairs':sum(s['edit_pairs'] for s in statistics.values()),
        'documents':sum(p['documents'] for p in all_partitions),'quota_complete':all(s['edit_pairs']==100 for s in statistics.values()),
        'collection_failures':failures,'builder_sha256':file_hash(Path(__file__)),
        'scope':'Real modified-file before/after/diff inputs, constructed contexts from real files, and separately labelled controlled stresses.',
        'limitations':['Recent Git history is a bounded sample, not production traffic.','Historical file-specific license exceptions are not independently audited.',
            'Context requests are constructed repository contexts, not captured conversations or tool outputs.','Stress transformations are correlated controlled cases.']}
    write_json(CORPUS/'manifest.json',manifest)
    codebench.event('scenarios_written',pairs=manifest['edit_pairs'],inputs=manifest['documents'],quota_complete=manifest['quota_complete'])


def configure():
    codebench.CORPUS=CORPUS
    codebench.RESULTS=RESULTS
    original=codebench.fingerprint
    codebench.fingerprint=lambda:original() | {'scenario_code_sha256':file_hash(Path(__file__))}


def audit():
    m=read_json(CORPUS/'manifest.json');ids=set();total=0
    index=CORPUS/'edit-index.jsonl.gz'
    if m.get('edit_index_sha256') and file_hash(index)!=m['edit_index_sha256']:raise ValueError('Edit index changed')
    pairs={r['id']:r for r in read_jsonl(index)}
    for part in m['partitions']:
        p=ROOT/part['path']
        if file_hash(p)!=part['sha256']:raise ValueError('Scenario partition changed')
        count=0
        for r in read_jsonl(p):
            if r['id'] in ids or sha256(r['text'].encode())!=r['sha256']:raise ValueError('Scenario integrity failed')
            if r.get('pair_id'):
                pair=pairs[r['pair_id']]
                variant=r['dataset'].removeprefix('real-edit-').removesuffix('-v0.1')
                raw=r['text'].encode()
                if sha256(raw)!=pair[variant+'_sha256']:raise ValueError('Edit text differs from frozen index')
                if variant in ['before','after']:
                    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\x00'+raw).hexdigest()
                    if blob!=pair[variant+'_blob']:raise ValueError('Edit provenance Git blob mismatch')
                if r['split']!=pair['split'] or r['source_id']!=pair['repository']:raise ValueError('Edit family/split mismatch')
            ids.add(r['id']);count+=1
        if count!=part['documents']:raise ValueError('Scenario count mismatch')
        total+=count
    if total!=m['documents'] or not m['quota_complete']:raise ValueError('Scenario quotas incomplete')
    write_json(CORPUS/'audit.json',{'at':codebench.now(),'manifest_sha256':file_hash(CORPUS/'manifest.json'),'inputs':total,
        'real_edit_pairs':len(pairs),'git_blob_hashes_verified':2*len(pairs),'integrity_pass':True})
    return total


def run():
    from .adapters import registry
    audit();configure();RESULTS.mkdir(parents=True,exist_ok=True)
    ids=[s['id'] for s in registry()];random.Random(20261021).shuffle(ids);states={}
    digest=file_hash(CORPUS/'manifest.json');fp=codebench.fingerprint()
    gate=Condition();active=[0]
    def capacity():
        main_status=ROOT/'results/code-realistic-v0.1/run-status.json'
        if main_status.exists():
            state=read_json(main_status).get('profiles',{})
            if len(state)>=len(ids) and all(s['status']=='completed' for s in state.values()):return 4
        return 2
    def evaluate(identifier):
        path=RESULTS/f'{identifier}.json'
        if path.exists():
            old=read_json(path)
            if old.get('status')=='completed' and old.get('manifest_sha256')==digest and old.get('run_fingerprint')==fp:return {'status':'completed','resumed':True}
        codebench.event('scenario_tokenizer_start',tokenizer=identifier)
        log=RESULTS/'checkpoints'/f'{identifier}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with gate:
            while active[0]>=capacity():gate.wait(timeout=5)
            active[0]+=1
        start=time.monotonic()
        try:
            with log.open('a') as stream:
                proc=subprocess.run([sys.executable,'-m','tokenizer_lab.code_scenarios','worker','--tokenizer',identifier],cwd=ROOT,
                    stdout=stream,stderr=subprocess.STDOUT,env=os.environ|{'TOKENIZER_LAB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'},timeout=7200)
            return {'status':'completed' if proc.returncode==0 else 'failed','exit_code':proc.returncode,'seconds':round(time.monotonic()-start,2)}
        finally:
            with gate:active[0]-=1;gate.notify_all()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(evaluate,i):i for i in ids}
        for future in as_completed(futures):
            identifier=futures[future]
            try:states[identifier]=future.result()
            except Exception as exc:states[identifier]={'status':'failed','error':str(exc)}
            write_json(RESULTS/'run-status.json',{'manifest_sha256':digest,'profiles':states,'at':codebench.now()})
            codebench.event('scenario_tokenizer_finished',tokenizer=identifier,**states[identifier])
    report()


def report():
    import csv
    from .adapters import registry
    m=read_json(CORPUS/'manifest.json');digest=file_hash(CORPUS/'manifest.json');rows=[]
    for s in registry():
        path=RESULTS/f"{s['id']}.json"
        if not path.exists():continue
        v=read_json(path)
        if v.get('status')!='completed' or v.get('manifest_sha256')!=digest:continue
        rows.extend(r for r in v['benchmarks'] if r['size_band']=='all')
    columns=['tokenizer','dataset','language','split','documents','utf8_bytes','tokens','bytes_per_token','exact','canonical_only','mismatches','errors']
    with (RESULTS/'by-scenario.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows({k:r.get(k) for k in columns} for r in rows)
    summary={'inputs_per_profile':m['documents'],'completed_profiles':len({r['tokenizer'] for r in rows}),'observations':sum(r['documents'] for r in rows),
        'errors':sum(r['errors'] for r in rows),'manifest_sha256':digest,'at':codebench.now()}
    write_json(RESULTS/'summary.json',summary);codebench.event('scenario_report',**summary)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['build','audit','run','worker','report']);p.add_argument('--tokenizer');a=p.parse_args()
    if a.command=='worker':
        configure();codebench.worker(a.tokenizer)
    else:{'build':build,'audit':audit,'run':run,'report':report}[a.command]()


if __name__=='__main__':main()
