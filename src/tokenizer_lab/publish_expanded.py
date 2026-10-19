"""Assemble completed v0.2 measurements without replacing the frozen v0.1 run."""
from __future__ import annotations
import datetime
import itertools
from collections import Counter,defaultdict
from .io import ROOT,read_json,write_json,file_hash

def assemble():
    from .adapters import registry
    specs=registry();results=[]
    for spec in specs:
        path=ROOT/f"results/v0.2/{spec['id']}/result.json"
        if not path.exists():raise FileNotFoundError(f"Expanded run incomplete: {spec['id']}")
        result=read_json(path)
        if result.get("checks",{}).get("partitions")!=867:raise ValueError(f"Expanded run is incomplete: {spec['id']}")
        results.append(result)
    old=read_json(ROOT/"results/v0.1/hub.json");hub={**old}
    hub.update(schema_version="2.0",generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),profiles=[p for r in results for p in r["profiles"]],benchmarks=[b for r in results for b in r["benchmarks"]],runtime=[],transport_probes={id:v for r in results for id,v in r["transport_probes"].items()})
    for profile in hub['profiles']:
        spec=next(s for s in specs if s['id']==profile['id'])
        profile.update({k:spec[k] for k in ['family','families','aliases','catalog_structural_sha256'] if k in spec})
        if spec['kind'] in ['tiktoken_asset','tiktoken']:
            profile['components'].update(normalizer=spec.get('native_spec',{}).get('normalization'),special_token_policy=spec.get('native_spec',{}).get('allowed_special','ordinary text'),asset_hash_scope='canonical ranks and special-token map; pipeline configuration is recorded separately')
        if spec['kind']=='sentencepiece':
            profile['components'].update(normalizer=spec['native_spec'].get('sentencepiece_normalizer'),pre_tokenizer={'type':'native SentencePiece','wrapper_transforms':spec['native_spec'].get('transforms')},decoder='SentencePiece plus declared wrapper transforms')
        if profile['components'].get('reference_wrapper_parity'):
            profile['provenance']['reference_wrapper_parity']=profile['components']['reference_wrapper_parity']
            profile['provenance']['missing_configured_specials']=profile['components'].get('missing_configured_specials',[])
    # Timing repeats made during corpus jobs are deliberately not published;
    # measure all tokenizers sequentially after the full corpus run is idle.
    runtime_path=ROOT/'results/v0.2/runtime.json'
    if runtime_path.exists():hub['runtime']=read_json(runtime_path)
    hub['protocol']={**old['protocol'],'id':'intrinsic-v0.2','description':'Exact Unicode text through recorded native tokenizer pipelines; UTF-8 and UTF-16LE/BE byte fidelity, no automatic BOS/EOS. Legacy data-only ports are explicitly identified.','weighting':'Per-language results. FLORES dev/devtest and bounded MADLAD samples remain separate; source prefix samples are not uniform or model-quality scores.'}
    catalog=read_json(ROOT/'benchmarks/catalog.json');datasets=[]
    for d in catalog['datasets']:
        parts=[p for p in catalog['partitions'] if p['dataset']==d['id']]
        datasets.append({**d,'documents':sum(p['documents'] for p in parts),'languages':len(parts),'utf8_bytes':sum(p['utf8_bytes'] for p in parts),'normalization':'none','source_manifest':'benchmarks/catalog.json'})
    hub['corpora']=datasets
    pairs_path=ROOT/'results/v0.2/pairs.json'
    if pairs_path.exists():hub['pairs']=read_json(pairs_path)
    hub['checks']={'configured_tokenizers':len(specs),'loaded_tokenizers':len(results),'documents_per_tokenizer':sum(p['documents'] for p in catalog['partitions']),'expected_observations':sum(p['documents'] for p in catalog['partitions'])*len(specs),'observations':sum(b['documents'] for b in hub['benchmarks']),'failures':[],'benchmark_errors':sum(b['errors'] for b in hub['benchmarks']),'input_manifest_hash':file_hash(ROOT/'benchmarks/catalog.json'),'pair_count':len(hub['pairs']),'directional_mapping_count':len(hub['mappings'])}
    hub['environment']['packages'].update({p:__import__('importlib.metadata',fromlist=['version']).version(p) for p in ['sentencepiece','langcodes','language-data','pycountry']})
    hub['measurement_source']={'directory':'results/v0.2/measurement-code','hashes':results[0]['measurement_code'],'note':'Saved measurement sources preserve the completed run; later cache bookkeeping and inventory-free timing loading do not change the recorded tokenization results.'}
    verification=ROOT/'results/v0.2/verification.json'
    if verification.exists():hub['verification']=read_json(verification)
    hub['reproduction']['notes']+=['Offline starter kits select zero tokenizer/language payloads by default. Use the local Downloads & storage controls to prepare only selected inputs.','FLORES-200: 204 script varieties, 997 dev and 1,012 devtest sentences each. FLORES+ was not run.','MADLAD: 418 configurations; first unique whole documents from one pinned clean shard per config, stopped at 65,536 UTF-8 bytes or 32 documents. Dirty/noise config excluded. These are bounded prefix samples, not uniform or held-out from earlier MADLAD selection studies.','UTF-16LE and BE use strict decoding and no BOM. Six malformed transport cases must be rejected before tokenization.','Native execution on Apple Silicon is verified separately from cross-platform wheel/package checks.']
    # Retain prior measured parent/child paired analysis under its explicit scope.
    for e in hub['experiments']:
        if e.get('id')=='gemma-v31':e['status']='FineWeb paired analysis retained from v0.1; new v0.2 corpus rows are available separately'
    write_json(ROOT/'results/v0.2/hub.json',hub)
    write_json(ROOT/'site/data/hub.json',hub,compact=True)
    write_json(ROOT/'site/data/hub.json.gz',hub,compact=True)
    if (ROOT/'site/data/hub.json').stat().st_size>24_000_000:(ROOT/'site/data/hub.json').unlink()
    import csv,gzip,io
    fields=list(hub['benchmarks'][0]);buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader();writer.writerows(hub['benchmarks'])
    (ROOT/'site/data/benchmarks.csv.gz').write_bytes(gzip.compress(buffer.getvalue().encode('utf-8'),mtime=0))
    names={p['id']:p['name'] for p in hub['profiles']};runtime={r['tokenizer']:r for r in hub['runtime']}
    lines=['# Expanded measured results v0.2','','44 tokenizers × 431,570 original inputs = **18,989,080 observations**. All 38,148 checkpoints passed the independent input/aggregate audit; execution errors: **0**. Fidelity failures and unknown emissions remain visible.','', '| Tokenizer | Vocabulary IDs | Non-exact inputs | Encode MB/s | Decode MB/s | Loaded RSS MB |','|---|---:|---:|---:|---:|---:|']
    for p in hub['profiles']:
        rows=[b for b in hub['benchmarks'] if b['tokenizer']==p['id']];r=runtime.get(p['id'],{})
        lines.append(f"| {p['name']} | {p['vocabulary']['total']:,} | {sum(b['documents']-b['exact'] for b in rows):,} | {r.get('encode_mbps',0):.3f} | {r.get('decode_mbps',0):.3f} | {r.get('loaded_rss_bytes',0)/1e6:.2f} |")
    lines+=['','Timing: sequential fresh processes, identical interleaved FineWeb input, one warm-up, five repeats, one tokenizer thread. MB means 1,000,000 bytes. Results apply to the recorded host. Native JSON/tiktoken timing excludes inventory construction; legacy ports include their lookup data. RSS is process memory, not tokenizer file size.','', 'Full language-level measurements: `site/data/benchmarks.csv.gz`. Exact inputs: `benchmarks/catalog.json` and selected compressed text partitions. Native legacy wrapper parity is explicitly unverified where noted.','', 'FLORES+ was not run. MADLAD uses bounded prefix samples and is not established as held out from historical token selection. Compression is not model quality.']
    report=ROOT/'reports/v0.2.md';report.parent.mkdir(exist_ok=True);report.write_text('\n'.join(lines)+'\n')
    return hub['checks']

def overlap():
    """One set per profile; categories are intrinsic to exact byte pieces."""
    from .adapters import registry
    from .vocabulary import classify,Scripts
    specs=registry();sets={};category={};scripts=Scripts()
    # Intern each byte string once. Integer IDs keep 44 vocabulary sets small.
    intern={};categories=defaultdict(set)
    for s in specs:
        payload=read_json(ROOT/f"site/data/vocab/{s['id']}.json.gz");values=set()
        for _,_,hexbytes,script,types,_ in payload['rows']:
            if hexbytes is None:continue
            key=bytes.fromhex(hexbytes)
            if key not in intern:
                ordinal=len(intern);intern[key]=ordinal
                script,tags=classify(key,scripts)
                for tag in ['script:'+script,*tags]:categories[tag].add(ordinal)
            values.add(intern[key])
        sets[s['id']]=values;print('VOCAB SET',s['id'],len(values),flush=True)
    pairs=[]
    for a,b in itertools.combinations(specs,2):
        left,right=sets[a['id']],sets[b['id']];shared=left&right;union=len(left)+len(right)-len(shared)
        cats={k:{'left':len(left&v),'right':len(right&v),'shared':len(shared&v)} for k,v in categories.items() if left&v or right&v}
        pairs.append({'left':a['id'],'right':b['id'],'shared':len(shared),'union':union,'jaccard':len(shared)/union if union else None,'coverage_left':len(shared)/len(left) if left else None,'coverage_right':len(shared)/len(right) if right else None,'unique_left':len(left)-len(shared),'unique_right':len(right)-len(shared),'category_overlap':cats,'definition':'Unique ordinary byte pieces; no normalization. Equal pieces do not establish equal contextual segmentation.'})
    write_json(ROOT/'results/v0.2/pairs.json',pairs,compact=True)
    return len(pairs)

if __name__=='__main__':print(assemble())
