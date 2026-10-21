"""Assemble the Kimi extension without overwriting frozen v0.2/v0.3 runs."""
from __future__ import annotations
import copy
import csv
import datetime
import gzip
import io
import itertools
from .io import ROOT,read_json,write_json,file_hash

BASE='v0.3'
VERSION='v0.4'

def new_ids():return read_json(ROOT/'configs/kimi-selection.json')['tokenizers']

def extend_pairs():
    from .adapters import registry
    from .vocabulary import compare_vocabularies
    base=read_json(ROOT/f'results/{BASE}/hub.json');ids=new_ids();pairs=list(base['pairs'])
    # All eight assets have identical ordinary ranks. Special IDs are excluded
    # from overlap, so calculate their shared byte-piece comparisons once.
    specs=[s for s in registry() if s['id'] in ids]
    assert len({s['sha256'] for s in specs})==1
    added=read_json(ROOT/f'site/data/vocab/{ids[0]}.json.gz')
    for profile in base['profiles']:
        old=read_json(ROOT/f"site/data/vocab/{profile['id']}.json.gz")
        comparison=compare_vocabularies(old,added)
        for id in ids:pairs.append(comparison|{'right':id})
    comparison=compare_vocabularies(added,added)
    for a,b in itertools.combinations(ids,2):pairs.append(comparison|{'left':a,'right':b})
    n=len(base['profiles'])+len(ids)
    assert len(pairs)==n*(n-1)//2
    assert len({frozenset((p['left'],p['right'])) for p in pairs})==len(pairs)
    write_json(ROOT/f'results/{VERSION}/pairs.json',pairs,compact=True)
    return pairs

def assemble():
    from .adapters import registry
    base=read_json(ROOT/f'results/{BASE}/hub.json');catalog=read_json(ROOT/'benchmarks/catalog.json')
    assert base['checks']['input_manifest_hash']==file_hash(ROOT/'benchmarks/catalog.json')
    ids=new_ids();parts={(p['dataset'],p['language']) for p in catalog['partitions']}
    results={id:read_json(ROOT/f'results/{VERSION}/{id}/result.json') for id in ids}
    for id,r in results.items():
        assert r['selection']['tokenizers']==[id]
        assert {(b['dataset'],b['language']) for b in r['benchmarks']}==parts
        assert len(r['checkpoints'])==len(parts) and r['network_downloads']==0
        assert len(r['profiles'])==len(r['runtime'])==1
        assert r['runtime'][0]['input_sha256']==base['runtime'][0]['input_sha256']
    hub=copy.deepcopy(base)
    hub['generated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    hub['profiles'] += [p for r in results.values() for p in r['profiles']]
    hub['benchmarks'] += [b for r in results.values() for b in r['benchmarks']]
    hub['runtime'] += [dict(t,measured_at=r['generated_at'],schedule='Kimi extension; sequential fresh subprocess, same frozen runtime input. Historical profiles were not re-timed.') for r in results.values() for t in r['runtime']]
    hub['transport_probes'].update({id:r['transport_probes'][id] for id,r in results.items()})
    # This alias changes labels and provenance only, never the frozen measurements.
    spec=next(s for s in registry() if s['id']=='qwen25')
    updates={k:spec[k] for k in ['name','families','aliases']}
    next(p for p in hub['profiles'] if p['id']=='qwen25').update(updates)
    hub['profile_metadata_updates']={'qwen25':updates}
    hub['pairs']=read_json(ROOT/f'results/{VERSION}/pairs.json')
    hub['result_paths'].update({id:f'results/{VERSION}/{id}/result.json' for id in ids})
    hub['measurement_source']={'note':'Frozen v0.2/v0.3 results are retained. New Kimi measurements and saved measurement sources are in v0.4. Use result_paths and measurement_sources for per-profile identities.'}
    hub['measurement_sources'].update({id:{'directory':f'results/{VERSION}/measurement-code','hashes':r['measurement_code']} for id,r in results.items()})
    count=len(hub['profiles']);docs=sum(p['documents'] for p in catalog['partitions'])
    hub['checks'].update(configured_tokenizers=count,loaded_tokenizers=count,
        expected_observations=docs*count,observations=sum(b['documents'] for b in hub['benchmarks']),
        benchmark_errors=sum(b['errors'] for b in hub['benchmarks']),pair_count=len(hub['pairs']))
    assert hub['checks']['observations']==hub['checks']['expected_observations']
    assert {p['id'] for p in hub['profiles']}=={s['id'] for s in registry()}
    verification=ROOT/f'results/{VERSION}/verification.json'
    hub['verification']=read_json(verification) if verification.exists() else {'status':'pending','scope':'Kimi extension audit'}
    hub['reproduction']['notes'] += [
        'v0.4 adds eight Kimi/Moonlight tokenizer profiles covering 15 public checkpoints. Kimi Dev is an exact tokenizer.json alias of the existing Qwen 2.5 profile; its measurements are reused and its label/aliases updated.',
        'All eight new profiles share 163,584 ordinary ranks but retain different special-ID maps, direct encode policies or reserved ranges. Four wrappers expose 258 special IDs, two more than the model config declares; tokenizer IDs are not an embedding compatibility guarantee.',
        'Kimi uses inspected native tiktoken text paths; full Transformers wrapper parity is not independently established. Automatic BOS/EOS and chat formatting are disabled. Seven direct encode paths recognize special spellings; K2 Base encodes them as ordinary text.',
        'Audio/image pipelines and API-only Kimi models without an inspected public tokenizer are outside this text benchmark. Historical profile runtime measurements were not repeated.' ]
    write_json(ROOT/f'results/{VERSION}/hub.json',hub)
    return hub

def verify_extension():
    from .verify_expanded import verify
    ids=new_ids();new=verify(VERSION,ids)
    base_path=ROOT/f'results/{BASE}/verification.json';base=read_json(base_path)
    old=read_json(ROOT/f'results/{BASE}/hub.json');hub=read_json(ROOT/f'results/{VERSION}/hub.json')
    assert base['status']=='passed' and base['catalog_sha256']==new['catalog_sha256']
    assert hub['benchmarks'][:len(old['benchmarks'])]==old['benchmarks']
    for before,after in zip(old['profiles'],hub['profiles']):
        update=hub['profile_metadata_updates'].get(before['id'],{})
        assert before|update==after
    for id,r in [(id,read_json(ROOT/f'results/{VERSION}/{id}/result.json')) for id in ids]:
        for name,digest in r['measurement_code'].items():
            assert file_hash(ROOT/f'results/{VERSION}/measurement-code'/name)==digest
    combined={**new,'tokenizers':base['tokenizers']+new['tokenizers'],
        **{k:base[k]+new[k] for k in ['checkpoints','observations','errors','malformed_transport_rejections']},
        'scope':f"{new['checkpoints']:,} new checkpoints independently audited. Prior {base['checkpoints']:,}-checkpoint verification retained; historical observations were not re-audited. Qwen 2.5 metadata adds Kimi Dev as an exact alias.",
        'base_hub_sha256':file_hash(ROOT/f'results/{BASE}/hub.json'),
        'base_verification_sha256':file_hash(base_path),
        'new_verification_sha256':file_hash(ROOT/f'results/{VERSION}/verification-new.json')}
    assert combined['observations']==hub['checks']['observations']
    write_json(ROOT/f'results/{VERSION}/verification.json',combined)
    return combined

def publish_local(hub):
    write_json(ROOT/'site/data/hub.json.gz',hub,compact=True)
    (ROOT/'site/data/hub.json').unlink(missing_ok=True)
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(hub['benchmarks'][0]))
    writer.writeheader();writer.writerows(hub['benchmarks'])
    (ROOT/'site/data/benchmarks.csv.gz').write_bytes(gzip.compress(buffer.getvalue().encode(),mtime=0))
    ids=new_ids();rows=[b for b in hub['benchmarks'] if b['tokenizer'] in ids]
    lines=['# Kimi tokenizer extension v0.4','',
        f"**{len(hub['profiles'])} profiles × {hub['checks']['documents_per_tokenizer']:,} inputs = {hub['checks']['observations']:,} observations.**",'',
        f"Added **{len(ids)} profiles**, covering **15 official checkpoints**, **{sum(b['documents'] for b in rows):,} new observations**, **{len(rows):,} independently audited partition checkpoints**, and **396 new vocabulary comparisons**. The explorer now has **{len(hub['pairs']):,} pairs**.",'',
        '| Tokenizer | Checkpoint aliases | Ordinary / special IDs | Total IDs | UTF-8 mismatches | UTF-16LE / BE mismatches | Encode MB/s | Decode MB/s | Loaded RSS MB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for id in ids:
        p=next(p for p in hub['profiles'] if p['id']==id);v=p['vocabulary'];r=next(r for r in hub['runtime'] if r['tokenizer']==id)
        b=[b for b in rows if b['tokenizer']==id];bad=lambda key:sum(x['documents']-x[key] for x in b)
        lines.append(f"| {p['name']} | {len(p['aliases'])} | {v['ordinary']:,} / {v['special']:,} | {v['total']:,} | {bad('exact'):,} | {bad('utf16le_exact'):,} / {bad('utf16be_exact'):,} | {r['encode_mbps']:.3f} | {r['decode_mbps']:.3f} | {r['loaded_rss_bytes']/1e6:.2f} |")
    dev=next(p for p in hub['profiles'] if p['id']=='qwen25')
    dev_rows=[b for b in hub['benchmarks'] if b['tokenizer']=='qwen25']
    changed=sum(b['documents']-b['exact'] for b in dev_rows)
    lines+=['',f"**Kimi Dev / Qwen 2.5 alias:** {dev['vocabulary']['total']:,} IDs; **{changed:,} / {sum(b['documents'] for b in dev_rows):,} inputs change bytes** ({100*changed/sum(b['documents'] for b in dev_rows):.4f}%). All {sum(b['canonical_only'] for b in dev_rows):,} are canonical-only NFC normalization changes; execution errors: {sum(b['errors'] for b in dev_rows)}. Its frozen measurements are reused, so this is separate from the eight zero-mismatch profiles above."]
    lines+=['',f"New execution errors: **{sum(b['errors'] for b in rows)}**. Each profile ran all **867 partitions**, including full FLORES-200 dev/devtest, bounded MADLAD samples, the pinned FineWeb mixture and constructed diagnostics. Each has **16 UTF-16 probes: 10 valid / 6 malformed rejected**. These are observed fidelity results, not proof for all Unicode strings.",'',
        '| Tokenizer | FineWeb tokens | FineWeb UTF-8 bytes/token | FLORES devtest tokens | MADLAD sample tokens |',
        '|---|---:|---:|---:|---:|']
    for id in ids:
        p=next(p for p in hub['profiles'] if p['id']==id)
        counts=[]
        for dataset in ['fineweb-mixture-v0.1','flores200-devtest','madlad400-sample-v0.2']:
            group=[b for b in rows if b['tokenizer']==id and b['dataset']==dataset]
            counts.append((sum(b['tokens'] for b in group),sum(b['utf8_bytes'] for b in group)))
        lines.append(f"| {p['name']} | {counts[0][0]:,} | {counts[0][1]/counts[0][0]:.4f} | {counts[1][0]:,} | {counts[2][0]:,} |")
    lines+=['', 'Pooled counts reflect this corpus language mixture. Compression is not model quality. Text inputs and source hashes are unchanged from v0.2.', '',
        'All eight profiles share **163,584 ordinary byte pieces**, SHA-256 `b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103`. They differ in full special-ID maps, reserved ranges or the direct encode policy. Base K2 uses ordinary text for special spellings; the other seven direct encode paths recognize reserved markers. No automatic BOS/EOS or chat template is used.', '',
        'Moonlight, Kimi VL, K2 Base and Kimi Linear wrappers expose **163,842 IDs**, including two reserved IDs beyond the checkpoint configs\' **163,840 vocabulary rows**. The other four expose 163,840 IDs. Tokenizer inventory and model embedding compatibility are distinct.', '',
        'Kimi Dev 72B tokenizer.json is byte-identical to the existing Qwen 2.5 profile (SHA-256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`). It is added as an alias and reuses its frozen measurements. It is not counted as a ninth new tokenizer.', '',
        'Inventory scope: all 19 public repositories returned for official author moonshotai on 2026-10-21; 16 text-capable checkpoints included, two speech pipelines and one vision-only encoder excluded. API-only Kimi models without an inspected public artifact are not represented. Multimodal profiles are tested on text only.', '',
        'Timing uses native tiktoken, one thread, one warm-up and five repeats in sequential fresh processes on the recorded Apple Silicon host. All use the same 216 interleaved FineWeb documents / 1,112,864 UTF-8 bytes and runtime input SHA-256 as prior releases. Historical rows were not re-timed. MB = 1,000,000 bytes; RSS is whole-process memory.', '',
        'Direct text encode/decode methods were inspected and their ranks, regex, special maps and 400,000/25,000-character safeguards reproduced without trust_remote_code. Full Transformers wrapper parity is not independently established. Chat rendering, images and audio were not benchmarked.', '',
        'Reproduce from explicit cached inputs with `configs/kimi-selection.json`. Run each selected tokenizer separately into `results/v0.4/<id>` with vocabulary/runtime enabled; `python -m tokenizer_lab.publish_kimi` assembles and audits those results. Exact revisions, artifact hashes, alias identities and exclusions: `configs/kimi-candidates.json`.', '',
        'Audit: `results/v0.4/verification-new.json`; combined audit chain: `results/v0.4/verification.json`. Frozen v0.2/v0.3 measurements remain intact. The Qwen 2.5 label/aliases are the only historical profile metadata changes. Website publishing and native Windows/Linux/Intel Mac checks remain deferred. Existing 0.2.1 starter ZIPs have not been rebuilt.' ]
    (ROOT/'reports/v0.4.md').write_text('\n'.join(lines)+'\n')

def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Replay the eight profiles from explicitly cached inputs before assembling')
    args=parser.parse_args(argv)
    if args.run:
        import os,shutil
        from .engine import run_selection
        os.environ.update(TOKENIZER_LAB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        selection=read_json(ROOT/'configs/kimi-selection.json')
        folder=ROOT/f'results/{VERSION}/measurement-code';folder.mkdir(parents=True,exist_ok=True)
        for name in ['engine.py','adapters.py','legacy.py','metrics.py','transport.py','io.py']:
            shutil.copyfile(ROOT/'src/tokenizer_lab'/name,folder/name)
        for id in new_ids():
            run_selection(selection|{'tokenizers':[id]},output=f'results/{VERSION}/{id}',vocabulary=True,runtime=True,
                progress=lambda p:print(p['message'],flush=True) if p['completed']%200==0 or p['completed']>=867 else None)
    from .saved_examples import build
    extend_pairs();assemble();verify_extension();examples=build(new_ids())
    hub=assemble();publish_local(hub)
    print({'checks':hub['checks'],'saved_examples':examples},flush=True)

if __name__=='__main__':main()
