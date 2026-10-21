"""Add reference tokenizers while retaining the frozen 44-profile v0.2 run.

Run after GPT-2 and Llama 3 complete the selection in expanded-selection.json:
    python -m tokenizer_lab.publish_references
"""
from __future__ import annotations
import csv
import datetime
import gzip
import io
from .io import ROOT,read_json,write_json,file_hash

NEW_IDS=('gpt2','llama3')

def extend_pairs():
    from .vocabulary import compare_vocabularies
    base=read_json(ROOT/'results/v0.2/hub.json')
    pairs=list(base['pairs'])
    new={id:read_json(ROOT/f'site/data/vocab/{id}.json.gz') for id in NEW_IDS}
    for profile in base['profiles']:
        old=read_json(ROOT/f"site/data/vocab/{profile['id']}.json.gz")
        for id in NEW_IDS:pairs.append(compare_vocabularies(old,new[id]))
    pairs.append(compare_vocabularies(new[NEW_IDS[0]],new[NEW_IDS[1]]))
    count=len(base['profiles'])+len(NEW_IDS)
    assert len(pairs)==count*(count-1)//2
    write_json(ROOT/'results/v0.3/pairs.json',pairs,compact=True)
    return pairs

def assemble():
    base=read_json(ROOT/'results/v0.2/hub.json')
    catalog=read_json(ROOT/'benchmarks/catalog.json')
    assert base['checks']['input_manifest_hash']==file_hash(ROOT/'benchmarks/catalog.json')
    parts={(p['dataset'],p['language']) for p in catalog['partitions']}
    results=[read_json(ROOT/f'results/v0.3/{id}/result.json') for id in NEW_IDS]
    for id,result in zip(NEW_IDS,results):
        assert result['selection']['tokenizers']==[id]
        assert {(b['dataset'],b['language']) for b in result['benchmarks']}==parts
        assert result['network_downloads']==0
        assert len(result['profiles'])==len(result['runtime'])==1
        assert result['runtime'][0]['input_sha256']==base['runtime'][0]['input_sha256']
    hub={**base,'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'profiles':base['profiles']+[p for r in results for p in r['profiles']],
         'benchmarks':base['benchmarks']+[b for r in results for b in r['benchmarks']],
         'runtime':base['runtime']+[dict(t,measured_at=r['generated_at'],schedule='Reference extension; sequential fresh process on the recorded host, same frozen input. Historical v0.2 timings were not rerun.') for r in results for t in r['runtime']],
         'transport_probes':base['transport_probes']|{id:r['transport_probes'][id] for id,r in zip(NEW_IDS,results)},
         'pairs':read_json(ROOT/'results/v0.3/pairs.json')}
    ids=[p['id'] for p in hub['profiles']];assert len(ids)==len(set(ids))
    hub['result_paths']={id:f"results/{'v0.3' if id in NEW_IDS else 'v0.2'}/{id}/result.json" for id in ids}
    hub['measurement_source']={'note':'v0.3 adds GPT-2 and original Llama 3. The 44 v0.2 measurements and their saved sources remain frozen; use measurement_sources and result_paths for per-profile provenance.'}
    hub['measurement_sources']={p['id']:base['measurement_source'] for p in base['profiles']}
    hub['measurement_sources'].update({id:{'directory':'results/v0.3/measurement-code','hashes':r['measurement_code']} for id,r in zip(NEW_IDS,results)})
    docs=sum(p['documents'] for p in catalog['partitions'])
    hub['checks']={**base['checks'],'configured_tokenizers':len(ids),'loaded_tokenizers':len(ids),
        'expected_observations':docs*len(ids),'observations':sum(b['documents'] for b in hub['benchmarks']),
        'benchmark_errors':sum(b['errors'] for b in hub['benchmarks']),'pair_count':len(hub['pairs'])}
    assert hub['checks']['observations']==hub['checks']['expected_observations']
    verification=ROOT/'results/v0.3/verification.json'
    hub['verification']=read_json(verification) if verification.exists() else {'status':'pending','scope':'Reference extension audit'}
    hub['reproduction']={**base['reproduction'],'notes':base['reproduction']['notes']+[
        'v0.3 adds native GPT-2 (50,257 IDs) and original Llama 3 (128,256 IDs), using the same frozen corpus and intrinsic-v0.2 protocol.',
        'Llama 3 ranks reconstructed from byte-identical public redistribution must match the original Meta tokenizer.model SHA-256. Ordinary text policy; no automatic BOS/EOS. Meta character/run splitting is retained.',
        'Historical v0.2 runtime rows remain from their original measurement date. The two reference timings use the same input hash and recorded host; timings are not a simultaneous controlled rerun of all 46 profiles.' ]}
    write_json(ROOT/'results/v0.3/hub.json',hub)
    return hub

def verify_extension():
    from .verify_expanded import verify
    new=verify('v0.3',NEW_IDS)
    base_path=ROOT/'results/v0.2/verification.json';base=read_json(base_path)
    oldhub=read_json(ROOT/'results/v0.2/hub.json');hub=read_json(ROOT/'results/v0.3/hub.json')
    assert base['status']=='passed' and base['catalog_sha256']==new['catalog_sha256']
    assert hub['profiles'][:len(oldhub['profiles'])]==oldhub['profiles']
    assert hub['benchmarks'][:len(oldhub['benchmarks'])]==oldhub['benchmarks']
    combined={**new,'tokenizers':base['tokenizers']+new['tokenizers'],
        **{k:base[k]+new[k] for k in ['checkpoints','observations','errors','malformed_transport_rejections']},
        'scope':'1,734 new checkpoints independently audited. Frozen v0.2 aggregates and prior 38,148-checkpoint verification retained unchanged; historical checkpoints were not re-audited.',
        'base_hub_sha256':file_hash(ROOT/'results/v0.2/hub.json'),'base_verification_sha256':file_hash(base_path),
        'new_verification_sha256':file_hash(ROOT/'results/v0.3/verification-new.json')}
    assert combined['observations']==hub['checks']['observations']
    write_json(ROOT/'results/v0.3/verification.json',combined)
    return combined

def publish_local(hub):
    write_json(ROOT/'site/data/hub.json.gz',hub,compact=True)
    plain=ROOT/'site/data/hub.json'
    if plain.exists():plain.unlink()
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(hub['benchmarks'][0]))
    writer.writeheader();writer.writerows(hub['benchmarks'])
    (ROOT/'site/data/benchmarks.csv.gz').write_bytes(gzip.compress(buffer.getvalue().encode(),mtime=0))
    lines=['# Reference tokenizer extension v0.3','',
        f"**{len(hub['profiles'])} profiles × {hub['checks']['documents_per_tokenizer']:,} inputs = {hub['checks']['observations']:,} observations.**",
        '', 'GPT-2 and original Llama 3 each ran all 867 partitions. The 44 existing profiles remain frozen in v0.2.', '',
        '| Tokenizer | Total IDs | Ordinary / special | Non-exact UTF-8 | Non-exact UTF-16LE / BE | Encode MB/s | Decode MB/s | Loaded RSS MB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for id in NEW_IDS:
        p=next(p for p in hub['profiles'] if p['id']==id);v=p['vocabulary']
        b=[b for b in hub['benchmarks'] if b['tokenizer']==id];r=next(r for r in hub['runtime'] if r['tokenizer']==id)
        bad=lambda key:sum(x['documents']-x[key] for x in b)
        lines.append(f"| {p['name']} | {v['total']:,} | {v['ordinary']:,} / {v['special']:,} | {bad('exact'):,} | {bad('utf16le_exact'):,} / {bad('utf16be_exact'):,} | {r['encode_mbps']:.3f} | {r['decode_mbps']:.3f} | {r['loaded_rss_bytes']/1e6:.2f} |")
    lines+=['','Pooled counts by dataset (language weights follow this fixed corpus; no overall winner score):','',
        '| Dataset | GPT-2 tokens | Llama 3 tokens | GPT-2 bytes/token | Llama 3 bytes/token | Llama 3 token reduction |',
        '|---|---:|---:|---:|---:|---:|']
    for dataset in ['fineweb-mixture-v0.1','flores200-devtest','madlad400-sample-v0.2']:
        rows=[[b for b in hub['benchmarks'] if b['tokenizer']==id and b['dataset']==dataset] for id in NEW_IDS]
        tokens=[sum(b['tokens'] for b in group) for group in rows]
        ratios=[sum(b['utf8_bytes'] for b in group)/n for group,n in zip(rows,tokens)]
        lines.append(f"| {dataset} | {tokens[0]:,} | {tokens[1]:,} | {ratios[0]:.4f} | {ratios[1]:.4f} | {100*(1-tokens[1]/tokens[0]):.2f}% |")
    lines+=['',f"Added **863,140 observations**, **1,734 audited checkpoints**, and **89 vocabulary comparisons**; **{hub['checks']['pair_count']:,} pairs** are available in total. Execution errors in the new observations: **{sum(b['errors'] for b in hub['benchmarks'] if b['tokenizer'] in NEW_IDS)}**.",
        '', 'Each new profile passed 16 UTF-16 transport probes: 10 valid cases and 6 malformed cases rejected before tokenization. Fidelity results concern the sampled inputs, not every possible Unicode string.',
        '', 'Timings use native tiktoken, one thread, one warm-up and five repeats in sequential fresh processes on Apple Silicon. Input: 216 interleaved FineWeb documents, 1,112,864 UTF-8 bytes, identical SHA-256 to v0.2. MB = 1,000,000 bytes. RSS is whole-process memory, not file size. Historical 44-profile timings were not rerun.',
        '', 'Llama 3: 128,000 ordinary ranks plus 256 special IDs. Data downloaded from the pinned NousResearch redistribution; all four files match the official Meta Git blobs. Reconstructed rank file SHA-256: `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55`. Meta reference splitting at 400,000 characters and 25,000 consecutive whitespace/non-whitespace characters is retained.',
        '', 'Both profiles use ordinary-text encoding, no automatic BOS/EOS, no chat template, and no normalization. Literal special-token spellings remain ordinary text. Model weights are not downloaded.',
        '', 'Reproduce: explicitly cache the two tokenizers and selected corpus payloads, run `tokenizer-lab run-selected --selection configs/reference-selection.json --vocabulary --runtime --output results/local/references`, then inspect the output. The extension assembler expects separate per-tokenizer result folders under `results/v0.3/`.',
        '', 'Evidence: `results/v0.3/verification-new.json`, combined prior/new audit in `results/v0.3/verification.json`, all aggregates in `results/v0.3/hub.json`. New checkpoints are local compressed observations; frozen inputs are identified by `benchmarks/catalog.json`. Private website publication and native Windows/Linux/Intel Mac checks remain deferred.']
    (ROOT/'reports/v0.3.md').write_text('\n'.join(lines)+'\n')

def main():
    from .saved_examples import build
    extend_pairs();assemble();verify_extension()
    examples=build(NEW_IDS)
    hub=assemble();publish_local(hub)
    print({'checks':hub['checks'],'saved_examples':examples},flush=True)

if __name__=='__main__':main()
