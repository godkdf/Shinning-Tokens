"""Audit the full expanded observation matrix against frozen input identities."""
from .io import ROOT,read_json,read_jsonl,file_hash,write_json
from .engine import read_partition
from .metrics import aggregate

def verify(version='v0.2',tokenizer_ids=None):
    hub=read_json(ROOT/f'results/{version}/hub.json');catalog=read_json(ROOT/'benchmarks/catalog.json')
    parts={(p['dataset'],p['language']):p for p in catalog['partitions']}
    observed=0;checkpoints=0;errors=0
    for profile in hub['profiles']:
        id=profile['id']
        if tokenizer_ids is not None and id not in tokenizer_ids:continue
        result=read_json(ROOT/hub.get('result_paths',{}).get(id,f'results/{version}/{id}/result.json'))
        benchmarks={(b['dataset'],b['language']):b for b in result['benchmarks']}
        assert set(benchmarks)==set(parts),f'Incomplete partition coverage: {id}'
        published={(b['dataset'],b['language']):b for b in hub['benchmarks'] if b['tokenizer']==id}
        assert published==benchmarks,f'Published aggregates changed: {id}'
        for checkpoint in result['checkpoints']:
            key=checkpoint['dataset'],checkpoint['language'];part=parts[key]
            path=ROOT/checkpoint['path'];assert file_hash(path)==checkpoint['sha256'],'Observation checkpoint hash changed'
            source={d['id']:d for d in read_partition(part)};rows=list(read_jsonl(path))
            assert len(rows)==len(source)==checkpoint['documents'],'Input/observation count differs'
            assert len({r['id'] for r in rows})==len(rows),'Duplicate observation IDs'
            for r in rows:
                d=source[r['id']];assert r['sha256']==d['sha256'] and r['utf8_bytes']==d['utf8_bytes'],'Changed input identity'
                assert r['tokenizer']==id and r['dataset']==key[0] and r['language']==key[1]
                assert r['tokens']>=0 and r['utf16le_bytes']==r['utf16be_bytes']==2*r['utf16_code_units']
                if r['error'] is None:assert r['exact']==r['utf16le_exact']==r['utf16be_exact'],'Transport fidelity inconsistency'
            assert aggregate(rows)==[benchmarks[key]],f'Aggregate differs from observations: {id}/{key}'
            observed+=len(rows);checkpoints+=1;errors+=sum(r['error'] is not None for r in rows)
        probes=result['transport_probes'][id]
        assert len(probes)==16 and sum(p.get('rejected',False) for p in probes)==6,'Malformed UTF-16 accepted'
        print('AUDITED',id,flush=True)
    count=len(tokenizer_ids) if tokenizer_ids is not None else len(hub['profiles'])
    assert observed==sum(p['documents'] for p in parts.values())*count
    if tokenizer_ids is None:assert observed==hub['checks']['observations']==hub['checks']['expected_observations']
    evidence={'status':'passed','tokenizers':count,'partitions_per_tokenizer':len(parts),'checkpoints':checkpoints,'observations':observed,'errors':errors,'malformed_transport_rejections':6*count,'catalog_sha256':file_hash(ROOT/'benchmarks/catalog.json')}
    write_json(ROOT/f'results/{version}'/('verification-new.json' if tokenizer_ids is not None else 'verification.json'),evidence);return evidence

if __name__=='__main__':print(verify())
