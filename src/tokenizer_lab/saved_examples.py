"""Small saved token visualizations, separate from canonical benchmark text."""
from collections import defaultdict
from .io import ROOT,read_json,write_json,file_hash
from .engine import read_partition

def build(tokenizer_ids=None):
    from .adapters import load_adapter,registry
    catalog=read_json(ROOT/'benchmarks/catalog.json');groups=defaultdict(dict)
    old=ROOT/'site/data/examples.json'
    if old.exists():
        for row in read_json(old).get('examples',[]):
            dataset=row['id'].split(':')[0];groups[(dataset,row['language'])][row['id']]=row
    # Extensions keep all previously saved examples and their measured encodings.
    if tokenizer_ids is not None:
        for part in read_json(ROOT/'site/data/text-index.json')['partitions']:
            path=ROOT/'site'/part.get('examples_path','')
            if path.is_file():
                for row in read_json(path)['examples']:
                    groups[(part['dataset'],part['language'])][row['id']]=row
    for part in catalog['partitions']:
        eligible=[r for r in read_partition(part) if len(r['text'])<=1500]
        if not eligible:continue
        row=min(eligible,key=lambda r:(len(r['text']),r['id']))
        groups[(part['dataset'],part['language'])].setdefault(row['id'],dict(row,encodings={}))
    for spec in registry():
        if tokenizer_ids is not None and spec['id'] not in tokenizer_ids:continue
        adapter=load_adapter(spec['id'],download=False)
        for items in groups.values():
            for row in items.values():
                if spec['id'] in row.get('encodings',{}):continue
                try:
                    ids=adapter.encode(row['text']);decoded=adapter.decode_bytes(ids)
                    encoding={'ids':ids,'pieces':adapter.pieces(ids),'decoded':decoded.decode('utf-8',errors='backslashreplace'),'exact':decoded==row['text'].encode(),'tokens':len(ids),'bytes_per_token':len(row['text'].encode())/len(ids) if ids else None}
                except Exception as exc:encoding={'error':f'{type(exc).__name__}: {exc}'}
                row.setdefault('encodings',{})[spec['id']]=encoding
        print('SAVED EXAMPLES',spec['id'],flush=True);del adapter
    index=read_json(ROOT/'site/data/text-index.json')
    for part in index['partitions']:
        items=groups.get((part['dataset'],part['language']))
        if not items:continue
        file=ROOT/f"site/data/encodings/{part['dataset']}/{part['language']}.json.gz"
        write_json(file,{'examples':list(items.values())},compact=True)
        part['examples_path']=str(file.relative_to(ROOT/'site'));part['examples_sha256']=file_hash(file);part['examples_bytes']=file.stat().st_size
    write_json(ROOT/'site/data/text-index.json',index)
    return {'partitions':sum(bool(p.get('examples_path')) for p in index['partitions']),'examples':sum(len(g) for g in groups.values())}

if __name__=='__main__':print(build())
