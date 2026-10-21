from collections import Counter
import hashlib
import pytest

from tokenizer_lab.code_scenarios import make_contexts, make_stresses
from tokenizer_lab.io import sha256,write_json,write_jsonl,file_hash


def documents():
    result=[]
    for split in ['development','evaluation']:
        for i in range(10):
            text=f'# {split} file {i}\r\nprint("e\u0301")\r\n'
            result.append({'id':split+str(i),'text':text,'utf8_bytes':len(text.encode()),'sha256':sha256(text.encode()),
                'split':split,'repository':'owner/'+split,'path':f'file{i}.py','commit':'a'*40,'source_url':'https://example.invalid/'+split+str(i)})
    return result


def test_contexts_preserve_payloads_and_family_split():
    docs=documents();rows=make_contexts(docs,'python')
    assert len(rows)==50
    assert Counter(r['split'] for r in rows)=={'development':10,'evaluation':40}
    assert len({r['id'] for r in rows})==50
    for r in rows:
        assert r['source_id']=='owner/'+r['split']
        for source in r['source_files']:
            doc=next(d for d in docs if d['sha256']==source['sha256'])
            assert doc['text'] in r['text']
        assert sha256(r['text'].encode())==r['sha256']


def test_stresses_keep_correlated_cases_explicit():
    rows=make_stresses(documents(),'python')
    assert len(rows)==50
    assert Counter(r['split'] for r in rows)=={'development':10,'evaluation':40}
    assert len({r['base_source_sha256'] for r in rows})==5
    assert len({r['transformation'] for r in rows})==10
    assert all(r['controlled'] for r in rows)
    assert any('\x00' in r['text'] for r in rows)
    assert any('\r\n' in r['text'] for r in rows)
    assert all(sha256(r['text'].encode())==r['sha256'] for r in rows)


@pytest.mark.parametrize('corrupt',[False,True])
def test_edit_audit_checks_git_blob_provenance(tmp_path,monkeypatch,corrupt):
    from tokenizer_lab import code_scenarios as s
    monkeypatch.setattr(s,'ROOT',tmp_path)
    monkeypatch.setattr(s,'CORPUS',tmp_path/'data')
    texts={'before':'print(1)\r\n','after':'print(2)\r\n','diff':'-print(1)\r\n+print(2)\r\n'}
    pair={'id':'python:edit','repository':'o/r','split':'evaluation'}
    for name,text in texts.items():
        raw=text.encode();pair[name+'_sha256']=sha256(raw)
        if name!='diff':pair[name+'_blob']=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\x00'+raw).hexdigest()
    if corrupt:pair['before_blob']='0'*40
    index=tmp_path/'data/edit-index.jsonl.gz';write_jsonl(index,[pair]);parts=[]
    for name,text in texts.items():
        p=tmp_path/f'data/{name}.jsonl.gz'
        write_jsonl(p,[{'id':name,'text':text,'sha256':sha256(text.encode()),'pair_id':pair['id'],
            'dataset':'real-edit-'+name+'-v0.1','source_id':'o/r','split':'evaluation'}])
        parts.append({'path':f'data/{name}.jsonl.gz','sha256':file_hash(p),'documents':1})
    write_json(tmp_path/'data/manifest.json',{'partitions':parts,'documents':3,'quota_complete':True,'edit_index_sha256':file_hash(index)})
    if corrupt:
        with pytest.raises(ValueError,match='Git blob mismatch'):s.audit()
    else:assert s.audit()==3
