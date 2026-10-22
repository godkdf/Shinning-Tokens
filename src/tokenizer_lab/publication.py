"""Build and audit local publication candidates. Never pushes or deploys."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from urllib.parse import unquote,urlsplit
import zipfile
from .io import ROOT,file_hash,read_json,write_json,sha256

TARGETS=('macos-arm64','macos-x86_64','windows-x86_64','linux-x86_64')
STATIC_BUDGET=24_000_000
PART_BYTES=22*1024*1024
PUBLIC_DIRS=('src','tests','docs','configs','scripts','.github','research','licenses','reports',
             'experiments/gemma-v31','experiments/embedding-mapping')
ROOT_FILES=('README.md','CHANGELOG.md','BENCHMARK.md','CONTRIBUTING.md','SECURITY.md',
            'THIRD_PARTY.md','CITATION.cff','LICENSE','pyproject.toml','uv.lock','.gitignore','.gitattributes')

def version(root=ROOT):return tomllib.loads((root/'pyproject.toml').read_text())['project']['version']
def destination(root=ROOT):return root/'artifacts/releases'/version(root)

def source_paths(root=ROOT):
    """Explicit public scope; a new untracked experiment is never auto-published."""
    paths={root/p for p in ROOT_FILES if (root/p).is_file()}
    for directory in PUBLIC_DIRS:
        paths.update(p for p in (root/directory).rglob('*') if p.is_file())
    for name in ['benchmarks/catalog.json','benchmarks/languages.json','site/data/hub.json.gz',
                 'site/data/benchmarks.csv.gz','site/data/text-index.json','site/data/languages.json',
                 'site/data/methods.json','site/data/offline-index.json']:
        if (root/name).is_file():paths.add(root/name)
    for directory in ['benchmarks/metadata','benchmarks/unicode']:
        paths.update(p for p in (root/directory).rglob('*') if p.is_file())
    for name in ['study.json.gz','measurements.csv.gz','report.md','guide.md','validation.json',
                 'summary.csv','source-report.md','source-by-language.csv.gz','scenarios-by-language.csv.gz']:
        path=root/'site/data/code-study'/name
        if path.is_file():paths.add(path)
    for p in (root/'benchmarks').rglob('*.json'):
        if p.name in ['manifest.json','source-lock.json']:paths.add(p)
    paths.update(p for p in (root/'site').glob('*') if p.is_file())
    for cohort in (root/'results').glob('v*'):
        for p in cohort.rglob('*'):
            if not p.is_file():continue
            relative=p.relative_to(cohort)
            if p.name=='runtime-input.json' or 'checkpoints' in relative.parts or p.name=='integration-validation.json':continue
            if p.suffix in ['.json','.py']:paths.add(p)
    excluded={'custom-tokenizers.json','.DS_Store','source-release.json','release-validation.json'}
    return sorted(p for p in paths if p.resolve().is_relative_to(root.resolve()) and p.name not in excluded and '__pycache__' not in p.parts
                  and p.suffix not in ['.pyc','.tmp','.zip','.safetensors','.bin','.npy','.npz']
                  and not p.name.startswith('.env') and not p.name.endswith(('.tar.gz','.jsonl.gz')))

def markdown_errors(root=ROOT):
    failures=[]
    for path in source_paths(root):
        if path.suffix.lower()!='.md':continue
        # Upstream notices/model cards retain their own repository-relative links.
        if path.is_relative_to(root/'licenses') or path.is_relative_to(root/'benchmarks/metadata'):continue
        body=re.sub(r'```.*?```','',path.read_text(encoding='utf-8'),flags=re.S)
        for match in re.finditer(r'(?<!!)\[[^\]]+\]\(([^\n)]+)\)',body):
            target=match[1].strip().split(' "',1)[0].strip('<>')
            if urlsplit(target).scheme or target.startswith('#'):continue
            relative=unquote(target.split('#',1)[0])
            if not relative:continue
            resolved=(path.parent/relative).resolve()
            if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
                failures.append({'file':str(path.relative_to(root)),'target':target})
    return failures

def member_manifest(root,paths):
    return {p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in paths}

def build_source(root=ROOT):
    folder=destination(root);folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'tokenizer-lab-{version(root)}-source.tar.gz'
    members=member_manifest(root,source_paths(root));temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as archive:
        for name in members:
            info=archive.gettarinfo(str(root/name),arcname=name)
            info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
            with (root/name).open('rb') as source:archive.addfile(info,source)
        data=(json.dumps(members,indent=2,sort_keys=True)+'\n').encode()
        info=tarfile.TarInfo('SOURCE-MANIFEST.json');info.size=len(data);info.mode=0o644
        archive.addfile(info,io.BytesIO(data))
    temporary.replace(path)
    if path.stat().st_size>STATIC_BUDGET:raise ValueError('Source snapshot exceeds the static-asset budget')
    with tarfile.open(path) as archive:
        for name,record in members.items():
            data=archive.extractfile(name).read()
            if len(data)!=record['bytes'] or sha256(data)!=record['sha256']:raise ValueError('Source archive member differs: '+name)
    shutil.copyfile(path,root/'site/data/source.tar.gz')
    result={'release':version(root),'file':'source.tar.gz','bytes':path.stat().st_size,
        'sha256':file_hash(path),'files':len(members)+1,
        'scope':'Curated source, documentation, manifests, notices, frozen aggregates and measuring sources. No original-text payloads, tokenizer inputs, model weights or observation checkpoints.'}
    write_json(root/'site/data/source-release.json',result)
    return result

def kit_fingerprint(root=ROOT):
    # Metadata indexes for the downloadable ZIPs/source have circular byte hashes;
    # portable kits deliberately omit those indexes and binary release payloads.
    paths=[p for p in source_paths(root) if not p.is_relative_to(root/'results')
           and not p.is_relative_to(root/'tests') and not p.is_relative_to(root/'reports')
           and not p.is_relative_to(root/'experiments')
           and p.name not in ['offline-index.json','benchmarks.csv.gz']]
    return sha256(json.dumps(member_manifest(root,paths),sort_keys=True,separators=(',',':')).encode())

def split_starter(root,path,target):
    directory=root/'site/data/offline'/target;directory.mkdir(parents=True,exist_ok=True);parts=[]
    with path.open('rb') as source:
        while data:=source.read(PART_BYTES):
            out=directory/f'{path.name}.part-{len(parts)+1:03d}';out.write_bytes(data)
            parts.append({'path':out.relative_to(root/'site').as_posix(),'bytes':len(data),'sha256':sha256(data)})
    return parts

def build_starters(root=ROOT):
    from . import offline
    if root!=offline.ROOT:raise ValueError('Build starters from the active project checkout')
    folder=destination(root);folder.mkdir(parents=True,exist_ok=True)
    fingerprint=kit_fingerprint(root);kits=[]
    old=read_json(root/'site/data/offline-index.json') if (root/'site/data/offline-index.json').exists() else {'kits':[]}
    for target in TARGETS:
        path=folder/f'tokenizer-lab-{version(root)}-starter-{target}.zip';meta=path.with_suffix('.manifest.json')
        result=read_json(meta) if meta.exists() else {}
        if not (path.exists() and result.get('source_fingerprint')==fingerprint and result.get('sha256')==file_hash(path)):
            with tempfile.TemporaryDirectory(prefix='build-',dir=folder) as work:
                built=Path(work)/path.name
                result=offline.build_bundle(target,tokenizers=[],languages=[],datasets=[],output=built)
                runtime=read_json(built.with_suffix('.manifest.json'))['runtime']
                built.replace(path)
            result.update(archive=path.name,source_fingerprint=fingerprint,runtime=runtime)
            write_json(meta,result)
        evidence=folder/f'native-{target}.json'
        verified=evidence.exists() and read_json(evidence).get('passed') is True and read_json(evidence).get('archive_sha256')==result['sha256']
        kits.append({'target':target,'filename':path.name,'sha256':result['sha256'],'bytes':result['bytes'],
            'installed_bytes':result['installed_bytes'],'minimum_os':result['runtime']['minimum_os'],
            'parts':split_starter(root,path,target),'tokenizers':0,'languages':0,'datasets':0,
            'native_execution_verified':verified,
            'release_url':f'https://github.com/godkdf/Shinning-Tokens/releases/download/v{version(root)}/{path.name}'})
        print('PREPARED',target,result['bytes'],flush=True)
    write_json(root/'site/data/offline-index.json',{'release':version(root),'available_profiles':54,'kits':kits,
        'note':'Zero tokenizer and corpus payloads. Current native verification is recorded per archive; cross-platform package checks do not establish native execution.'})
    current={p['path'] for k in kits for p in k['parts']}
    # Only remove reproducible old part copies; original historical ZIPs remain.
    for kit in old['kits']:
        for part in kit['parts']:
            p=root/'site'/part['path']
            if part['path'] not in current and p.is_file() and file_hash(p)==part['sha256']:p.unlink()
    return {'release':version(root),'kits':len(kits),'source_fingerprint':fingerprint}

def audit(root=ROOT,source_only=False):
    from .adapters import registry
    failures=[];release=version(root);hub=read_json(root/'results/v0.4/hub.json');native_verified=[];archive_files_checked=0
    specs=read_json(root/'configs/registry.json')['tokenizers'];ids={p['id'] for p in hub['profiles']}
    if ids!={s['id'] for s in specs}:failures.append('Registry and current measured profile sets differ')
    if len(hub['pairs'])!=len(ids)*(len(ids)-1)//2:failures.append('Vocabulary comparison graph is incomplete')
    if sum(b['documents'] for b in hub['benchmarks'])!=hub['checks']['observations']:failures.append('Observation denominator differs')
    if read_json(root/'site/data/hub.json.gz')!=hub:failures.append('Explorer and current scientific aggregate differ')
    if hub['checks']['input_manifest_hash']!=file_hash(root/'benchmarks/catalog.json'):failures.append('Frozen input catalog changed')
    if f'version: {release}\n' not in (root/'CITATION.cff').read_text():failures.append('Citation version differs')
    if f'## {release}' not in (root/'CHANGELOG.md').read_text():failures.append('Changelog version missing')
    lock=tomllib.loads((root/'uv.lock').read_text())
    if next(p['version'] for p in lock['package'] if p['name']=='shining-tokenizer-lab')!=release:failures.append('Lockfile project version differs')
    for cohort,base in [('v0.4','v0.3'),('v0.3','v0.2')]:
        record=read_json(root/f'results/{cohort}/verification.json')
        if record['status']!='passed' or record['base_hub_sha256']!=file_hash(root/f'results/{base}/hub.json') or record['base_verification_sha256']!=file_hash(root/f'results/{base}/verification.json'):
            failures.append('Broken inherited audit chain: '+cohort)
    broken=markdown_errors(root)
    if broken:failures.append({'broken_local_markdown_links':broken})
    sensitive=re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')
    for path in source_paths(root):
        if path.stat().st_size>=100*1024*1024:failures.append('Regular Git/source file exceeds 100 MiB: '+str(path.relative_to(root)))
        if not path.is_relative_to(root/'licenses') and path.suffix in ['.py','.js','.md','.json','.yml','.toml'] and sensitive.search(path.read_bytes()):
            failures.append('Credential-shaped content in '+str(path.relative_to(root)))
    if (root/'.git').exists():
        tracked=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
        for relative in filter(None,tracked):
            path=root/relative
            if path.is_file() and path.stat().st_size>=100*1024*1024:failures.append('Tracked file exceeds 100 MiB: '+relative)
    if not source_only:
        index=read_json(root/'site/data/offline-index.json')
        if index['release']!=release or {k['target'] for k in index['kits']}!=set(TARGETS):failures.append('Starter release/targets differ')
        for kit in index['kits']:
            digest=hashlib.sha256();size=0
            for part in kit['parts']:
                path=root/'site'/part['path']
                if not path.exists() or file_hash(path)!=part['sha256']:failures.append('Missing/corrupt starter part: '+part['path']);continue
                data=path.read_bytes();digest.update(data);size+=len(data)
            if size!=kit['bytes'] or digest.hexdigest()!=kit['sha256']:failures.append('Starter reconstruction differs: '+kit['target'])
            meta=destination(root)/Path(kit['filename']).with_suffix('.manifest.json')
            if not meta.exists() or read_json(meta).get('source_fingerprint')!=kit_fingerprint(root):failures.append('Starter source is stale: '+kit['target'])
            archive_path=destination(root)/kit['filename']
            if not archive_path.exists() or file_hash(archive_path)!=kit['sha256']:
                failures.append('Starter ZIP differs: '+kit['target']);continue
            with zipfile.ZipFile(archive_path) as archive:
                manifest=read_json_stream(archive,'Tokenizer Lab/offline-manifest.json')
                if manifest['target']!=kit['target'] or manifest['tokenizers'] or manifest['languages'] or manifest['datasets']:
                    failures.append('Starter selection or target differs: '+kit['target'])
                if manifest['payload_sizes']['tokenizer_input_bytes'] or manifest['payload_sizes']['compressed_text_bytes']:
                    failures.append('Starter unexpectedly contains input payloads: '+kit['target'])
                catalog=read_json_stream(archive,'Tokenizer Lab/configs/registry.json')
                if {s['id'] for s in catalog['tokenizers']}!=ids:failures.append('Starter catalog differs: '+kit['target'])
                for name,record in manifest['files'].items():
                    with archive.open('Tokenizer Lab/'+name) as file:digest=hashlib.file_digest(file,'sha256').hexdigest()
                    if digest!=record['sha256']:failures.append('Starter member differs: '+kit['target']+'/'+name)
                    archive_files_checked+=1
            if kit['native_execution_verified']:
                proof=destination(root)/f"native-{kit['target']}.json"
                if not proof.exists() or not read_json(proof).get('passed') or read_json(proof).get('archive_sha256')!=kit['sha256']:
                    failures.append('Native verification has no matching evidence: '+kit['target'])
                else:native_verified.append(kit['target'])
        source=read_json(root/'site/data/source-release.json')
        path=root/'site/data/source.tar.gz'
        if source['release']!=release or file_hash(path)!=source['sha256']:failures.append('Source release/hash differs')
        with tarfile.open(path) as archive:
            members=json.load(archive.extractfile('SOURCE-MANIFEST.json'))
            for name,record in members.items():
                if not (root/name).is_file() or file_hash(root/name)!=record['sha256'] or sha256(archive.extractfile(name).read())!=record['sha256']:
                    failures.append('Stale source member: '+name)
        for path in (root/'site').rglob('*'):
            if path.is_file() and path.stat().st_size>STATIC_BUDGET:failures.append('Static asset exceeds budget: '+str(path.relative_to(root)))
    result={'passed':not failures,'release':release,'result_cohort':'v0.4','protocol':hub['protocol']['id'],
        'profiles':len(ids),'observations':hub['checks']['observations'],'aggregate_rows':len(hub['benchmarks']),
        'vocabulary_pairs':len(hub['pairs']),'source_only':source_only,'failures':failures,
        'external_links_checked':False,'archive_member_hashes_checked':archive_files_checked,'native_verified':native_verified,
        'native_deferred':['windows-x86_64','linux-x86_64','macos-x86_64'],
        'publication_performed':False}
    folder=destination(root);folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/('validation-source.json' if source_only else 'validation.json'),result)
    return result

def read_json_stream(archive,name):
    with archive.open(name) as file:return json.load(file)

def verify_release(root=ROOT):
    folder=destination(root);manifest=read_json(folder/'release-manifest.json');failures=[]
    for record in manifest['artifacts']:
        name=record['file'];path=folder/name
        if Path(name).name!=name or not path.is_file() or path.stat().st_size!=record['bytes'] or file_hash(path)!=record['sha256']:
            failures.append(name)
    return {'passed':not failures,'release':manifest['release'],'artifacts_checked':len(manifest['artifacts']),'failures':failures}

def package(root=ROOT):
    evidence=audit(root)
    if not evidence['passed']:raise ValueError('Release audit failed; inspect validation.json')
    folder=destination(root);release=version(root);path=folder/f'tokenizer-lab-{release}-explorer.zip'
    paths=sorted(p for p in (root/'site').rglob('*') if p.is_file() and '.DS_Store' not in p.parts and '__pycache__' not in p.parts)
    members=member_manifest(root,paths)
    with zipfile.ZipFile(path.with_suffix('.tmp'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for p in paths:
            archive.write(p,p.relative_to(root).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix in ['.gz','.png'] or '.zip.part-' in p.name else zipfile.ZIP_DEFLATED)
        archive.writestr('EXPLORER-MANIFEST.json',json.dumps(members,indent=2,sort_keys=True)+'\n')
    path.with_suffix('.tmp').replace(path)
    with zipfile.ZipFile(path) as archive:
        for name,record in members.items():
            with archive.open(name) as file:
                digest=hashlib.file_digest(file,'sha256').hexdigest()
            if digest!=record['sha256']:raise ValueError('Explorer ZIP member differs: '+name)
    notes=f'''# Tokenizer Lab {release}\n\nA reproducible tokenizer comparison hub: **54 profiles, 23,304,780 observations, 1,431 vocabulary pairs**. No model training or weights are required.\n\nIncludes native GPT-2 and Llama 3, eight Kimi/Moonlight profiles and distinct GLM/Qwen/DeepSeek/MiMo/MiniMax generations. Benchmarks cover the pinned FineWeb mixture, FLORES-200 dev/devtest, bounded MADLAD samples and Unicode/number/code diagnostics. UTF-8, UTF-16LE and UTF-16BE fidelity are reported separately.\n\nChoose a source archive, a starter for your OS, or the larger full saved explorer. Starters include **0 tokenizer and 0 benchmark-language payloads**; downloads are explicit and cached runs work offline. SHA256SUMS and release-manifest.json identify every attached artifact.\n\nLimits: compression is not model quality; FLORES+ was not run; MADLAD is a bounded prefix sample. Historical timing cohorts were not repeated simultaneously. Some full upstream wrapper parity remains unverified. Kimi Dev/Qwen 2.5 changes **11,521/431,570** inputs through NFC normalization; these remain byte-exact failures. Gemma V3.1's reconstructed parent is not independently verified against Google's gated download.\n\nNative execution on Windows, Linux and Intel Mac remains deferred; runtime/wheel/package checks do not establish that evidence. The Apple Silicon validation file records the current archive tested. Source code is Apache-2.0; upstream tokenizer/dataset licenses and notices remain applicable.\n\nThis directory is a local publication candidate. Creating it does not publish a GitHub release, change repository visibility or deploy a website.\n'''
    notes=notes.split('\n\nThis directory is a local publication candidate.',1)[0]+'\n'
    (folder/'RELEASE_NOTES.md').write_text(notes)
    artifacts=[]
    allowed={f'tokenizer-lab-{release}-source.tar.gz',path.name,'RELEASE_NOTES.md','validation.json','validation-source.json','test-results.json'}
    for target in TARGETS:
        allowed|={f'tokenizer-lab-{release}-starter-{target}.zip',f'tokenizer-lab-{release}-starter-{target}.manifest.json',f'native-{target}.json'}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.name in allowed:
            artifacts.append({'file':p.name,'bytes':p.stat().st_size,'sha256':file_hash(p)})
    write_json(folder/'release-manifest.json',{'release':release,'repository':'https://github.com/godkdf/Shinning-Tokens',
        'source_archive_sha256':file_hash(folder/f'tokenizer-lab-{release}-source.tar.gz'),
        'explorer_manifest_sha256':sha256(json.dumps(members,sort_keys=True,separators=(',',':')).encode()),
        'artifacts':artifacts,'publication_performed':False})
    checks=[*artifacts,{'file':'release-manifest.json','sha256':file_hash(folder/'release-manifest.json')}]
    (folder/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['file']}\n" for r in checks))
    verified=verify_release(root)
    if not verified['passed']:raise ValueError('Release artifacts differ after packaging')
    return {'release':release,'directory':str(folder),'artifacts':len(artifacts),'explorer_bytes':path.stat().st_size,'publication_performed':False}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ['source','starters','package','verify']:sub.add_parser(name)
    check=sub.add_parser('audit');check.add_argument('--source-only',action='store_true')
    args=parser.parse_args(argv)
    result={'source':build_source,'starters':build_starters,'package':package,'verify':verify_release}.get(args.command,lambda:audit(source_only=args.source_only))()
    print(json.dumps(result,indent=2));return 0 if result.get('passed',True) else 1

if __name__=='__main__':raise SystemExit(main())
