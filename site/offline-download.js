const host=document.querySelector('[data-offline-downloads]');
const bytes=n=>`${(n/1e6).toFixed(2)} MB`;
const names={'windows-x86_64':'Windows · Intel / AMD','linux-x86_64':'Linux · Intel / AMD','macos-arm64':'Mac · Apple Silicon','macos-x86_64':'Mac · Intel'};
const node=(tag,text,className)=>{const n=document.createElement(tag);if(text)n.textContent=text;if(className)n.className=className;return n;};
async function digest(data){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',data))].map(b=>b.toString(16).padStart(2,'0')).join('');}
async function render(){
 try{
  const response=await fetch('./data/offline-index.json');if(!response.ok)throw new Error('Offline packages are not present in this source checkout. See docs/OFFLINE.md to build a selected package.');
  const index=await response.json();const grid=node('div','','two-column');
  for(const kit of index.kits){
   const card=node('section','','card');card.append(node('h2',names[kit.target]||kit.target),node('p',`Release ${index.release}`,'note'),node('p',`${bytes(kit.bytes)} download · ${bytes(kit.installed_bytes)} extracted`),node('p','0 tokenizers · 0 benchmarks · 0 language files','note'));
   const button=node('button','Download starter ZIP ↓','button primary');button.type='button';const status=node('p','','note');const cancel=node('button','Cancel','button small');cancel.type='button';cancel.hidden=true;let controller;
   cancel.onclick=()=>controller?.abort();
   button.onclick=async()=>{
    controller=new AbortController();button.disabled=true;cancel.hidden=false;
    try{
     const parts=[];let received=0;
     for(let i=0;i<kit.parts.length;i++){
      status.textContent=`Downloading part ${i+1} / ${kit.parts.length} · ${bytes(received)} / ${bytes(kit.bytes)}`;
      const r=await fetch('./'+kit.parts[i].path,{signal:controller.signal});if(!r.ok)throw new Error(`Part ${i+1}: HTTP ${r.status}`);
      const part=await r.arrayBuffer();if(part.byteLength!==kit.parts[i].bytes||await digest(part)!==kit.parts[i].sha256)throw new Error(`Part ${i+1} failed its integrity check`);
      received+=part.byteLength;parts.push(part);
     }
     const blob=new Blob(parts,{type:'application/zip'});if(blob.size!==kit.bytes)throw new Error('Incomplete ZIP');
     status.textContent='Verifying complete ZIP…';if(await digest(await blob.arrayBuffer())!==kit.sha256)throw new Error('ZIP integrity check failed');
     const url=URL.createObjectURL(blob);const a=node('a');a.href=url;a.download=kit.filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);status.textContent=`Verified ${bytes(blob.size)}. Extract the ZIP and open Launch Tokenizer Lab.`;
    }catch(error){status.textContent=error.name==='AbortError'?'Download cancelled. No further parts will be fetched.':error.message;}finally{button.disabled=false;cancel.hidden=true;}
   };
   const detail=node('details');detail.append(node('summary','Integrity & system requirements'),node('p',kit.minimum_os||''),node('code',kit.sha256));card.append(button,cancel,status,detail);if(kit.release_url){const direct=node('a','Download from GitHub release ↗','inline-link');direct.href=kit.release_url;direct.rel='noopener noreferrer';card.append(direct);}grid.append(card);
  }
  host.replaceChildren(grid);
 }catch(error){host.replaceChildren(node('p',error.message,'callout'));}
}
render();
