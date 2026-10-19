function replace(target,...children){target.replaceChildren(...children.filter(child=>child!==null&&child!==undefined&&child!==false));}
// Shared by the standalone offline launcher and the explorer's Local runs view.
function node(tag,attrs={},...children){
  const result=document.createElement(tag);
  for(const child of children.flat(Infinity))if(child!==null&&child!==undefined&&child!==false)result.append(child instanceof Node?child:document.createTextNode(String(child)));
  for(const [key,value]of Object.entries(attrs)){
    if(value===null||value===undefined||value===false)continue;
    if(key.startsWith('on')&&typeof value==='function')result.addEventListener(key.slice(2).toLowerCase(),value);
    else if(key==='checked'||key==='disabled'||key==='value')result[key]=value;
    else result.setAttribute(key,value===true?'':String(value));
  }
  return result;
}
const count=value=>Number(value||0).toLocaleString('en-US');
const values=value=>Array.isArray(value)?value:[];
const normalize=value=>String(value).normalize('NFD').replace(/\p{Diacritic}/gu,'').toLowerCase();
const action=(label,handler,kind='')=>node('button',{type:'button',class:`button ${kind}`,onClick:handler},label);
const field=(label,input)=>node('label',{class:'field'},node('span',{class:'field-label'},label),input);
async function api(path,body,{method=body?'POST':'GET'}={}){
  const response=await fetch(path,{method,...(body?{headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{}),signal:AbortSignal.timeout(20000)});
  let payload;try{payload=await response.json();}catch{throw new Error(`The local service returned no JSON (${response.status}).`);}
  if(!response.ok)throw new Error(payload.error||`Request failed (${response.status}).`);
  return payload;
}
function savedSelection(){try{return JSON.parse(localStorage.getItem('tokenizer-lab-selection-v02')||'{}');}catch{return {};}}
function helpBlock(){return node('div',{class:'callout'},node('strong',{},'Use the local launch package.'),node('p',{},'Open the launcher for Windows, Linux, Apple silicon, or Intel Mac after preparing its matching environment. Keep the downloaded tokenizer assets and corpus files beside the application. The local service performs the computation on your machine.'),node('a',{href:'#reproduce'},'Installation and offline bundle details →'));}
export async function mountLocalLauncher(root,{languages:fallbackLanguages=[]}={}){
  replace(root,node('p',{class:'muted',role:'status'},'Checking the local tokenizer service…'));
  let health,catalog;
  try{
    health=await api('/api/health');
    if(health.ok!==true&&health.service!=='tokenizer-lab')throw new Error('This address does not provide the Tokenizer Lab service.');
    catalog=await api('/api/catalog');
  }catch(error){
    replace(root,node('div',{class:'card'},node('h2',{},'Start the Lab on your computer'),node('p',{class:'muted'},'This explorer shows published results. Benchmark runs and local tokenizer files require the local service.'),node('p',{class:'note'},error.message),helpBlock()));return;
  }
  let tokenizers=values(catalog.tokenizers),languages=values(catalog.languages).length?catalog.languages:fallbackLanguages,datasets=values(catalog.datasets);
  const previous=savedSelection();
  const selected={tokenizers:new Set(values(previous.tokenizers).filter(id=>tokenizers.some(item=>item.id===id))),languages:new Set(values(previous.languages).filter(id=>languages.some(item=>item.id===id))),datasets:new Set(values(previous.datasets).filter(id=>datasets.some(item=>item.id===id)))};
  const customPaths=new Map();
  const status=node('div',{class:'conversion-status'},node('span',{class:'status-dot'}),health.offline===true?'Offline service · cached runs only':'Local service connected');
  const errorBox=node('div',{'aria-live':'polite'}),runOutput=node('div',{'aria-live':'polite'}),selectionSummary=node('div',{class:'selection-summary'});
  const submit=action('Run selected benchmarks →',run,'primary');
  let running=false,downloadBusy=false,currentPlan=null,planKey='',lastSelectionKey='';
  const planHost=node('div',{'aria-live':'polite'}),downloadOutput=node('div',{'aria-live':'polite'});
  const review=action('Review selected files & download sizes',reviewPlan);
  const selectionPayload=()=>({tokenizers:[...selected.tokenizers].filter(id=>!customPaths.has(id)),datasets:[...selected.datasets],languages:[...selected.languages]});
  const selectionKey=()=>JSON.stringify({...selectionPayload(),custom:[...customPaths.keys()].filter(id=>selected.tokenizers.has(id))});
  const byteSize=value=>typeof value!=='number'||!Number.isFinite(value)?'size unknown':value>=1e9?`${(value/1e9).toFixed(2)} GB`:value>=1e6?`${(value/1e6).toFixed(2)} MB`:value>=1e3?`${(value/1e3).toFixed(1)} kB`:`${count(value)} B`;
  const planReady=()=>currentPlan&&planKey===selectionKey()&&!values(currentPlan.missing).length&&values(currentPlan.files).every(file=>file.cached===true);
  const remember=()=>{try{localStorage.setItem('tokenizer-lab-selection-v02',JSON.stringify(Object.fromEntries(Object.entries(selected).map(([key,value])=>[key,[...value]]))));}catch{}};
  function refreshSummary(){
    selectionSummary.textContent=`${count(selected.tokenizers.size)} tokenizers · ${count(selected.languages.size)} languages · ${count(selected.datasets.size)} datasets selected`;
    const key=selectionKey();
    if(key!==lastSelectionKey){currentPlan=null;planKey='';replace(planHost,node('p',{class:'note'},'Review the selected files to see exact download and cached sizes before downloading or running.'));lastSelectionKey=key;}
    submit.disabled=running||downloadBusy||!selected.tokenizers.size||!selected.languages.size||!selected.datasets.size||!planReady();
    review.disabled=running||downloadBusy||!(selected.tokenizers.size+selected.languages.size+selected.datasets.size);
    if(planReady())selectionSummary.textContent+=' · all required files cached';
    remember();
  }
  function picker(title,items,key,{continentFilter=false}={}){
    const search=node('input',{class:'input',type:'search',placeholder:key==='languages'?'English, العربية, isiZulu, eng_Latn…':`Find ${title.toLowerCase()}…`,'aria-label':`Search ${title.toLowerCase()}`});
    const list=node('div',{class:'selection-list',role:'group','aria-label':title}),summary=node('span',{class:'note'});
    const continent=node('select',{class:'input','aria-label':'Filter launch languages by continent'},node('option',{value:''},'All continents'),[...new Set(items.flatMap(item=>values(item.continents)))].sort().map(value=>node('option',{value},value)));
    function matching(){return items.filter(item=>normalize([item.name,item.id,...values(item.aliases)].join(' ')).includes(normalize(search.value))&&(!continentFilter||!continent.value||values(item.continents).includes(continent.value)));}
    function unavailable(){return false;}
    function draw(){
      const matches=matching();summary.textContent=`${count(matches.length)} shown · ${count(selected[key].size)} selected`;
      replace(list,...matches.map(item=>{
        const disabled=unavailable(item),input=node('input',{type:'checkbox',value:item.id,checked:selected[key].has(item.id),disabled,onChange:()=>{input.checked?selected[key].add(item.id):selected[key].delete(item.id);refreshSummary();summary.textContent=`${count(matches.length)} shown · ${count(selected[key].size)} selected`;}});
        return node('label',{class:`selection-option ${disabled?'unavailable':''}`},input,node('span',{},node('strong',{},item.name||item.id),node('small',{},item.name&&item.name!==item.id?item.id:''),key==='languages'&&values(item.continents).length?node('small',{},item.continents.join(' · ')):null),key==='tokenizers'?node('span',{class:`badge ${item.cached?'':'badge-warn'}`},item.cached?'Cached':'Needs download'):null);
      }));
      if(!matches.length)list.append(node('p',{class:'note'},'No matches. Try a language name, alias, or code.'));
    }
    search.addEventListener('input',draw);continent.addEventListener('change',draw);
    const controls=node('div',{class:'button-row selection-controls'},action('Select all shown',()=>{for(const item of matching())if(!unavailable(item))selected[key].add(item.id);draw();refreshSummary();},'small'),action('Clear shown',()=>{for(const item of matching())selected[key].delete(item.id);draw();refreshSummary();},'small'),summary);
    draw();return {element:node('section',{class:'card selection-card'},node('h3',{},title),search,continentFilter?continent:null,controls,list),draw};
  }
  const tokenHost=node('div',{}),languageHost=node('div',{}),datasetHost=node('div',{});
  const drawTokenizers=()=>{replace(tokenHost,picker('Tokenizers',tokenizers,'tokenizers').element);};
  drawTokenizers();languageHost.append(picker('Languages',languages,'languages',{continentFilter:true}).element);datasetHost.append(picker('Benchmark datasets',datasets,'datasets').element);
  const customPath=node('input',{class:'input',type:'text',placeholder:'Full path to your tokenizer.json','aria-label':'Absolute path to custom tokenizer JSON',autocomplete:'off',spellcheck:'false'});
  const customName=node('input',{class:'input',type:'text',placeholder:'My tokenizer','aria-label':'Custom tokenizer display name',maxlength:100});
  const customStatus=node('div',{'aria-live':'polite'});
  const register=action('Register local tokenizer',async()=>{
    const path=customPath.value.trim(),name=customName.value.trim();
    if(!path||!name){replace(customStatus,node('p',{class:'note'},'Enter a display name and the full path to a local tokenizer.json file.'));return;}
    if(!/^(\/|[a-z]:[\\/]|\\\\)/i.test(path)){replace(customStatus,node('p',{class:'note'},'Use an absolute path, such as C:\\TokenizerLab\\tokenizer.json or /home/me/tokenizer.json.'));return;}
    register.disabled=true;replace(customStatus,node('p',{class:'note'},'Reading local tokenizer metadata…'));
    try{
      const payload=await api('/api/custom-tokenizers',{path,name}),spec=payload.tokenizer||payload;
      if(!spec.id)throw new Error('The registration response is missing a tokenizer ID.');
      customPaths.set(spec.id,path);
      tokenizers=[...tokenizers.filter(item=>item.id!==spec.id),{...spec,name:spec.name||name,cached:true}];
      selected.tokenizers.add(spec.id);drawTokenizers();refreshSummary();
      replace(customStatus,node('p',{class:'note'},`Registered ${spec.name||name} (${spec.id}). It is selected above.`));
    }catch(error){replace(customStatus,node('div',{class:'callout error'},error.message));}finally{register.disabled=false;}
  });
  replace(root,status,node('p',{class:'note'},'Selections are saved in this browser. Download required assets and datasets while online before preparing an offline package. Nothing downloads automatically. Review a precise file plan, then download only the selected tokenizer and language files. Offline runs use cached assets.'),node('div',{class:'launch-grid'},tokenHost,languageHost,datasetHost),node('section',{class:'card custom-tokenizer-card'},node('h3',{},'Add your own tokenizer'),node('p',{class:'muted'},'Use a Hugging Face-compatible tokenizer.json already on this computer. Registration reads tokenizer data; it does not load remote model code.'),node('div',{class:'control-row'},field('Display name',customName),field('Absolute tokenizer.json path',customPath)),register,customStatus),node('section',{class:'card download-manager'},node('h3',{},'Downloads & storage'),node('p',{class:'muted'},'Choose only the tokenizers, datasets, and languages you need. Each language has its own compressed text file. A reviewed plan lists every selected file before download.'),review,planHost,downloadOutput),node('div',{class:'run-controls'},selectionSummary,submit),errorBox,runOutput,node('p',{class:'note'},'Continent groups overlap. A language may be used in several continents. UTF-8 and UTF-16 fidelity compare exact transport bytes of reconstructed Unicode text.'));
  refreshSummary();
  async function refreshCatalog(){
    const updated=await api('/api/catalog');
    const custom=tokenizers.filter(item=>customPaths.has(item.id));
    tokenizers=[...values(updated.tokenizers).filter(item=>!customPaths.has(item.id)),...custom];
    languages=values(updated.languages).length?updated.languages:languages;
    datasets=values(updated.datasets).length?updated.datasets:datasets;
    drawTokenizers();replace(languageHost,picker('Languages',languages,'languages',{continentFilter:true}).element);replace(datasetHost,picker('Benchmark datasets',datasets,'datasets').element);
  }
  async function reviewPlan(){
    review.disabled=true;replace(planHost,node('p',{class:'note',role:'status'},'Checking selected file identities and sizes…'));
    const key=selectionKey();
    try{
      const plan=await api('/api/downloads/plan',selectionPayload());
      if(key!==selectionKey()){currentPlan=null;replace(planHost,node('p',{class:'note'},'Selection changed. Review the new file plan.'));return;}
      if(!Array.isArray(plan.files))throw new Error('Download plan has no files array.');
      currentPlan=plan;planKey=key;drawPlan();
    }catch(error){currentPlan=null;replace(planHost,node('div',{class:'callout error'},error.message));}finally{refreshSummary();}
  }
  function drawPlan(){
    const plan=currentPlan,files=values(plan.files),cached=files.filter(file=>file.cached===true),needed=files.filter(file=>file.cached!==true),missing=values(plan.missing);
    const downloadBytes=plan.download_bytes,cachedBytes=plan.cached_bytes,totalBytes=plan.total_bytes;
    const download=action(`Download ${count(needed.length)} selected files · ${byteSize(downloadBytes)}`,startDownload,'primary');
    download.disabled=downloadBusy||running||!needed.length||missing.length>0;
    const remove=action(`Remove ${count(cached.length)} selected cached files · ${byteSize(cachedBytes)}`,async()=>{
      remove.disabled=true;replace(downloadOutput,node('p',{class:'note'},'Removing only the selected known cached payloads…'));
      try{
        const result=await api('/api/cache',selectionPayload(),{method:'DELETE'});
        replace(downloadOutput,node('p',{class:'note'},`Removed ${count(Array.isArray(result.removed_files)?result.removed_files.length:result.removed_files)} files · freed ${byteSize(result.freed_bytes)}.`));
        await refreshCatalog();await reviewPlan();
      }catch(error){replace(downloadOutput,node('div',{class:'callout error'},error.message));}finally{remove.disabled=false;}
    },'small danger');
    remove.disabled=downloadBusy||running||!cached.length;
    const fileRows=files.map(file=>node('tr',{},node('td',{},node('strong',{},file.id||file.kind||'File'),node('div',{class:'row-subtitle'},file.kind||'')),node('td',{class:'wrap'},node('code',{},file.path||'Path not supplied')),node('td',{class:'numeric'},byteSize(file.bytes)),node('td',{},node('span',{class:`badge ${file.cached?'':'badge-warn'}`},file.cached?'Cached':'Needs download'))));
    replace(planHost,node('div',{class:'download-totals'},node('div',{},node('strong',{},count(files.length)),node('span',{},'selected files')),node('div',{},node('strong',{},byteSize(downloadBytes)),node('span',{},'to download')),node('div',{},node('strong',{},byteSize(cachedBytes)),node('span',{},'already cached')),node('div',{},node('strong',{},byteSize(totalBytes)),node('span',{},'installed payload'))),node('div',{class:'button-row'},download,remove),plan.network_note?node('div',{class:'callout'},plan.network_note):null,typeof plan.transient_download_bytes==='number'&&plan.transient_download_bytes>0?node('p',{class:'note'},`Transient source download: ${byteSize(plan.transient_download_bytes)}. See the source note for retention and shared costs.`):null,health.offline===true&&needed.length?node('p',{class:'note'},'Benchmark runs are in offline mode. This explicit download requires network access; it only fetches the files listed above. After caching, benchmark runs stay offline.'):null,missing.length?node('div',{class:'callout warning'},node('strong',{},`${count(missing.length)} selected inputs are unavailable.`),node('pre',{},JSON.stringify(missing,null,2))):null,node('details',{class:'details-section'},node('summary',{},`Inspect ${count(files.length)} selected files`),node('div',{class:'table-wrap'},node('table',{},node('thead',{},node('tr',{},['File','Local path','Installed size','State'].map(label=>node('th',{},label)))),node('tbody',{},fileRows)))),node('p',{class:'note'},'File sizes describe installed payloads. Network transfer can differ when a shared source archive or bounded source stream is needed. “Size unknown” does not mean zero. Cache removal affects only known selected payloads.'));
  }
  async function startDownload(){
    if(!currentPlan||planKey!==selectionKey())return;
    downloadBusy=true;refreshSummary();drawPlan();
    const payload=selectionPayload();
    try{
      const job=await api('/api/downloads',payload);
      if(!job.id)throw new Error('The download service returned no job ID.');
      await pollDownload(job.id);
      await refreshCatalog();await reviewPlan();
    }catch(error){replace(downloadOutput,node('div',{class:'callout error'},error.message));}finally{downloadBusy=false;refreshSummary();if(currentPlan)drawPlan();}
  }
  async function pollDownload(id){
    while(root.isConnected){
      const job=await api(`/api/downloads/${encodeURIComponent(id)}`);
      const status=String(job.status).toLowerCase(),finished=['complete','completed','success','succeeded','done','failed','error','cancelled','canceled'].includes(status);
      const progress=job.progress;
      const complete=Number(progress?.completed??progress?.current),total=Number(progress?.total);
      const message=typeof progress==='string'?progress:progress?.message||(Number.isFinite(complete)&&Number.isFinite(total)?`${count(complete)} / ${count(total)} files`:progress?JSON.stringify(progress):'Waiting for download progress');
      const cancel=action('Cancel selected download',async()=>{
        cancel.disabled=true;
        try{await api(`/api/downloads/${encodeURIComponent(id)}/cancel`,{});}catch(error){downloadOutput.append(node('div',{class:'callout error'},error.message));cancel.disabled=false;}
      },'small');
      replace(downloadOutput,node('div',{class:'download-progress'},node('div',{class:'card-header'},node('strong',{},`Download ${job.status||'queued'}`),finished?null:cancel),node('p',{role:'status'},message),Number.isFinite(complete)&&total>0?node('progress',{max:total,value:complete,'aria-label':'Selected download progress'}):null,job.error?node('div',{class:'callout error'},typeof job.error==='string'?job.error:JSON.stringify(job.error)):null));
      if(finished)return;
      await new Promise(resolve=>setTimeout(resolve,1000));
    }
  }
  async function run(){
    if(submit.disabled)return;
    running=true;refreshSummary();if(currentPlan)drawPlan();replace(errorBox,);replace(runOutput,node('p',{class:'muted',role:'status'},'Starting the local run…'));
    try{
      const custom=[...selected.tokenizers].filter(id=>customPaths.has(id));
      const payload={tokenizers:[...selected.tokenizers].filter(id=>!customPaths.has(id)),languages:[...selected.languages],datasets:[...selected.datasets],custom_tokenizers:custom.map(id=>customPaths.get(id))};
      const job=await api('/api/runs',payload);
      if(!job.id)throw new Error('The local service returned no run ID.');
      await poll(job.id);
    }catch(error){replace(errorBox,node('div',{class:'callout error'},error.message));}finally{running=false;refreshSummary();if(currentPlan)drawPlan();}
  }
  async function poll(id){
    while(root.isConnected){
      const job=await api(`/api/runs/${encodeURIComponent(id)}`);
      const complete=['complete','completed','success','succeeded','done'].includes(String(job.status).toLowerCase()),failed=['failed','error','cancelled'].includes(String(job.status).toLowerCase());
      const progress=job.progress;
      const done=Number(progress?.completed??progress?.current),total=Number(progress?.total);
      const progressLabel=typeof progress==='string'?progress:progress?.message || (Number.isFinite(done)&&Number.isFinite(total)?`${count(done)} / ${count(total)} work units`:progress?JSON.stringify(progress):'Waiting for progress');
      const controls=[];
      if(job.result){
        controls.push(action('Save run result as JSON ↓',()=>{
          const objectUrl=URL.createObjectURL(new Blob([JSON.stringify(job.result,null,2)],{type:'application/json'}));
          const anchor=node('a',{href:objectUrl,download:`tokenizer-lab-${id}.json`});anchor.click();setTimeout(()=>URL.revokeObjectURL(objectUrl),1000);
        },'small'));
        if(job.result.url&&String(job.result.url).startsWith('/'))controls.push(node('a',{class:'button small',href:job.result.url},'Open run results →'));
      }
      replace(runOutput,node('section',{class:'card run-result'},node('div',{class:'card-header'},node('h3',{},complete?'Benchmark completed':failed?'Benchmark stopped':'Benchmark running'),node('span',{class:`badge ${failed?'badge-red':''}`},job.status||'queued')),node('p',{class:'note mono'},`Run ${id}`),node('p',{role:'status'},progressLabel),Number.isFinite(done)&&total>0?node('progress',{max:total,value:done,'aria-label':'Benchmark progress'}):null,job.error?node('div',{class:'callout error'},typeof job.error==='string'?job.error:JSON.stringify(job.error)):null,controls.length?node('div',{class:'button-row'},controls):null,job.result?node('details',{},node('summary',{},'Inspect run result'),node('pre',{},JSON.stringify(job.result,null,2))):null));
      if(complete||failed)return;
      await new Promise(resolve=>setTimeout(resolve,1500));
    }
  }
}
const standalone=document.querySelector('[data-local-launcher]');
if(standalone)mountLocalLauncher(standalone);
