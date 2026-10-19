const REPO = 'https://github.com/godkdf/Shinning-Tokens';
const main = document.querySelector('#main');
const nav = document.querySelector('#navigation');
const menuButton = document.querySelector('.mobile-menu');
const labels = {overview:'Overview',compare:'Compare tokenizers',languages:'Languages',vocabulary:'Vocabulary explorer',numbers:'Numbers & code',convert:'Convert sequences',experiments:'Experiments',research:'Research library',reproduce:'Reproduce & contribute'};
const state = {hub:null,examples:[],methods:[],errors:{},vocab:new Map(),backend:false};
let routeVersion = 0;

// All dataset strings are inserted as text nodes, including token surfaces.
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null || value === false) continue;
    if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === 'value') node.value = value;
    else if (key === 'checked') node.checked = value;
    else if (key === 'disabled') node.disabled = value;
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  return node;
}
const text = (value) => value == null ? 'Not measured' : typeof value === 'object' ? JSON.stringify(value) : String(value);
const num = value => typeof value === 'number' && Number.isFinite(value);
const fmt = (value, digits = 0) => num(value) ? value.toLocaleString('en-US',{maximumFractionDigits:digits,minimumFractionDigits:digits}) : '—';
const ratio = (value, digits = 1) => num(value) ? `${fmt(value * 100,digits)}%` : '—';
const size = value => !num(value) ? '—' : value >= 1e9 ? `${fmt(value / 1e9,2)} GB` : value >= 1e6 ? `${fmt(value / 1e6,2)} MB` : value >= 1e3 ? `${fmt(value / 1e3,1)} kB` : `${fmt(value)} B`;
const list = value => Array.isArray(value) ? value : value == null ? [] : [value];
const titleCase = value => String(value).replaceAll('_',' ').replace(/\b\w/g, char => char.toUpperCase());
const profiles = () => state.hub?.profiles || [];
const profile = id => profiles().find(item => item.id === id);
const name = id => profile(id)?.name || id || 'Unknown tokenizer';
const escapeSpace = value => String(value).replaceAll('\r','␍').replaceAll('\n','↵').replaceAll('\t','⇥');
const routeHref = (route, params = {}) => `#${route}${Object.keys(params).length ? `?${new URLSearchParams(params)}` : ''}`;
function url(value) {
  if (!value) return null;
  try {const result = new URL(value, location.href); return ['http:','https:'].includes(result.protocol) ? result.href : null;} catch {return null;}
}
function link(label, href, attrs = {}) { return el('a',{href:url(href) || '#reproduce',...attrs},label); }
function docsLink(path, label = 'Read the method ↗') {
  return link(label,path?.startsWith('http') ? path : `${REPO}/blob/main/${path || 'BENCHMARK.md'}`,{target:'_blank',rel:'noopener noreferrer'});
}
function badge(label, kind = '') { return el('span',{class:`badge ${kind ? `badge-${kind}` : ''}`},label); }
function button(label, action, kind = '') { return el('button',{class:`button ${kind}`,type:'button',onClick:action},label); }
function callout(message, kind = '') { return el('div',{class:`callout ${kind}`},message); }
function empty(title, description) { return el('div',{class:'empty'},el('h3',{},title),description ? el('p',{},description) : null); }
function section(title, content, trailing) {
  return el('section',{class:'section'},el('div',{class:'section-heading'},el('h2',{},title),trailing),content);
}
function card(title, children, kicker) {
  return el('div',{class:'card'},kicker ? el('div',{class:'card-kicker'},kicker) : null,title ? el('h3',{},title) : null,children);
}
function heading(eyebrow, title, intro, action) {
  return el('div',{class:'page-header'},el('div',{class:'eyebrow'},eyebrow),el('div',{class:'header-line'},el('div',{},el('h1',{},title),el('p',{class:'page-intro'},intro)),action));
}
function stat(label, value, note) { return el('div',{class:'stat'},el('div',{class:'stat-label'},label),el('div',{class:'stat-value'},value),el('div',{class:'stat-note'},note)); }
function select(label, options, value, callback, className = '') {
  const id = `field-${label.toLowerCase().replace(/[^a-z0-9]+/g,'-')}`;
  const input = el('select',{id,value,onChange:event=>callback(event.target.value)},options.map(option=>el('option',{value:option.value},option.label)));
  return el('div',{class:`field ${className}`},el('label',{for:id},label),input);
}
const tokenizerOptions = () => profiles().map(p=>({value:p.id,label:p.name}));
function table(headers, rows, classes = '') {
  return el('div',{class:'table-wrap'},el('table',{class:classes},el('thead',{},el('tr',{},headers.map(item=>el('th',{scope:'col',class:typeof item === 'object' ? item.class : null},typeof item === 'object' ? item.label : item)))),el('tbody',{},rows)));
}
const cell = (value, classes) => el('td',{class:classes},value);
const ncell = value => cell(value,'numeric');
function sum(items, key) {const values = items.map(row=>row[key]).filter(num);return values.length ? values.reduce((a,b)=>a+b,0) : null;}
function aggregate(id, dataset = '') {
  const rows=(state.hub.benchmarks || []).filter(row=>row.tokenizer===id && (!dataset || row.dataset===dataset));
  const bytes=sum(rows,'utf8_bytes'),tokens=sum(rows,'tokens');
  return {rows,documents:sum(rows,'documents'),bytes,tokens,bpt:tokens>0&&num(bytes)&&!sum(rows,'errors')?bytes/tokens:null,exact:sum(rows,'exact'),canonical_only:sum(rows,'canonical_only'),mismatches:sum(rows,'mismatches'),errors:sum(rows,'errors')};
}
function benchmarkDatasets() {return [...new Set((state.hub.benchmarks || []).map(row=>row.dataset))].sort();}
function datasetOptions() {return [{value:'',label:'All published suites'},...benchmarkDatasets().map(id=>({value:id,label:id}))];}
function defaultDataset() {return benchmarkDatasets().find(id=>/fineweb.*(mixture|pinned|v0|launch)/i.test(id)) || benchmarkDatasets().find(id=>/fineweb/i.test(id)) || '';}
function dataNotice() {
  return el('div',{class:'data-notice'},el('span',{},`Bundle ${state.hub.schema_version || 'unversioned'} · Generated ${state.hub.generated_at ? new Date(state.hub.generated_at).toLocaleDateString('en-US',{year:'numeric',month:'short',day:'numeric'}) : 'date not supplied'}`),link('Download source measurements ↓','./data/hub.json',{download:'hub.json'}));
}
function chart(entries, maxItems = 12, formatter = value=>fmt(value)) {
  const measured=Object.entries(entries || {}).filter(([,v])=>num(v)).sort((a,b)=>b[1]-a[1]);
  if (!measured.length) return empty('No classified measurements','This analysis has not been included in the current bundle.');
  const max=Math.max(...measured.map(([,value])=>value),1);
  return el('div',{},measured.slice(0,maxItems).map(([label,value])=>el('div',{class:'chart-row'},el('span',{class:'chart-label',title:label},titleCase(label)),el('div',{class:'chart-track'},el('div',{class:'chart-bar',style:`width:${Math.max(0,value / max * 100)}%`})),el('span',{class:'chart-value'},formatter(value)))),measured.length>maxItems?el('p',{class:'note'},`${fmt(measured.length-maxItems)} further categories are available in the data download.`):null);
}
function tokenRendering(encoding, limit = 180) {
  if (!encoding) return empty('Encoding not published','This example does not have a saved encoding for this tokenizer.');
  const pieces=encoding.pieces || encoding.tokens_text || [];
  const ids=encoding.ids || [];
  return el('div',{},el('div',{class:'token-stream'},pieces.slice(0,limit).map((piece,index)=>el('span',{class:'token-piece',title:`Token ID ${ids[index] ?? 'unknown'} · ${String(piece)}`},escapeSpace(piece)))),pieces.length>limit?el('p',{class:'note'},`Showing ${limit} of ${fmt(pieces.length)} pieces. The complete sequence is in the download.`):null,el('div',{class:'token-ids'},`IDs: ${ids.length ? ids.join(' · ') : 'Not included'}`),el('div',{class:`fidelity ${encoding.exact===false ? 'fidelity-fail' : encoding.exact===true ? 'fidelity-pass' : ''}`},encoding.exact===true?'✓ Exact text reconstruction':encoding.exact===false?'△ Reconstruction differs from original text':'Reconstruction not checked'));
}
function examplesPanel(ids, choices = state.examples, initial = '') {
  if (!choices.length) return empty('Saved examples are not available',state.errors.examples || 'Generate the example bundle to inspect identical text across tokenizers.');
  if(!initial&&choices.some(x=>x.id==='unicode-diagnostics-v0.1:mixed:1-raw'))initial='unicode-diagnostics-v0.1:mixed:1-raw';
  const output=el('div',{});
  function draw(id) {
    const item=choices.find(example=>example.id===id)||choices[0];
    output.replaceChildren(el('div',{class:'source-text',lang:item.language || ''},item.text),el('div',{class:ids.length>1?'two-column':''},ids.map(tokenizer=>{
      const encoding=item.encodings?.[tokenizer];
      return card(null,[el('div',{class:'example-heading'},el('h3',{},name(tokenizer)),el('span',{class:'metric-inline'},`${fmt(encoding?.tokens ?? encoding?.ids?.length)} tokens · ${fmt(encoding?.bytes_per_token,2)} B/token`)),tokenRendering(encoding)]);
    })));
  }
  draw(initial);
  return el('div',{},el('div',{class:'control-row'},select('Saved text',choices.map(item=>({value:item.id,label:`${item.label || item.id}${item.language ? ` · ${item.language}` : ''}`})),choices.find(x=>x.id===initial)?.id || choices[0].id,draw,'example-picker')),output,el('p',{class:'note'},'Colored pieces show saved tokenizer output; colors are visual separators, not linguistic categories. Hover over a piece to inspect its token ID.'));
}

function overview() {
  const rows=state.hub.benchmarks || [];
  const languages=new Set(rows.filter(row=>row.dataset===defaultDataset()).map(row=>row.language));
  const corpusUnits=new Map();
  for(const row of rows) corpusUnits.set(`${row.dataset}/${row.language}`,Math.max(corpusUnits.get(`${row.dataset}/${row.language}`)||0,row.utf8_bytes||0));
  const controls=profiles().filter(p=>/control|parent/i.test(p.role || ''));
  const primary=profiles().filter(p=>!controls.includes(p));
  main.append(heading('Open tokenizer research','See how text becomes tokens.','Compare the vocabularies, tradeoffs, and multilingual behavior of real tokenizers. Every measurement leads back to text, methods, and reproducible artifacts.',el('a',{class:'button primary',href:'#compare'},'Compare tokenizers',el('span',{'aria-hidden':'true'},'↗'))));
  main.append(el('div',{class:'stats-grid'},stat('Tokenizer profiles',fmt(profiles().length),`${fmt(primary.length)} primary${controls.length?` · ${controls.length} parent control`:''}`),stat('Language / script varieties',fmt(languages.size),'Across the published benchmark suites'),stat('Distinct tokenizer pairs',fmt((state.hub.pairs || []).length),'Vocabulary overlap in both directions'),stat('Text represented',size([...corpusUnits.values()].reduce((a,b)=>a+b,0)),'Counted once per dataset and language')));
  main.append(section('Start with the question',el('div',{class:'three-column'},[
    ['01','How do they compare?','Inspect compression, fidelity, speed, and memory side by side.','#compare','Compare measurements →'],
    ['02','What is in the vocabulary?','Explore scripts, number patterns, code pieces, and shared bytes.','#vocabulary','Open the vocabulary →'],
    ['03','How can we improve them?','Follow vocabulary expansion and cross-tokenizer transfer research.','#experiments','Explore the experiments →']
  ].map(([number,title,description,href,label])=>card(title,[el('p',{class:'muted'},description),el('a',{class:'inline-link',href},label)],number)))));
  const tableRows=profiles().map(p=>{
    const measured=aggregate(p.id,defaultDataset());
    return el('tr',{},cell(el('div',{class:'profile-header'},el('div',{},el('a',{class:'row-name',href:routeHref('compare',{left:p.id})},p.name),el('div',{class:'row-subtitle'},p.components?.model || p.role || 'Tokenizer')))),ncell(fmt(p.vocabulary?.total)),ncell(fmt(p.vocabulary?.special)),ncell(fmt(measured.bpt,3)),cell(p.role?badge(titleCase(p.role),/control|parent/i.test(p.role)?'neutral':''):badge('Reference','neutral')),cell(el('a',{class:'inline-link',href:routeHref('vocabulary',{tokenizer:p.id})},'Inspect →')));
  });
  main.append(section('A common workspace. Different tradeoffs.',table([{label:'Tokenizer'},{label:'Vocabulary IDs',class:'numeric'},{label:'Special IDs',class:'numeric'},{label:'Bytes / token',class:'numeric'},'Role',''],tableRows),el('a',{href:'#compare'},'All measurements ↗')));
  main.append(el('p',{class:'note'},`Bytes/token uses ${defaultDataset() || 'all published suites'}; pooled original bytes divided by emitted tokens. Higher means fewer tokens for the same text. Vocabulary size, fidelity, and processing cost remain separate measurements.`));
  main.append(section('Look inside the same text',examplesPanel(primary.slice(0,2).map(p=>p.id)),link('Example data ↓','./data/examples.json',{download:'examples.json'})));
  main.append(section('Research with a visible paper trail',el('div',{class:'split-panel'},el('div',{},el('div',{class:'card-kicker'},'An open lab, not a single score'),el('p',{class:'panel-lead'},'Understand the gain. Find the regression. Make the next experiment easier.'),el('p',{class:'muted'},'Token efficiency is one observation. Exact reconstruction, language differences, vocabulary cost, and runtime tell the rest of the story.')),el('div',{},el('h3',{},'Build on the result'),el('div',{class:'step'},el('span',{class:'step-number'},'1'),el('div',{},el('h4',{},'Inspect the artifact'),el('p',{},'Pinned tokenizer revisions, exact bytes, and documented corpus provenance.'))),el('div',{class:'step'},el('span',{class:'step-number'},'2'),el('div',{},el('h4',{},'Change one idea'),el('p',{},'Add a vocabulary experiment, a corpus, an analysis, or a converter.'))),el('a',{class:'button',href:'#reproduce'},'Reproduce & contribute →')))));
  main.append(dataNotice());
}

function compare(params) {
  const left=profile(params.get('left'))?.id || profiles()[0]?.id;
  const right=profile(params.get('right'))?.id || profiles().find(p=>p.id!==left)?.id || left;
  const dataset=params.has('dataset')?params.get('dataset'):defaultDataset();
  main.append(heading('Tokenizer comparison','Compare the tradeoffs.','Same text, different representations. Inspect each measurement separately, then follow differences back to the vocabulary and source examples.'));
  const change=(key,value)=>location.hash=routeHref('compare',{left,right,dataset,[key]:value});
  main.append(el('div',{class:'filter-bar'},el('div',{class:'control-row'},select('Tokenizer A',tokenizerOptions(),left,value=>change('left',value)),select('Tokenizer B',tokenizerOptions(),right,value=>change('right',value)),select('Benchmark suite',datasetOptions(),dataset,value=>change('dataset',value)))));
  const a=aggregate(left,dataset),b=aggregate(right,dataset);
  const runtime=id=>(state.hub.runtime || []).find(item=>item.tokenizer===id);
  const metrics=[
    ['Vocabulary IDs',p=>fmt(profile(p)?.vocabulary?.total),'All ordinary and special token IDs.'],
    ['Ordinary / special IDs',p=>`${fmt(profile(p)?.vocabulary?.ordinary)} / ${fmt(profile(p)?.vocabulary?.special)}`,'Special IDs include structural or control tokens.'],
    ['Unique byte pieces',p=>fmt(profile(p)?.vocabulary?.unique_byte_pieces),'Recoverable ordinary byte sequences; duplicates counted once.'],
    ['Bytes per token',p=>fmt((p===left?a:b).bpt,4),'Original UTF-8 bytes ÷ total emitted tokens. Higher means more compact text.'],
    ['Emitted tokens',p=>fmt((p===left?a:b).tokens),'Total across the selected benchmark suite.'],
    ['Exact round trips',p=>{const data=p===left?a:b;return `${fmt(data.exact)} / ${fmt(data.documents)}`;},'Reconstructed original UTF-8 text without a byte changing.'],
    ['Normalization-only changes',p=>fmt((p===left?a:b).canonical_only),'Different original bytes with a canonically equivalent Unicode form.'],
    ['Other mismatches',p=>fmt((p===left?a:b).mismatches),'Differences beyond canonical Unicode equivalence.'],
    ['Encode / decode errors',p=>fmt((p===left?a:b).errors),'Failed inputs are not silently removed.'],
    ['Encoding · MB/s',p=>fmt(runtime(p)?.encode_mbps,2),'Original UTF-8 MB processed per second; recorded implementation and machine.'],
    ['Decoding · MB/s',p=>fmt(runtime(p)?.decode_mbps,2),'Decoded bytes produced per second; check fidelity alongside speed.'],
    ['Loaded memory',p=>size(runtime(p)?.loaded_rss_bytes),'Process resident memory after loading; see memory protocol.'],
    ['Peak memory',p=>size(runtime(p)?.peak_rss_bytes),'Peak process resident memory during the measurement.'],
    ['Load time · seconds',p=>fmt(runtime(p)?.load_seconds,3),'Recorded loading measurement; cache conditions matter.']
  ];
  main.append(table(['Measurement',{label:name(left),class:'numeric'},{label:name(right),class:'numeric'},'What it tells you'],metrics.map(([label,get,help])=>el('tr',{},cell(label),ncell(get(left)),ncell(get(right)),cell(help,'long-cell muted')))));
  if(!a.rows.length || !b.rows.length) main.append(callout('Some corpus measurements are absent for this selection. A dash means not measured; it never means zero.','warning'));
  main.append(el('div',{class:'metric-help'},el('span',{},el('strong',{},'No overall winner. '),'A smaller token count can coexist with a larger vocabulary or altered text.'),el('span',{},el('strong',{},'MB = 1,000,000 bytes. '),'Runtime applies to the recorded hardware and implementation.')));
  const pair=(state.hub.pairs || []).find(p=>p.left===left&&p.right===right || p.left===right&&p.right===left);
  main.append(section('How much vocabulary is shared?',pair?pairPanel(pair,left,right):empty('This pair has no overlap report',left===right?'Choose two different tokenizers.':'Generate the byte-level pair analysis to compare vocabularies.')));
  main.append(section('The same words. Different pieces.',examplesPanel([left,right],state.examples,params.get('example'))));
  main.append(section('Tokenizer identity',el('div',{class:'two-column'},profilePanel(profile(left)),profilePanel(profile(right)))));
  main.append(dataNotice());
}
function pairPanel(pair,left,right) {
  const forward=pair.left===left;
  const covLeft=forward?pair.coverage_left:pair.coverage_right,covRight=forward?pair.coverage_right:pair.coverage_left;
  const leftTotal=profile(left)?.vocabulary?.unique_byte_pieces,rightTotal=profile(right)?.vocabulary?.unique_byte_pieces;
  const leftOnly=num(leftTotal)?leftTotal-pair.shared:0,rightOnly=num(rightTotal)?rightTotal-pair.shared:0;
  const union=pair.union || 1;
  const visual=el('div',{class:'overlap-visual','aria-label':`${fmt(pair.shared)} shared byte pieces`},el('span',{style:`width:${Math.max(0,leftOnly/union*100)}%`,title:`${name(left)} only: ${fmt(leftOnly)}`}),el('span',{style:`width:${Math.max(0,pair.shared/union*100)}%`,title:`Shared: ${fmt(pair.shared)}`}),el('span',{style:`width:${Math.max(0,rightOnly/union*100)}%`,title:`${name(right)} only: ${fmt(rightOnly)}`}));
  return el('div',{},el('div',{class:'two-column'},card('Shared byte pieces',[el('div',{class:'overlap-metric'},fmt(pair.shared)),el('p',{class:'overlap-label'},`${ratio(pair.jaccard)} of the combined distinct byte vocabulary`),visual,el('div',{class:'chart-legend'},el('span',{},`${name(left)} only`),el('span',{},'Shared'),el('span',{},`${name(right)} only`))]),card('Coverage runs in both directions',[el('div',{class:'identity-grid'},el('dt',{},`Of ${name(left)}`),el('dd',{},el('strong',{},ratio(covLeft)),' also occurs in ',name(right)),el('dt',{},`Of ${name(right)}`),el('dd',{},el('strong',{},ratio(covRight)),' also occurs in ',name(left))),el('p',{class:'note'},'Byte identity is a vocabulary property. Shared pieces can still be selected differently in real sentences. Special tokens are excluded.')])),pair.category_overlap&&Object.keys(pair.category_overlap).length?el('details',{class:'details-section'},el('summary',{},'Shared pieces by category'),el('div',{},typeof Object.values(pair.category_overlap)[0]==='number'?chart(pair.category_overlap,30):el('pre',{},JSON.stringify(pair.category_overlap,null,2)))):null,el('p',{class:'note'},'Native vocabulary markers are interpreted by tokenizer-specific byte adapters. Incomplete UTF-8 pieces remain bytes; they are not replaced with the Unicode replacement character.'));
}
function profilePanel(p) {
  if(!p) return empty('Profile unavailable');
  return card(p.name,[p.role?badge(titleCase(p.role),'neutral'):null,el('dl',{class:'identity-grid'},el('dt',{},'Source'),el('dd',{},p.repo?link(p.repo,p.repo.startsWith('http')?p.repo:`https://huggingface.co/${p.repo}`,{target:'_blank',rel:'noopener noreferrer'}):'Local artifact'),el('dt',{},'Revision'),el('dd',{},el('code',{},p.revision || 'Local artifact')),el('dt',{},'SHA-256'),el('dd',{},el('code',{},p.artifact_sha256 || 'Not recorded')),el('dt',{},'Model'),el('dd',{},text(p.components?.model)),el('dt',{},'Normalizer'),el('dd',{},text(p.components?.normalizer)),el('dt',{},'Pre-tokenizer'),el('dd',{},text(p.components?.pre_tokenizer)),el('dt',{},'Decoder'),el('dd',{},text(p.components?.decoder))),p.provenance?el('details',{},el('summary',{},'Provenance & implementation'),el('div',{},el('pre',{},typeof p.provenance==='string'?p.provenance:JSON.stringify(p.provenance,null,2)))):null,el('div',{class:'button-row'},el('a',{class:'button small',href:routeHref('vocabulary',{tokenizer:p.id})},'Explore vocabulary →'),link('Vocabulary data ↓',`./data/vocab/${p.id}.json.gz`,{class:'button small',download:`${p.id}.json.gz`}))]);
}

function evidenceSummary(method) {
  if(typeof method.evidence!=='object')return text(method.evidence);
  return el('div',{},Object.entries(method.evidence).map(([key,value])=>el('p',{class:'note'},el('strong',{},`${titleCase(key)}: `),value.summary || value.status)));
}

function languages(params) {
  const dataset=params.has('dataset')?params.get('dataset'):defaultDataset();
  const metric=params.get('metric') || 'bytes_per_token';
  const search=params.get('search') || '';
  main.append(heading('Multilingual behavior','Language by language.','A pooled average can hide who benefits and who pays the cost. Compare the same saved documents within each language and script variety.'));
  const change=(key,value)=>location.hash=routeHref('languages',{dataset,metric,search,[key]:value});
  const query=el('input',{class:'input',type:'search',placeholder:'Find a language or script…',value:search,'aria-label':'Find a language or script'});
  const output=el('div',{});
  main.append(el('div',{class:'filter-bar'},el('div',{class:'control-row'},select('Benchmark suite',datasetOptions(),dataset,v=>change('dataset',v)),select('Measurement',[{value:'bytes_per_token',label:'Bytes per token'},{value:'tokens_per_kb',label:'Tokens per 1,000 UTF-8 bytes'},{value:'exact_rate',label:'Exact round-trip rate'},{value:'mismatches',label:'Non-exact documents'},{value:'token_p90',label:'90th percentile document tokens'},{value:'token_max',label:'Maximum document tokens'}],metric,v=>change('metric',v)),el('div',{class:'field'},el('label',{},'Language filter'),query))));
  main.append(output);
  function draw() {
    const rows=(state.hub.benchmarks || []).filter(row=>(!dataset || row.dataset===dataset)&&row.language.toLowerCase().includes(query.value.toLowerCase()));
    if(!rows.length){output.replaceChildren(empty('No matching language measurements','Choose another suite or generate the pinned language-mixture benchmark.'));return;}
    const keys=[...new Set(rows.map(row=>`${row.dataset}|||${row.language}`))].sort();
    const metricValue=row=>!row?null:metric==='tokens_per_kb'&&row.utf8_bytes>0?row.tokens/row.utf8_bytes*1000:metric==='exact_rate'&&row.documents>0?row.exact/row.documents:metric==='mismatches'?row.documents-row.exact:row[metric];
    const resultRows=keys.map(key=>{
      const [source,language]=key.split('|||');
      const selected=profiles().map(p=>rows.find(row=>row.tokenizer===p.id&&row.language===language&&row.dataset===source));
      const values=selected.map(metricValue);
      const numericValues=values.filter(num),low=Math.min(...numericValues),high=Math.max(...numericValues);
      return el('tr',{},cell(el('div',{},el('span',{class:'row-name'},language),!dataset?el('div',{class:'row-subtitle'},source):null)),...values.map(value=>{
        const strength=num(value)?(high>low?(value-low)/(high-low):.5):0;
        return el('td',{class:'numeric heat-cell',style:num(value)?`background-color:rgba(80,132,95,${.04+strength*.2})`:null,title:num(value)?'Shading shows relative magnitude within this row, not an overall quality score.':'Not measured'},metric==='exact_rate'?ratio(value,2):fmt(value,['bytes_per_token','tokens_per_kb'].includes(metric)?3:0));
      }),ncell(fmt(selected.find(Boolean)?.documents)),ncell(size(selected.find(Boolean)?.utf8_bytes)));
    });
    output.replaceChildren(el('div',{class:'table-toolbar'},`${fmt(keys.length)} language / dataset rows`,el('span',{},'Shading compares magnitude within a row')),table(['Language / script',...profiles().map(p=>({label:p.name,class:'numeric'})),{label:'Documents',class:'numeric'},{label:'Original bytes',class:'numeric'}],resultRows,'heatmap'),el('p',{class:'note'},'Higher bytes/token means fewer pieces for the same original text. More tokens or more altered documents are not improvements. Shading indicates magnitude, with no automatic “best tokenizer” label.'));
  }
  query.addEventListener('input',draw);draw();
  main.append(section('What exactly was measured?',corporaTable()));
  main.append(dataNotice());
}
function corporaTable() {
  const corpora=state.hub.corpora || [];
  if(!corpora.length) return empty('Corpus metadata is not included','A pinned evaluation must include document identities, sampling rules, source revisions, and byte counts.');
  return table(['Corpus','Purpose / status','Size','Provenance'],corpora.map(c=>el('tr',{},cell(el('div',{},el('strong',{},c.name || c.dataset || c.id),el('div',{class:'row-subtitle'},c.version || c.revision || ''))),cell(el('div',{},c.status?badge(titleCase(c.status),/historic|selection|pending/i.test(c.status)?'warn':'neutral'):null,el('div',{class:'row-subtitle'},c.purpose || c.description || '')),'long-cell'),cell(el('div',{},num(c.documents)?`${fmt(c.documents)} documents`:null,el('div',{class:'row-subtitle'},size(c.actual_utf8_bytes || c.utf8_bytes || c.bytes)))),cell(c.docs_path?docsLink(c.docs_path,'Manifest & method ↗'):c.source?link('Source ↗',c.source,{target:'_blank',rel:'noopener noreferrer'}):el('span',{class:'muted'},'See bundle metadata')))));
}

async function loadVocab(id) {
  if(state.vocab.has(id)) return state.vocab.get(id);
  const response=await fetch(`./data/vocab/${encodeURIComponent(id)}.json.gz`);
  if(!response.ok) throw new Error(`Vocabulary download returned ${response.status}. Generate the vocabulary bundle for ${name(id)}.`);
  const buffer=await response.arrayBuffer();
  let decoded;
  const bytes=new Uint8Array(buffer);
  if(bytes[0]===0x1f&&bytes[1]===0x8b){
    if(!globalThis.DecompressionStream) throw new Error('This browser cannot unpack gzip vocabulary files. Use a current browser or download the file for local inspection.');
    decoded=await new Response(new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
  }else decoded=new TextDecoder().decode(buffer);
  const data=JSON.parse(decoded);
  if(!Array.isArray(data.fields)||!Array.isArray(data.rows)) throw new Error('The vocabulary file does not match the fields/rows schema.');
  const indices=Object.fromEntries(data.fields.map((key,index)=>[key,index]));
  const result={rows:data.rows,indices};
  // Avoid keeping every large vocabulary in memory as the reader explores.
  if(state.vocab.size>=2) state.vocab.delete(state.vocab.keys().next().value);
  state.vocab.set(id,result);return result;
}
async function vocabulary(params,version) {
  const id=profile(params.get('tokenizer'))?.id || profiles()[0]?.id;
  main.append(heading('Vocabulary explorer','The building blocks, exposed.','Search exact native token surfaces, IDs, and byte sequences. Filter by script or structural category, and inspect the pieces a tokenizer can actually emit.'));
  main.append(el('div',{class:'control-row'},select('Tokenizer',tokenizerOptions(),id,value=>location.hash=routeHref('vocabulary',{tokenizer:value})),link('Download vocabulary ↓',`./data/vocab/${id}.json.gz`,{class:'button',download:`${id}.json.gz`})));
  const output=el('div',{},el('div',{class:'initial-loading',role:'status'},el('span',{class:'loading-orbit'}),`Loading ${name(id)} vocabulary…`));
  main.append(output);
  try {
    const data=await loadVocab(id);if(version!==routeVersion)return;
    const {indices:ix,rows}=data;
    const get=(row,key)=>row[ix[key]];
    const scripts=[...new Set(rows.map(row=>get(row,'script')).filter(Boolean))].sort();
    const types=[...new Set(rows.flatMap(row=>list(get(row,'types'))))].sort();
    let page=0,pageSize=50,sort='id',script=params.get('script') || '',category=params.get('type') || '';
    const search=el('input',{type:'search',placeholder:'Text, token ID, or hex bytes…',value:params.get('q') || '','aria-label':'Search token vocabulary'});
    const searchMode=el('select',{'aria-label':'Search by',value:'text'},el('option',{value:'text'},'Token text'),el('option',{value:'id'},'Token ID'),el('option',{value:'hex'},'Byte hex'));
    const scriptSelect=el('select',{'aria-label':'Script',value:script},el('option',{value:''},'All scripts'),scripts.map(item=>el('option',{value:item},titleCase(item))));
    const categorySelect=el('select',{'aria-label':'Token category',value:category},el('option',{value:''},'All categories'),types.map(item=>el('option',{value:item},titleCase(item))));
    const sortSelect=el('select',{'aria-label':'Sort vocabulary',value:'id'},el('option',{value:'id'},'Token ID ↑'),el('option',{value:'surface'},'Token text A–Z'),el('option',{value:'length_desc'},'Longest byte pieces first'),el('option',{value:'length_asc'},'Shortest byte pieces first'));
    const result=el('div',{'aria-live':'polite'});
    const rowDetail=el('div',{});
    const filters=el('div',{class:'filter-bar'},el('div',{class:'control-row'},el('div',{class:'field wide'},el('label',{},'Search'),search),el('div',{class:'field'},el('label',{},'Search by'),searchMode),el('div',{class:'field'},el('label',{},'Script'),scriptSelect),el('div',{class:'field'},el('label',{},'Category'),categorySelect),el('div',{class:'field'},el('label',{},'Sort'),sortSelect)));
    output.replaceChildren(filters,result,rowDetail);
    function draw() {
      const query=search.value;
      const mode=searchMode.value;
      let filtered=rows.filter(row=>{
        if(script&&get(row,'script')!==script)return false;
        if(category&&!list(get(row,'types')).includes(category))return false;
        if(!query)return true;
        if(mode==='id')return String(get(row,'id'))===query.trim();
        if(mode==='hex')return String(get(row,'bytes_hex') || '').toLowerCase().includes(query.replace(/\s|0x/g,'').toLowerCase());
        return String(get(row,'surface')).includes(query);
      });
      if(sort!=='id')filtered.sort((a,b)=>sort==='surface'?String(get(a,'surface')).localeCompare(String(get(b,'surface'))):sort==='length_desc'?(get(b,'byte_length') || 0)-(get(a,'byte_length') || 0):(get(a,'byte_length') || 0)-(get(b,'byte_length') || 0));
      else filtered.sort((a,b)=>get(a,'id')-get(b,'id'));
      const pages=Math.max(1,Math.ceil(filtered.length/pageSize));page=Math.min(page,pages-1);
      const shown=filtered.slice(page*pageSize,(page+1)*pageSize);
      function inspect(row) {
        const surface=String(get(row,'surface'));
        rowDetail.replaceChildren(section(`Token ${get(row,'id')} · ${name(id)}`,card(null,[el('p',{class:'source-text mono'},escapeSpace(surface)),el('dl',{class:'identity-grid'},el('dt',{},'Native surface'),el('dd',{class:'mono'},JSON.stringify(surface)),el('dt',{},'Bytes · hex'),el('dd',{class:'mono'},get(row,'bytes_hex') || 'No ordinary byte representation'),el('dt',{},'Characters'),el('dd',{class:'mono'},Array.from(surface).map(char=>`U+${char.codePointAt(0).toString(16).toUpperCase().padStart(4,'0')}`).join(' ')),el('dt',{},'Byte length'),el('dd',{},text(get(row,'byte_length'))),el('dt',{},'Script'),el('dd',{},titleCase(get(row,'script') || 'Unclassified')),el('dt',{},'Categories'),el('dd',{},list(get(row,'types')).join(', '))),button('Copy token surface',async event=>{await copy(surface,event.currentTarget);},'small')])));
        rowDetail.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'nearest'});
      }
      const pager=el('div',{class:'pagination'},el('span',{},`Page ${fmt(page+1)} of ${fmt(pages)}`),el('select',{'aria-label':'Tokens per page',value:pageSize,onChange:event=>{pageSize=Number(event.target.value);page=0;draw();}},[25,50,100].map(n=>el('option',{value:n},`${n} per page`))),button('← Previous',()=>{page--;draw();},'small'),button('Next →',()=>{page++;draw();},'small'));
      pager.querySelectorAll('button')[0].disabled=page===0;pager.querySelectorAll('button')[1].disabled=page>=pages-1;
      result.replaceChildren(el('div',{class:'table-toolbar'},el('span',{},`${fmt(filtered.length)} matching token IDs · ${fmt(rows.length)} total`),el('span',{},'Only this selected vocabulary is downloaded')),shown.length?table(['Token ID','Native token surface','Byte hex','Script','Categories',''],shown.map(row=>el('tr',{},ncell(fmt(get(row,'id'))),cell(escapeSpace(get(row,'surface')),'token-surface'),cell(get(row,'bytes_hex') ? el('code',{title:get(row,'bytes_hex')},String(get(row,'bytes_hex')).slice(0,40)+(String(get(row,'bytes_hex')).length>40?'…':'')) : el('span',{class:'muted'},'—')),cell(titleCase(get(row,'script') || 'Unclassified')),cell(el('div',{class:'tiny-tags'},list(get(row,'types')).map(type=>el('span',{},type)))),cell(button('Inspect',()=>inspect(row),'small'))))):empty('No matching tokens','Try a shorter exact text search, another script, or another category.'),pager,el('p',{class:'note'},'Text search is case-sensitive. ↵, ⇥, and ␍ make line breaks, tabs, and carriage returns visible; the stored surface is unchanged. Byte hex describes decoded pieces, while native surface preserves tokenizer-specific markers.'));
    }
    let timer;
    search.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(()=>{page=0;draw();},160);});
    searchMode.addEventListener('change',()=>{page=0;draw();});
    scriptSelect.addEventListener('change',()=>{script=scriptSelect.value;page=0;draw();});
    categorySelect.addEventListener('change',()=>{category=categorySelect.value;page=0;draw();});
    sortSelect.addEventListener('change',()=>{sort=sortSelect.value;page=0;draw();});draw();
  }catch(error){if(version===routeVersion)output.replaceChildren(callout(error.message,'error'),link('Download the vocabulary file directly',`./data/vocab/${id}.json.gz`,{class:'button',download:`${id}.json.gz`}));}
}

function numbers(params) {
  const id=profile(params.get('tokenizer'))?.id || profiles()[0]?.id;
  const p=profile(id),counts=p?.vocabulary?.type_counts || {};
  main.append(heading('Structural vocabulary','Numbers, code, and the space between.','Count the available pieces, then inspect what they do on text. A code-associated vocabulary entry is a documented heuristic, not proof that a tokenizer understands code.'));
  main.append(el('div',{class:'control-row'},select('Tokenizer',tokenizerOptions(),id,value=>location.hash=routeHref('numbers',{tokenizer:value})),el('a',{class:'button',href:routeHref('vocabulary',{tokenizer:id})},'Inspect individual pieces →')));
  const categories=Object.keys(counts);
  const groups=[['Numerical pieces',key=>/num|digit|decimal|integer|percent|fraction|scientific/i.test(key),'ASCII and other Unicode digits, numerical patterns, and mixed alphanumeric pieces.'],['Code-associated pieces',key=>/code|operator|bracket|indent|keyword|identifier|markup|escape|punct/i.test(key),'Operators and delimiters, identifier fragments, keywords, and markup under the published rules.'],['Whitespace & Unicode structure',key=>/space|newline|tab|control|combining|emoji|joiner|byte|mixed|invalid/i.test(key),'Spacing, line breaks, combining marks, byte fragments, and other structural pieces.']];
  main.append(el('div',{class:'three-column'},groups.map(([label,match,description])=>card(label,[el('p',{class:'muted'},description),chart(Object.fromEntries(categories.filter(match).map(key=>[key,counts[key]])),12),el('p',{class:'note'},'Categories can overlap. Counts are token IDs, not observed corpus frequencies.')]))));
  main.append(section('Every published category',table(['Category',{label:'Token IDs',class:'numeric'},{label:'Share of vocabulary IDs',class:'numeric'},''],categories.sort().map(key=>el('tr',{},cell(titleCase(key)),ncell(fmt(counts[key])),ncell(p.vocabulary?.total?ratio(counts[key]/p.vocabulary.total,2):'—'),cell(el('a',{class:'inline-link',href:routeHref('vocabulary',{tokenizer:id,type:key})},'Explore →')))))));
  main.append(callout('Script and category counts describe vocabulary contents. Actual performance depends on the segmentation algorithm and text. “if” can occur in code and prose; Common and Inherited Unicode characters do not establish language coverage.'));
  const exampleChoices=state.examples.filter(example=>/code|num|digit|json|math|unicode|space|emoji|structur/i.test(`${example.id} ${example.label} ${example.category || ''}`));
  main.append(section('Put the pieces into context',examplesPanel([id],exampleChoices)));
  const measured=(state.hub.benchmarks || []).filter(row=>row.tokenizer===id&&/code|num|digit|structur|unicode/i.test(row.dataset));
  main.append(section('Measured structural suites',measured.length?table(['Suite','Language',{label:'Documents',class:'numeric'},{label:'Tokens',class:'numeric'},{label:'Bytes/token',class:'numeric'},{label:'Exact',class:'numeric'}],measured.map(row=>el('tr',{},cell(row.dataset),cell(row.language),ncell(fmt(row.documents)),ncell(fmt(row.tokens)),ncell(fmt(row.bytes_per_token,3)),ncell(`${fmt(row.exact)} / ${fmt(row.documents)}`)))):empty('No structural benchmark rows published','Saved examples remain inspectable. Full suite measurements will appear here when included in the generated bundle.')));
  main.append(dataNotice());
}

async function copy(value,buttonNode) {
  try {await navigator.clipboard.writeText(value);if(buttonNode){const old=buttonNode.textContent;buttonNode.textContent='Copied';setTimeout(()=>buttonNode.textContent=old,1600);}}catch{if(buttonNode)buttonNode.textContent='Select and copy text';}
}
function convert(params) {
  const source=profile(params.get('source'))?.id || profiles()[0]?.id;
  const target=profile(params.get('target'))?.id || profiles().find(p=>p.id!==source)?.id || source;
  main.append(heading('Sequence conversion','Same text. A different token space.','Decode a complete source sequence, retokenize with the target, and check the reconstructed text. Sequence conversion preserves the available text; it cannot restore information already removed by source normalization.'));
  const controls=el('div',{class:'control-row'},select('Source tokenizer',tokenizerOptions(),source,value=>location.hash=routeHref('convert',{source:value,target})),select('Target tokenizer',tokenizerOptions(),target,value=>location.hash=routeHref('convert',{source,target:value})));
  main.append(controls);
  const savedOutput=el('div',{});
  const choices=state.examples.filter(example=>example.encodings?.[source]&&example.encodings?.[target]);
  choices.sort((a,b)=>(b.id==='unicode-diagnostics-v0.1:mixed:1-raw')-(a.id==='unicode-diagnostics-v0.1:mixed:1-raw'));
  if(choices.length){
    function draw(id){const example=choices.find(item=>item.id===id)||choices[0];savedOutput.replaceChildren(el('div',{class:'source-text'},example.text),el('div',{class:'two-column'},...[source,target].map(tokenizer=>card(name(tokenizer),[el('span',{class:'metric-inline'},`${fmt(example.encodings[tokenizer].tokens ?? example.encodings[tokenizer].ids.length)} tokens`),tokenRendering(example.encodings[tokenizer])]))),el('p',{class:'note'},'Saved encodings are measured independently from the original text. Use the local converter for source-ID → decode → target-ID conversion, including any source normalization.'));}
    draw(choices[0].id);
    main.append(section('Inspect a saved text example',el('div',{},el('div',{class:'control-row'},select('Saved text',choices.map(example=>({value:example.id,label:example.label || example.id})),choices[0].id,draw)),savedOutput)));
  }else main.append(empty('No compatible saved example','Generate encodings for this source and target pair.'));
  const apiStatus=el('div',{class:'conversion-status'},el('span',{class:'status-dot'}),state.backend?'Local conversion service is connected':'Static explorer · custom text needs the local service');
  const inputMode=el('select',{'aria-label':'Conversion input mode'},el('option',{value:'text'},'UTF-8 text'),el('option',{value:'ids'},'Source token IDs'));
  const input=el('textarea',{class:'input',rows:5,placeholder:'Enter text to encode with the source tokenizer, or switch to source token IDs.','aria-label':'Conversion input'});
  const output=el('div',{class:'conversion-result','aria-live':'polite'});
  const submit=button('Convert sequence →',async()=>{
    submit.disabled=true;output.replaceChildren(el('p',{class:'muted'},'Loading tokenizers and converting the full sequence…'));
    try{
      const payload={source,target};
      if(inputMode.value==='ids'){
        let ids;try{ids=JSON.parse(input.value);}catch{ids=input.value.trim().split(/[\s,]+/).filter(Boolean).map(Number);}
        if(!Array.isArray(ids)||!ids.length||ids.some(value=>!Number.isSafeInteger(value)||value<0))throw new Error('Enter source token IDs as a JSON array or a list of nonnegative integers separated by spaces or commas.');
        payload.ids=ids;
      }else payload.text=input.value;
      const response=await fetch('/api/convert',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      const result=await response.json();if(!response.ok)throw new Error(result.error || `Conversion failed (${response.status}).`);
      const sourceResult=result.source_encoding || result.encodings?.[source] || result.source_result || (typeof result.source==='object'?result.source:null);
      const targetResult=result.target_encoding || result.encodings?.[target] || result.target_result || (typeof result.target==='object'?result.target:null);
      output.replaceChildren(el('div',{class:'two-column'},card(name(source),sourceResult?tokenRendering(sourceResult):el('pre',{},JSON.stringify(result.source_ids || result.ids || [],null,2))),card(name(target),targetResult?tokenRendering(targetResult):el('pre',{},JSON.stringify(result.target_ids || [],null,2)))),result.text!=null||result.decoded!=null?card('Decoded source text',el('p',{class:'source-text'},result.text ?? result.decoded)):null,el('details',{class:'details-section'},el('summary',{},'Inspect complete API result'),el('div',{},el('pre',{},JSON.stringify(result,null,2)))));
    }catch(error){output.replaceChildren(callout(error.message,'error'));}finally{submit.disabled=!state.backend;}
  },'primary');
  submit.disabled=!state.backend;
  main.append(section('Convert your own input',card(null,[apiStatus,el('div',{class:'control-row'},el('div',{class:'field'},el('label',{},'Input type'),inputMode)),input,el('div',{class:'button-row',style:'margin-top:15px'},submit,el('a',{class:'button',href:'#reproduce'},'Run the local service →')),!state.backend?el('p',{class:'note'},'The hosted explorer never invents custom outputs. Run “uv run tokenizer-lab serve”, then open the local address for live conversion.'):null,output])));
  main.append(callout('A single source piece may map to several target pieces, and its neighbors can change the segmentation. A static token-by-token replacement table cannot reliably substitute for whole-sequence retokenization. This tool converts text representations; embedding transfer is a separate research task.'));
  const mapping=(state.hub.mappings || []).find(m=>m.source===source&&m.target===target);
  if(mapping)main.append(section('Inspect the standalone vocabulary map',card(`${name(source)} → ${name(target)}`,[el('p',{},`${fmt(mapping.rows)} source IDs, with byte matching and standalone target encodings.`),table(['Diagnostic status','Source IDs'],Object.entries(mapping.counts).map(([key,value])=>el('tr',{},cell(titleCase(key)),ncell(fmt(value))))),link('Download full mapping · JSON.GZ ↓',`./${mapping.path}`,{class:'button',download:`${source}--${target}.json.gz`}),el('p',{class:'note'},'Every valid standalone conversion is labelled for context dependence. Byte fragments need neighboring pieces; special/control IDs need a model-specific policy.')])));
  main.append(el('p',{class:'note'},el('a',{href:'#research'},'Explore embedding initialization and cross-tokenizer distillation →')));
}

function experiments() {
  main.append(heading('Experiments & case studies','A change you can inspect.','Vocabulary expansion, token-space mapping, and diagnostic studies. Follow the parent artifact, hypothesis, observed result, and limits before building on an experiment.'));
  const items=state.hub.experiments || [];
  if(!items.length){main.append(empty('Experiment reports are not in this bundle','The research pipeline can include a case study once its source artifacts and measurements are linked.'));return;}
  for(const [index,item]of items.entries()){
    const parent=item.parent || item.parent_tokenizer,candidate=item.child || item.candidate || item.tokenizer;
    const metricItems=item.metrics || item.results;
    const contents=[el('div',{class:'card-header'},el('div',{},el('div',{class:'experiment-index'},`EXPERIMENT ${String(index+1).padStart(2,'0')}`),el('h2',{},item.title || item.name || item.id)),badge(titleCase(item.status || item.evidence || 'Documented'),'historic selection development'.split(' ').some(v=>String(item.status).toLowerCase().includes(v))?'warn':'neutral')),el('p',{},item.summary || item.description || ''),item.hypothesis?el('div',{class:'quote'},item.hypothesis):null,parent||candidate?el('div',{class:'experiment-route'},parent?name(typeof parent==='object'?parent.id:parent):'Source artifact',' → ',candidate?name(typeof candidate==='object'?candidate.id:candidate):'Modified artifact'):null];
    if(metricItems){
      if(Array.isArray(metricItems))contents.push(table(['Measurement','Value','Context'],metricItems.map(metric=>el('tr',{},cell(metric.label || metric.name),cell(text(metric.value),'numeric'),cell(metric.context || metric.note || '','long-cell')))));
      else contents.push(el('dl',{class:'identity-grid'},Object.entries(metricItems).flatMap(([key,value])=>[el('dt',{},titleCase(key)),el('dd',{},text(value))])));
    }
    if(item.selection_data||item.evaluation_data)contents.push(el('dl',{class:'identity-grid'},el('dt',{},'Selection data'),el('dd',{},text(item.selection_data)),el('dt',{},'Evaluation data'),el('dd',{},text(item.evaluation_data))));
    if(item.limitations)contents.push(callout(el('div',{},el('strong',{},'Interpretation limits'),el('ul',{class:'text-list'},list(item.limitations).map(value=>el('li',{},text(value))))),'warning'));
    if(item.docs_path)contents.push(el('div',{class:'button-row',style:'margin-top:20px'},docsLink(item.docs_path,'Read the full experiment ↗')));
    if(item.paired_results?.length) {
      const rows=item.paired_results.map(r=>el('tr',{},cell(r.dataset),cell(r.language),ncell(fmt(r.documents)),ncell(fmt(r.reduction_percent,3)),ncell(r.ci95.map(v=>fmt(v,3)).join(' to '))));
      contents.push(section('Paired token-count change',table(['Suite','Language','Exact paired documents','Fewer tokens · %','95% interval'],rows)));
      contents.push(el('p',{class:'note'},'Positive = fewer tokens. Negative = more tokens. Exact-pair scope; source-group bootstrap, 1,000 replicates; no multiple-comparison correction.'));
    }
    if(item.historical_note)contents.push(callout(item.historical_note,'warning'));
    main.append(el('section',{class:'section card experiment-card',id:item.id || `experiment-${index}`},contents));
  }
  main.append(callout('Historical exploratory results remain labeled as historical. Measurements on selection data are not held-out evaluation, and intrinsic token-count gains alone do not establish better model learning.'));
  main.append(dataNotice());
}

function research() {
  main.append(heading('Research library','Beyond the token boundary.','Understand what a method transfers: vocabulary pieces, embedding vectors, an embedding-space mapping, or model behavior. Requirements and evidence stay visible beside each method.'));
  main.append(el('div',{class:'split-panel'},el('div',{},el('div',{class:'card-kicker'},'Three different jobs'),el('p',{class:'panel-lead'},'Changing token IDs is different from transferring what a model has learned.')),el('div',{},el('p',{},el('strong',{},'Retokenization '),'preserves available text across vocabularies.'),el('p',{},el('strong',{},'Embedding transfer '),'constructs vectors for a new token space.'),el('p',{},el('strong',{},'Distillation '),'trains a student to match a teacher’s behavior.'))));
  if(!state.methods.length){main.append(section('Methods',empty('Research catalog unavailable',state.errors.methods || 'Generate the cited methods bundle.')));return;}
  const filter=el('input',{class:'input',type:'search',placeholder:'Search methods, requirements, or evidence…','aria-label':'Search research methods'});
  const output=el('div',{class:'two-column section'});
  main.append(el('div',{class:'filter-bar'},el('div',{class:'field'},el('label',{},'Find a method'),filter)),output);
  function draw(){
    const query=filter.value.toLowerCase();
    const items=state.methods.filter(method=>JSON.stringify(method).toLowerCase().includes(query));
    output.replaceChildren(...items.map(method=>el('article',{class:'card method-card'},el('div',{class:'card-header'},el('div',{class:'card-kicker'},method.category || 'Transfer method'),badge(typeof method.evidence==='string'?method.evidence:method.evidence?.status || 'See source','neutral')),el('h2',{},method.name),el('p',{},method.summary),el('div',{class:'method-meta'},el('span',{},'Requires'),el('div',{},Array.isArray(method.requirements)?method.requirements.join(' · '):text(method.requirements)),el('span',{},'Compute'),el('div',{},typeof method.compute==='object' ? el('div',{},el('p',{},method.compute.cpu_feasibility),el('p',{},method.compute.training_required)) : text(method.compute)),el('span',{},'Evidence'),el('div',{},evidenceSummary(method))),method.notes?el('p',{class:'note'},text(method.notes)):null,el('div',{class:'chips'},list(method.sources).map((source,index)=>{
      const href=typeof source==='string'?source:source.url || source.href;
      return link(typeof source==='object'?(source.label || source.title || `Source ${index+1}`):`Source ${index+1}`,href,{class:'chip',target:'_blank',rel:'noopener noreferrer'});
    })),method.docs_path?el('div',{style:'margin-top:17px'},docsLink(method.docs_path,'Method & reproduction notes ↗')):null)));
    if(!items.length)output.append(empty('No matching methods','Try a method name such as WECHSEL, FOCUS, or mapping.'));
  }
  filter.addEventListener('input',draw);draw();
  main.append(callout('Paper-reported evidence, an available implementation, and a result reproduced in this Lab are different evidence levels. The method records identify which one is supplied. Running tokenizers needs no model training; several transfer methods do.'));
}

function codeBlock(command) {return el('div',{class:'code-block'},el('pre',{},command),button('Copy',event=>copy(command,event.currentTarget),'small'));}
function reproduce() {
  const protocol=state.hub.protocol || {};
  main.append(heading('Methods, data & contribution','Make every result reusable.','Run the same analysis, inspect the original observations, and contribute a tokenizer or experiment with a complete paper trail. The published bundle is generated from pinned inputs.'));
  main.append(el('div',{class:'two-column'},card('Start from the repository',[el('p',{class:'muted'},'The evaluation code, configuration, corpus rules, and report generation live together.'),codeBlock(`git clone ${REPO}.git\ncd Shinning-Tokens\nuv sync\nuv run tokenizer-lab doctor`),el('p',{class:'note'},'The environment check reports tokenizer access, local artifacts, and available dependencies before a full run.'),docsLink('README.md','Installation & quickstart ↗')],'01 / Install'),card('Rebuild the research bundle',[el('p',{class:'muted'},'The launch configuration pins tokenizer identities and corpus recipes. Downloads and cached artifacts are separate from Git.'),codeBlock('uv run tokenizer-lab run --config configs/launch.json\nuv run tokenizer-lab build-site\nuv run tokenizer-lab serve'),el('p',{class:'note'},'Open http://localhost:8765 for the explorer and arbitrary-input conversion. The hosted site uses the saved result bundle.'),docsLink('BENCHMARK.md','Read the measurement protocol ↗')],'02 / Reproduce')));
  const commands=state.hub.reproduction?.commands || [];
  if(commands.length)main.append(section('More reproducible commands',el('div',{class:'two-column'},commands.map(command=>card(command.label || 'Command',[command.description?el('p',{class:'muted'},command.description):null,codeBlock(command.command)])))));
  main.append(section('The measurement contract',card(protocol.title || 'Pinned inputs. Explicit measurements.',[protocol.id?el('span',{class:'corpus-chip'},protocol.id):null,el('p',{class:'muted',style:'margin-top:15px'},protocol.description || 'All tokenizers see identical original text. Fidelity compares original and reconstructed UTF-8 bytes. Vocabulary, corpus behavior, and runtime measurements are reported independently.'),el('div',{class:'two-column'},el('div',{},el('h4',{},'Text behavior'),el('ul',{class:'text-list'},el('li',{},'Original UTF-8 bytes, emitted token counts, and exact reconstruction.'),el('li',{},'Canonical-only differences remain failures of byte-exact fidelity.'),el('li',{},'Document-level observations preserve regressions and long tails.'),el('li',{},protocol.weighting || 'Aggregate weighting is explicit; each language remains inspectable.'))),el('div',{},el('h4',{},'Runtime & capacity'),el('ul',{class:'text-list'},el('li',{},'Vocabulary ID count, special IDs, distinct byte pieces, and artifact identity.'),el('li',{},'Encode/decode throughput uses decimal UTF-8 MB per second.'),el('li',{},'Recorded implementation, threads, machine, warm-up, and repeat count.'),el('li',{},'Process memory is distinct from tokenizer file size.')))),docsLink(protocol.docs_path || 'BENCHMARK.md','Full definitions, edge cases & exclusions ↗')])));
  main.append(section('Corpus provenance',corporaTable()));
  main.append(section('Download the evidence',card(null,el('div',{class:'download-list'},[
    ['Measurements & tokenizer profiles','hub.json','Generated aggregates, identities, runtime, overlap, and provenance'],
    ['Saved tokenization examples','examples.json','Original text, IDs, pieces, and reconstruction checks'],
    ['Research methods catalog','methods.json','Cited methods, requirements, and evidence labels'],
    ['Complete per-document observations','observations.jsonl.gz','All observations, input hashes, fidelity failures and counts'],
    ['Frozen Gemma V3.1 inputs','gemma-v31-inputs.tar.gz','Exact candidate and reconstructed control; Gemma 3 agreement and modification notices included'],
    ['Complete source snapshot','source.tar.gz','The initial source release, frozen corpora, configuration, tests and reports']
  ].map(([label,file,description])=>el('div',{class:'download-row'},el('div',{},el('strong',{},label),el('div',{},el('small',{},description))),link(file.endsWith('.tar.gz')?'TAR.GZ ↓':file.endsWith('.jsonl.gz')?'JSONL.GZ ↓':'JSON ↓',`./data/${file}`,{download:file}))),profiles().map(p=>el('div',{class:'download-row'},el('div',{},el('strong',{},`${p.name} vocabulary`),el('div',{},el('small',{},'Exact native surfaces, byte hex, scripts, categories, and IDs'))),link('JSON.GZ ↓',`./data/vocab/${p.id}.json.gz`,{download:`${p.id}.json.gz`})))))));
  const checks=state.hub.checks || {};
  if(Object.keys(checks).length)main.append(section('Published bundle checks',card(null,el('div',{class:'check-grid'},Object.entries(checks).map(([key,value])=>el('div',{class:'check-row'},el('span',{},titleCase(key)),el('span',{class:value===true?'pass':value===false?'fail':'muted'},typeof value==='boolean'?(value?'✓ Passed':'△ Failed'):text(value))))))));
  main.append(section('Contribute something the next person can use',el('div',{class:'three-column'},card('A tokenizer or analysis',[el('p',{class:'muted'},'Supply an adapter, an exact identity, byte semantics, and a reproducible measurement. Add a useful diagnostic even when it reveals no improvement.'),docsLink('CONTRIBUTING.md','Contribution guide ↗')]),card('An experiment',[el('p',{class:'muted'},'Declare the parent, hypothesis, deterministic construction, selection data, evaluation data, gains, regressions, and limitations.'),docsLink('docs/experiments/README.md','Experiment format ↗')]),card('A research reproduction',[el('p',{class:'muted'},'Cite the original method. Record dependencies and compute requirements. Separate published claims from results measured here.'),el('a',{class:'inline-link',href:'#research'},'Browse methods →')]))));
  main.append(dataNotice());
}

function renderRoute() {
  if(!state.hub)return;
  routeVersion++;
  const [rawRoute,rawParams='']=(location.hash.slice(1)||'overview').split('?');
  const route=labels[rawRoute]?rawRoute:'overview';
  const params=new URLSearchParams(rawParams);
  document.querySelectorAll('[data-route]').forEach(anchor=>{const active=anchor.dataset.route===route;anchor.classList.toggle('active',active);if(active)anchor.setAttribute('aria-current','page');else anchor.removeAttribute('aria-current');});
  document.querySelector('#current-page').textContent=labels[route];
  document.title=`${labels[route]} · Tokenizer Lab`;
  nav.classList.remove('open');menuButton.setAttribute('aria-expanded','false');
  main.replaceChildren();
  if(!profiles().length){main.append(heading('Published data required','The lab is ready for its bundle.','Tokenizer profiles and measurements have not been generated yet.'),callout('Run the documented pipeline to create site/data/hub.json. No placeholder scores are displayed.','warning'),el('a',{class:'button',href:REPO},'Open source repository ↗'));return;}
  const handlers={overview,compare,languages,vocabulary,numbers,convert,experiments,research,reproduce};
  Promise.resolve(handlers[route](params,routeVersion)).catch(error=>{main.append(callout(`This view could not render: ${error.message}. The downloadable JSON remains available.`,'error'));console.error(error);});
}
async function fetchJSON(path) {const response=await fetch(path);if(!response.ok)throw new Error(`${path}: HTTP ${response.status}`);return response.json();}
async function bootstrap() {
  const results=await Promise.allSettled([fetchJSON('./data/hub.json'),fetchJSON('./data/examples.json'),fetchJSON('./data/methods.json'),fetch('/api/health').then(async response=>{if(!response.ok)return false;const payload=await response.json();return payload.ok===true || payload.status==='ok' || payload.service==='tokenizer-lab';})]);
  if(results[0].status==='rejected'){
    main.replaceChildren(heading('A reproducible research hub','The data bundle is not available.','Generate the pinned measurements or serve this directory through the local Tokenizer Lab server.'),callout(results[0].reason.message,'error'),codeBlock('uv run tokenizer-lab build-site\nuv run tokenizer-lab serve'),link('Read setup instructions ↗',`${REPO}/blob/main/README.md`,{class:'button'}));return;
  }
  state.hub=results[0].value;
  if(results[1].status==='fulfilled')state.examples=results[1].value.examples || [];else state.errors.examples=results[1].reason.message;
  if(results[2].status==='fulfilled')state.methods=results[2].value.methods || [];else state.errors.methods=results[2].reason.message;
  if(results[3].status==='fulfilled')state.backend=results[3].value;
  document.querySelector('#release-label').textContent=state.hub.protocol?.id || `Bundle ${state.hub.schema_version || 'v0.1'}`;
  renderRoute();
}
menuButton.addEventListener('click',()=>{const open=nav.classList.toggle('open');menuButton.setAttribute('aria-expanded',String(open));});
window.addEventListener('hashchange',()=>{renderRoute();window.scrollTo({top:0,behavior:'instant'});});
bootstrap();
