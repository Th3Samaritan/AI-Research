/* Token Journey · explore an architecture in 3D, one level at a time: model → layer → part. */
(function(){
"use strict";
const TJ=window.TJ,$=s=>document.querySelector(s);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const N=n=>Math.round(n).toLocaleString('en-US');
const fB=b=>b>=1e9?(b/1e9).toFixed(2)+' GB':b>=1e6?(b/1e6).toFixed(1)+' MB':b>=1e3?(b/1e3).toFixed(1)+' KB':N(b)+' B';
const fT=s=>s>=1e-3?(s*1e3).toFixed(2)+' ms':s>=1e-6?(s*1e6).toFixed(1)+' µs':(s*1e9).toFixed(0)+' ns';
const fK=n=>n>=1048576?(n/1048576)+'M':n>=1024?(n/1024)+'K':String(n);
const CTX=[256,1024,4096,8192,16384,32768,65536,131072];
const D=TJ.toyDims;
TJ.css=css;TJ.fB=fB;TJ.fT=fT;

/* ---------- shared helpers other sections reuse ---------- */
function orbit(canvas,cam,target,st,onChange,onClick){
  let drag=null;
  canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,t:st.theta,p:st.phi,moved:false};canvas.setPointerCapture(e.pointerId);if(st.onGrab)st.onGrab();});
  canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>5)drag.moved=true;if(!drag.moved)return;st.theta=drag.t-dx*0.008;st.phi=Math.max(0.15,Math.min(1.52,drag.p-dy*0.006));if(st.lim){st.theta=Math.max(st.lim[0],Math.min(st.lim[1],st.theta));st.phi=Math.max(st.lim[2],Math.min(st.lim[3],st.phi));}onChange();});
  const end=e=>{if(drag&&!drag.moved&&onClick)onClick(e);drag=null;};canvas.addEventListener('pointerup',end);canvas.addEventListener('pointercancel',()=>{drag=null;});
  return ()=>{cam.position.set(target.x+st.r*Math.sin(st.phi)*Math.sin(st.theta),target.y+st.r*Math.cos(st.phi),target.z+st.r*Math.sin(st.phi)*Math.cos(st.theta));cam.lookAt(target);};
}
function visible(el,cb){if(!('IntersectionObserver' in window)){cb(true);return;}new IntersectionObserver(es=>es.forEach(e=>cb(e.isIntersecting)),{threshold:0.02}).observe(el);}
function makeScene(cv,stage,st,target,onClick){
  const renderer=new THREE.WebGLRenderer({canvas:cv,antialias:true,alpha:true});renderer.setPixelRatio(Math.min(2,window.devicePixelRatio||1));
  const scene=new THREE.Scene(),cam=new THREE.PerspectiveCamera(38,1,0.05,500);
  scene.add(new THREE.AmbientLight(0xffffff,0.7));const dl=new THREE.DirectionalLight(0xffffff,0.6);dl.position.set(6,14,9);scene.add(dl);
  const S={renderer,scene,cam,stage,st,target,cv};
  S.render=()=>renderer.render(scene,cam);
  S.place=orbit(cv,cam,target,st,()=>{S.place();S.dirty=true;},onClick);
  S.resize=()=>{const w=stage.clientWidth,h=stage.clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix();S.dirty=true;};
  window.addEventListener('resize',S.resize);
  visible(stage,v=>{S.vis=v;if(v)S.resize();});
  return S;
}
const col=n=>new THREE.Color(css(n));
const lam=(n,o)=>new THREE.MeshLambertMaterial(Object.assign({color:col(n)},o||{}));
function clear(g){while(g.children.length){const c=g.children.pop();if(c.children&&c.children.length)clear(c);if(c.geometry)c.geometry.dispose();if(c.material&&c.material.dispose)c.material.dispose();}}
function project(S,v){const p=v.clone().project(S.cam);return {x:(p.x+1)/2*S.stage.clientWidth,y:(1-p.y)/2*S.stage.clientHeight,ok:p.z<1&&p.z>-1};}
TJ.three={orbit,visible,makeScene,col,lam,clear,project};

/* signed value → colour (copper positive, teal negative) */
function vcol(v,scale){const a=Math.min(1,Math.abs(v)/(scale||2));const base=col('--sunk'),c=col(v>=0?'--copper':'--teal');return base.lerp(c,0.25+0.75*a);}
function strip(vec,label,scale){
  if(!vec)return '';const m=scale||Math.max(1e-6,...[...vec].map(Math.abs));
  const cells=[...vec].map(v=>{const a=Math.min(1,Math.abs(v)/m);return `<i title="${v.toFixed(3)}" style="background:color-mix(in srgb, var(${v>=0?'--copper':'--teal'}) ${Math.round(15+85*a)}%, var(--sunk))"></i>`;}).join('');
  return `<div class="vstrip"><span>${label}</span><div style="grid-template-columns:repeat(${vec.length},1fr)">${cells}</div></div>`;
}
TJ.strip=strip;
const fmtV=v=>'['+[...v].slice(0,6).map(x=>x.toFixed(2)).join(', ')+(v.length>6?', …':'')+']';
const rmsOf=v=>Math.sqrt([...v].reduce((s,x)=>s+x*x,0)/v.length);
/* ---------- state ---------- */
const J={archId:'qwen3-8b',focus:0,toks:[],gen:[],rec:null,W:{},pin:null,choice:null,totalReal:1,
  lv:{level:0,layer:0,part:null},compare:false,cmp:['qwen3-8b','gemma3-4b','deepseek-v32'],
  steps:[],idx:0,t:1,playing:false};
TJ.J=J;
const COL={full:'--copper',local:'--teal',linear:'--good',dense:'--steel',moe:'--gold',norm:'--violet',q:'--copper',k:'--teal',v:'--teal2',emb:'--steel',lm:'--gold',out:'--copper',spine:'--copper'};
const NAME={tok:'Tokenize',emb:'Embed',in:'Enter',norm1:'RMSNorm',norm2:'RMSNorm',qkv:'Q · K · V',rope:'RoPE',compress:'Compress',up:'Absorb',index:'Indexer',score:'Scores',softmax:'Softmax',mix:'Mix values',decay:'Decay',write:'Write',read:'Read',oproj:'Out proj',add1:'+ residual',add2:'+ residual',gu:'Gate · Up',act:'Activation',down:'Down proj',route:'Router',experts:'Experts',combine:'Combine',layer:'Layer',fnorm:'Final norm',lmhead:'LM head',sample:'Sample',attn:'Attention',ffn:'FFN',out:'Output'};

/* ---------- controls ---------- */
function fillControls(){
  $('#jArch').innerHTML=TJ.ARCH.map(a=>`<option value="${a.id}"${a.id===J.archId?' selected':''}>${a.name}</option>`).join('');
  $('#jHw').innerHTML=TJ.HW.map((h,i)=>`<option value="${i}">${h.name}</option>`).join('');
  $('#jFmt').innerHTML=TJ.FMT.map(f=>`<option value="${f.id}"${f.id==='bf16'?' selected':''}>${f.name}</option>`).join('');
  $('#jAct').innerHTML='<option value="">Model default</option>'+Object.entries(TJ.ACTS).map(([k,v])=>`<option value="${k}">${v.name}</option>`).join('');
  $('#jCmpPick').innerHTML=TJ.ARCH.map(a=>`<label class="chk"><input type="checkbox" value="${a.id}"${J.cmp.includes(a.id)?' checked':''}> ${a.short}</label>`).join('');
}
const opt=()=>({bpw:TJ.FMT.find(f=>f.id===$('#jFmt').value).bpw,kvB:2,ctx:CTX[+$('#jCtx').value]});
const hw=()=>TJ.HW[+$('#jHw').value];
const actId=()=>$('#jAct').value||J.a.act;
function actName(){return TJ.ACTS[actId()].name.split(' ·')[0];}

/* ---------- run the toy model ---------- */
function compute(){
  const a=J.a;if(!J.W[a.id])J.W[a.id]=TJ.buildToy(a);
  const toks=allToks();
  J.rec=TJ.forward(a,J.W[a.id],toks,actId());
  const temp=+$('#jTemp').value/10;J.probs=TJ.probs(J.rec.logits,temp);
  if($('#jGreedy').checked){let b=0;J.probs.forEach((p,i)=>{if(p>J.probs[b])b=i;});J.choice=b;}
  else{const r=TJ.mulberry(TJ.hash($('#jPrompt').value)+toks.length*7919);let u=r(),c=0;J.choice=J.probs.length-1;for(let i=0;i<J.probs.length;i++){c+=J.probs[i];if(u<=c){J.choice=i;break;}}}
}
function allToks(){return J.toks.concat(J.gen);}
function layerStages(i){
  const a=J.a,k=TJ.kind(a,i),s=['in','norm1'];
  if(k.mix==='linear')s.push('qkv','decay','write','read','oproj');
  else if(a.mla){s.push('qkv','compress','rope','up');if(a.dsa)s.push('index');s.push('score','softmax','mix','oproj');}
  else s.push('qkv','rope','score','softmax','mix','oproj');
  s.push('add1','norm2');
  if(k.ffn==='moe')s.push('route','experts','combine');else s.push('gu','act','down');
  s.push('add2');
  return s.map(kind=>({kind,layer:i}));
}

/* ---------- descriptions ---------- */
function mixColor(a,k){return k.mix==='linear'?COL.linear:k.mix==='local'?COL.local:COL.full;}
function mixName(a,k){
  if(k.mix==='linear')return 'Gated DeltaNet (linear attention)';
  if(a.dsa)return 'MLA + DeepSeek Sparse Attention';
  if(a.mla)return 'Multi-head latent attention (MLA)';
  if(k.mix==='local')return 'Sliding-window attention (local)';
  if(a.attn==='hybrid'||a.attn==='swa')return 'Full attention (global)';
  return a.attn==='mha'?'Multi-head attention':a.attn==='mqa'?'Multi-query attention':'Grouped-query attention';
}
function mixShort(a,k){return k.mix==='linear'?'DeltaNet':k.mix==='local'?'local':a.dsa?'MLA+DSA':a.mla?'MLA':(a.attn==='hybrid'||a.attn==='swa')?'global':a.attn.toUpperCase();}
function ffnName(a,k){return k.ffn==='moe'?`Mixture of experts · ${a.moe.E} experts, top-${a.moe.k}${a.moe.s?' + 1 shared':''}`:`Dense FFN · ${actName()}`;}
function mixSub(a,k){
  if(k.mix==='linear'){const l=a.lin;return `${l.kh} key heads · ${l.vh} value heads · fixed ${l.dk}×${l.dv} state per head`;}
  if(a.mla){const m=a.mla;return `${a.H} heads · caches ${m.kvr}+${m.rope} numbers per token${a.dsa?` · top-${N(a.dsa.topk)} tokens`:''}`;}
  return `${a.H} query heads · ${a.kv} K/V head${a.kv>1?'s':''} · head dim ${a.hd}${k.mix==='local'?` · last ${N(a.window)} tokens`:''}`;
}
function runsOf(a){const r=[];for(let i=0;i<a.L;i++){const k=TJ.kind(a,i),t=mixShort(a,k)+'|'+k.ffn;if(r.length&&r[r.length-1].t===t)r[r.length-1].e=i;else r.push({t,s:i,e:i,k});}return r;}
function patternOf(a){
  for(let p=2;p<=8;p++){let ok=true;for(let i=p;i<a.L;i++){const x=TJ.kind(a,i),y=TJ.kind(a,i%p);if(x.mix!==y.mix||x.ffn!==y.ffn){ok=false;break;}}if(ok)return p;}
  return 0;
}

/* ---------- 3D building blocks ---------- */
let S=null,CUR=null;
const tip=()=>$('#jTip');
function level(){return {g:new THREE.Group(),labels:[],picks:[],steps:[],extra:{}};}
function M(color,op,basic){const o={color:col(color),transparent:true,opacity:op==null?1:op};const m=basic?new THREE.MeshBasicMaterial(o):new THREE.MeshLambertMaterial(o);m.userData.base=o.opacity;return m;}
function edges(mesh,op,color){const e=new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry),new THREE.LineBasicMaterial({color:col(color||'--ink'),transparent:true,opacity:op==null?0.22:op}));e.material.userData.base=e.material.opacity;e.userData.isEdge=true;mesh.add(e);return e;}
function bx(L,w,h,d,color,x,y,z,o){o=o||{};const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),M(color,o.op,o.basic));m.position.set(x,y,z);if(o.edge!==false)edges(m,o.edgeOp);if(o.pick){m.userData.pick=o.pick;L.picks.push(m);}L.g.add(m);return m;}
function plate(L,w,h,color,x,y,z,o){
  o=o||{};const m=bx(L,w,h,0.08,color,x,y,z,Object.assign({op:0.85},o));
  const rows=Math.min(o.rows||6,10),cols=Math.min(o.cols||10,16),pts=[];
  for(let r=1;r<rows;r++){const yy=-h/2+r*h/rows;pts.push(-w/2,yy,0.045,w/2,yy,0.045);}
  for(let c=1;c<cols;c++){const xx=-w/2+c*w/cols;pts.push(xx,-h/2,0.045,xx,h/2,0.045);}
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pts,3));
  const ls=new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:col('--ink'),transparent:true,opacity:0.13}));ls.material.userData.base=0.13;ls.userData.isEdge=true;m.add(ls);return m;
}
function vline(L,pts,color,op,dashed){
  const g=new THREE.BufferGeometry().setFromPoints(pts.map(p=>new THREE.Vector3(...p)));
  const m=dashed?new THREE.LineDashedMaterial({color:col(color),dashSize:0.15,gapSize:0.1,transparent:true,opacity:op==null?0.8:op}):new THREE.LineBasicMaterial({color:col(color),transparent:true,opacity:op==null?0.8:op});
  m.userData.base=m.opacity;const l=new THREE.Line(g,m);if(dashed)l.computeLineDistances();L.g.add(l);return l;
}
function arrow(L,pts,color,o){
  o=o||{};const l=vline(L,pts,color||'--muted',o.op==null?0.75:o.op,o.dashed);
  const a=new THREE.Vector3(...pts[pts.length-2]),b=new THREE.Vector3(...pts[pts.length-1]),dir=b.clone().sub(a).normalize();
  const cone=new THREE.Mesh(new THREE.ConeGeometry(0.09,0.24,12),M(color||'--muted',o.op==null?0.85:o.op,true));
  cone.position.copy(b).addScaledVector(dir,-0.12);cone.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir);L.g.add(cone);
  if(o.label){const m=new THREE.Vector3(...pts[o.at==null?Math.floor((pts.length-1)/2):o.at]).lerp(new THREE.Vector3(...pts[(o.at==null?Math.floor((pts.length-1)/2):o.at)+1]),0.5);lab(L,o.label,m,{side:o.side||'r',pri:o.pri==null?1:o.pri,cls:'shape'});}
  return l;
}
function lab(L,html,pos,o){o=o||{};const el=document.createElement('div');el.className='lbl3 s-'+(o.side||'c')+(o.cls?' '+o.cls:'');el.innerHTML=html;$('#jLabels').appendChild(el);const it={el,pos:pos.clone?pos.clone():new THREE.Vector3(...pos),side:o.side||'c',pri:o.pri||0};L.labels.push(it);return it;}
function vec(L,vals,x,y,z,o){
  o=o||{};const n=vals?vals.length:16,w=o.w||3.2,bw=w/n,bars=[];
  for(let i=0;i<n;i++){const m=new THREE.Mesh(new THREE.BoxGeometry(bw*0.78,1,o.d||0.22),M('--copper',1));const px=o.axis==='z'?x:x-w/2+bw*(i+0.5),pz=o.axis==='z'?z-w/2+bw*(i+0.5):z;m.position.set(px,y,pz);if(o.pick){m.userData.pick=o.pick;L.picks.push(m);}L.g.add(m);bars.push(m);}
  const V={bars,y,k:o.k||0.28,set(v,op){bars.forEach((b,i)=>{const val=v?v[i%v.length]:0,h=Math.max(-0.9,Math.min(0.9,val*V.k));b.scale.y=Math.abs(h)+0.02;b.position.y=V.y+h/2;b.material.color.copy(vcol(val,2.2));b.material.opacity=op==null?1:op;b.material.userData.base=b.material.opacity;b.visible=(op==null?1:op)>0.01;});}};
  V.set(vals);return V;
}
function headRow(L,n,x,y,z,color,o){
  o=o||{};const maxShow=o.max||32,show=n>maxShow?16:n,sp=o.sp||0.32,arr=[];const w=(show-1)*sp+(n>maxShow?sp*2:0);
  for(let i=0;i<show;i++){const xi=x-w/2+i*sp+(n>maxShow&&i>=show/2?sp*2:0);arr.push(bx(L,o.bw||0.2,o.h||0.6,o.bd||0.2,color,xi,y+(o.h||0.6)/2,z,{pick:o.pick,edgeOp:0.15}));}
  if(n>maxShow)lab(L,'…',new THREE.Vector3(x-w/2+(show/2)*sp+sp*0.5,y+0.3,z),{side:'c',pri:2});
  return {arr,w,show};
}
/* mini map of the whole model, for orientation inside a layer */
function mini(L,a,x,y0,h,cur){
  const p=h/a.L;
  for(let i=0;i<a.L;i++){const k=TJ.kind(a,i);const m=bx(L,1.3,p*0.7,0.7,i===cur?'--gold':mixColor(a,k),x,y0+i*p+p*0.35,0,{op:i===cur?1:0.55,edge:false,pick:{type:'mini',layer:i,tip:`Layer ${i+1} · ${mixShort(a,k)} + ${k.ffn==='moe'?'MoE':'dense FFN'} · click to open`}});}
  const fr=bx(L,1.6,h+0.3,1.0,'--ink',x,y0+h/2,0,{op:0.04,edgeOp:0.3,pick:{type:'model',tip:'Back to the whole model'}});
  lab(L,`<b>${a.short}</b><br>layer ${cur+1} of ${a.L}`,new THREE.Vector3(x,y0+h+0.35,0),{side:'t',pri:3});
}

/* ---------- LEVEL 0: the model ---------- */
function stack(L,a,x0,o){
  const W=o.W,DZ=Math.max(1.6,W*0.42),p=o.pitch;let y=0;const items={};
  const put=(id,h,color,pick)=>{const m=bx(L,W,h,DZ,color,x0,y+h/2,0,{pick,op:0.92});items[id]={m,y:y+h/2,h};y+=h+0.22;return m;};
  put('tok',0.4,'--steel',{type:'tok',tip:'Tokenizer · click to open'});
  put('emb',0.6,'--steel',{type:'emb',tip:'Embedding table · click to open'});
  const ly0=y+0.1;L.extra.layers=L.extra.layers||{};const lm=[];
  for(let i=0;i<a.L;i++){const k=TJ.kind(a,i),yy=ly0+i*p,hh=p*0.4,tp=`Layer ${i+1} · ${mixName(a,k)} + ${k.ffn==='moe'?'MoE':'dense'} FFN · click to open`;
    const m1=bx(L,W,hh,DZ,mixColor(a,k),x0,yy+hh/2,0,{edgeOp:0.1,pick:{type:'layer',layer:i,arch:a.id,tip:tp}});
    const m2=bx(L,W,hh,DZ,k.ffn==='moe'?COL.moe:COL.dense,x0,yy+hh*1.5,0,{edgeOp:0.1,pick:{type:'layer',layer:i,arch:a.id,tip:tp}});lm.push([m1,m2]);}
  L.extra.layers[a.id]=lm;
  y=ly0+a.L*p+0.15;
  put('fnorm',0.22,COL.norm,{type:'fnorm',tip:'Final RMSNorm'});
  put('lm',0.6,COL.lm,{type:'lm',tip:'LM head · click to open'});
  put('out',0.32,'--copper',{type:'lm',tip:'Softmax → next token · click to open'});
  return {items,ly0,ly1:ly0+a.L*p,top:y,W,DZ,p};
}
function buildModel(L){
  if(J.compare){
    const ids=J.cmp.slice(0,3),as=ids.map(TJ.arch),maxL=Math.max(...as.map(a=>a.L)),p=13/maxL;let x=0,prevW=0;
    as.forEach((a,n)=>{const W=1.4+a.h/1500;if(n)x+=prevW/2+6.5+W/2;prevW=W;const s=stack(L,a,x,{W,pitch:p});const tt=TJ.totals(a);
      lab(L,`<b>${a.short}</b><br>${a.L} layers · width ${N(a.h)}<br>${(tt.stored/1e9).toFixed(1)} B params${a.moe?` (${(tt.active/1e9).toFixed(1)} B active)`:''}`,new THREE.Vector3(x,s.top+0.3,0),{side:'t',pri:5});
      const cnt={};for(let i=0;i<a.L;i++){const k=TJ.kind(a,i),n=mixShort(a,k);cnt[n]=(cnt[n]||0)+1;}const moeN=a.moe?a.L-(a.moe.dense||0):0;
      lab(L,Object.entries(cnt).map(([n,c])=>`${c} × ${n}`).join(' + ')+'<br>'+(a.moe?`${a.moe.dense?a.moe.dense+' dense FFN, ':''}${moeN} MoE (${a.moe.E} experts)`:'all dense FFN'),new THREE.Vector3(x,-0.2,s.DZ/2),{side:'b',pri:3});});
    L.extra.cmpN=as.length;
    return;
  }
  const a=J.a,s=stack(L,a,0,{W:5.2,pitch:Math.min(0.42,13/a.L)}),tt=TJ.totals(a);L.extra.stack=s;
  const R=s.W/2,Z=s.DZ/2;
  lab(L,`<b>${a.name}</b><br>${a.L} layers · width ${N(a.h)} · ${(tt.stored/1e9).toFixed(2)} B parameters${a.moe?`, ${(tt.active/1e9).toFixed(1)} B used per token`:''}`,new THREE.Vector3(0,s.top+0.35,0),{side:'t',pri:6});
  lab(L,`<b>Tokenizer</b> · text → integer IDs (${N(a.V)} entries)`,new THREE.Vector3(R,s.items.tok.y,Z),{side:'r',pri:4});
  lab(L,`<b>Embedding</b> · ID → vector of ${N(a.h)} numbers`,new THREE.Vector3(R,s.items.emb.y,Z),{side:'r',pri:4});
  lab(L,`<b>Final RMSNorm</b>`,new THREE.Vector3(R,s.items.fnorm.y,Z),{side:'r',pri:3});
  lab(L,`<b>LM head</b> · vector → ${N(a.V)} scores${a.tied?' (reuses the embedding)':''}`,new THREE.Vector3(R,s.items.lm.y,Z),{side:'r',pri:4});
  lab(L,`<b>Softmax</b> → probability of each next token`,new THREE.Vector3(R,s.items.out.y,Z),{side:'r',pri:3});
  // brackets + descriptions of the layer pattern
  const bxR=R+0.35,runs=runsOf(a),per=patternOf(a);
  const bracket=(y0,y1)=>vline(L,[[bxR,y0,Z],[bxR+0.2,y0,Z],[bxR+0.2,y1,Z],[bxR,y1,Z]],'--muted',0.8);
  const yOf=i=>s.ly0+i*s.p;
  if(runs.length<=6){runs.forEach(r=>{bracket(yOf(r.s)+0.02,yOf(r.e+1)-s.p*0.15);
    lab(L,`<b>Layers ${r.s+1}${r.e>r.s?'–'+(r.e+1):''}</b> · ${mixName(a,r.k)} + ${r.k.ffn==='moe'?'mixture-of-experts':'dense'} FFN`,new THREE.Vector3(bxR+0.2,(yOf(r.s)+yOf(r.e+1))/2,Z),{side:'r',pri:3});});}
  else{bracket(yOf(0),yOf(a.L)-s.p*0.15);
    const blk=[];for(let i=0;i<per;i++){const k=TJ.kind(a,i);const n=mixShort(a,k);if(blk.length&&blk[blk.length-1].n===n)blk[blk.length-1].c++;else blk.push({n,c:1,k});}
    const rep=Math.floor(a.L/per),rem=a.L%per;
    lab(L,`<b>Layers 1–${a.L}</b> · a ${per}-layer pattern, repeated ${rep}×${rem?` (+${rem})`:''}:<br>${blk.map(b=>`${b.c} × ${mixName(a,b.k)}`).join(' → ')}<br>every layer has a ${TJ.kind(a,0).ffn==='moe'?'MoE':'dense'} FFN`,new THREE.Vector3(bxR+0.2,(yOf(0)+yOf(a.L))/2,Z),{side:'r',pri:3});}
  const stepN=a.L>40?10:a.L>20?5:4;
  for(let i=0;i<a.L;i++)if(i===0||(i+1)%stepN===0||i===a.L-1)lab(L,String(i+1),new THREE.Vector3(-R-0.1,yOf(i)+s.p*0.4,Z),{side:'l',pri:2,cls:'tick'});
  lab(L,'layer',new THREE.Vector3(-R-0.1,yOf(a.L)+0.15,Z),{side:'l',pri:1,cls:'tick'});
  // play path: tokenizer → embedding → every layer → head
  const ball=addBall(L);
  const P=(y)=>new THREE.Vector3(R+0.0,y,Z+0.25);
  L.steps.push({kind:'tok',li:-1,pos:P(s.items.tok.y),dur:1.2,hl:[s.items.tok.m]});
  L.steps.push({kind:'emb',li:-1,pos:P(s.items.emb.y),dur:1.2,hl:[s.items.emb.m]});
  const lms=L.extra.layers[a.id];
  for(let i=0;i<a.L;i++)L.steps.push({kind:'layer',li:i,pos:P(yOf(i)+s.p*0.4),dur:Math.max(0.12,3.2/a.L),hl:lms[i],label:String(i+1)});
  L.steps.push({kind:'fnorm',li:-1,pos:P(s.items.fnorm.y),dur:0.8,hl:[s.items.fnorm.m]});
  L.steps.push({kind:'lmhead',li:-1,pos:P(s.items.lm.y),dur:1.2,hl:[s.items.lm.m]});
  L.steps.push({kind:'sample',li:-1,pos:P(s.items.out.y),dur:1.2,hl:[s.items.out.m]});
  L.ball=ball;
}
function addBall(L){const b=new THREE.Mesh(new THREE.SphereGeometry(0.16,18,12),M('--gold',1,true));b.userData.noFit=true;L.g.add(b);return b;}

/* ---------- LEVEL 1: one layer ---------- */
function buildLayer(L,li){
  const a=J.a,k=TJ.kind(a,li),R=J.rec.layers[li],SX=-4.4,BW=5.2,BD=2.4,BX=0.4;
  let y=0;const rows={};
  const box=(id,h,color,pick)=>{const m=bx(L,BW,h,BD,color,BX,y+h/2,0,{pick,op:0.9});rows[id]={m,y0:y,y1:y+h,yc:y+h/2};y+=h;return m;};
  rows.in={yc:0};y=0.9;
  box('norm1',0.32,COL.norm,{type:'norm1',tip:'RMSNorm before attention'});y+=0.55;
  box('attn',1.9,mixColor(a,k),{type:'attn',open:'attn',tip:'Click to open the attention block'});
  if(a.sandwich){y+=0.3;box('post1',0.26,COL.norm,{type:'norm1',tip:'Gemma post-attention norm'});}
  y+=0.6;rows.add1={yc:y};y+=0.75;
  box('norm2',0.32,COL.norm,{type:'norm2',tip:'RMSNorm before the FFN'});y+=0.55;
  box('ffn',1.9,k.ffn==='moe'?COL.moe:COL.dense,{type:'ffn',open:'ffn',tip:'Click to open the FFN'});
  if(a.sandwich){y+=0.3;box('post2',0.26,COL.norm,{type:'norm2',tip:'Gemma post-FFN norm'});}
  y+=0.6;rows.add2={yc:y};y+=0.9;rows.out={yc:y};
  // residual spine
  const sp=new THREE.Mesh(new THREE.CylinderGeometry(0.08,0.08,y,14),M(COL.spine,0.75));sp.position.set(SX,y/2,0);sp.userData.pick={type:'residual',tip:'The residual stream'};L.picks.push(sp);L.g.add(sp);
  const add=(id,kind)=>{const m=new THREE.Mesh(new THREE.SphereGeometry(0.28,20,14),M('--gold',1));m.position.set(SX,rows[id].yc,0);m.userData.pick={type:kind,tip:'Residual add: x ← x + block(x)'};L.picks.push(m);L.g.add(m);rows[id].m=m;
    lab(L,'<b>+</b> residual add',new THREE.Vector3(SX-0.35,rows[id].yc,0),{side:'l',pri:3});};
  add('add1','add1');add('add2','add2');
  const vin=vec(L,R.xin,SX,0,0,{axis:'z',w:2.2,k:0.22});const vout=vec(L,R.xout,SX,y,0,{axis:'z',w:2.2,k:0.22});
  lab(L,`<b>x</b> from ${li?'layer '+li:'the embedding'} · [1 × ${N(a.h)}]`,new THREE.Vector3(SX-0.4,0,0),{side:'l',pri:4});
  lab(L,`<b>x</b> to ${li<a.L-1?'layer '+(li+2):'the final norm'}`,new THREE.Vector3(SX-0.4,y,0),{side:'l',pri:4});
  // flow arrows: branch off the spine, go up through norm + block, come back to the add
  const L0=BX-BW/2,top1=a.sandwich?rows.post1.y1:rows.attn.y1,top2=a.sandwich?rows.post2.y1:rows.ffn.y1;
  arrow(L,[[SX,rows.norm1.yc,0],[L0,rows.norm1.yc,0]],'--muted',{label:`[1 × ${N(a.h)}]`,side:'t',pri:1});
  arrow(L,[[BX,rows.norm1.y1,0],[BX,rows.attn.y0,0]],'--muted');
  arrow(L,[[BX,top1,0],[BX,rows.add1.yc,0],[SX+0.3,rows.add1.yc,0]],'--muted',{label:'attention output',at:1,side:'t',pri:1});
  arrow(L,[[SX,rows.add1.yc+0.3,0],[SX,rows.norm2.yc-0.05,0]],'--copper',{op:0});
  arrow(L,[[SX,rows.norm2.yc,0],[L0,rows.norm2.yc,0]],'--muted');
  arrow(L,[[BX,rows.norm2.y1,0],[BX,rows.ffn.y0,0]],'--muted');
  arrow(L,[[BX,top2,0],[BX,rows.add2.yc,0],[SX+0.3,rows.add2.yc,0]],'--muted',{label:'FFN output',at:1,side:'t',pri:1});
  // callouts
  const RX=BX+BW/2,Z=BD/2,mp=TJ.mixParams(a,li),fp=TJ.ffnParams(a,li),W=opt().bpw/8;
  lab(L,'<b>RMSNorm</b> · rescale x to unit size',new THREE.Vector3(RX,rows.norm1.yc,Z),{side:'r',pri:3});
  lab(L,`<b>${mixName(a,k)}</b><br>${mixSub(a,k)}<br>${N((mp.qkv+(mp.up||0)+(mp.idx||0)+mp.o)/1e6)} M weights${a.qkNorm&&k.mix!=='linear'?' · QK-Norm':''}${a.attn==='hybrid'&&k.mix==='full'?' · output gate':''}<br><span class="open">click to open ▸</span>`,new THREE.Vector3(RX,rows.attn.yc,Z),{side:'r',pri:5});
  if(a.sandwich){lab(L,'<b>Post-norm</b> (Gemma sandwich)',new THREE.Vector3(RX,rows.post1.yc,Z),{side:'r',pri:2});lab(L,'<b>Post-norm</b>',new THREE.Vector3(RX,rows.post2.yc,Z),{side:'r',pri:2});}
  lab(L,'<b>RMSNorm</b>',new THREE.Vector3(RX,rows.norm2.yc,Z),{side:'r',pri:3});
  lab(L,`<b>${ffnName(a,k)}</b><br>${k.ffn==='moe'?`each expert ${N(a.ffn)} wide · ${N(fp.stored/1e6)} M stored, ${N(fp.active/1e6)} M used`:`${N(a.h)} → ${N((a.moe&&a.moe.denseFfn)||a.ffn)} → ${N(a.h)} · ${N(fp.stored/1e6)} M weights`}<br><span class="open">click to open ▸</span>`,new THREE.Vector3(RX,rows.ffn.yc,Z),{side:'r',pri:5});
  lab(L,'residual stream',new THREE.Vector3(SX,y*0.5,0.3),{side:'l',pri:2,cls:'tick'});
  mini(L,a,-8.4,0.4,y-0.8,li);
  // play
  L.ball=addBall(L);
  const sp1=(id)=>new THREE.Vector3(SX,rows[id].yc,0.35);const bp=(id,dy)=>new THREE.Vector3(BX-BW/2-0.25,rows[id].yc+(dy||0),Z);
  const ra=a.sandwich?rows.post1:rows.attn,rf=a.sandwich?rows.post2:rows.ffn;
  L.steps.push({kind:'in',li,pos:new THREE.Vector3(SX,0.2,0.35),dur:0.8,on:t=>vin.set(R.xin,1)});
  L.steps.push({kind:'norm1',li,pos:bp('norm1'),dur:1.1,hl:[rows.norm1.m]});
  L.steps.push({kind:'attn',li,pos:bp('attn'),dur:1.6,hl:[rows.attn.m]});
  L.steps.push({kind:'add1',li,pos:sp1('add1'),dur:1.1,hl:[rows.add1.m]});
  L.steps.push({kind:'norm2',li,pos:bp('norm2'),dur:1.1,hl:[rows.norm2.m]});
  L.steps.push({kind:'ffn',li,pos:bp('ffn'),dur:1.6,hl:[rows.ffn.m]});
  L.steps.push({kind:'add2',li,pos:sp1('add2'),dur:1.1,hl:[rows.add2.m]});
  L.steps.push({kind:'out',li,pos:new THREE.Vector3(SX,y-0.2,0.35),dur:0.8,on:t=>vout.set(R.xout,t)});
}

/* ---------- LEVEL 2: inside a part ---------- */
function rowY(n){return n*1.95;}
function titleLab(L,html,x,y){lab(L,html,new THREE.Vector3(x,y,0),{side:'t',pri:6});}
function buildAttnStd(L,li){
  const a=J.a,k=TJ.kind(a,li),R=J.rec.layers[li],M2=R.mix,toks=allToks(),T=toks.length,Hq=a.H,Hk=a.kv,grp=Hq/Hk;
  const sp=Hq>16?0.26:0.36,qW=(Hq-1)*sp;
  // input
  const vin=vec(L,R.n1,0,0,0,{w:4,k:0.25,pick:{type:'norm1',tip:'Normalized input x̂'}});
  lab(L,`<b>x̂</b> normalized input · [1 × ${N(a.h)}]`,new THREE.Vector3(2.2,0,0),{side:'r',pri:4});
  // projections
  const y1=rowY(1)+0.2,wq=Math.min(4.2,1.2+qW*0.5),wk=Math.max(0.5,wq*Hk/Hq*(a.attn==='hybrid'?2:1));
  const pq=plate(L,wq,1.0,COL.q,-wq/2-0.2-wk,y1,0,{rows:6,cols:12,pick:{type:'qkv',tip:'W_q'}});
  const pk=plate(L,wk,1.0,COL.k,-wk/2,y1,0,{rows:6,cols:4,pick:{type:'qkv',tip:'W_k'}});
  const pv=plate(L,wk,1.0,COL.v,wk/2+0.2,y1,0,{rows:6,cols:4,pick:{type:'qkv',tip:'W_v'}});
  pq.position.x=-wk-0.3-wq/2;pk.position.x=-wk/2-0.1;pv.position.x=wk/2+0.1;
  lab(L,`W_q · ${N(a.h)} × ${N(Hq*a.hd*(a.attn==='hybrid'?2:1))}`,new THREE.Vector3(pq.position.x,y1-0.62,0),{side:'b',pri:3,cls:'shape'});
  lab(L,`W_k, W_v · ${N(a.h)} × ${N(Hk*a.hd)} each`,new THREE.Vector3(wk*0.5+0.6,y1,0),{side:'r',pri:3,cls:'shape'});
  arrow(L,[[0,0.5,0],[0,y1-0.55,0]],'--muted');
  // heads: query heads in groups, each group above its K/V pair
  const y2=rowY(2)+0.5,qs=[],kvx=[];
  const gw=grp*sp,gap=Hk>1?Math.max(0.5,sp*1.6):0,total=Hk*gw+(Hk-1)*gap;let show=Hq<=32;
  const qn=show?Hq:16,gN=show?Hk:Math.max(1,Math.round(16/grp));
  let x=-total/2;
  for(let g=0;g<Hk&&qs.length<qn;g++){const x0=x;for(let j=0;j<grp&&qs.length<qn;j++){qs.push(bx(L,sp*0.62,0.55,0.2,COL.q,x+sp*0.31,y2+0.28,0.7,{edgeOp:0.15,pick:{type:'qheads',tip:'Query head '+(qs.length+1)}}));x+=sp;}
    const cx=(x0+x-sp)/2+sp*0.31;kvx.push(cx);vline(L,[[x0,y2-0.05,0.7],[x-sp+sp*0.62,y2-0.05,0.7]],'--violet',0.9);vline(L,[[cx,y2-0.05,0.7],[cx,y2-0.05,-0.15]],'--violet',0.6);x+=gap;}
  const ks=[],vs=[];kvx.forEach((cx,g)=>{ks.push(bx(L,0.18,0.55,0.2,COL.k,cx-0.11,y2+0.28,-0.3,{edgeOp:0.15,pick:{type:'cache',s:T-1,g:Math.min(g,M2.kvT-1),tip:'K head '+(g+1)}}));vs.push(bx(L,0.18,0.55,0.2,COL.v,cx+0.11,y2+0.28,-0.3,{edgeOp:0.15,pick:{type:'cache',s:T-1,g:Math.min(g,M2.kvT-1),tip:'V head '+(g+1)}}));});
  arrow(L,[[0,y1+0.55,0],[0,y2-0.15,0]],'--muted',{label:`Q [${Hq} × ${a.hd}] · K, V [${Hk} × ${a.hd}]`,side:'r',pri:2});
  lab(L,`<b>${Hq} query heads</b>${Hk<Hq?` · every ${grp} share one K/V pair (violet)`:' · each has its own K/V'}${a.qkNorm?' · QK-Norm':''} · RoPE on q and k`,new THREE.Vector3(total/2+0.3,y2+0.4,0.7),{side:'r',pri:4});
  if(!show)lab(L,`showing ${qs.length} of ${Hq} heads`,new THREE.Vector3(0,y2+1.0,0.7),{side:'t',pri:2,cls:'tick'});
  // KV cache shelf behind each K/V head
  const shelf=[],dz=Math.min(0.32,4.2/T),zc0=-0.75;const local=k.mix==='local';
  kvx.forEach((cx,g)=>{for(let s=0;s<T;s++){const age=T-1-s,z=zc0-age*dz;if(s===T-1)continue;const out=local&&age>=D.WIN;
    const kc=bx(L,0.18,0.22,dz*0.8,COL.k,cx-0.11,y2+0.15,z,{op:out?0.18:0.9,edgeOp:0.08,pick:{type:'cache',s,g:Math.min(g,M2.kvT-1),tip:`cached K,V of "${toks[s].text}"`}});
    const vc=bx(L,0.18,0.22,dz*0.8,COL.v,cx+0.11,y2+0.15,z,{op:out?0.18:0.9,edgeOp:0.08,pick:{type:'cache',s,g:Math.min(g,M2.kvT-1),tip:`cached K,V of "${toks[s].text}"`}});shelf.push({kc,vc,s,g});}});
  const zEnd=zc0-(T-1)*dz;
  lab(L,`<b>KV cache</b> · one K and V per past token (${T-1} here), ×${a.L} layers`,new THREE.Vector3(kvx[kvx.length-1]+0.4,y2+0.15,zEnd),{side:'r',pri:4});
  if(local){const zw=zc0-(D.WIN-1)*dz;vline(L,[[kvx[0]-0.4,y2-0.25,-0.3],[kvx[0]-0.4,y2-0.25,zw-dz/2]],'--gold',1);lab(L,`window: last ${N(a.window)} tokens (toy: ${D.WIN})`,new THREE.Vector3(kvx[0]-0.45,y2-0.25,(zw-0.3)/2),{side:'l',pri:4});}
  // scores grid
  const y3=rowY(3)+0.9,rowsN=Math.min(Hq,16),cs=Math.min(0.3,4.8/T),tiles=[];
  for(let j=0;j<rowsN;j++)for(let s=0;s<T;s++){const tj=Math.floor(j*D.NQ/rowsN);const m=bx(L,cs*0.9,0.16,0.06,'--gold',-(T-1)*cs/2+s*cs,y3+(rowsN-1-j)*0.18,0.7,{edge:false,pick:{type:'score',tip:'score of head '+(j+1)+' for "'+toks[s].text+'"'}});tiles.push({m,tj,s});}
  for(let s=0;s<T;s++)lab(L,toks[s].text,new THREE.Vector3(-(T-1)*cs/2+s*cs,y3-0.18,0.7),{side:'b',pri:1,cls:'tok'});
  const gTop=y3+rowsN*0.18;
  lab(L,`<b>scores = softmax(q·kᵀ / √${a.hd})</b><br>rows: query heads · columns: tokens it can see`,new THREE.Vector3((T-1)*cs/2+0.3,y3+rowsN*0.09,0.7),{side:'r',pri:4});
  arrow(L,[[0,y2+0.65,0.7],[0,y3-0.45,0.7]],'--muted',{label:'q·k for every cached token',side:'r',pri:1});
  // weighted sum → heads → W_o → output
  const y4=gTop+0.75,ho=headRow(L,Hq,0,y4,0.4,COL.q,{h:0.3,sp:Math.min(sp,0.3),pick:{type:'mix',tip:'head outputs'}});
  lab(L,`<b>Σ weights × V</b> → ${Hq} head outputs, concatenated [1 × ${N(Hq*a.hd)}]`,new THREE.Vector3(ho.w/2+0.3,y4+0.15,0.4),{side:'r',pri:4});
  arrow(L,[[0,gTop+0.05,0.7],[0,y4-0.05,0.5]],'--muted');
  let y5=y4+0.95;
  if(a.attn==='hybrid'){plate(L,2.4,0.3,'--good',0,y5,0.4,{rows:1,cols:8,pick:{type:'oproj',tip:'sigmoid output gate'}});lab(L,'<b>output gate</b> σ(W_g·x̂) × heads',new THREE.Vector3(1.4,y5,0.4),{side:'r',pri:3});y5+=0.6;}
  const po=plate(L,3.2,0.7,COL.q,0,y5+0.3,0.4,{rows:4,cols:10,pick:{type:'oproj',tip:'W_o'}});
  lab(L,`<b>W_o</b> · ${N(Hq*a.hd)} × ${N(a.h)} mixes the heads back`,new THREE.Vector3(1.8,y5+0.3,0.4),{side:'r',pri:4});
  arrow(L,[[0,y4+0.35,0.4],[0,y5-0.08,0.4]],'--muted');
  const y6=y5+1.35,vout=vec(L,R.attn,0,y6,0.4,{w:4,k:0.25,pick:{type:'oproj',tip:'attention output'}});
  lab(L,`<b>attention output</b> → added to the residual stream`,new THREE.Vector3(2.2,y6,0.4),{side:'r',pri:4});
  arrow(L,[[0,y5+0.68,0.4],[0,y6-0.25,0.4]],'--muted');
  titleLab(L,`<b>${mixName(a,k)}</b> · layer ${li+1}`,0,y6+0.9);
  // play steps
  L.ball=addBall(L);const P=(y,z)=>new THREE.Vector3(-0.0,y,z==null?1.1:z);
  const qH=v=>qs.forEach((m,j)=>{m.scale.y=Math.max(0.02,v);m.position.y=y2+0.28*m.scale.y;});
  const newKV=v=>[...ks,...vs].forEach(m=>{m.scale.y=Math.max(0.02,v);m.position.y=y2+0.28*m.scale.y;m.material.emissive=col('--gold');m.material.emissiveIntensity=0.35*v;});
  const sc=(phase,t)=>tiles.forEach(({m,tj,s})=>{const raw=M2.scores[tj][s];if(raw==null){m.material.opacity=0.06;m.material.color.copy(col('--line'));return;}
    const row=M2.scores[tj].filter(v=>v!=null),mx=Math.max(...row),pre=Math.exp(raw-mx),w=M2.w[tj][s]||0;
    const v=phase===0?0:phase===1?pre*t:pre+(w*1.6-pre)*t;m.material.color.copy(new THREE.Color(css('--sunk')).lerp(col('--gold'),Math.min(1,0.15+v)));m.material.opacity=phase===0?0.25:1;});
  const hv=v=>ho.arr.forEach((m,j)=>{const tj=Math.floor(j*D.NQ/Hq),val=Math.hypot(M2.heads[tj*2],M2.heads[tj*2+1]);m.scale.y=Math.max(0.02,v*(0.4+val*0.8));m.position.y=y4+0.15*m.scale.y;});
  L.steps.push({kind:'norm1',li,pos:P(0.3),dur:1.0,hl:vin.bars,on:t=>{}});
  L.steps.push({kind:'qkv',li,pos:P(y1),dur:1.6,hl:[pq,pk,pv],on:t=>{qH(t);newKV(t);}});
  L.steps.push({kind:'rope',li,pos:P(y2+0.4),dur:1.3,hl:[...qs,...ks]});
  L.steps.push({kind:'score',li,pos:P(y3+0.6),dur:1.7,hl:shelf.map(c=>c.kc),on:t=>sc(1,t)});
  L.steps.push({kind:'softmax',li,pos:P(y3+1.2),dur:1.5,hl:tiles.map(t=>t.m),on:t=>sc(2,t)});
  L.steps.push({kind:'mix',li,pos:P(y4+0.2),dur:1.5,hl:shelf.map(c=>c.vc),on:t=>hv(t)});
  L.steps.push({kind:'oproj',li,pos:P(y5+0.3),dur:1.3,hl:[po],on:t=>vout.set(R.attn,t)});
  L.reset=()=>{qH(0);newKV(0);sc(0,0);hv(0);vout.set(R.attn,0);};
}
function buildAttnMLA(L,li){
  const a=J.a,m=a.mla,R=J.rec.layers[li],M2=R.mix,toks=allToks(),T=toks.length;
  const vin=vec(L,R.n1,0,0,0,{w:4,k:0.25,pick:{type:'norm1',tip:'x̂'}});
  lab(L,`<b>x̂</b> normalized input · [1 × ${N(a.h)}]`,new THREE.Vector3(2.2,0,0),{side:'r',pri:4});
  const QX=-2.6,KX=2.4,y1=rowY(1)+0.2;
  const pdq=plate(L,2.0,0.8,COL.q,QX,y1,0,{rows:5,cols:8,pick:{type:'qkv',tip:'W_DQ'}});
  const pdk=plate(L,1.3,0.8,'--violet',KX,y1,0,{rows:5,cols:5,pick:{type:'compress',tip:'W_DKV'}});
  lab(L,`<b>query path</b> W_DQ ${N(a.h)} → ${N(m.qr)}`,new THREE.Vector3(QX-1.1,y1,0),{side:'l',pri:4});
  lab(L,`<b>key/value path</b> W_DKV ${N(a.h)} → ${m.kvr} + ${m.rope}`,new THREE.Vector3(KX+0.75,y1,0),{side:'r',pri:4});
  arrow(L,[[0,0.45,0],[QX,y1-0.45,0]],'--muted');arrow(L,[[0,0.45,0],[KX,y1-0.45,0]],'--muted');
  // latent written to cache; cache shelf runs back in depth
  const y2=rowY(2)+0.4,dz=Math.min(0.34,4.2/T),lat=[];
  for(let s=0;s<T;s++){const z=-(T-1-s)*dz,isNew=s===T-1;
    const c=bx(L,0.9,0.3,dz*0.8,'--violet',KX-0.2,y2,z,{op:0.85,edgeOp:0.1,pick:{type:isNew?'compress':'cache',s,tip:`latent of "${toks[s].text}"`}});
    const r=bx(L,0.22,0.3,dz*0.8,'--gold',KX+0.42,y2,z,{op:0.9,edgeOp:0.1,pick:{type:isNew?'compress':'cache',s,tip:`RoPE key of "${toks[s].text}"`}});lat.push({c,r,s});}
  arrow(L,[[KX,y1+0.45,0],[KX,y2-0.2,0]],'--muted',{label:'c (512) + k_rope (64)',side:'r',pri:2});
  lab(L,`<b>latent cache</b> · ${m.kvr+m.rope} numbers per token (MHA would need ${N(2*a.H*a.hd)})`,new THREE.Vector3(KX+0.6,y2+0.05,-(T-1)*dz),{side:'r',pri:4});
  // query heads with nope/rope parts
  const y3=rowY(3)+0.3,qn=16,qs=[];const sp=0.28;
  for(let j=0;j<qn;j++){const x=QX-((qn-1)*sp)/2+j*sp+(j>=qn/2?0.3:0);qs.push(bx(L,0.16,0.4,0.18,COL.q,x,y3+0.2,0,{edgeOp:0.12,pick:{type:'qkv',tip:'query head'}}));bx(L,0.16,0.18,0.18,'--gold',x,y3+0.52,0,{edgeOp:0.12});}
  lab(L,'…',new THREE.Vector3(QX+0.15,y3+0.3,0),{side:'c',pri:2});
  lab(L,`<b>${a.H} query heads</b> (16 shown) · ${m.nope} content + ${m.rope} RoPE dims`,new THREE.Vector3(QX,y3+0.95,0),{side:'t',pri:4});
  arrow(L,[[QX,y1+0.45,0],[QX,y3-0.1,0]],'--muted',{label:`c_q [${N(m.qr)}] → W_UQ`,side:'l',pri:2});
  const ab=vline(L,[[QX+2.4,y3+0.3,0],[KX-0.7,y2+0.2,0]],'--violet',0.9,true);
  lab(L,'W_UK folded into the query: heads score the latent directly',new THREE.Vector3((QX+2.4+KX-0.7)/2,(y3+y2)/2+0.35,0),{side:'t',pri:3,cls:'shape'});
  // DSA indexer
  let yI=null,ibars=[];
  if(a.dsa){yI=y2+0.65;for(let s=0;s<T;s++){const z=-(T-1-s)*dz;ibars.push(bx(L,0.14,1,dz*0.6,'--steel',KX+1.0,yI,z,{edgeOp:0.1,pick:{type:'indexer',s,tip:`indexer score of "${toks[s].text}"`}}));}
    lab(L,`<b>lightning indexer</b> · ${a.dsa.heads} heads × ${a.dsa.dim}, FP8 · keeps top-${N(a.dsa.topk)} (toy ${D.TOPK})`,new THREE.Vector3(KX+1.2,yI+0.6,0.2),{side:'r',pri:4});}
  // scores
  const y4=rowY(4)+0.6,cs=Math.min(0.3,4.8/T),tiles=[];
  for(let j=0;j<16;j++)for(let s=0;s<T;s++){const tj=Math.floor(j*D.NQ/16);tiles.push({m:bx(L,cs*0.9,0.16,0.06,'--gold',-(T-1)*cs/2+s*cs,y4+(15-j)*0.18,0.5,{edge:false,pick:{type:'score',tip:'score'}}),tj,s});}
  for(let s=0;s<T;s++)lab(L,toks[s].text,new THREE.Vector3(-(T-1)*cs/2+s*cs,y4-0.18,0.5),{side:'b',pri:1,cls:'tok'});
  const gTop=y4+16*0.18;
  lab(L,`<b>scores</b> = q·(W_UK c) + q_rope·k_rope${a.dsa?'<br>only selected tokens get a score':''}`,new THREE.Vector3((T-1)*cs/2+0.3,y4+1.4,0.5),{side:'r',pri:4});
  arrow(L,[[QX,y3+0.65,0],[-0.6,y4-0.4,0.5]],'--muted');
  const y5=gTop+0.7,ho=headRow(L,a.H,0,y5,0.4,COL.q,{h:0.3,sp:0.26,pick:{type:'mix',tip:'head outputs'}});
  lab(L,`<b>Σ weights × latent</b>, then W_UV (folded into W_O)`,new THREE.Vector3(ho.w/2+0.3,y5+0.15,0.4),{side:'r',pri:4});
  arrow(L,[[0,gTop+0.05,0.5],[0,y5-0.05,0.4]],'--muted');
  const po=plate(L,3.2,0.7,COL.q,0,y5+1.1,0.4,{rows:4,cols:10,pick:{type:'oproj',tip:'W_O'}});
  lab(L,`<b>W_O</b> · ${N(a.H*m.v)} × ${N(a.h)}`,new THREE.Vector3(1.8,y5+1.1,0.4),{side:'r',pri:4});
  arrow(L,[[0,y5+0.35,0.4],[0,y5+0.72,0.4]],'--muted');
  const y6=y5+2.2,vout=vec(L,R.attn,0,y6,0.4,{w:4,k:0.25,pick:{type:'oproj',tip:'attention output'}});
  lab(L,'<b>attention output</b> → residual stream',new THREE.Vector3(2.2,y6,0.4),{side:'r',pri:4});
  arrow(L,[[0,y5+1.48,0.4],[0,y6-0.25,0.4]],'--muted');
  titleLab(L,`<b>${mixName(a,TJ.kind(a,li))}</b> · layer ${li+1}`,0,y6+0.9);
  L.ball=addBall(L);const P=(x,y,z)=>new THREE.Vector3(x,y,z==null?0.9:z);
  const sel=M2.sel?new Set(M2.sel):null;
  const newLat=v=>{const n=lat[T-1];[n.c,n.r].forEach(mm=>{mm.scale.y=Math.max(0.02,v);mm.material.emissive=col('--gold');mm.material.emissiveIntensity=0.4*v;});};
  const idx=v=>{if(!a.dsa)return;const mx=Math.max(...M2.idx.map(Math.abs),1e-6);ibars.forEach((b,s)=>{const h=Math.max(0.02,Math.abs(M2.idx[s])/mx*0.9*v);b.scale.y=h;b.position.y=yI+h/2;b.material.color.copy(col(sel.has(s)&&v>0.5?'--gold':'--steel'));});
    lat.forEach(({c,r,s})=>{const o=sel.has(s)?1:1-0.8*v;c.material.opacity=0.85*o;r.material.opacity=0.9*o;});};
  const sc=(ph,t)=>tiles.forEach(({m,tj,s})=>{const raw=M2.scores[tj][s];if(raw==null){m.material.opacity=0.06;return;}const row=M2.scores[tj].filter(v=>v!=null),mx=Math.max(...row),pre=Math.exp(raw-mx),w=M2.w[tj][s]||0;const v=ph===0?0:ph===1?pre*t:pre+(w*1.6-pre)*t;m.material.color.copy(new THREE.Color(css('--sunk')).lerp(col('--gold'),Math.min(1,0.15+v)));m.material.opacity=ph===0?0.25:1;});
  const hv=v=>ho.arr.forEach((mm,j)=>{const tj=Math.floor(j*D.NQ/ho.arr.length),val=Math.hypot(M2.heads[tj*2],M2.heads[tj*2+1]);mm.scale.y=Math.max(0.02,v*(0.4+val*0.8));mm.position.y=y5+0.15*mm.scale.y;});
  L.steps.push({kind:'norm1',li,pos:P(0,0.3),dur:1.0,hl:vin.bars});
  L.steps.push({kind:'qkv',li,pos:P(QX,y1),dur:1.4,hl:[pdq]});
  L.steps.push({kind:'compress',li,pos:P(KX,y2),dur:1.6,hl:[pdk],on:t=>newLat(t)});
  L.steps.push({kind:'rope',li,pos:P(QX,y3+0.5),dur:1.2,hl:qs});
  L.steps.push({kind:'up',li,pos:P((QX+KX)/2,(y3+y2)/2+0.2),dur:1.4,hl:[ab]});
  if(a.dsa)L.steps.push({kind:'index',li,pos:P(KX+1,yI+0.4),dur:1.8,hl:ibars,on:t=>idx(t)});
  L.steps.push({kind:'score',li,pos:P(0,y4+1.2,0.9),dur:1.6,hl:lat.map(l=>l.c),on:t=>sc(1,t)});
  L.steps.push({kind:'softmax',li,pos:P(0,y4+2.2,0.9),dur:1.4,hl:tiles.map(t=>t.m),on:t=>sc(2,t)});
  L.steps.push({kind:'mix',li,pos:P(0,y5+0.2),dur:1.4,on:t=>hv(t)});
  L.steps.push({kind:'oproj',li,pos:P(0,y5+1.1),dur:1.3,hl:[po],on:t=>vout.set(R.attn,t)});
  L.reset=()=>{newLat(0);idx(0);sc(0,0);hv(0);vout.set(R.attn,0);};
}
function buildAttnLin(L,li){
  const a=J.a,l=a.lin,R=J.rec.layers[li],M2=R.mix,toks=allToks(),T=toks.length;
  const vin=vec(L,R.n1,0,0,0,{w:4,k:0.25,pick:{type:'norm1',tip:'x̂'}});
  lab(L,`<b>x̂</b> normalized input · [1 × ${N(a.h)}]`,new THREE.Vector3(2.2,0,0),{side:'r',pri:4});
  const y1=rowY(1)+0.2,specs=[['W_q',l.kh*l.dk,COL.q],['W_k',l.kh*l.dk,COL.k],['W_v',l.vh*l.dv,COL.v],['W_z (gate)',l.vh*l.dv,'--good'],['W_α, W_β',l.vh*2,'--gold']];
  let x=-3.4;const pls=[];specs.forEach(([n,w,c])=>{const ww=Math.max(0.35,w/4096*1.3);const p=plate(L,ww,0.8,c,x+ww/2,y1,0,{rows:5,cols:4,pick:{type:'qkv',tip:n}});pls.push(p);lab(L,n,new THREE.Vector3(x+ww/2,y1-0.5,0),{side:'b',pri:3,cls:'shape'});x+=ww+0.35;});
  arrow(L,[[0,0.45,0],[0,y1-0.45,0]],'--muted');
  lab(L,`${l.kh} key heads × ${l.dk} · ${l.vh} value heads × ${l.dv}`,new THREE.Vector3(x+0.1,y1,0),{side:'r',pri:3,cls:'shape'});
  // state grids
  const y2=rowY(2)+0.9,grids=[],gauges=[],gs=0.34;
  for(let h=0;h<D.LH;h++){const x0=-3+h*1.95;const cells=[];
    for(let r=0;r<D.LD;r++)for(let c=0;c<D.LD;c++)cells.push(bx(L,gs*0.85,1,gs*0.85,'--good',x0+c*gs,y2,-0.5+r*gs,{edgeOp:0.12,pick:{type:'state',tip:'state entry S['+r+','+c+'] of head '+(h+1)}}));
    grids.push(cells);
    const ga=bx(L,0.12,1,0.12,'--teal',x0-0.42,y2,0.9,{edge:false,pick:{type:'decay',tip:'α: how much memory is kept'}}),gb=bx(L,0.12,1,0.12,'--gold',x0-0.24,y2,0.9,{edge:false,pick:{type:'write',tip:'β: write strength'}});gauges.push([ga,gb]);}
  lab(L,`<b>state S</b>, one per head · real ${l.vh} heads × ${l.dk}×${l.dv} fp32 (4 heads at 4×4 shown)`,new THREE.Vector3(4.9,y2-0.3,0),{side:'r',pri:4});
  arrow(L,[[0,y1+0.45,0],[0,y2-0.45,0.9]],'--muted',{label:'q, k (L2-normalized), v, α, β',side:'l',pri:2});
  // past tokens: a fading trace, no cache
  const y2b=y2+1.45,chips=[],cw=Math.min(0.5,6/T);
  for(let s=0;s<T;s++)chips.push(bx(L,cw*0.85,0.16,0.4,'--steel',-3+s*cw,y2b,-1.4,{edgeOp:0.1,pick:{type:'pastTok',s,tip:`"${toks[s].text}" survives only as a trace in S`}}));
  lab(L,'<b>no KV cache</b>: past tokens live on only as a fading trace inside S',new THREE.Vector3(-3+T*cw,y2b,-1.4),{side:'r',pri:3});
  // read, gate, out
  const y3=rowY(4)+0.3,vo=vec(L,M2.heads,0,y3,0.4,{w:4,k:0.6,pick:{type:'read',tip:'o = S·q'}});
  lab(L,'<b>read</b> o = S · q (one small product per head)',new THREE.Vector3(2.2,y3,0.4),{side:'r',pri:4});
  arrow(L,[[0,y2+0.6,0.4],[0,y3-0.3,0.4]],'--muted');
  const pg=plate(L,2.6,0.3,'--good',0,y3+0.9,0.4,{rows:1,cols:8,pick:{type:'read',tip:'gated RMSNorm'}});
  lab(L,'<b>gated RMSNorm</b> · norm(o) × SiLU(z)',new THREE.Vector3(1.5,y3+0.9,0.4),{side:'r',pri:3});
  const po=plate(L,3.2,0.7,COL.q,0,y3+1.8,0.4,{rows:4,cols:10,pick:{type:'oproj',tip:'W_o'}});
  lab(L,`<b>W_o</b> · ${N(l.vh*l.dv)} × ${N(a.h)}`,new THREE.Vector3(1.8,y3+1.8,0.4),{side:'r',pri:4});
  arrow(L,[[0,y3+0.3,0.4],[0,y3+0.72,0.4]],'--muted');arrow(L,[[0,y3+1.07,0.4],[0,y3+1.42,0.4]],'--muted');
  const y4=y3+2.9,vout=vec(L,R.attn,0,y4,0.4,{w:4,k:0.25,pick:{type:'oproj',tip:'output'}});
  lab(L,'<b>output</b> → residual stream',new THREE.Vector3(2.2,y4,0.4),{side:'r',pri:4});
  arrow(L,[[0,y3+2.18,0.4],[0,y4-0.25,0.4]],'--muted');
  titleLab(L,`<b>Gated DeltaNet</b> · layer ${li+1}`,0,y4+0.9);
  L.ball=addBall(L);const P=(x,y,z)=>new THREE.Vector3(x,y,z==null?1.2:z);
  const am=M2.alpha.reduce((s,v)=>s+v,0)/M2.alpha.length;
  const setS=(pd,pw)=>grids.forEach((cells,h)=>cells.forEach((m,i)=>{const v0=M2.Sprev[h][i],v1=M2.Sdecay[h][i],v2=M2.Snew[h][i];const v=pw>0?v1+(v2-v1)*pw:v0+(v1-v0)*pd;const hh=Math.max(0.03,Math.min(1.1,Math.abs(v)*1.2));m.scale.y=hh;m.position.y=y2+(v>=0?hh/2:-hh/2);m.material.color.copy(vcol(v,0.6));}));
  const g=v=>gauges.forEach(([ga,gb],h)=>{ga.scale.y=Math.max(0.02,M2.alpha[h]*v);ga.position.y=y2-0.5+ga.scale.y/2;gb.scale.y=Math.max(0.02,M2.beta[h]*v);gb.position.y=y2-0.5+gb.scale.y/2;});
  const fade=(pd,pw)=>chips.forEach((m,s)=>{const age=T-1-s;m.material.opacity=s===T-1?pw:Math.pow(am,age+pd);m.material.color.copy(col(s===T-1?'--gold':'--steel'));});
  let PD=0,PW=0;
  L.steps.push({kind:'norm1',li,pos:P(0,0.3),dur:1.0,hl:vin.bars});
  L.steps.push({kind:'qkv',li,pos:P(0,y1),dur:1.5,hl:pls,on:t=>g(t)});
  L.steps.push({kind:'decay',li,pos:P(0,y2+0.6),dur:1.7,hl:gauges.map(x=>x[0]),on:t=>{PD=t;setS(PD,PW);fade(PD,PW);}});
  L.steps.push({kind:'write',li,pos:P(1,y2+0.6),dur:1.9,hl:gauges.map(x=>x[1]),on:t=>{PW=t;setS(PD,PW);fade(PD,PW);}});
  L.steps.push({kind:'read',li,pos:P(0,y3+0.2,0.9),dur:1.3,hl:[pg],on:t=>vo.set(M2.heads,t)});
  L.steps.push({kind:'oproj',li,pos:P(0,y3+1.8,0.9),dur:1.3,hl:[po],on:t=>vout.set(R.attn,t)});
  L.reset=()=>{PD=0;PW=0;g(0);setS(0,0);fade(0,0);vo.set(M2.heads,0);vout.set(R.attn,0);};
}
function buildFFN(L,li){
  const a=J.a,k=TJ.kind(a,li),R=J.rec.layers[li],F=R.ffn,A=TJ.ACTS[actId()];
  if(k.ffn==='moe')return buildMoE(L,li);
  const FF=(a.moe&&a.moe.denseFfn)||a.ffn;
  const vin=vec(L,R.n2,0,0,0,{w:4,k:0.25,pick:{type:'norm2',tip:'x̂'}});
  lab(L,`<b>x̂</b> normalized input · [1 × ${N(a.h)}]`,new THREE.Vector3(2.2,0,0),{side:'r',pri:4});
  const GX=A.glu?-2.6:0,UX=A.glu?2.6:0,y1=rowY(1)+0.2;
  const pg=A.glu?plate(L,3.0,0.8,'--violet',GX,y1,0,{rows:5,cols:14,pick:{type:'gu',tip:'W_gate'}}):null;
  const pu=plate(L,3.0,0.8,COL.dense,UX,y1,0,{rows:5,cols:14,pick:{type:'gu',tip:'W_up'}});
  if(pg)lab(L,`W_gate · ${N(a.h)} × ${N(FF)}`,new THREE.Vector3(GX,y1-0.5,0),{side:'b',pri:3,cls:'shape'});
  lab(L,`W_up · ${N(a.h)} × ${N(FF)}`,new THREE.Vector3(UX,y1-0.5,0),{side:'b',pri:3,cls:'shape'});
  if(pg)arrow(L,[[0,0.45,0],[GX,y1-0.45,0]],'--muted');arrow(L,[[0,0.45,0],[UX,y1-0.45,0]],'--muted');
  const y2=rowY(2)+0.4;
  const gv=A.glu?vec(L,F.g,GX,y2,0,{w:3.4,k:0.3,d:0.3,pick:{type:'gate',tip:'gate neurons'}}):null;
  const uv=vec(L,F.u,UX,y2,0,{w:3.4,k:0.3,d:0.3,pick:{type:'up',tip:'up neurons'}});
  if(gv)lab(L,`<b>gate</b> · ${N(FF)} neurons (32 shown)`,new THREE.Vector3(GX,y2+0.75,0),{side:'t',pri:3});
  lab(L,`<b>up</b> · ${N(FF)} neurons (32 shown)`,new THREE.Vector3(UX,y2+0.75,0),{side:'t',pri:3});
  if(pg)arrow(L,[[GX,y1+0.45,0],[GX,y2-0.6,0]],'--muted');arrow(L,[[UX,y1+0.45,0],[UX,y2-0.6,0]],'--muted');
  const y3=rowY(3)+0.6;
  const av=A.glu?vec(L,F.av,GX,y3,0,{w:3.4,k:0.3,d:0.3,pick:{type:'act',tip:'act(gate)'}}):null;
  if(av){lab(L,`<b>act(gate)</b>`,new THREE.Vector3(GX,y3+0.75,0),{side:'t',pri:2});
    arrow(L,[[GX,y2+0.95,0],[GX,y3-0.6,0]],'--muted',{label:actLabel(),side:'l',pri:3});}
  const y4=rowY(4)+0.8,hv=vec(L,F.h,0,y4,0,{w:3.4,k:0.3,d:0.3,pick:{type:'hidden',tip:'h'}});
  const mul=new THREE.Mesh(new THREE.SphereGeometry(0.2,16,10),M('--gold',1));mul.position.set(0,(y3+y4)/2,0);L.g.add(mul);
  lab(L,A.glu?'⊙ multiply':'',new THREE.Vector3(0.3,(y3+y4)/2,0),{side:'r',pri:2});
  if(A.glu){arrow(L,[[GX,y3+0.45,0],[-0.25,(y3+y4)/2,0]],'--muted');arrow(L,[[UX,y2+0.95,0],[UX,y3,0],[0.25,(y3+y4)/2,0]],'--muted');}
  else arrow(L,[[UX,y2+0.95,0],[0,(y3+y4)/2-0.25,0]],'--muted',{label:actLabel(),side:'r',pri:3});
  arrow(L,[[0,(y3+y4)/2+0.22,0],[0,y4-0.6,0]],'--muted');
  lab(L,`<b>h</b> = ${A.glu?'act(gate) ⊙ up':'act(up)'}`,new THREE.Vector3(1.9,y4,0),{side:'r',pri:4});
  const pd=plate(L,3.0,0.8,COL.dense,0,y4+1.3,0,{rows:5,cols:14,pick:{type:'down',tip:'W_down'}});
  lab(L,`<b>W_down</b> · ${N(FF)} × ${N(a.h)}`,new THREE.Vector3(1.6,y4+1.3,0),{side:'r',pri:4});
  arrow(L,[[0,y4+0.5,0],[0,y4+0.88,0]],'--muted');
  const y5=y4+2.5,vout=vec(L,R.ffnOut,0,y5,0,{w:4,k:0.25,pick:{type:'down',tip:'FFN output'}});
  lab(L,'<b>FFN output</b> → residual stream',new THREE.Vector3(2.2,y5,0),{side:'r',pri:4});
  arrow(L,[[0,y4+1.72,0],[0,y5-0.25,0]],'--muted');
  titleLab(L,`<b>Dense FFN · ${A.name.split(' ·')[0]}</b> · layer ${li+1}`,0,y5+0.9);
  L.ball=addBall(L);const P=(x,y)=>new THREE.Vector3(x,y,0.7);
  L.steps.push({kind:'norm2',li,pos:P(0,0.3),dur:1.0,hl:vin.bars});
  L.steps.push({kind:'gu',li,pos:P(0,y2),dur:1.6,hl:[pu].concat(pg?[pg]:[]),on:t=>{if(gv)gv.set(F.g,t);uv.set(F.u,t);}});
  L.steps.push({kind:'act',li,pos:P(GX,y3),dur:1.6,hl:av?av.bars:uv.bars,on:t=>{if(av)av.set(F.av,t);}});
  L.steps.push({kind:'act',li,pos:P(0,y4),dur:1.3,hl:[mul],on:t=>hv.set(F.h,t)});
  L.steps.push({kind:'down',li,pos:P(0,y4+1.3),dur:1.3,hl:[pd],on:t=>vout.set(R.ffnOut,t)});
  L.reset=()=>{if(gv)gv.set(F.g,0);uv.set(F.u,0);if(av)av.set(F.av,0);hv.set(F.h,0);vout.set(R.ffnOut,0);};
}
function actLabel(){const id=actId();return {swiglu:'SiLU(z) = z·σ(z)',geglu:'GELU(z)',reglu:'ReLU(z)',gelu:'GELU(z)',relu:'ReLU(z)',silu:'SiLU(z)'}[id];}
function buildMoE(L,li){
  const a=J.a,m=a.moe,R=J.rec.layers[li],F=R.ffn;
  const vin=vec(L,R.n2,0,0,0,{w:4,k:0.25,pick:{type:'norm2',tip:'x̂'}});
  lab(L,`<b>x̂</b> normalized input · [1 × ${N(a.h)}]`,new THREE.Vector3(2.2,0,0),{side:'r',pri:4});
  const y1=rowY(1)+0.1,pr=plate(L,1.8,0.6,'--violet',0,y1,0,{rows:4,cols:8,pick:{type:'route',tip:'router'}});
  lab(L,`<b>router</b> · ${N(a.h)} × ${m.E} → one score per expert (${m.score})`,new THREE.Vector3(1.0,y1,0),{side:'r',pri:4});
  arrow(L,[[0,0.45,0],[0,y1-0.35,0]],'--muted');
  // expert floor: real count, flat grid
  const cols=16,rowsN=Math.ceil(m.E/cols),ts=0.32,y2=rowY(2)+0.5;
  const r=TJ.mulberry(TJ.hash(allToks().map(t=>t.text).join(' ')+li));const sc=Array.from({length:m.E},()=>r());
  const chosen=sc.map((v,i)=>[v,i]).sort((p,q)=>q[0]-p[0]).slice(0,m.k).map(p=>p[1]);const cs=new Set(chosen);
  const tiles=[],bars=[];
  for(let e=0;e<m.E;e++){const cx=-(cols-1)*ts/2+(e%cols)*ts,cz=-((rowsN-1)*ts)/2+Math.floor(e/cols)*ts-0.6;
    tiles.push(bx(L,ts*0.85,0.12,ts*0.85,'--steel',cx,y2,cz,{edgeOp:0.08,pick:{type:'expert',e,tip:`expert ${e+1}${cs.has(e)?' · chosen for this token':''}`}}));
    bars.push(bx(L,ts*0.4,1,ts*0.4,'--gold',cx,y2+0.06,cz,{edge:false}));}
  lab(L,`<b>${m.E} experts</b>, each a ${N(a.ffn)}-wide ${actName()} FFN<br>bar height = router score · gold = top-${m.k} (which ones is illustrative)`,new THREE.Vector3((cols-1)*ts/2+0.4,y2,-0.6),{side:'r',pri:4});
  let sh=null;if(m.s){sh=bx(L,0.7,0.7,0.7,'--teal',-(cols-1)*ts/2-1.1,y2+0.35,-0.6,{pick:{type:'shared',tip:'shared expert: runs for every token'}});lab(L,'<b>shared expert</b><br>always runs',new THREE.Vector3(-(cols-1)*ts/2-1.5,y2+0.35,-0.6),{side:'l',pri:4});}
  arrow(L,[[0,y1+0.3,0],[0,y2-0.1,0.9]],'--muted');
  // chosen experts lifted into a row
  const y3=rowY(4)+0.2,lift=[];const n=chosen.length+(m.s?1:0),sp=0.75;
  chosen.forEach((e,i)=>{const x=-(n-1)*sp/2+i*sp;lift.push(bx(L,0.55,0.55,0.55,'--gold',x,y3,0,{pick:{type:'experts',tip:`expert ${e+1} runs`}}));lab(L,'E'+(e+1),new THREE.Vector3(x,y3-0.45,0),{side:'b',pri:1,cls:'tok'});});
  if(m.s){const x=-(n-1)*sp/2+chosen.length*sp;lift.push(bx(L,0.55,0.55,0.55,'--teal',x,y3,0,{pick:{type:'shared',tip:'shared expert'}}));lab(L,'shared',new THREE.Vector3(x,y3-0.45,0),{side:'b',pri:1,cls:'tok'});}
  lab(L,`<b>only these ${m.k}${m.s?' + 1':''} run</b> · ${fB(TJ.ffnParams(a,li).active*opt().bpw/8)} read of ${fB(TJ.ffnParams(a,li).stored*opt().bpw/8)} stored`,new THREE.Vector3((n-1)*sp/2+0.5,y3,0),{side:'r',pri:4});
  arrow(L,[[0,y2+0.3,0.9],[0,y3-0.55,0]],'--muted');
  const y4=y3+1.3,cmb=new THREE.Mesh(new THREE.SphereGeometry(0.24,18,12),M('--gold',1));cmb.position.set(0,y4,0);cmb.userData.pick={type:'combine',tip:'weighted sum'};L.picks.push(cmb);L.g.add(cmb);
  lab(L,`<b>Σ wₑ · expertₑ(x̂)</b>${m.s?' + shared(x̂)':''}${m.score==='sigmoid'?' · weights × 2.5':''}`,new THREE.Vector3(0.4,y4,0),{side:'r',pri:4});
  lift.forEach(b=>vline(L,[[b.position.x,y3+0.3,0],[0,y4-0.2,0]],'--muted',0.5));
  const y5=y4+1.2,vout=vec(L,R.ffnOut,0,y5,0,{w:4,k:0.25,pick:{type:'combine',tip:'MoE output'}});
  lab(L,'<b>MoE output</b> → residual stream',new THREE.Vector3(2.2,y5,0),{side:'r',pri:4});
  arrow(L,[[0,y4+0.25,0],[0,y5-0.25,0]],'--muted');
  titleLab(L,`<b>${ffnName(a,TJ.kind(a,li))}</b> · layer ${li+1}`,0,y5+0.9);
  L.ball=addBall(L);const P=(x,y,z)=>new THREE.Vector3(x,y,z==null?0.9:z);
  const route=v=>{bars.forEach((b,e)=>{const h=Math.max(0.02,sc[e]*0.9*v*(cs.has(e)?1.6:0.7));b.scale.y=h;b.position.y=y2+0.06+h/2;});tiles.forEach((t,e)=>t.material.color.copy(col(cs.has(e)&&v>0.6?'--gold':'--steel')));};
  const run=v=>{lift.forEach(b=>{b.scale.setScalar(Math.max(0.02,v));});if(sh){sh.material.emissive=col('--teal');sh.material.emissiveIntensity=0.5*v;}};
  L.steps.push({kind:'norm2',li,pos:P(0,0.3),dur:1.0,hl:vin.bars});
  L.steps.push({kind:'route',li,pos:P(0,y2+0.6,1.4),dur:1.8,hl:[pr],on:t=>route(t)});
  L.steps.push({kind:'experts',li,pos:P(0,y3+0.5,0.6),dur:1.7,hl:lift,on:t=>run(t)});
  L.steps.push({kind:'combine',li,pos:P(0,y4,0.6),dur:1.3,hl:[cmb],on:t=>vout.set(R.ffnOut,t)});
  L.reset=()=>{route(0);run(0);vout.set(R.ffnOut,0);};
}
function buildEmb(L){
  const a=J.a,toks=allToks(),T=toks.length,tk=toks[T-1];
  const cw=Math.min(0.9,8/T),chips=[];
  toks.forEach((t,i)=>{const x=-(T-1)*cw/2+i*cw;chips.push(bx(L,cw*0.85,0.32,0.4,i===T-1?'--gold':'--steel',x,0,0,{pick:{type:'tok',tip:`"${t.text}" → ID ${N(t.rid)}`}}));lab(L,t.text,new THREE.Vector3(x,-0.3,0.2),{side:'b',pri:2,cls:'tok'});});
  lab(L,'<b>tokens</b> · the tokenizer splits text into pieces it knows',new THREE.Vector3((T-1)*cw/2+0.6,0,0),{side:'r',pri:4});
  const y1=rowY(1)+1.6,W=4.6,H=2.8,pt=plate(L,W,H,'--steel',0,y1,0,{rows:10,cols:14,pick:{type:'emb',tip:'embedding table'}});
  const ry=y1-H/2+(tk.rid/a.V)*H;const row=bx(L,W+0.15,0.12,0.16,'--gold',0,ry,0.06,{basic:true,edge:false});
  lab(L,`<b>embedding table</b> · ${N(a.V)} rows × ${N(a.h)} columns · ${fB(a.V*a.h*opt().bpw/8)}`,new THREE.Vector3(W/2+0.3,y1+0.4,0),{side:'r',pri:4});
  lab(L,`row ${N(tk.rid)} = "${tk.text}"`,new THREE.Vector3(W/2+0.3,ry,0),{side:'r',pri:5,cls:'shape'});
  arrow(L,[[chips[T-1].position.x,0.2,0],[0,ry,0.15]],'--gold',{op:0.9});
  const y2=y1+H/2+0.9,v1=vec(L,J.rec.embRaw,0,y2,0,{w:4,k:0.3,pick:{type:'emb',tip:'the looked-up row'}});
  lab(L,`<b>one row read</b> → vector of ${N(a.h)} numbers`,new THREE.Vector3(2.2,y2,0),{side:'r',pri:4});
  arrow(L,[[0,ry+0.1,0.15],[0,y2-0.3,0]],'--muted');
  let y3=y2,v2=null;
  if(a.embScale){y3=y2+1.3;v2=vec(L,J.rec.emb,0,y3,0,{w:4,k:0.3});lab(L,`<b>× √${N(a.h)} = ${Math.sqrt(a.h).toFixed(1)}</b> (Gemma scales the embedding)`,new THREE.Vector3(2.2,y3,0),{side:'r',pri:4});arrow(L,[[0,y2+0.35,0],[0,y3-0.3,0]],'--muted');}
  titleLab(L,'<b>Tokenizer → embedding</b> · this becomes the residual stream',0,y3+0.9);
  L.ball=addBall(L);
  L.steps.push({kind:'tok',li:-1,pos:new THREE.Vector3(chips[T-1].position.x,0.4,0.5),dur:1.3,hl:chips});
  L.steps.push({kind:'emb',li:-1,pos:new THREE.Vector3(0,ry,0.6),dur:1.5,hl:[row],on:t=>v1.set(J.rec.embRaw,t)});
  if(v2)L.steps.push({kind:'emb',li:-1,pos:new THREE.Vector3(0,y3,0.6),dur:1.2,on:t=>v2.set(J.rec.emb,t)});
  L.reset=()=>{v1.set(J.rec.embRaw,0);if(v2)v2.set(J.rec.emb,0);};
}
function buildLM(L){
  const a=J.a,last=J.rec.layers[a.L-1].xout;
  const v0=vec(L,last,0,0,0,{w:4,k:0.2,pick:{type:'fnorm',tip:'output of the last layer'}});
  lab(L,`<b>x</b> after layer ${a.L}`,new THREE.Vector3(2.2,0,0),{side:'r',pri:4});
  const pn=plate(L,2.4,0.28,COL.norm,0,0.9,0,{rows:1,cols:8,pick:{type:'fnorm',tip:'final RMSNorm'}});lab(L,'<b>final RMSNorm</b>',new THREE.Vector3(1.4,0.9,0),{side:'r',pri:3});
  arrow(L,[[0,0.35,0],[0,0.72,0]],'--muted');
  const y1=rowY(1)+0.9,pw=plate(L,6.0,1.1,COL.lm,0,y1,0,{rows:6,cols:16,pick:{type:'lmhead',tip:'unembedding matrix'}});
  lab(L,`<b>LM head</b> · ${N(a.h)} × ${N(a.V)} · ${fB(a.V*a.h*opt().bpw/8)}${a.tied?' · same matrix as the embedding':''}`,new THREE.Vector3(3.15,y1,0),{side:'r',pri:4});
  arrow(L,[[0,1.08,0],[0,y1-0.6,0]],'--muted');
  const top=[...J.probs].map((p,i)=>[p,i]).sort((x,y)=>y[0]-x[0]).slice(0,10);
  const y2=rowY(3)+0.2,lb=[],pb=[],sp=0.62;const mxl=Math.max(...top.map(t=>J.rec.logits[t[1]])),mnl=Math.min(...J.rec.logits);
  top.forEach(([p,i],n)=>{const x=-(top.length-1)*sp/2+n*sp;
    lb.push(bx(L,0.3,1,0.3,'--steel',x,y2,0,{pick:{type:'lmhead',tip:`logit of "${TJ.WORDS[i]}"`}}));
    pb.push(bx(L,0.3,1,0.3,i===J.choice?'--copper':'--gold',x,y2+2.2,0,{pick:{type:'sample',tip:`P("${TJ.WORDS[i]}") = ${(p*100).toFixed(1)}%`}}));
    lab(L,`${TJ.WORDS[i]}`,new THREE.Vector3(x,y2-0.15,0.25),{side:'b',pri:2,cls:'tok'});});
  lab(L,`<b>logits</b> · one score per vocabulary entry (top 10 of ${N(a.V)} shown)`,new THREE.Vector3((top.length-1)*sp/2+0.5,y2+0.4,0),{side:'r',pri:4});
  lab(L,`<b>softmax(logits / T)</b><br>→ probabilities · chosen: "${TJ.WORDS[J.choice]}"`,new THREE.Vector3(-(top.length-1)*sp/2-0.4,y2+2.5,0),{side:'l',pri:4});
  arrow(L,[[0,y1+0.6,0],[0,y2-0.25,0.4]],'--muted');
  titleLab(L,'<b>Final norm → LM head → next token</b>',0,y2+3.6);
  L.ball=addBall(L);
  const setL=v=>lb.forEach((b,n)=>{const h=Math.max(0.03,(J.rec.logits[top[n][1]]-mnl)/(mxl-mnl+1e-6)*1.4*v);b.scale.y=h;b.position.y=y2+h/2;});
  const setP=v=>pb.forEach((b,n)=>{const h=Math.max(0.03,top[n][0]/top[0][0]*1.1*v);b.scale.y=h;b.position.y=y2+2.0+h/2;});
  L.steps.push({kind:'fnorm',li:-1,pos:new THREE.Vector3(0,0.9,0.5),dur:1.0,hl:[pn]});
  L.steps.push({kind:'lmhead',li:-1,pos:new THREE.Vector3(0,y1,0.5),dur:1.6,hl:[pw],on:t=>setL(t)});
  L.steps.push({kind:'sample',li:-1,pos:new THREE.Vector3(0,y2+2.6,0.6),dur:1.8,hl:pb,on:t=>setP(t)});
  L.reset=()=>{setL(0);setP(0);};
}
function buildPart(L,li,part){
  const a=J.a;
  if(part==='emb'||part==='tok')return buildEmb(L);
  if(part==='lm')return buildLM(L);
  if(part==='ffn')buildFFN(L,li);
  else{const k=TJ.kind(a,li);if(k.mix==='linear')buildAttnLin(L,li);else if(a.mla)buildAttnMLA(L,li);else buildAttnStd(L,li);}
  const bb=new THREE.Box3();L.g.children.forEach(c=>{if(!c.userData.noFit)bb.expandByObject(c);});
  mini(L,a,bb.min.x-2.8,bb.min.y+0.3,Math.min(7,bb.max.y-bb.min.y-1),li);
}

/* ---------- navigation ---------- */
const PRESET=[{th:0.5,ph:1.3},{th:0.45,ph:1.36},{th:0.55,ph:1.36}];
function build(){
  if(!S)return;
  if(CUR){S.scene.remove(CUR.g);clear(CUR.g);}
  $('#jLabels').innerHTML='';
  const L=level(),lv=J.lv;
  if(lv.level===0)buildModel(L);else if(lv.level===1)buildLayer(L,lv.layer);else buildPart(L,lv.layer,lv.part);
  S.scene.add(L.g);CUR=L;J.focus=lv.layer;hovered=null;tip().hidden=true;
  fit(L,lv.level===0&&J.compare?{th:0.18,ph:1.32}:PRESET[lv.level]);
  J.steps=L.steps;J.idx=Math.max(0,L.steps.length-1);J.t=1;J.playing=false;
  if(L.reset)L.reset();
  setStep(J.idx,1,true);
  chips();crumbs();legend();
  if(L.ball)L.ball.visible=false;
  S.dirty=true;
}
function fit(L,p){
  const box=new THREE.Box3();L.g.children.forEach(c=>{if(!c.userData.noFit)box.expandByObject(c);});
  const c=box.getCenter(new THREE.Vector3()),s=box.getSize(new THREE.Vector3());
  const asp=S.stage.clientWidth/Math.max(1,S.stage.clientHeight),tn=Math.tan(19*Math.PI/180);
  const right=J.lv.level===0?(J.compare?0.05:0.9):J.lv.level===2?0.4:0.6;
  const dy=(s.y/2)/tn,dx=((s.x*(1+right))/2)/(tn*asp);
  S.goal={t:[c.x+s.x*right*0.45,c.y+s.y*0.06,c.z],r:Math.max(14,Math.max(dy,dx)*(J.lv.level===0?1.2:J.lv.level===2?1.1:0.9)+s.z*0.3),th:p.th,ph:p.ph};
  S.st.lim=[p.th-0.75,p.th+0.75,0.85,1.5];
}
function go(spec){Object.assign(J.lv,spec);J.pin=null;build();}
function crumbs(){
  const a=J.a,lv=J.lv,parts=[`<button type="button" data-go="0">${J.compare?'Compare':a.short}</button>`];
  if(lv.level>=1&&lv.part!=='emb'&&lv.part!=='tok'&&lv.part!=='lm')parts.push(`<button type="button" data-go="1">Layer ${lv.layer+1}</button>`);
  if(lv.level===2)parts.push(`<b>${lv.part==='attn'?mixShort(a,TJ.kind(a,lv.layer))+' attention':lv.part==='ffn'?(TJ.kind(a,lv.layer).ffn==='moe'?'MoE':'FFN'):lv.part==='lm'?'LM head':'Embedding'}</b>`);
  $('#jCrumbs').innerHTML=parts.join('<span>›</span>');
  $('#jBack').disabled=lv.level===0;
  const inL=lv.level>=1&&!['emb','tok','lm'].includes(lv.part);
  $('#jPrevL').hidden=$('#jNextL').hidden=!inL;
  $('#jCrumbs').querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>{const g=+b.dataset.go;go(g===0?{level:0,part:null}:{level:1,part:null});});
}
function legend(){
  const a=J.a,lv=J.lv;let it=[];
  if(lv.level===0){const kinds=new Set();(J.compare?J.cmp.map(TJ.arch):[a]).forEach(x=>{for(let i=0;i<x.L;i++){const k=TJ.kind(x,i);kinds.add(k.mix==='linear'?'linear':k.mix==='local'?'local':'full');kinds.add(k.ffn);}});
    if(kinds.has('full'))it.push([COL.full,'Attention half of a layer (full / global)']);if(kinds.has('local'))it.push([COL.local,'Attention: sliding window (local)']);if(kinds.has('linear'))it.push([COL.linear,'Attention: Gated DeltaNet (linear)']);
    if(kinds.has('dense'))it.push([COL.dense,'FFN half of a layer (dense)']);if(kinds.has('moe'))it.push([COL.moe,'FFN half: mixture of experts']);it.push([COL.norm,'RMSNorm'],[COL.lm,'LM head']);}
  else if(lv.level===1)it=[[COL.norm,'RMSNorm'],[mixColor(a,TJ.kind(a,lv.layer)),'Token mixing (attention)'],[TJ.kind(a,lv.layer).ffn==='moe'?COL.moe:COL.dense,'FFN'],['--gold','Residual add'],[COL.spine,'Residual stream']];
  else it=[[COL.q,'Queries / output projection'],[COL.k,'Keys'],[COL.v,'Values · cache'],['--violet','Latent / grouping / gate'],['--gold','Selected · scores · ball = your token'],['--steel','Inactive · not selected']];
  it.push(['linear-gradient(90deg,var(--teal),var(--copper))','Vector bars: negative → positive']);
  $('#jLegend').innerHTML=it.map(([c,t])=>`<span><i style="background:${c.startsWith('--')?`var(${c})`:c}"></i>${t}</span>`).join('');
}

/* ---------- play ---------- */
function chips(){$('#jChips').innerHTML=J.steps.map((s,i)=>`<button type="button" data-i="${i}" class="${s.kind==='layer'?'mini':''}">${s.label||NAME[s.kind]||s.kind}</button>`).join('');}
function setStep(idx,t,quiet){
  const st=J.steps;if(!st.length)return;
  if(CUR&&CUR.reset)CUR.reset();
  for(let i=0;i<idx;i++)if(st[i].on)st[i].on(1);
  if(st[idx].on)st[idx].on(t);
  J.idx=idx;J.t=t;
  const prev=idx>0?st[idx-1].pos:st[0].pos.clone().add(new THREE.Vector3(0,-0.6,0));
  if(CUR.ball){CUR.ball.visible=!quiet;CUR.ball.position.copy(prev).lerp(st[idx].pos,ease(t));}
  CUR.picks.forEach(m=>{if(m.material.emissive&&!m.userData.keepEm){m.material.emissive.setHex(0);}});
  (st[idx].hl||[]).forEach(m=>{if(m&&m.material&&m.material.emissive&&!quiet){m.material.emissive.copy(col('--gold'));m.material.emissiveIntensity=0.35+0.25*Math.sin(t*Math.PI);}});
  document.querySelectorAll('#jChips button').forEach((b,i)=>b.classList.toggle('on',i===idx&&!quiet));
  const on=document.querySelector('#jChips button.on');if(on&&J.playing)on.scrollIntoView({block:'nearest',inline:'nearest'});
  if(!J.pin&&!quiet)inspect({kind:st[idx].kind,layer:st[idx].li});
  $('#jStageName').textContent=quiet?'':((st[idx].li>=0?`Layer ${st[idx].li+1} · `:'')+(NAME[st[idx].kind]||st[idx].kind));
  costHighlight(st[idx]);
  S.dirty=true;
}
function ease(t){return t<0.5?2*t*t:1-Math.pow(-2*t+2,2)/2;}
function play(on){J.playing=on;$('#jPlay').textContent=on?'❚❚ Pause':'▶ Play';}

/* ---------- labels: placed every frame, lower priority hides on overlap ---------- */
function placeLabels(){
  if(!CUR)return;const W=S.stage.clientWidth,H=S.stage.clientHeight,placed=[];
  const its=CUR.labels.slice().sort((p,q)=>q.pri-p.pri);
  for(const it of its){
    const p=project(S,it.pos);if(!p.ok){it.el.style.visibility='hidden';continue;}
    if(it.w==null||it.dirty){it.w=it.el.offsetWidth;it.h=it.el.offsetHeight;}
    let x=p.x,y=p.y;
    if(it.side==='r'){x+=10;y-=it.h/2;}else if(it.side==='l'){x-=10+it.w;y-=it.h/2;}else if(it.side==='t'){x-=it.w/2;y-=it.h+6;}else if(it.side==='b'){x-=it.w/2;y+=6;}else{x-=it.w/2;y-=it.h/2;}
    const r={x0:x-2,y0:y-1,x1:x+it.w+2,y1:y+it.h+1};
    if(r.x1>W-4){const d=r.x1-(W-4);x-=d;r.x0-=d;r.x1-=d;}if(r.x0<4){const d=4-r.x0;x+=d;r.x0+=d;r.x1+=d;}
    if(it.pri>=6&&r.y0<50){const d=50-r.y0;y+=d;r.y0+=d;r.y1+=d;}
    const off=r.y0<46||r.y1>H;
    const hit=placed.some(q=>!(r.x1<q.x0||r.x0>q.x1||r.y1<q.y0||r.y0>q.y1));
    if((hit||off)&&it.pri<6){it.el.style.visibility='hidden';continue;}
    placed.push(r);it.el.style.visibility='';it.el.style.transform=`translate(${Math.round(x)}px,${Math.round(y)}px)`;
  }
}

/* ---------- picking & hover ---------- */
const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();
function hitAt(e){if(!CUR)return null;const r=S.cv.getBoundingClientRect();mouse.x=(e.clientX-r.left)/r.width*2-1;mouse.y=-(e.clientY-r.top)/r.height*2+1;ray.setFromCamera(mouse,S.cam);const h=ray.intersectObjects(CUR.picks,false).filter(x=>x.object.visible);return h.length?h[0].object:null;}
let hovered=null;
function onHover(e){
  const o=hitAt(e),t=tip();
  if(hovered&&hovered!==o){hovered.userData.hover=false;if(hovered.material.emissive&&!(J.steps[J.idx]&&(J.steps[J.idx].hl||[]).includes(hovered)))hovered.material.emissive.setHex(0);if(hovered.userData.pick.type==='layer'&&CUR.extra.layers)lightLayer(hovered.userData.pick,false);}
  hovered=o;
  if(!o){t.hidden=true;S.cv.style.cursor='';S.dirty=true;return;}
  const pk=o.userData.pick;S.cv.style.cursor='pointer';
  if(J.lv.level===0&&pk.type==='layer')lightLayer(pk,true);else if(o.material.emissive){o.material.emissive.copy(col('--gold'));o.material.emissiveIntensity=0.25;}
  const r=S.stage.getBoundingClientRect();t.hidden=false;t.textContent=pk.tip||NAME[pk.type]||pk.type;t.style.transform=`translate(${Math.min(e.clientX-r.left+14,r.width-t.offsetWidth-6)}px,${e.clientY-r.top+14}px)`;S.dirty=true;
}
function lightLayer(pk,on){const ms=CUR.extra.layers&&CUR.extra.layers[pk.arch||J.a.id];if(!ms||!ms[pk.layer])return;ms[pk.layer].forEach(m=>{m.material.emissive.copy(on?col('--gold'):new THREE.Color(0));m.material.emissiveIntensity=on?0.45:0;});}
function onClick(e){
  const o=hitAt(e);if(!o)return;const pk=o.userData.pick;tip().hidden=true;
  if(pk.type==='layer'){if(pk.arch&&pk.arch!==J.a.id){$('#jArch').value=pk.arch;J.a=TJ.arch(pk.arch);run(true);}J.compare=false;$('#jCompare').checked=false;$('#jCmpPick').hidden=true;go({level:1,layer:pk.layer,part:null});return;}
  if(pk.type==='mini'){go({level:1,layer:pk.layer,part:null});return;}
  if(pk.type==='model'){go({level:0,part:null});return;}
  if(pk.open){go({level:2,part:pk.open});return;}
  if(J.lv.level===0&&['tok','emb','lm'].includes(pk.type)){if(J.compare)return;go({level:2,part:pk.type==='lm'?'lm':'emb',layer:J.lv.layer});return;}
  J.pin=pk;inspect({kind:pk.type,layer:J.lv.level?J.lv.layer:(pk.layer!=null?pk.layer:-1),pick:pk});
}

/* ---------- loop ---------- */
let last=0;
function loop(ts){
  const dt=Math.min(0.05,(ts-last)/1000||0);last=ts;
  if(S&&S.vis){
    if(S.goal){const g=S.goal,st=S.st,tg=S.target,f=reduce?1:0.12;tg.x+=(g.t[0]-tg.x)*f;tg.y+=(g.t[1]-tg.y)*f;tg.z+=(g.t[2]-tg.z)*f;st.r+=(g.r-st.r)*f;st.theta+=(g.th-st.theta)*f;st.phi+=(g.ph-st.phi)*f;
      if(Math.abs(g.r-st.r)<0.01&&Math.abs(g.t[1]-tg.y)<0.005&&Math.abs(g.th-st.theta)<0.002)S.goal=null;S.place();S.dirty=true;}
    if(J.playing&&J.steps.length){const sp=+$('#jSpeed').value;let t=J.t+dt*sp/J.steps[J.idx].dur,i=J.idx;
      if(t>=1){if(i<J.steps.length-1){i++;t=0;}else{t=1;play(false);$('#jNext').classList.add('pulse');}}
      setStep(i,t);}
    if(S.dirty){S.render();placeLabels();S.dirty=false;}
  }
  requestAnimationFrame(loop);
}

/* ---------- actions ---------- */
function tokChips(){$('#jToks').innerHTML=allToks().map((t,i,arr)=>`<span class="tk${t.gen?' gen':''}${i===arr.length-1?' cur':''}">${t.text}</span>`).join('')+`<span class="tk next">→ ${J.choice!=null?TJ.WORDS[J.choice]:'?'}</span>`;}
function refresh(){compute();tokChips();costPanel();build();document.dispatchEvent(new Event('tj-run'));}
function run(quiet){J.toks=TJ.tokenize($('#jPrompt').value||'the model',J.a);J.gen=[];if(!quiet){refresh();setStep(0,0);play(!reduce);}else{compute();tokChips();costPanel();}}
function next(){
  if(!J.rec)return;const w=TJ.WORDS[J.choice];J.gen.push({text:w,id:J.choice,rid:TJ.hash(J.a.id+w)%J.a.V,gen:true});
  if(J.toks.length+J.gen.length>14)J.toks.shift();
  $('#jNext').classList.remove('pulse');refresh();setStep(0,0);play(!reduce);
}
function setArch(){J.a=TJ.arch($('#jArch').value);$('#jArchBlurb').textContent=J.a.blurb;J.lv={level:0,layer:Math.min(J.lv.layer,J.a.L-1),part:null};run(true);build();}
TJ.journeyRec=()=>J.rec;TJ.journeyFocus=()=>J.focus;TJ.journeyArch=()=>J.a;

function init(){
  fillControls();J.a=TJ.arch(J.archId);$('#jArchBlurb').textContent=J.a.blurb;
  if(window.THREE){
    const st={theta:PRESET[0].th,phi:PRESET[0].ph,r:30,onGrab:()=>{S&&(S.goal=null);}},target=new THREE.Vector3(0,7,0);
    S=makeScene($('#jCanvas'),$('#jStage'),st,target,onClick);S.place();S.resize();
    S.cv.addEventListener('pointermove',e=>{if(e.buttons)return;onHover(e);});S.cv.addEventListener('pointerleave',()=>{tip().hidden=true;});
    window.addEventListener('resize',()=>{if(CUR)fit(CUR,J.lv.level===0&&J.compare?{th:0.18,ph:1.32}:PRESET[J.lv.level]);});
  }else $('#jStage').insertAdjacentHTML('beforeend','<p class="hint">3D needs WebGL (three.js failed to load).</p>');
  J.toks=TJ.tokenize($('#jPrompt').value,J.a);compute();tokChips();costPanel();build();
  $('#jArch').onchange=setArch;$('#jRun').onclick=()=>run();$('#jNext').onclick=next;
  $('#jPrompt').addEventListener('keydown',e=>{if(e.key==='Enter')run();});
  $('#jPlay').onclick=()=>{if(!J.steps.length)return;if(!J.playing&&J.idx>=J.steps.length-1&&J.t>=1)setStep(0,0);J.pin=null;play(!J.playing);};
  $('#jPrev').onclick=()=>{play(false);J.pin=null;setStep(Math.max(0,J.t>0.05&&J.t<1?J.idx:J.idx-1),1);};
  $('#jStep').onclick=()=>{play(false);J.pin=null;setStep(J.t<1?J.idx:Math.min(J.steps.length-1,J.idx+1),1);};
  $('#jRestart').onclick=()=>{J.pin=null;setStep(0,0);play(true);};
  $('#jChips').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;play(false);J.pin=null;setStep(+b.dataset.i,1);});
  $('#jBack').onclick=()=>{const lv=J.lv;if(lv.level===2&&!['emb','tok','lm'].includes(lv.part))go({level:1,part:null});else go({level:0,part:null});};
  $('#jPrevL').onclick=()=>{if(J.lv.layer>0)go({layer:J.lv.layer-1});};
  $('#jNextL').onclick=()=>{if(J.lv.layer<J.a.L-1)go({layer:J.lv.layer+1});};
  $('#jReset').onclick=()=>{if(CUR){fit(CUR,J.lv.level===0&&J.compare?{th:0.18,ph:1.32}:PRESET[J.lv.level]);}};
  $('#jZoomIn').onclick=()=>{S.goal=null;S.st.r*=0.85;S.place();S.dirty=true;};
  $('#jZoomOut').onclick=()=>{S.goal=null;S.st.r*=1.18;S.place();S.dirty=true;};
  $('#jCompare').onchange=()=>{J.compare=$('#jCompare').checked;$('#jCmpPick').hidden=!J.compare;go({level:0,part:null});};
  $('#jCmpPick').addEventListener('change',e=>{const ids=[...document.querySelectorAll('#jCmpPick input:checked')].map(i=>i.value);if(ids.length>3){e.target.checked=false;return;}if(!ids.length){e.target.checked=true;return;}J.cmp=ids;go({level:0});});
  ['#jHw','#jFmt','#jPhase'].forEach(s=>$(s).addEventListener('change',()=>{costPanel();if(J.lv.level)build();}));
  $('#jCtx').addEventListener('input',()=>{$('#jCtxO').textContent=N(CTX[+$('#jCtx').value])+' tokens';costPanel();});
  $('#jAct').addEventListener('change',()=>{refresh();document.dispatchEvent(new Event('tj-act'));});
  $('#jTemp').addEventListener('input',()=>{$('#jTempO').textContent=(+$('#jTemp').value/10).toFixed(1);compute();tokChips();if(J.lv.part==='lm')build();});
  $('#jGreedy').addEventListener('change',()=>{compute();tokChips();if(J.lv.part==='lm')build();});
  $('#jCtxO').textContent=N(CTX[+$('#jCtx').value])+' tokens';
  requestAnimationFrame(loop);
}
TJ.retheme=()=>{if(!S)return;build();costPanel();};
TJ.setAct=v=>{$('#jAct').value=v;refresh();};
TJ.initJourney=init;
TJ.go=go;TJ.stepTo=i=>{play(false);setStep(Math.min(i,J.steps.length-1),1);};TJ.snapCam=()=>{if(!S||!S.goal)return;const g=S.goal;S.target.set(...g.t);S.st.r=g.r;S.st.theta=g.th;S.st.phi=g.ph;S.goal=null;S.place();S.dirty=true;};


/* ---------- inspector ---------- */
function rows(list){return `<div class="receipt">${list.map(([a,b])=>`<div class="row"><span>${a}</span><span>${b}</span></div>`).join('')}</div>`;}
function partCost(a,li,part,o,H,ph){
  const ks=layerStages(li).map(s=>s.kind),A=['qkv','rope','compress','up','index','score','softmax','mix','decay','write','read','oproj'],F=['gu','act','down','route','experts','combine'];
  const want=part==='attn'?A:part==='ffn'?F:ks;let c={bytes:0,flops:0},t=0,tm=0,tc=0;
  ks.filter(k=>want.includes(k)).forEach(k=>{const x=TJ.stageCost(a,{kind:k,layer:li},o),y=TJ.time(x,H,ph,o.ctx);c.bytes+=x.bytes;c.flops+=x.flops;t+=y.t;tm+=y.tm;tc+=y.tc;});
  return {cost:c,time:{t,tm,tc,bound:tm>=tc?'memory':'compute'}};
}
function inspect(target){
  const a=J.a,o=opt(),H=hw(),ph=$('#jPhase').value;let kind=target.kind,li=target.layer==null?-1:target.layer;
  if(kind==='in'||kind==='out')kind='residual';
  if(kind==='mini'||kind==='model'){kind='layer';li=target.pick&&target.pick.layer!=null?target.pick.layer:li;}
  if(li<0&&!['tok','emb','fnorm','lmhead','sample','lm'].includes(kind))li=J.lv.level?J.lv.layer:-1;
  if(kind==='lm')kind='lmhead';
  const R=li>=0?J.rec.layers[li]:null,k=li>=0?TJ.kind(a,li):null;
  if((kind==='layer'||kind==='residual')&&li<0)li=0;
  const map={qheads:'qkv',cache:'score',state:'write',pastTok:'decay',absorb:'up',indexer:'index',gate:'gu',up:'gu',hidden:'act',expert:'experts',router:'route',shared:'experts',residual:'in'};
  let cs;
  if(kind==='attn'||kind==='ffn'||kind==='layer')cs=partCost(a,li,kind==='layer'?'all':kind,o,H,ph);
  else{const s={kind:map[kind]||kind,layer:li};const c=TJ.stageCost(a,s,o);cs={cost:c,time:TJ.time(c,H,ph,o.ctx)};}
  const info=explain(kind,R||J.rec.layers[0],k||TJ.kind(a,0),li,target.pick);
  const c=cs.cost,tm=cs.time;
  const costHtml=c.bytes||c.flops?rows([[`Bytes read (${a.short}, ${$('#jFmt').selectedOptions[0].text.split(' ·')[0]}, ${fK(o.ctx)} ctx)`,fB(c.bytes)],[`Math`,`${(c.flops/1e9).toFixed(c.flops<1e9?3:1)} GFLOP`],[`Time on ${H.name.split(' ·')[0]}`,`${fT(tm.t)} <span class="pill ${tm.bound==='memory'?'warn':'good'}">${tm.bound}-bound</span>`],[`Share of this token's time`,`${(tm.t/J.totalReal*100).toFixed(2)}%`]]):'';
  $('#jInspector').innerHTML=`<div class="ins-head"><span class="eyebrow">${li>=0?`Layer ${li+1} · `:''}${target.pick?'clicked':'now playing'}</span>${J.pin?'<button type="button" id="jUnpin">Follow animation</button>':''}</div>
   <h3>${info.title}</h3><div class="what">${info.what}</div>
   ${info.toy?`<div class="toyv"><span class="tag">Toy numbers (16-dim stand-in)</span>${info.toy}</div>`:''}
   ${info.real?`<span class="tag">${a.short}, real shapes</span>${rows(info.real)}`:''}
   ${costHtml?`<span class="tag">Cost for one token</span>${costHtml}`:''}
   ${info.extra||''}`;
  const u=$('#jUnpin');if(u)u.onclick=()=>{J.pin=null;setStep(J.idx,J.t);};
  const fb=$('#jFocusBtn');if(fb)fb.onclick=()=>go({level:1,layer:+fb.dataset.l,part:null});
  const sp=$('#jSpark');if(sp)drawSpark(sp,li);
}

function explain(kind,R,k,li,pick){
  const a=J.a,T=J.rec.T,toks=allToks(),tk=toks[T-1],o=opt(),W=o.bpw/8;
  const gqa=a.kv?a.H/a.kv:null;
  switch(kind){
  case 'tok':return {title:'1 · Tokenize',what:`<p>The text is cut into pieces the model knows, and each piece becomes an integer ID. ${a.short} has a <b>${N(a.V)}</b>-entry vocabulary. A word it doesn't have is split into sub-word pieces (<code>##</code> marks a continuation). The model never sees letters, only these IDs.</p><p>All ${T} tokens go through the model. The one we follow is the last, <b>"${tk.text}"</b>, because its output picks the next word.</p>`,
    toy:`<div class="tkrow">${toks.map(t=>`<span>${t.text}<small>${N(t.rid)}</small></span>`).join('')}</div><p class="note">The ID numbers are illustrative. The real tokenizer uses byte-level BPE.</p>`};
  case 'emb':return {title:'2 · Embedding lookup',what:`<p>The ID selects one row of a ${N(a.V)} × ${N(a.h)} table. There's no math here, just a memory read of ${N(a.h)} numbers. That row is the token's starting vector, and it becomes the residual stream.</p>${a.embScale?`<p>Gemma then multiplies the row by √${N(a.h)} = ${Math.sqrt(a.h).toFixed(1)} so the stream starts at a sensible scale. Watch the packet grow (toy: ×4).</p>`:''}`,
    toy:strip(J.rec.embRaw,'row')+(a.embScale?strip(J.rec.emb,'× √d'):''),real:[['Table size',`${N(a.V)} × ${N(a.h)} = ${N(a.V*a.h)} weights`],['Stored',fB(a.V*a.h*W)],['Read for this token',`one row = ${fB(a.h*W)}`]]};
  case 'residual':case 'in':return {title:'The residual stream',what:`<p>Each token carries a single vector of ${N(a.h)} numbers up through every layer. Blocks never replace it. Each one reads a <i>normalized copy</i>, computes something, and <b>adds</b> the result back: <code>x ← x + f(norm(x))</code>.</p><p>That's why ${a.L} layers can train at all: the gradient has a straight path down the stream. It's also why the stream's size keeps growing as layers keep adding to it (chart below).</p>`,
    toy:R?strip(R.xin,'x in'):'',extra:`<span class="tag">Stream size ‖x‖ after each layer (toy run)</span><canvas id="jSpark" width="420" height="110"></canvas>`};
  case 'norm1':case 'norm2':case 'fnorm':{const x=kind==='norm1'?R.xin:kind==='norm2'?R.xmid:J.rec.layers[a.L-1].xout,y=kind==='norm1'?R.n1:kind==='norm2'?R.n2:J.rec.final;
    return {title:kind==='fnorm'?'Final RMSNorm':'RMSNorm',what:`<p>The vector is divided by its root-mean-square, then multiplied by a learned gain γ:</p><pre>x̂ = x / √(mean(x²) + ε) · γ</pre><p>This way the next block always sees inputs at the same scale, however much the stream has grown. Here the rms goes from <b>${rmsOf(x).toFixed(2)}</b> to <b>${rmsOf(y).toFixed(2)}</b>.</p>${a.sandwich&&kind!=='fnorm'?'<p>Gemma also normalizes each block\'s <i>output</i> before adding it back (a "sandwich" norm).</p>':''}`,
      toy:strip(x,'before')+strip(y,'after'),real:[['γ weights',N(a.h)],['Cost','negligible: one pass over '+N(a.h)+' numbers']]};}
  case 'qkv':case 'qheads':{
    if(k.mix==='linear'){const l=a.lin,M=R.mix;return {title:'Gated DeltaNet: q, k, v and two gates',what:`<p>This layer has <b>no KV cache</b>. It projects the token into queries, keys and values (k and q are L2-normalized), plus two scalars per head:</p><ul><li><b>α</b> (decay, teal gauge): how much of the old memory to keep</li><li><b>β</b> (write strength, gold gauge): how hard to write this token in</li></ul><p>Toy α = ${M.alpha.map(x=>x.toFixed(2)).join(', ')}; β = ${M.beta.map(x=>x.toFixed(2)).join(', ')}.</p>`,
      toy:strip(R.n1,'x̂ in')+M.k.map((v,h)=>strip(v,'k'+(h+1))).join(''),real:[['Key heads × dim',`${l.kh} × ${l.dk}`],['Value heads × dim',`${l.vh} × ${l.dv}`],['In-projection',`${N(a.h)} → ${N(l.kh*l.dk*2+l.vh*l.dv*2)} (+ gates)`]]};}
    if(a.mla){const m=a.mla;return {title:'MLA: queries through a low-rank bottleneck',what:`<p>The query is first squeezed to a ${N(m.qr)}-number latent, then expanded to ${a.H} heads. Each head gets ${m.nope} "content" dims plus ${m.rope} RoPE dims. This saves weights. The big saving is on the K/V side (next steps).</p>`,
      toy:strip(R.n1,'x̂ in'),real:[['W_DQ',`${N(a.h)} × ${N(m.qr)}`],['W_UQ',`${N(m.qr)} × ${a.H}·(${m.nope}+${m.rope})`],['Heads',a.H]]};}
    const M=R.mix;return {title:'Project to Q, K, V; write K, V to the cache',what:`<p>Three matrix multiplications turn the normalized vector into <b>${a.H} query heads</b> and <b>${a.kv} key/value heads</b> of ${a.hd} numbers each.</p><ul><li><b>Q</b>: "what am I looking for?"</li><li><b>K</b>: "what do I contain?"</li><li><b>V</b>: "what I hand over if chosen"</li></ul><p>The new K and V are <b>appended to the KV cache</b> (the gold-edged cubes at the right), so later tokens can read them without recomputing.</p>${a.attn==='mha'?'<p>Multi-head: every query head has its own K and V, the most expensive cache possible.</p>':a.attn==='mqa'?`<p>Multi-query: all ${a.H} heads share <b>one</b> K/V head. The cache is ${a.H}× smaller than MHA's.</p>`:`<p>Grouped-query: every ${gqa} query heads share one K/V head (violet lines). The cache is <b>${gqa}× smaller</b> than MHA's, and the math is the same.</p>`}${a.qkNorm?'<p>QK-Norm: each q and k head is RMS-normalized first, so scores cannot blow up.</p>':''}`,
      toy:strip(R.n1,'x̂ in')+M.q.slice(0,4).map((q,j)=>`<div class="note">q head ${j+1}: [${q[0].toFixed(2)}, ${q[1].toFixed(2)}] → uses K/V head ${Math.floor(j*M.kvT/D.NQ)+1}</div>`).join(''),
      real:[['W_q',`${N(a.h)} × ${a.H}·${a.hd} = ${N(a.h*a.H*a.hd)}`],['W_k, W_v',`${N(a.h)} × ${a.kv}·${a.hd} each`],['Cache written this token, this layer',`2 × ${a.kv} × ${a.hd} × 2 B = ${N(2*a.kv*a.hd*2)} B`],['Same × all layers',`${fB(2*a.kv*a.hd*2*a.L)} per token`]]};}
  case 'rope':{const ang=(T-1)*0.6;const M=R.mix;let q1=M.type==='std'?M.q[0]:null;
    return {title:'RoPE: position by rotation',what:`<p>Attention by itself doesn't know word order. RoPE rotates each pair of numbers in q and k by an angle proportional to the token's position. When q·k is taken, only the <i>difference</i> of the angles survives, so scores depend on how far apart two tokens are.</p><pre>[a, b] → [a·cosθ − b·sinθ, a·sinθ + b·cosθ],  θ = position × frequency</pre><p>This token is at position ${T-1}, so in the toy its pair turns by ${ang.toFixed(2)} rad. The real model uses ${a.mla?a.mla.rope/2:(a.hd/2)} frequencies per head, from fast (nearby words) to slow (long range).${a.attn==='swa'?' Gemma 3 uses base 10K on local layers and 1M on global ones.':''}${a.mla?' In MLA only the 64-dim RoPE part is rotated, and the rotated key is shared by all heads.':''}</p>`,
      toy:q1?`<div class="note">q head 1 before: [${(q1[0]*Math.cos(-ang)-q1[1]*Math.sin(-ang)).toFixed(2)}, ${(q1[0]*Math.sin(-ang)+q1[1]*Math.cos(-ang)).toFixed(2)}] → after: [${q1[0].toFixed(2)}, ${q1[1].toFixed(2)}]</div>`:''};}
  case 'compress':{const m=a.mla,M=R.mix;return {title:'MLA: compress K and V into one latent',what:`<p>MLA doesn't store ${a.H} heads × (K + V). It stores one <b>${m.kvr}-number latent c</b> plus a <b>${m.rope}-number RoPE key</b> shared by every head. That's ${m.kvr+m.rope} numbers per token per layer, compared with ${N(2*a.H*a.hd)} for MHA at this size: <b>${(2*a.H*a.hd/(m.kvr+m.rope)).toFixed(0)}× smaller</b>.</p><p>The violet column is c and the gold cube is the RoPE key. Only these go into the cache.</p>`,
    toy:strip(M.c[T-1],'c (latent)')+strip(M.kr[T-1],'k rope'),real:[['W_DKV',`${N(a.h)} × ${m.kvr+m.rope}`],['Cache per token per layer',`${m.kvr+m.rope} × 2 B = ${N((m.kvr+m.rope)*2)} B`],['× ${a.L} layers',fB((m.kvr+m.rope)*2*a.L)+' per token']]};}
  case 'up':case 'absorb':{const m=a.mla;return {title:'Absorb the up-projection',what:`<p>Each head <i>could</i> rebuild its K and V from c: <code>k_h = W_UK,h · c</code>, <code>v_h = W_UV,h · c</code>. At inference that never happens. Because <code>q·(W_UK c) = (W_UKᵀ q)·c</code>, W_UK is folded into the query and W_UV into the output projection. Heads score directly against the small cached latent (the violet plate), and the full K/V are never built.</p><p>The trade: a little more compute per head in exchange for a far smaller cache read. That's a good deal, because decode is memory-bound.</p>`,
    real:[['W_UK + W_UV',`${m.kvr} × ${a.H}·(${m.nope}+${m.v})`],['Per-head score dims',`${m.kvr} (latent) + ${m.rope} (RoPE)`]]};}
  case 'index':case 'indexer':{const M=R.mix,dd=a.dsa;return {title:'DSA lightning indexer: pick the top-k',what:`<p>A tiny scorer (${dd.heads} heads, ${dd.dim}-dim keys stored in FP8) rates <b>every</b> past token:</p><pre>I(t,s) = Σ_h w_h · ReLU(q_h · k_s)</pre><p>Only the top <b>${N(dd.topk)}</b> go on to real MLA attention. The rest (dimmed) are skipped. The indexer still reads all ${fK(o.ctx)} keys, but at ${dd.dim} B per token in FP8 instead of ${N((a.mla.kvr+a.mla.rope)*2)} B for the full latent.</p><p>Toy: top-${D.TOPK} of ${T}${T<=D.TOPK?'. Type a longer prompt (over 4 tokens) to see tokens get dropped.':''}.</p>`,
    toy:`<div class="note">indexer scores: ${M.idx.map((v,s)=>`<b style="color:var(${M.sel.includes(s)?'--gold':'--muted'})">${toks[s].text}:${v.toFixed(2)}</b>`).join(' · ')}</div>`,
    real:[['Indexer read at '+fK(o.ctx),fB(o.ctx*dd.dim)],['MLA read (top-'+N(dd.topk)+')',fB(Math.min(o.ctx,dd.topk)*(a.mla.kvr+a.mla.rope)*2)],['Dense MLA would read',fB(o.ctx*(a.mla.kvr+a.mla.rope)*2)]]};}
  case 'score':case 'softmax':case 'mix':case 'cache':{
    const M=R.mix;
    if(kind==='cache'&&pick){const s=pick.s;let txt=`<p>Token <b>"${toks[s].text}"</b>, position ${s}. ${s===T-1?'This is the token being processed. Its K and V were just written.':'Written when that token was processed, then reused unchanged by every later token.'}</p>`;
      if(M.type==='std'){const g=pick.g;const ws=[];for(let j=0;j<D.NQ;j++)if(Math.floor(j*M.kvT/D.NQ)===g)ws.push(`head ${j+1}: ${((M.w[j][s]||0)*100).toFixed(1)}%`);txt+=`<p>K/V head ${g+1}. Its group's attention to this token: ${ws.join(', ')}.</p>`;
        return {title:'A cached key and value',what:txt,toy:`<div class="note">K = [${M.K[s][g].map(x=>x.toFixed(2)).join(', ')}] · V = [${M.V[s][g].map(x=>x.toFixed(2)).join(', ')}]</div>`};}
      return {title:'A cached latent',what:txt,toy:strip(M.c[s],'c')};}
    const j=0,row=M.scores[j],w=M.w[j];
    const tbl=`<div class="tablewrap"><table><tr><th>token</th><th>score q₁·k</th><th>weight</th></tr>${row.map((v,s)=>`<tr${v==null?' style="opacity:.4"':''}><td>${toks[s].text}</td><td class="num">${v==null?(M.local?'outside window':'not selected'):v.toFixed(3)}</td><td class="num">${v==null?'0':((w[s]||0)*100).toFixed(1)+'%'}</td></tr>`).join('')}</table></div>`;
    if(kind==='score'){const lk=k.mix==='local';return {title:'Score every visible key',what:`<p>Each query head takes a dot product with every key it can see: <code>score = q·k / √d</code>. A high score means "this past token is relevant to me." The line brightness is the raw score.</p><p>To do this the GPU has to <b>read the K cache from memory</b>: at ${fK(o.ctx)} context that's ${fB(TJ.cacheRead(a,li,o.ctx,2)/2)} for this layer alone. That's why tokens slow down as a chat gets longer.</p>${lk?`<p>This is a <b>local</b> layer: only the last ${N(a.window)} tokens are visible (toy: ${D.WIN}). Its cache read stops growing past that point.</p>`:''}${a.kv&&a.kv<a.H?`<p>The ${gqa} heads in a group read the <i>same</i> keys, so each K is fetched once per group, not once per head.</p>`:''}`,toy:tbl};}
    if(kind==='softmax')return {title:'Softmax: scores → weights',what:`<p><code>wᵢ = e^{sᵢ} / Σⱼ e^{sⱼ}</code>. The weights are positive and sum to 1, and large scores dominate. Line brightness is now the weight, and the value cubes swell in proportion to how much their group attends to them.</p>`,toy:tbl};
    return {title:'Weighted sum of values',what:`<p>Each head blends the V vectors using its weights, <code>out_h = Σ wᵢ vᵢ</code>, and the ${a.H} head outputs are concatenated into one ${N(a.H*(a.mla?a.mla.v:a.hd))}-number vector. That's everything this token learned from the context in this layer.</p><p>The V half of the cache is read here: another ${fB(TJ.cacheRead(a,li,o.ctx,2)/2)} at ${fK(o.ctx)}.</p>`,toy:strip(M.heads,'heads')};}
  case 'decay':case 'pastTok':{const M=R.mix;return {title:'Forget a little: S ← α·S',what:`<p>The whole state matrix of each head is multiplied by its α (here ${M.alpha.map(x=>x.toFixed(2)).join(', ')}). Every earlier token's trace fades a bit at each step. The past-token chips show this: a token <i>n</i> steps back keeps about α<sup>n</sup> of its influence.</p><p>This is how a fixed-size memory makes room: older details fade unless they're written again.</p>`};}
  case 'write':case 'state':{const M=R.mix,l=a.lin;return {title:'Delta-rule write into a fixed state',what:`<p>The state first <i>predicts</i> this token's value from its key, <code>S·k</code>. Only the <b>error</b> gets written:</p><pre>S ← S + β · (v − S·k) · kᵀ</pre><p>If the memory already knew this association, almost nothing changes. Instead of piling up, the memory is <b>corrected</b>. Cube height is each state entry.</p><p>Real size: ${l.vh} heads × ${l.dk}×${l.dv} fp32 = <b>${fB(l.vh*l.dk*l.dv*4)}</b> per layer, whether the context is 10 tokens or 1M.</p>`,
    toy:M.Snew.slice(0,2).map((s,h)=>strip(s,'S'+(h+1)+' new')).join('')};}
  case 'read':{const M=R.mix;return {title:'Read: o = S · q',what:`<p>The query reads the memory with one small matrix-vector product per head. There's no loop over past tokens and nothing grows with context. Then a gated RMSNorm and the output projection follow. A cost that stays flat however long the chat gets is why Qwen3.5 uses these for 3 of every 4 layers.</p>`,toy:strip(M.heads,'o')};}
  case 'oproj':{const M=R.mix;return {title:'Output projection',what:`<p>W_O mixes the heads back into ${N(a.h)} numbers. That's the attention block's contribution, and it's what gets added to the stream.${a.attn==='hybrid'&&k.mix==='full'?' Qwen3.5\'s full-attention layers first multiply by a sigmoid output gate, which lets a head switch itself off.':''}${a.sandwich?' Gemma normalizes this output before the add.':''}</p>`,toy:strip(M.heads,'heads')+strip(R.attn,'W_O·')};}
  case 'add1':case 'add2':{const x=kind==='add1'?R.xin:R.xmid,y=kind==='add1'?R.xmid:R.xout,f=kind==='add1'?R.attn:R.ffnOut;
    return {title:'Residual add',what:`<p><code>x ← x + block(x)</code>. The block wrote a correction on top of the stream, and nothing was overwritten. Stream size ‖x‖: <b>${TJ.norm2(x).toFixed(2)} → ${TJ.norm2(y).toFixed(2)}</b>. The block's contribution was ${TJ.norm2(f).toFixed(2)}.</p>`,toy:strip(x,'x')+strip(f,'+ block')+strip(y,'= new x'),extra:`<span class="tag">Stream size after each layer</span><canvas id="jSpark" width="420" height="110"></canvas>`};}
  case 'gu':case 'gate':case 'up':{const F=R.ffn,FF=(a.moe&&a.moe.denseFfn&&k.ffn==='dense')?a.moe.denseFfn:a.ffn;const lp=(3*a.h*FF)/(3*a.h*FF+(a.kv?a.h*(a.H+2*a.kv)*a.hd+a.H*a.hd*a.h:0));
    return {title:'Gate and up projections',what:`<p>Two matrices expand ${N(a.h)} numbers into ${N(FF)} each (toy: 16 → 32). Think of the ${N(FF)} hidden neurons as feature detectors. The FFN is where most of the weights live${a.kv?`: about <b>${(lp*100).toFixed(0)}%</b> of this layer's bytes`:''}, and it's where most of the model's "knowledge" is stored.</p>`,
      toy:strip(F.g,'gate')+strip(F.u,'up'),real:[['W_gate, W_up',`${N(a.h)} × ${N(FF)} each`],['Bytes',fB(2*a.h*FF*W)]]};}
  case 'act':case 'hidden':{const F=R.ffn,A=TJ.ACTS[actId()];const z=[...F.h].filter(v=>Math.abs(v)<0.05).length;
    return {title:'Activation: '+A.name,what:`<p>${A.glu?`Gated: <code>h = act(gate) ⊙ up</code>. act(gate) works like a soft per-neuron switch, and up carries the content.`:'Not gated: <code>h = act(up)</code>.'} Used by ${A.who}.</p><ul><li><b>SiLU</b> z·σ(z): smooth, lets small negatives through, and never fully "dies"</li><li><b>GELU</b>: almost the same shape (tanh approximation)</li><li><b>ReLU</b>: zeroes every negative; cheap, but neurons can get stuck at 0</li></ul><p>Toy: ${z} of 32 neurons are ≈ 0 for this token. You can switch the activation above and watch the numbers change, or see it as a 3D surface in section 2.</p>`,
      toy:strip(F.g,'gate')+strip(F.av,'act(gate)')+strip(F.h,'h')};}
  case 'down':{const F=R.ffn;return {title:'Down projection',what:`<p>W_down maps the hidden neurons back to ${N(a.h)} numbers: the FFN's contribution to the stream.${a.sandwich?' (Gemma normalizes it first.)':''}</p>`,toy:F.h?strip(F.h,'h')+strip(R.ffnOut,'W_down·'):''};}
  case 'route':case 'router':{const F=R.ffn,m=a.moe;return {title:'Router: choose experts',what:`<p>A ${N(a.h)} × ${m.E} matrix scores every expert for this token. ${m.score==='softmax'?'Qwen3 applies a softmax over all experts, keeps the top 8, and renormalizes.':'DeepSeek applies a sigmoid to each expert, adds a per-expert balancing bias (used for <i>choosing</i> only), keeps the top 8, renormalizes, and scales by 2.5.'} Gold bars are the winners.</p>`,
    toy:`<div class="note">probabilities: ${F.probs.map((p,e)=>`<b style="color:var(${F.sel.includes(e)?'--gold':'--muted'})">E${e+1} ${p.toFixed(2)}</b>`).join(' · ')}</div>`,real:[['Router',`${N(a.h)} × ${m.E}`],['Chosen per token',`${m.k}${m.s?' + 1 shared':''} of ${m.E}`]]};}
  case 'experts':case 'expert':case 'shared':case 'combine':{const F=R.ffn,m=a.moe,f=TJ.stageCost(a,{kind:'experts',layer:li},o);const stored=(m.E+m.s)*3*a.h*a.ffn*W;
    return {title:kind==='combine'?'Combine the chosen experts':'Only the chosen experts run',what:`<p>Each expert is a small ${actName()} FFN, ${N(a.ffn)} wide. Only the ${m.k} chosen${m.s?' plus the shared one':''} are computed, so this layer reads <b>${fB(f.bytes)}</b> of the <b>${fB(stored)}</b> it stores (${(f.bytes/stored*100).toFixed(1)}%).</p><p>That's why MoE is fast for its size, and also why it still needs memory for <i>all</i> the experts. Every token can pick different ones.</p>${kind==='combine'?`<p>Output = Σ wₑ · expertₑ(x)${m.s?' + shared(x)':''}. Toy weights: ${F.w.map(x=>x.toFixed(2)).join(', ')}.</p>`:''}`,
      toy:strip(R.ffnOut,'output')};}
  case 'layer':{const kk=TJ.kind(a,li);return {title:`Layer ${li+1}`,what:`<p>${kk.mix==='linear'?'Gated DeltaNet (fixed state)':kk.mix==='local'?'Sliding-window attention':a.mla?(a.dsa?'MLA with sparse top-k':'Multi-head latent attention'):'Full attention'} followed by ${kk.ffn==='moe'?'a mixture-of-experts FFN':'a dense FFN'}, each wrapped in an RMSNorm and a residual add.</p><p>Stream size ${TJ.norm2(R.xin).toFixed(2)} → ${TJ.norm2(R.xout).toFixed(2)}.</p>${J.lv.level!==1||J.lv.layer!==li?`<button type="button" id="jFocusBtn" data-l="${li}">Open layer ${li+1} ▸</button>`:''}`,toy:strip(R.xin,'in')+strip(R.xout,'out')};}
  case 'attn':return {title:mixName(a,k),what:`<p>This is the only step where a token looks at <i>other</i> tokens. ${mixSub(a,k)}.</p><p>${a.blurb}</p>${J.lv.level===1?'<p><b>Click the block</b> to open it and watch every operation.</p>':''}`};
  case 'ffn':return {title:ffnName(a,k),what:`<p>Each token is processed on its own here, with no mixing across tokens. This is where most of the weights live, and where most of each token's bytes are read.</p>${k.ffn==='moe'?`<p>Only ${a.moe.k} of ${a.moe.E} experts run for each token${a.moe.s?', plus one shared expert':''}.</p>`:''}${J.lv.level===1?'<p><b>Click the block</b> to open it.</p>':''}`};
  case 'lmhead':return {title:'LM head: one score per word',what:`<p>The final vector is multiplied by the ${N(a.V)} × ${N(a.h)} unembedding matrix, giving one <b>logit</b> per vocabulary entry. Reading it costs <b>${fB(a.V*a.h*W)}</b>, often the single biggest matrix in a small model.${a.tied?' Gemma reuses the embedding table here ("tied"), which saves memory but not reading time.':''}</p>`,real:[['Matrix',`${N(a.V)} × ${N(a.h)}`],['Bytes',fB(a.V*a.h*W)]]};
  case 'sample':{const tp=+$('#jTemp').value/10;return {title:'Sample the next token',what:`<p>Divide the logits by temperature T = ${tp.toFixed(1)}, then apply a softmax to get probabilities. ${$('#jGreedy').checked?'Greedy: take the largest.':'Draw one at random using those probabilities.'} Lower T sharpens the distribution and higher T flattens it.</p><p>Chosen: <b>"${TJ.WORDS[J.choice]}"</b>. Press <b>Generate next token</b> and the trip starts again, for that one new token only. Everything before it is already in the cache.</p><p class="note">The toy's weights are random, so its word choices are noise. In ${a.short} the same pipeline produces sensible text because the weights were trained.</p>`};}
  }
  return {title:kind,what:''};
}
function drawSpark(cv,li){
  const g=cv.getContext('2d'),W=cv.width,H=cv.height,L=J.rec.layers;g.clearRect(0,0,W,H);
  const v=L.map(l=>l.norms.out),mx=Math.max(...v)*1.1;
  g.strokeStyle=css('--line');g.beginPath();g.moveTo(24,H-16);g.lineTo(W-6,H-16);g.stroke();
  g.fillStyle=css('--muted');g.font='10px '+css('--mono');g.fillText('‖x‖',2,12);g.fillText('layer 1',24,H-3);g.fillText(String(L.length),W-24,H-3);
  g.strokeStyle=css('--copper');g.lineWidth=2;g.beginPath();v.forEach((y,i)=>{const X=24+i/(v.length-1||1)*(W-32),Y=H-16-y/mx*(H-26);i?g.lineTo(X,Y):g.moveTo(X,Y);});g.stroke();
  if(li>=0){const X=24+li/(v.length-1||1)*(W-32),Y=H-16-v[li]/mx*(H-26);g.fillStyle=css('--gold');g.beginPath();g.arc(X,Y,4,0,7);g.fill();}
}

/* ---------- cost panel ---------- */
const CAT={emb:'Embedding',attnw:'Attention weights',cache:'Cache / state reads',ffn:'FFN / experts',lm:'LM head',misc:'Norms, adds, sampling'};
function catOf(k){if(k==='emb')return 'emb';if(['qkv','oproj','up','compress','rope'].includes(k))return 'attnw';if(['score','mix','index','decay','write','read','softmax'].includes(k))return 'cache';if(['gu','down','act','route','experts','combine'].includes(k))return 'ffn';if(k==='lmhead')return 'lm';return 'misc';}
function costPanel(){
  const a=J.a,o=opt(),H=hw(),ph=$('#jPhase').value;const sum={},bytes={};Object.keys(CAT).forEach(c=>{sum[c]=0;bytes[c]=0;});
  // full forward, every layer in detail, for an honest breakdown
  const st=[{kind:'emb',layer:-1}];for(let i=0;i<a.L;i++)st.push(...layerStages(i));st.push({kind:'fnorm',layer:-1},{kind:'lmhead',layer:-1},{kind:'sample',layer:-1});
  let tot=0,tb=0;st.forEach(s=>{const c=TJ.stageCost(a,s,o),t=TJ.time(c,H,ph,o.ctx);const k=catOf(s.kind);sum[k]+=t.t;bytes[k]+=c.bytes;tot+=t.t;tb+=c.bytes;});J.totalReal=tot;
  const tt=TJ.totals(a),stored=tt.stored*o.bpw/8,fits=stored/1073741824<=H.mem*0.92;
  const cols={emb:'--steel',attnw:'--violet',cache:'--teal',ffn:'--copper',lm:'--gold',misc:'--line'};
  $('#jCostBar').innerHTML=Object.keys(CAT).map(k=>`<i data-c="${k}" style="flex:${Math.max(sum[k],tot*0.004)};background:var(${cols[k]})" title="${CAT[k]}: ${fT(sum[k])}"></i>`).join('');
  $('#jCostLegend').innerHTML=Object.keys(CAT).map(k=>`<span data-c="${k}"><i style="background:var(${cols[k]})"></i>${CAT[k]} <b>${fT(sum[k])}</b> · ${fB(bytes[k])}</span>`).join('');
  const P=o.ctx;
  $('#jCostOut').innerHTML=`<div class="metric"><span class="k">${ph==='prefill'?`Prefill of ${fK(P)} tokens`:'One decode step'}</span><span class="v">${fT(tot)}</span><span class="s">ideal, at full bandwidth</span></div>
   <div class="metric"><span class="k">${ph==='prefill'?'Prompt tokens / s':'Tokens / s ceiling'}</span><span class="v">${N(ph==='prefill'?P/tot:1/tot)}</span><span class="s">real runs reach ~70–80%</span></div>
   <div class="metric"><span class="k">Bytes read per step</span><span class="v">${fB(tb)}</span><span class="s">${N(tt.active/1e6)} M weights active</span></div>
   <div class="metric"><span class="k">Weights stored</span><span class="v">${fB(stored)}</span><span class="s"><span class="pill ${fits?'good':'bad'}">${fits?'fits':'does not fit'}</span> in ${H.mem} GB</span></div>`;
}
function costHighlight(st){const c=catOf(st.kind==='layer'||st.kind==='attn'?'qkv':st.kind==='ffn'?'gu':st.kind);document.querySelectorAll('#jCostBar i,#jCostLegend span').forEach(e=>e.classList.toggle('on',st.kind!=='layer'&&e.dataset.c===c));}
})();
