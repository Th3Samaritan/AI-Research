/* Token Journey · the 3D walk-through: tower of layers, magnified layer, packets, inspector, cost. */
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
  canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>5)drag.moved=true;if(!drag.moved)return;st.theta=drag.t-dx*0.008;st.phi=Math.max(0.15,Math.min(1.52,drag.p-dy*0.006));onChange();});
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
const J={archId:'qwen3-8b',focus:0,toks:[],gen:[],rec:null,W:{},stages:[],idx:0,t:0,playing:false,pin:null,choice:null,first:true};
TJ.J=J;
const VIEWS={all:{t:[-1.6,6.8,0],r:31,th:0.42,ph:1.22},tower:{t:[-8,6.2,0],r:17,th:0.35,ph:1.25},emb:{t:[-6,0.6,0],r:12,th:0.4,ph:1.12},
  attn:{t:[2.6,3.9,0],r:13,th:0.5,ph:1.12},ffn:{t:[2.6,9.1,0],r:13,th:0.5,ph:1.1},out:{t:[1.2,13.2,0],r:12.5,th:0.4,ph:1.18}};
const TX=-8.2,BX=-4,Y={in:0.8,n1:1.5,a0:2.1,a1:5.9,add1:6.3,n2:6.95,f0:7.4,f1:10.9,add2:11.35,top:11.95,fn:12.45,lm:13.1};
const NX=BX+1.7;

/* ---------- controls ---------- */
function fillControls(){
  $('#jArch').innerHTML=TJ.ARCH.map(a=>`<option value="${a.id}"${a.id===J.archId?' selected':''}>${a.name}</option>`).join('');
  $('#jHw').innerHTML=TJ.HW.map((h,i)=>`<option value="${i}">${h.name}</option>`).join('');
  $('#jFmt').innerHTML=TJ.FMT.map(f=>`<option value="${f.id}"${f.id==='bf16'?' selected':''}>${f.name}</option>`).join('');
  $('#jAct').innerHTML='<option value="">Model default</option>'+Object.entries(TJ.ACTS).map(([k,v])=>`<option value="${k}">${v.name}</option>`).join('');
}
function focusOptions(){
  const a=J.a;$('#jFocus').innerHTML=Array.from({length:a.L},(_,i)=>{const k=TJ.kind(a,i);return `<option value="${i}"${i===J.focus?' selected':''}>Layer ${i+1} · ${k.mix==='linear'?'DeltaNet':k.mix==='local'?'local attn':a.attn==='mla'||a.attn==='dsa'?(a.dsa?'MLA+DSA':'MLA'):'full attn'} · ${k.ffn==='moe'?'MoE':'dense FFN'}</option>`;}).join('');
}
const opt=()=>({bpw:TJ.FMT.find(f=>f.id===$('#jFmt').value).bpw,kvB:2,ctx:CTX[+$('#jCtx').value]});
const hw=()=>TJ.HW[+$('#jHw').value];
const actId=()=>$('#jAct').value||J.a.act;

/* ---------- run the toy model ---------- */
function compute(){
  const a=J.a;if(!J.W[a.id])J.W[a.id]=TJ.buildToy(a);
  const toks=J.toks.concat(J.gen);
  J.rec=TJ.forward(a,J.W[a.id],toks,actId());
  const temp=+$('#jTemp').value/10;J.probs=TJ.probs(J.rec.logits,temp);
  if($('#jGreedy').checked){let b=0;J.probs.forEach((p,i)=>{if(p>J.probs[b])b=i;});J.choice=b;}
  else{const r=TJ.mulberry(TJ.hash($('#jPrompt').value)+toks.length*7919);let u=r(),c=0;J.choice=J.probs.length-1;for(let i=0;i<J.probs.length;i++){c+=J.probs[i];if(u<=c){J.choice=i;break;}}}
}
function allToks(){return J.toks.concat(J.gen);}

/* ---------- stages ---------- */
const BASE={tok:1.4,emb:1.8,in:0.6,norm1:1.3,qkv:1.8,rope:1.4,compress:1.5,up:1.4,index:1.9,score:1.7,softmax:1.5,mix:1.5,decay:1.7,write:1.9,read:1.3,oproj:1.3,add1:1.3,norm2:1.2,gu:1.7,act:1.7,down:1.4,route:1.7,experts:1.7,combine:1.3,add2:1.3,layer:0.26,fnorm:1.0,lmhead:1.7,sample:2.0};
const NAME={tok:'Tokenize',emb:'Embed',in:'Enter layer',norm1:'RMSNorm',qkv:'Q · K · V',rope:'RoPE',compress:'Compress (latent)',up:'Absorb W_UK',index:'Indexer top-k',score:'Scores q·k',softmax:'Softmax',mix:'Mix values',decay:'Decay α',write:'Delta write',read:'Read S·q',oproj:'Out proj',add1:'+ residual',norm2:'RMSNorm',gu:'Gate · Up',act:'Activation',down:'Down proj',route:'Router',experts:'Experts',combine:'Combine',add2:'+ residual',layer:'Layer',fnorm:'Final norm',lmhead:'LM head',sample:'Sample'};
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
function buildStages(){
  const a=J.a,st=[{kind:'tok',layer:-1},{kind:'emb',layer:-1}];
  const all=$('#jAll').checked;
  for(let i=0;i<a.L;i++){if(all||i===J.focus)st.push(...layerStages(i));else st.push({kind:'layer',layer:i});}
  st.push({kind:'fnorm',layer:-1},{kind:'lmhead',layer:-1},{kind:'sample',layer:-1});
  const o=opt(),h=hw(),ph=$('#jPhase').value,P=o.ctx;
  st.forEach(s=>{s.cost=TJ.stageCost(a,s,o);s.time=TJ.time(s.cost,h,ph,P);});
  const tot=st.reduce((x,s)=>x+s.time.t,0);
  const real=$('#jPace').value==='real';
  st.forEach(s=>{s.dur=real?Math.max(0.07,s.time.t/tot*32):BASE[s.kind];});
  J.stages=st;J.totalReal=tot;
  $('#jChips').innerHTML=st.map((s,i)=>`<button type="button" data-i="${i}" class="${s.kind==='layer'?'mini':''}" title="${s.layer>=0?'Layer '+(s.layer+1)+' · ':''}${NAME[s.kind]}">${s.kind==='layer'?(s.layer+1):NAME[s.kind]}</button>`).join('');
}

/* ---------- 3D ---------- */
let S=null,G=null;const anchors=[];
function label(text,pos,cls){const el=document.createElement('div');el.className='lbl3'+(cls?' '+cls:'');el.textContent=text;$('#jLabels').appendChild(el);anchors.push({el,pos});return el;}
function box(w,h,d,mat,x,y,z,pick){const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),mat);m.position.set(x,y,z);if(pick)m.userData.pick=pick;return m;}
function line(pts,color,op){const g=new THREE.BufferGeometry().setFromPoints(pts.map(p=>new THREE.Vector3(...p)));return new THREE.Line(g,new THREE.LineBasicMaterial({color:col(color),transparent:true,opacity:op==null?1:op}));}

function packet(){
  const g=new THREE.Group(),bars=[];
  const plate=new THREE.Mesh(new THREE.BoxGeometry(0.34,0.03,D.D*0.2+0.1),lam('--ink',{transparent:true,opacity:0.25}));g.add(plate);
  for(let i=0;i<D.D;i++){const m=new THREE.Mesh(new THREE.BoxGeometry(0.14,1,0.14),new THREE.MeshLambertMaterial({color:0xffffff,transparent:true,opacity:1}));m.position.z=-((D.D-1)*0.2)/2+i*0.2;g.add(m);bars.push(m);}
  g.userData={bars,plate};
  return g;
}
function setPacket(p,v,op){
  const {bars,plate}=p.userData;
  bars.forEach((b,i)=>{const x=v?v[i]:0,h=Math.max(-1.1,Math.min(1.1,x*0.32));b.scale.y=Math.abs(h)+0.02;b.position.y=h/2;b.material.color.copy(vcol(x,2.2));b.material.opacity=op;});
  plate.material.opacity=0.25*op;p.visible=op>0.01;
}
const lerpV=(a,b,t)=>a&&b?Float32Array.from(a,(x,i)=>x+(b[i]-x)*t):(b||a);

function buildStatic(){
  if(!S)return;clear(G.st);clear(G.dt);$('#jLabels').innerHTML='';anchors.length=0;J.pick=[];
  const a=J.a,g=G.st,sh=10/a.L*0.74;
  // tower
  J.slabs=[];
  for(let i=0;i<a.L;i++){
    const k=TJ.kind(a,i),y=1+i*(10/a.L)+sh/2;
    const c=k.mix==='linear'?'--violet':k.mix==='local'?'--teal':'--copper';
    const m=box(2.4,sh,1.4,lam(c,{transparent:true,opacity:i===J.focus?1:0.78}),TX,y,0,{type:'layer',layer:i});g.add(m);J.pick.push(m);J.slabs.push(m);
    if(k.ffn==='moe'){const s=box(2.42,sh*0.35,1.42,lam('--gold'),TX,y-sh*0.3,0);g.add(s);}
    if(i===J.focus){const e=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2.6,sh+0.08,1.6)),new THREE.LineBasicMaterial({color:col('--ink')}));e.position.set(TX,y,0);g.add(e);
      // magnifier lines from the focused slab to the detail area
      [[1.3,sh/2,0.8,Y.a0],[1.3,-sh/2,0.8,Y.in],[1.3,sh/2,-0.8,Y.add2],[1.3,-sh/2,-0.8,Y.in]].forEach(([dx,dy,dz,ty])=>g.add(line([[TX+dx,y+dy,dz],[BX-0.4,ty,dz*2]],'--ink',0.18)));}
  }
  label(`${a.L} layers`,new THREE.Vector3(TX,11.4,0),'b');
  label('Tower: the whole model',new THREE.Vector3(TX,-1.25,0));
  // embedding table
  const et=box(3,0.42,2,lam('--steel',{transparent:true,opacity:0.85}),TX,0.2,0,{type:'emb'});g.add(et);J.pick.push(et);
  for(let r=0;r<9;r++)g.add(line([[TX-1.5,0.42,-1+r*0.25],[TX+1.5,0.42,-1+r*0.25]],'--ink',0.25));
  J.embRow=box(3.05,0.06,0.22,new THREE.MeshBasicMaterial({color:col('--gold')}),TX,0.45,0);g.add(J.embRow);
  label(`Embedding ${N(a.V)} × ${N(a.h)}`,new THREE.Vector3(TX,-0.25,1.3));
  // residual bus
  const bus=new THREE.Mesh(new THREE.CylinderGeometry(0.07,0.07,Y.fn-0.4,12),lam('--copper',{transparent:true,opacity:0.55}));bus.position.set(BX,(Y.fn+0.4)/2,0);bus.userData.pick={type:'residual'};g.add(bus);J.pick.push(bus);
  label('Residual stream',new THREE.Vector3(BX,Y.fn+0.25,0),'b');
  // norms and adds
  const ring=(y,kind)=>{const m=new THREE.Mesh(new THREE.TorusGeometry(0.42,0.07,10,28),lam('--violet'));m.position.set(NX,y,0);m.rotation.y=Math.PI/2;m.userData.pick={type:kind};g.add(m);J.pick.push(m);g.add(line([[BX,y,0],[NX-0.4,y,0]],'--violet',0.6));};
  ring(Y.n1,'norm1');ring(Y.n2,'norm2');
  const fnr=new THREE.Mesh(new THREE.TorusGeometry(0.42,0.07,10,28),lam('--violet'));fnr.position.set(BX,Y.fn,0);fnr.rotation.x=Math.PI/2;fnr.userData.pick={type:'fnorm'};g.add(fnr);J.pick.push(fnr);
  const add=(y,kind)=>{const m=new THREE.Mesh(new THREE.SphereGeometry(0.26,18,12),lam('--gold'));m.position.set(BX,y,0);m.userData.pick={type:kind};g.add(m);J.pick.push(m);label('+',new THREE.Vector3(BX-0.55,y,0),'b');};
  add(Y.add1,'add1');add(Y.add2,'add2');
  label('RMSNorm',new THREE.Vector3(NX,Y.n1-0.6,0));label('RMSNorm',new THREE.Vector3(NX,Y.n2-0.6,0));
  // region frames (pickable, lowest priority)
  const frame=(y0,y1,kind,txt)=>{const h=y1-y0;const m=box(8.2,h,4.6,new THREE.MeshBasicMaterial({color:col('--ink'),transparent:true,opacity:0.03,depthWrite:false}),2.9,y0+h/2,0,{type:kind,region:true});g.add(m);J.pick.push(m);
    const e=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(8.2,h,4.6)),new THREE.LineBasicMaterial({color:col('--line')}));e.position.copy(m.position);g.add(e);
    label(txt,new THREE.Vector3(0.6,y1+0.3,2.3),'b');
    g.add(line([[BX,y0+0.1,0],[-1.2,y0+0.1,0]],'--line',0.8));g.add(line([[-1.2,y1-0.1,0],[BX,kind==='attn'?Y.add1:Y.add2,0]],'--line',0.8));};
  const k=TJ.kind(a,J.focus);
  const mixTxt=k.mix==='linear'?'Gated DeltaNet (linear attention)':a.dsa?'MLA + DeepSeek Sparse Attention':a.mla?'Multi-head latent attention':k.mix==='local'?`Sliding-window attention · ${a.H}Q/${a.kv}KV`:`${a.attn==='mha'?'Multi-head':a.attn==='mqa'?'Multi-query':'Grouped-query'} attention · ${a.H}Q / ${a.kv}KV`;
  frame(Y.a0,Y.a1,'attn',`Layer ${J.focus+1} · ${mixTxt}`);
  frame(Y.f0,Y.f1,'ffn',k.ffn==='moe'?`Mixture of experts · ${a.moe.E} experts, top-${a.moe.k}${a.moe.s?' + 1 shared':''}`:`Feed-forward · ${actName()}`);
  // LM head
  const lm=box(7,0.45,2.2,lam('--gold',{transparent:true,opacity:0.8}),1.6,Y.lm,0,{type:'lmhead'});g.add(lm);J.pick.push(lm);
  label(`LM head ${N(a.h)} → ${N(a.V)}${a.tied?' (tied to embedding)':''}`,new THREE.Vector3(1.6,Y.lm-0.45,1.2));
  // packets
  J.A=packet();J.B=packet();J.A.position.set(BX,Y.in,0);g.add(J.A);g.add(J.B);
  J.ball=new THREE.Mesh(new THREE.SphereGeometry(0.2,18,12),new THREE.MeshBasicMaterial({color:col('--gold')}));g.add(J.ball);
  J.ghosts=[];for(let i=0;i<13;i++){const s=new THREE.Mesh(new THREE.SphereGeometry(0.11,10,8),new THREE.MeshBasicMaterial({color:col('--gold'),transparent:true,opacity:0.5}));g.add(s);J.ghosts.push(s);}
  buildDetail();
}
function actName(){return TJ.ACTS[actId()].name.split(' ·')[0];}

/* the magnified layer: everything depends on the focused layer's record */
function buildDetail(){
  clear(G.dt);J.dpick=[];J.d={};
  const a=J.a,R=J.rec.layers[J.focus],g=G.dt,T=J.rec.T,d=J.d;
  const zq=j=>-1.75+j*0.5;
  if(R.mix.type==='lin'){
    const M=R.mix;d.grids=[];d.chips=[];d.gauges=[];
    for(let h=0;h<D.LH;h++){
      const cells=[];const x0=0.6+h*1.65;
      for(let r=0;r<D.LD;r++)for(let c=0;c<D.LD;c++){const m=box(0.3,1,0.3,new THREE.MeshLambertMaterial({color:0xffffff}),x0+c*0.36,3.0,-0.55+r*0.36,{type:'state',head:h});g.add(m);J.dpick.push(m);cells.push(m);}
      d.grids.push(cells);
      const ga=box(0.12,1,0.12,lam('--teal'),x0-0.35,3.0,-1.2);const gb=box(0.12,1,0.12,lam('--gold'),x0-0.15,3.0,-1.2);g.add(ga);g.add(gb);d.gauges.push([ga,gb]);
      label(`state S${h+1} (4×4)`,new THREE.Vector3(x0+0.5,2.55,1.1));
    }
    const qx=-0.4;d.q=[];for(let h=0;h<D.LH;h++){const m=new THREE.Mesh(new THREE.CylinderGeometry(0.1,0.1,1,12),lam('--copper'));m.position.set(qx,4.4,zq(h*2+0.5));m.userData.pick={type:'qheads'};g.add(m);J.dpick.push(m);d.q.push(m);}
    const toks=allToks(),dx=Math.min(0.5,6.4/T);
    for(let s=0;s<T;s++){const m=box(dx*0.8,0.18,0.5,lam('--steel',{transparent:true,opacity:1}),0.4+s*dx,4.9,1.6,{type:'pastTok',s});g.add(m);J.dpick.push(m);d.chips.push(m);}
    label('past tokens: only a fading trace survives in S',new THREE.Vector3(0.4,5.25,1.6));
    label('α keep · β write',new THREE.Vector3(0.3,2.45,-1.4));
  }else{
    const M=R.mix,isMla=M.type==='mla';const nq=D.NQ;d.q=[];
    for(let j=0;j<nq;j++){const m=new THREE.Mesh(new THREE.CylinderGeometry(0.1,0.1,1,12),lam('--copper'));m.position.set(-0.4,3.6,zq(j));m.userData.pick={type:'qheads'};g.add(m);J.dpick.push(m);d.q.push(m);}
    label(`${nq} query heads (real ${a.H})`,new THREE.Vector3(-0.4,4.75,-2.3));
    const dx=Math.min(0.42,5.6/T),x0=1.2;d.K=[];d.V=[];d.lines=[];d.lat=[];
    if(isMla){
      const plate=box(0.08,1.6,4.0,lam('--violet',{transparent:true,opacity:0.35}),0.45,3.6,0,{type:'absorb'});g.add(plate);J.dpick.push(plate);d.plate=plate;
      for(let s=0;s<T;s++){const colm=[];for(let c=0;c<D.LC;c++){const m=box(dx*0.7,0.2,0.36,new THREE.MeshLambertMaterial({color:0xffffff,transparent:true}),x0+s*dx,2.85+c*0.24,0,{type:'cache',s});g.add(m);J.dpick.push(m);colm.push(m);}
        const kr=box(dx*0.7,0.2,0.36,new THREE.MeshLambertMaterial({color:col('--gold'),transparent:true}),x0+s*dx,2.85+D.LC*0.24+0.05,0,{type:'cache',s});g.add(kr);J.dpick.push(kr);colm.push(kr);d.lat.push(colm);}
      label('latent cache: 4 + 1 RoPE pair per token (real 512 + 64)',new THREE.Vector3(x0+2,4.25,0.6));
      for(let j=0;j<nq;j++)for(let s=0;s<T;s++){const l=line([[-0.3,3.6,zq(j)],[0.45,3.6,zq(j)*0.6],[x0+s*dx,3.4,0]],'--copper',0);g.add(l);d.lines.push({l,j,s});}
      if(M.idx){d.idx=[];for(let s=0;s<T;s++){const m=box(dx*0.6,1,0.2,new THREE.MeshLambertMaterial({color:col('--steel')}),x0+s*dx,4.75,1.4,{type:'indexer',s});g.add(m);J.dpick.push(m);d.idx.push(m);}
        label(`lightning indexer → top-${D.TOPK} (real top-2,048)`,new THREE.Vector3(x0+2,5.45,1.4));}
    }else{
      const kvT=M.kvT,zg=gi=>{let z=0,c=0;for(let j=0;j<nq;j++)if(Math.floor(j*kvT/nq)===gi){z+=zq(j);c++;}return z/c;};
      for(let gi=0;gi<kvT;gi++){
        for(let j=0;j<nq;j++)if(Math.floor(j*kvT/nq)===gi)g.add(line([[-0.3,3.15,zq(j)],[x0-0.35,3.15,zg(gi)]],'--violet',0.55));
        const rowK=[],rowV=[];
        for(let s=0;s<T;s++){
          const kc=box(dx*0.72,0.26,0.26,new THREE.MeshLambertMaterial({color:col('--teal'),transparent:true}),x0+s*dx,3.6,zg(gi),{type:'cache',s,g:gi});
          const vc=box(dx*0.72,0.26,0.26,new THREE.MeshLambertMaterial({color:col('--teal2'),transparent:true}),x0+s*dx,2.9,zg(gi),{type:'cache',s,g:gi});
          g.add(kc);g.add(vc);J.dpick.push(kc,vc);rowK.push(kc);rowV.push(vc);
        }
        d.K.push(rowK);d.V.push(rowV);
      }
      label(`KV cache · ${kvT} K/V head${kvT>1?'s':''} (real ${a.kv}) · top K, bottom V`,new THREE.Vector3(x0+2.4,2.35,-2.0));
      for(let j=0;j<nq;j++){const gi=Math.floor(j*kvT/nq);for(let s=0;s<T;s++){const l=line([[-0.3,3.95,zq(j)],[x0+s*dx,3.75,zg(gi)]],'--copper',0);g.add(l);d.lines.push({l,j,s});}}
      if(M.local)label(`local layer: sees only the last ${D.WIN} tokens (real ${N(a.window)})`,new THREE.Vector3(x0+2,5.2,0));
    }
    d.dx=dx;d.x0=x0;
    label('new token →',new THREE.Vector3(x0+(T-1)*dx,2.45,2.1));
  }
  // FFN
  const F=R.ffn;
  if(F.type==='dense'){
    d.gate=[];d.up=[];d.h=[];
    const mk=(arr,y,kind)=>{for(let i=0;i<D.F;i++){const m=box(0.34,1,0.34,new THREE.MeshLambertMaterial({color:0xffffff}),0.3+(i%8)*0.62,y,-1.05+Math.floor(i/8)*0.7,{type:kind,i});g.add(m);J.dpick.push(m);arr.push(m);}};
    mk(d.gate,7.95,'gate');mk(d.up,8.85,'up');mk(d.h,10.05,'hidden');
    label('gate = W_g·x (32; real '+N((a.moe&&a.moe.denseFfn)||a.ffn)+')',new THREE.Vector3(7.2,7.95,0.9));
    label('up = W_u·x',new THREE.Vector3(6.6,8.85,0.9));
    label('act(gate) × up → W_down',new THREE.Vector3(5.0,10.65,0.9));
    const pl=box(0.1,1.4,3.0,lam('--violet',{transparent:true,opacity:0.35}),5.8,9.4,0,{type:'down'});g.add(pl);J.dpick.push(pl);
  }else{
    d.ex=[];d.rb=[];
    for(let e=0;e<D.E;e++){const x=0.4+(e%4)*1.35,z=e<4?-0.9:0.9;
      const m=box(0.9,0.9,0.9,new THREE.MeshLambertMaterial({color:col('--steel'),emissive:0x000000}),x,8.4,z,{type:'expert',e});g.add(m);J.dpick.push(m);d.ex.push(m);
      const b=box(0.22,1,0.22,lam('--gold'),x,9.0,z);g.add(b);d.rb.push(b);}
    const r=box(0.3,1.2,2.6,lam('--violet',{transparent:true,opacity:0.5}),-0.7,8.4,0,{type:'router'});g.add(r);J.dpick.push(r);d.router=r;
    if(F.shared){const m=box(1.0,1.0,1.0,new THREE.MeshLambertMaterial({color:col('--teal'),emissive:0x000000}),6.2,8.4,0,{type:'shared'});g.add(m);J.dpick.push(m);d.sh=m;label('shared expert (always on)',new THREE.Vector3(6.2,7.65,0));}
    label(`8 toy experts, top-2 (real ${a.moe.E}, top-${a.moe.k})`,new THREE.Vector3(2.4,10.3,1.6));
    d.elines=F.sel.map(e=>{const l=line([[-0.55,8.4,0],[0.4+(e%4)*1.35,8.4,e<4?-0.9:0.9]],'--gold',0);g.add(l);return l;});
  }
  // output bars
  const top=[...J.probs].map((p,i)=>[p,i]).sort((x,y)=>y[0]-x[0]).slice(0,7);
  if(!top.some(t=>t[1]===J.choice))top[6]=[J.probs[J.choice],J.choice];
  d.out=top.map(([p,i],n)=>{const m=box(0.55,1,0.55,lam(i===J.choice?'--copper':'--steel'),-1.6+n*1.05,Y.lm+0.4,0,{type:'sample'});g.add(m);J.dpick.push(m);
    const el=label(`${TJ.WORDS[i]} ${(p*100).toFixed(0)}%`,new THREE.Vector3(-1.6+n*1.05,Y.lm+(n%2?0.15:0.75),0.9),i===J.choice?'b':'');el.dataset.out='1';return {m,p,i,el};});
}

/* ---------- applying a stage at progress t ---------- */
function ease(t){return t<0.5?2*t*t:1-Math.pow(-2*t+2,2)/2;}
function apply(){
  if(!S||!J.stages.length)return;
  const st=J.stages[J.idx],t=ease(Math.min(1,J.t)),R=st.layer>=0?J.rec.layers[st.layer]:null,a=J.a,d=J.d,FR=J.rec.layers[J.focus];
  const A=J.A,B=J.B,k=st.kind;
  const at=(p,x,y,z)=>p.position.set(x,y,z||0);
  const mv=(p,f,to)=>at(p,f[0]+(to[0]-f[0])*t,f[1]+(to[1]-f[1])*t,(f[2]||0)+((to[2]||0)-(f[2]||0))*t);
  let av=null,aop=1,bv=null,bop=0;
  // tower ball
  const sh=10/a.L;const ly=i=>1+i*sh+sh*0.37;
  if(st.layer>=0)at(J.ball,TX+1.35,ly(st.layer)+(k==='layer'||k==='add2'?t*sh*0.5:0),0.75);else at(J.ball,TX+1.35,k==='tok'||k==='emb'?0.5:11.3,0.75);
  J.ghosts.forEach((gh,i)=>{const on=J.first&&i<J.rec.T-1&&J.ball.position.y>0.6&&J.ball.position.y<11.2;gh.visible=on;if(on)gh.position.set(TX+1.35,J.ball.position.y,0.75-(i+1)*0.13);});
  // embedding row
  const tk=allToks()[J.rec.T-1];J.embRow.position.z=-0.95+(tk.id%40)/40*1.9;J.embRow.visible=k==='emb'||k==='tok';
  // focus-layer detail visuals: defaults
  const before=(kind)=>{const ls=layerStages(J.focus).map(s=>s.kind);const ci=st.layer===J.focus?ls.indexOf(k):(st.layer>J.focus||st.layer<0&&J.idx>2?99:-1);const ki=ls.indexOf(kind);return ci>ki?1:ci===ki?t:0;};
  const dim=(st.layer>=0&&st.layer!==J.focus)||st.layer<0;
  detailVisuals(before,dim);
  switch(k){
    case 'tok':av=null;aop=0;break;
    case 'emb':av=lerpV(J.rec.embRaw,J.rec.emb,t);mv(A,[TX,0.7,0],[BX,Y.in,0]);break;
    case 'in':av=R.xin;mv(A,[BX,Y.in],[BX,Y.n1]);break;
    case 'norm1':av=R.xin;at(A,BX,Y.n1);bv=lerpV(R.xin,R.n1,t);bop=1;mv(B,[BX,Y.n1],[NX,Y.n1]);break;
    case 'qkv':case 'compress':case 'rope':case 'up':case 'index':case 'decay':case 'write':
      av=R.xin;at(A,BX,Y.n1);bv=R.n1;bop=1;
      if(k==='qkv')mv(B,[NX,Y.n1],[-1.4,2.4]);else at(B,-1.4,2.4);break;
    case 'score':case 'softmax':av=R.xin;at(A,BX,Y.n1);bv=R.n1;bop=1;at(B,-1.4,2.4);break;
    case 'mix':case 'read':av=R.xin;at(A,BX,Y.n1);bv=lerpV(R.n1,R.mix.heads,t);bop=1;mv(B,[-1.4,2.4],[6.4,5.3]);break;
    case 'oproj':av=R.xin;at(A,BX,Y.n1);bv=lerpV(R.mix.heads,R.attn,t);bop=1;mv(B,[6.4,5.3],[BX+0.9,Y.add1]);break;
    case 'add1':av=lerpV(R.xin,R.xmid,t);mv(A,[BX,Y.n1],[BX,Y.add1]);bv=R.attn;bop=1-t;mv(B,[BX+0.9,Y.add1],[BX,Y.add1]);break;
    case 'norm2':av=R.xmid;at(A,BX,Y.add1+0.3);mv(A,[BX,Y.add1],[BX,Y.n2]);bv=lerpV(R.xmid,R.n2,t);bop=1;mv(B,[BX,Y.n2],[NX,Y.n2]);break;
    case 'gu':case 'route':av=R.xmid;at(A,BX,Y.n2);bv=R.n2;bop=1;mv(B,[NX,Y.n2],[-1.6,7.7]);break;
    case 'act':case 'experts':av=R.xmid;at(A,BX,Y.n2);bv=R.n2;bop=1;mv(B,[-1.6,7.7],[-1.6,9.6]);break;
    case 'down':case 'combine':av=R.xmid;at(A,BX,Y.n2);bv=lerpV(R.n2,R.ffnOut,t);bop=1;mv(B,[-1.6,9.6],[BX+0.9,Y.add2]);break;
    case 'add2':av=lerpV(R.xmid,R.xout,t);mv(A,[BX,Y.n2],[BX,Y.top]);bv=R.ffnOut;bop=1-t;mv(B,[BX+0.9,Y.add2],[BX,Y.add2]);break;
    case 'layer':av=lerpV(R.xin,R.xout,t);mv(A,[BX,Y.in],[BX,Y.top]);break;
    case 'fnorm':{const L=J.rec.layers[a.L-1].xout;av=lerpV(L,J.rec.final,t);mv(A,[BX,Y.top],[BX,Y.fn+0.2]);break;}
    case 'lmhead':av=J.rec.final;mv(A,[BX,Y.fn+0.2],[1.6,Y.lm+0.35]);aop=1-t*0.6;break;
    case 'sample':av=J.rec.final;at(A,1.6,Y.lm+0.35);aop=0.4*(1-t);break;
  }
  setPacket(A,av,aop);setPacket(B,bv,bop);
  // output bars
  const showOut=k==='lmhead'?t*0.5:k==='sample'?0.5+t*0.5:0;
  d.out.forEach(o=>{const h=Math.max(0.02,(k==='sample'?o.p:Math.max(...J.probs)*0.6)*3.2*showOut*1.6);o.m.scale.y=h;o.m.position.y=Y.lm+0.25+h/2;o.m.visible=showOut>0;o.el.style.display=showOut>0.5?'':'none';});
  // tokens overlay
  document.querySelectorAll('#jToks .tk').forEach((el,i,arr)=>el.classList.toggle('cur',i===arr.length-1&&(k==='tok'||k==='emb')));
  document.querySelectorAll('#jChips button').forEach((b,i)=>b.classList.toggle('on',i===J.idx));
  // slab highlight
  J.slabs.forEach((m,i)=>{m.material.emissive=new THREE.Color(i===st.layer?col('--gold'):0x000000);m.material.emissiveIntensity=0.5;});
  if(!J.pin)inspect({stage:st});
  costHighlight(st);
  $('#jStageName').textContent=(st.layer>=0?`Layer ${st.layer+1} of ${a.L} · `:'')+NAME[k];
  $('#jReal').textContent=`real ${fT(st.time.t)} · ${st.time.bound}-bound`;
  if($('#jFollow').checked&&!J.userCam)goView(viewFor(k));
  S.dirty=true;
}
function viewFor(k){if(k==='tok'||k==='emb')return 'emb';if(k==='layer')return 'tower';if(['fnorm','lmhead','sample'].includes(k))return 'out';if(['norm2','gu','act','down','route','experts','combine','add2'].includes(k))return 'ffn';return 'attn';}

function detailVisuals(before,dim){
  const d=J.d,R=J.rec.layers[J.focus],M=R.mix,F=R.ffn,T=J.rec.T,a=J.a;
  const op=dim?0.35:1;
  if(M.type==='lin'){
    const pd=before('decay'),pw=before('write'),pq=before('qkv');
    d.grids.forEach((cells,h)=>cells.forEach((m,i)=>{const v0=M.Sprev[h][i],v1=M.Sdecay[h][i],v2=M.Snew[h][i];const v=pw>0?v1+(v2-v1)*pw:v0+(v1-v0)*pd;const hh=Math.max(0.03,Math.abs(v)*1.6);m.scale.y=hh;m.position.y=3.0+(v>=0?hh/2:-hh/2);m.material.color.copy(vcol(v,0.6));}));
    d.gauges.forEach(([ga,gb],h)=>{const al=M.alpha[h]*pq,be=M.beta[h]*pq;ga.scale.y=Math.max(0.02,al*1.2);ga.position.y=2.4+ga.scale.y/2;gb.scale.y=Math.max(0.02,be*1.2);gb.position.y=2.4+gb.scale.y/2;});
    const am=M.alpha.reduce((s,x)=>s+x,0)/M.alpha.length;
    d.chips.forEach((m,s)=>{const age=T-1-s;m.material.opacity=(s===T-1?pw:Math.pow(am,age+pd))*op;m.material.color.copy(col(s===T-1?'--gold':'--steel'));});
    d.q.forEach((m,h)=>{const v=pq*(0.4+Math.abs(M.q[h][0])*0.8);m.scale.y=Math.max(0.02,v);m.position.y=4.4;});
  }else{
    const pq=before('qkv'),ps=before('score'),pm=before('softmax'),pc=M.type==='mla'?before('compress'):pq,pi=before('index');
    d.q.forEach((m,j)=>{const qv=M.type==='mla'?0.9:Math.hypot(M.q[j][0],M.q[j][1]);const h=Math.max(0.02,pq*(0.3+qv*0.35));m.scale.y=h;m.position.y=3.6+h/2-0.1;});
    const sel=M.sel?new Set(M.sel):null;
    if(M.type==='mla'){
      d.lat.forEach((cm,s)=>cm.forEach((m,c)=>{const isNew=s===T-1;const v=c<D.LC?M.c[s][c]:M.kr[s][0];m.material.color.copy(c<D.LC?vcol(v,1.5).lerp(col('--violet'),0.45):col('--gold'));
        let o=isNew?pc:1;if(sel&&!sel.has(s)&&pi>0)o*=1-0.75*pi;m.material.opacity=o*op;m.scale.set(1,isNew?Math.max(0.01,pc):1,1);}));
      if(d.plate)d.plate.material.opacity=(0.15+0.5*before('up'))*op;
      if(d.idx){const mx=Math.max(...M.idx.map(Math.abs),1e-6);d.idx.forEach((m,s)=>{const h=Math.max(0.02,Math.abs(M.idx[s])/mx*0.9*pi);m.scale.y=h;m.position.y=4.6+h/2;m.material.color.copy(col(sel.has(s)&&pi>0.5?'--gold':'--steel'));});}
    }else{
      d.K.forEach((row,gi)=>row.forEach((m,s)=>{const isNew=s===T-1;const vis=!M.local||T-1-s<D.WIN;const o=(isNew?pq:1)*(vis?1:0.15)*op;m.material.opacity=o;m.scale.y=isNew?Math.max(0.01,pq):1;}));
      d.V.forEach((row,gi)=>row.forEach((m,s)=>{const isNew=s===T-1;const vis=!M.local||T-1-s<D.WIN;
        let wsum=0,c=0;for(let j=0;j<D.NQ;j++)if(Math.floor(j*M.kvT/D.NQ)===gi){wsum+=M.w[j][s]||0;c++;}
        const grow=1+pm*(wsum/c)*2.2;m.scale.set(1,(isNew?Math.max(0.01,pq):1)*grow,1);m.material.opacity=(isNew?pq:1)*(vis?1:0.15)*op;}));
    }
    d.lines.forEach(({l,j,s})=>{
      const raw=M.scores[j][s];if(raw==null){l.material.opacity=0;return;}
      const row=M.scores[j].filter(v=>v!=null),mx=Math.max(...row);
      const pre=Math.exp(raw-mx)*0.5,post=M.w[j][s];
      const o=ps<1?ps*pre:(pre+(post-pre)*pm);l.material.opacity=Math.min(1,o*1.3)*(before('oproj')>0.9?0.3:1)*op;});
  }
  if(F.type==='dense'){
    const pg=before('gu'),pa=before('act');const sc=x=>Math.max(-1.3,Math.min(1.3,x*0.45));
    const setb=(m,v,p,y)=>{const h=sc(v)*p;m.scale.y=Math.abs(h)+0.02;m.position.y=y+h/2;m.material.color.copy(vcol(v,2.5));};
    d.gate.forEach((m,i)=>setb(m,pa>0?F.g[i]+(F.av[i]-F.g[i])*pa:F.g[i],pg,7.95));
    d.up.forEach((m,i)=>setb(m,F.u[i],pg,8.85));
    d.h.forEach((m,i)=>setb(m,F.h[i],pa,10.05));
  }else{
    const pr=before('route'),pe=before('experts');const sel=new Set(F.sel);const mx=Math.max(...F.probs);
    d.rb.forEach((b,e)=>{const h=Math.max(0.02,F.probs[e]/mx*0.9*pr);b.scale.y=h;b.position.y=8.9+h/2;b.material.color.copy(col(sel.has(e)&&pr>0.6?'--gold':'--steel'));});
    d.ex.forEach((m,e)=>{m.material.emissive=new THREE.Color(sel.has(e)?col('--gold'):0x000000);m.material.emissiveIntensity=sel.has(e)?pe*0.7:0;m.material.color.copy(col(sel.has(e)&&pe>0?'--gold':'--steel'));});
    d.elines.forEach(l=>l.material.opacity=pe);
    if(d.sh){d.sh.material.emissive=col('--teal');d.sh.material.emissiveIntensity=pe*0.6;}
  }
}

/* ---------- camera ---------- */
function goView(v){const p=VIEWS[v];if(!p)return;J.camGoal=p;}
function camStep(){
  if(!J.camGoal)return;const p=J.camGoal,st=S.st,tg=S.target,f=reduce?1:0.07;
  tg.x+=(p.t[0]-tg.x)*f;tg.y+=(p.t[1]-tg.y)*f;tg.z+=(p.t[2]-tg.z)*f;st.r+=(p.r-st.r)*f;st.theta+=(p.th-st.theta)*f;st.phi+=(p.ph-st.phi)*f;
  if(Math.abs(p.r-st.r)<0.02&&Math.abs(p.t[1]-tg.y)<0.01)J.camGoal=null;
  S.place();S.dirty=true;
}

/* ---------- loop ---------- */
let last=0;
function loop(ts){
  const dt=Math.min(0.05,(ts-last)/1000||0);last=ts;
  if(S&&S.vis){
    if(J.playing&&J.stages.length){
      const sp=+$('#jSpeed').value;J.t+=dt*sp/J.stages[J.idx].dur;
      if(J.t>=1){if(J.idx<J.stages.length-1){J.idx++;J.t=0;}else{J.t=1;J.playing=false;onDone();}}
      apply();
    }
    camStep();
    if(S.dirty){S.render();placeLabels();S.dirty=false;}
  }
  requestAnimationFrame(loop);
}
function placeLabels(){anchors.forEach(({el,pos})=>{const p=project(S,pos);if(!p.ok){el.style.visibility='hidden';return;}el.style.visibility='';el.style.transform=`translate(${p.x}px,${p.y}px) translate(-50%,-50%)`;});}
function onDone(){$('#jPlay').textContent='▶ Play';$('#jNext').classList.add('pulse');}

/* ---------- inspector ---------- */
function rows(list){return `<div class="receipt">${list.map(([a,b])=>`<div class="row"><span>${a}</span><span>${b}</span></div>`).join('')}</div>`;}
function inspect(target){
  const a=J.a,o=opt(),H=hw();let st=target.stage,kind=st?st.kind:target.pick.type,li=st?st.layer:(target.pick.layer!=null?target.pick.layer:J.focus);
  if(kind==='layer'&&target.pick)li=target.pick.layer;
  if(li<0&&['qheads','cache','state','pastTok','absorb','indexer','gate','up','hidden','expert','router','shared','attn','ffn','norm1','norm2','add1','add2','down'].includes(kind))li=J.focus;
  const R=li>=0?J.rec.layers[li]:null,k=li>=0?TJ.kind(a,li):null;
  const cost=st?st:(()=>{const map={qheads:'qkv',cache:'score',state:'write',pastTok:'decay',absorb:'up',indexer:'index',gate:'gu',up:'gu',hidden:'act',expert:'experts',router:'route',shared:'experts',attn:'layer',ffn:'layer',residual:'in'};const kk=map[kind]||kind;const s={kind:kk,layer:li};s.cost=TJ.stageCost(a,s,o);s.time=TJ.time(s.cost,H,$('#jPhase').value,o.ctx);return s;})();
  const info=explain(kind,R,k,li,target.pick);
  const c=cost.cost||{bytes:0,flops:0},tm=cost.time||{t:0,bound:'memory',tm:0,tc:0};
  const costHtml=c.bytes||c.flops?rows([[`Bytes read (${a.short}, ${$('#jFmt').selectedOptions[0].text.split(' ·')[0]}, ${fK(o.ctx)} ctx)`,fB(c.bytes)],[`Math`,`${(c.flops/1e9).toFixed(c.flops<1e9?3:1)} GFLOP`],[`Time on ${H.name.split(' ·')[0]}`,`${fT(tm.t)} <span class="pill ${tm.bound==='memory'?'warn':'good'}">${tm.bound}-bound</span>`],[`Share of this token's time`,`${(tm.t/J.totalReal*100).toFixed(2)}%`]]):'';
  $('#jInspector').innerHTML=`<div class="ins-head"><span class="eyebrow">${li>=0?`Layer ${li+1} · `:''}${target.pick?'clicked':'now playing'}</span>${J.pin?'<button type="button" id="jUnpin">Follow animation</button>':''}</div>
   <h3>${info.title}</h3><div class="what">${info.what}</div>
   ${info.toy?`<div class="toyv"><span class="tag">Toy numbers (16-dim stand-in)</span>${info.toy}</div>`:''}
   ${info.real?`<span class="tag">${a.short}, real shapes</span>${rows(info.real)}`:''}
   ${costHtml?`<span class="tag">Cost for one token</span>${costHtml}`:''}
   ${info.extra||''}`;
  const u=$('#jUnpin');if(u)u.onclick=()=>{J.pin=null;apply();};
  const fb=$('#jFocusBtn');if(fb)fb.onclick=()=>{J.focus=+fb.dataset.l;$('#jFocus').value=J.focus;J.pin=null;restart(true);};
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
  case 'layer':{const kk=TJ.kind(a,li);return {title:`Layer ${li+1}`,what:`<p>${kk.mix==='linear'?'Gated DeltaNet (fixed state)':kk.mix==='local'?'Sliding-window attention':a.mla?(a.dsa?'MLA with sparse top-k':'Multi-head latent attention'):'Full attention'} followed by ${kk.ffn==='moe'?'a mixture-of-experts FFN':'a dense FFN'}. The steps are the same as in the magnified layer, just fast-forwarded here.</p><p>Stream size ${TJ.norm2(R.xin).toFixed(2)} → ${TJ.norm2(R.xout).toFixed(2)}.</p>${li!==J.focus?`<button type="button" id="jFocusBtn" data-l="${li}">Magnify this layer</button>`:''}`,toy:strip(R.xin,'in')+strip(R.xout,'out')};}
  case 'attn':return {title:'The attention block',what:`<p>This is the only place where a token looks at other tokens. ${a.blurb}</p><p>Click the parts: query heads, cached keys and values, the indexer or the state grids.</p>`};
  case 'ffn':return {title:'The feed-forward block',what:'<p>Each token is processed on its own, with no mixing across tokens. This is where most of the parameters, and most of the bytes read, live.</p>'};
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
  let tot=0,tb=0;st.forEach(s=>{const c=TJ.stageCost(a,s,o),t=TJ.time(c,H,ph,o.ctx);const k=catOf(s.kind);sum[k]+=t.t;bytes[k]+=c.bytes;tot+=t.t;tb+=c.bytes;});
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
function costHighlight(st){const c=catOf(st.kind==='layer'?'gu':st.kind);document.querySelectorAll('#jCostBar i,#jCostLegend span').forEach(e=>e.classList.toggle('on',st.kind!=='layer'&&e.dataset.c===c));}

/* ---------- picking ---------- */
const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();
function onClick(e){
  const r=S.cv.getBoundingClientRect();mouse.x=(e.clientX-r.left)/r.width*2-1;mouse.y=-(e.clientY-r.top)/r.height*2+1;
  ray.setFromCamera(mouse,S.cam);const hits=ray.intersectObjects(J.pick.concat(J.dpick||[]),false).filter(h=>h.object.visible&&h.object.userData.pick);
  if(!hits.length)return;const h=hits.find(x=>!x.object.userData.pick.region)||hits[0];
  J.pin=h.object.userData.pick;inspect({pick:J.pin});
}

/* ---------- actions ---------- */
function tokChips(){$('#jToks').innerHTML=allToks().map((t,i)=>`<span class="tk${t.gen?' gen':''}">${t.text}</span>`).join('')+(J.stages.length?`<span class="tk next">→ ?</span>`:'');}
function restart(rebuild){
  compute();if(rebuild)buildStatic();else buildDetail();buildStages();tokChips();
  J.idx=0;J.t=0;J.playing=!reduce;$('#jPlay').textContent=J.playing?'❚❚ Pause':'▶ Play';J.userCam=false;$('#jNext').classList.remove('pulse');
  costPanel();apply();document.dispatchEvent(new Event('tj-run'));
}
function run(){J.toks=TJ.tokenize($('#jPrompt').value||'the model',J.a);J.gen=[];J.first=true;restart(true);}
function next(){
  if(!J.rec)return;const w=TJ.WORDS[J.choice];J.gen.push({text:w,id:J.choice,rid:TJ.hash(J.a.id+w)%J.a.V,gen:true});
  if(J.toks.length+J.gen.length>14)J.toks.shift();
  J.first=false;restart(true);J.idx=2;J.t=0;apply();
}
function setArch(){J.a=TJ.arch($('#jArch').value);J.focus=J.a.moe&&J.a.moe.dense?3:0;focusOptions();$('#jArchBlurb').textContent=J.a.blurb;run();}
TJ.journeyRec=()=>J.rec;TJ.journeyFocus=()=>J.focus;TJ.journeyArch=()=>J.a;

function init(){
  fillControls();J.a=TJ.arch(J.archId);focusOptions();$('#jArchBlurb').textContent=J.a.blurb;
  if(!window.THREE){$('#jStage').insertAdjacentHTML('beforeend','<p class="hint">3D needs WebGL (three.js failed to load).</p>');}
  else{
    const st={theta:VIEWS.all.th,phi:VIEWS.all.ph,r:VIEWS.all.r,onGrab:()=>{J.camGoal=null;J.userCam=true;}},target=new THREE.Vector3(...VIEWS.all.t);
    S=makeScene($('#jCanvas'),$('#jStage'),st,target,onClick);G={st:new THREE.Group(),dt:new THREE.Group()};S.scene.add(G.st);S.scene.add(G.dt);S.place();S.resize();
  }
  J.toks=TJ.tokenize($('#jPrompt').value,J.a);compute();buildStatic();buildStages();tokChips();costPanel();J.idx=0;J.t=0;apply();
  $('#jArch').onchange=setArch;$('#jRun').onclick=run;$('#jNext').onclick=next;
  $('#jPrompt').addEventListener('keydown',e=>{if(e.key==='Enter')run();});
  $('#jPlay').onclick=()=>{if(J.idx>=J.stages.length-1&&J.t>=1){J.idx=0;J.t=0;}J.playing=!J.playing;$('#jPlay').textContent=J.playing?'❚❚ Pause':'▶ Play';J.userCam=false;};
  $('#jPrev').onclick=()=>{J.playing=false;$('#jPlay').textContent='▶ Play';if(J.t>0.05)J.t=0;else J.idx=Math.max(0,J.idx-1);J.t=0;J.pin=null;J.userCam=false;apply();J.t=1;apply();};
  $('#jStep').onclick=()=>{J.playing=false;$('#jPlay').textContent='▶ Play';if(J.t<1)J.t=1;else if(J.idx<J.stages.length-1){J.idx++;J.t=1;}J.pin=null;J.userCam=false;apply();};
  $('#jRestart').onclick=()=>{J.idx=0;J.t=0;J.playing=true;$('#jPlay').textContent='❚❚ Pause';J.pin=null;J.userCam=false;};
  $('#jChips').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;J.idx=+b.dataset.i;J.t=1;J.playing=false;$('#jPlay').textContent='▶ Play';J.pin=null;J.userCam=false;apply();});
  $('#jFocus').onchange=()=>{J.focus=+$('#jFocus').value;J.pin=null;restart(true);};
  ['#jAll','#jPace'].forEach(s=>$(s).addEventListener('change',()=>{buildStages();J.idx=Math.min(J.idx,J.stages.length-1);apply();}));
  ['#jHw','#jFmt','#jPhase'].forEach(s=>$(s).addEventListener('change',()=>{buildStages();costPanel();apply();}));
  $('#jCtx').addEventListener('input',()=>{$('#jCtxO').textContent=N(CTX[+$('#jCtx').value])+' tokens';buildStages();costPanel();apply();});
  $('#jAct').addEventListener('change',()=>{restart(true);document.dispatchEvent(new Event('tj-act'));});
  $('#jTemp').addEventListener('input',()=>{$('#jTempO').textContent=(+$('#jTemp').value/10).toFixed(1);compute();buildDetail();apply();});
  $('#jGreedy').addEventListener('change',()=>{compute();buildDetail();apply();});
  document.querySelectorAll('#jViews button').forEach(b=>b.onclick=()=>{J.userCam=true;goView(b.dataset.v);});
  $('#jZoomIn').onclick=()=>{S.st.r=Math.max(5,S.st.r*0.85);J.camGoal=null;J.userCam=true;S.place();S.dirty=true;};
  $('#jZoomOut').onclick=()=>{S.st.r=Math.min(60,S.st.r*1.18);J.camGoal=null;J.userCam=true;S.place();S.dirty=true;};
  $('#jCtxO').textContent=N(CTX[+$('#jCtx').value])+' tokens';
  if(!reduce){J.playing=true;$('#jPlay').textContent='❚❚ Pause';}
  requestAnimationFrame(loop);
}
TJ.retheme=()=>{if(!S)return;buildStatic();apply();costPanel();};
TJ.setAct=v=>{$('#jAct').value=v;restart(true);};
TJ.initJourney=init;
})();
