/* Part 4 · Why it's built this way: cost core, what-if labs, side-by-side 3D races. */
(function(){
"use strict";
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const N=n=>Math.round(n).toLocaleString('en-US');
const GiB=1073741824,KiB=1024,MiB=1048576;
const fB=b=>b>=GiB?(b/GiB).toFixed(2)+' GiB':b>=MiB?(b/MiB).toFixed(1)+' MiB':b>=KiB?(b/KiB).toFixed(1)+' KiB':N(b)+' B';
const fG=b=>(b/1e9).toFixed(2)+' GB';
const fT=s=>s>=1?s.toFixed(2)+' s':s>=1e-3?(s*1e3).toFixed(1)+' ms':(s*1e6).toFixed(0)+' µs';
const fK=n=>n>=1048576?(n/1048576)+'M':n>=1024?(n/1024)+'K':String(n);
const pct=(a,b)=>{if(!a)return b?'from 0':'same';const p=(b/a-1)*100;return (p>0?'+':'')+(Math.abs(p)<10?p.toFixed(1):p.toFixed(0))+'%';};

/* ================= data ================= */
const DEVS=[
 {name:'RTX 4060 Ti · 16 GB · 288 GB/s',mem:16,bw:288e9,fl:88e12},
 {name:'RTX 3090 · 24 GB · 936 GB/s',mem:24,bw:936e9,fl:71e12},
 {name:'RTX 4090 · 24 GB · 1,008 GB/s',mem:24,bw:1008e9,fl:165e12},
 {name:'MacBook M3 Pro · 36 GB · 150 GB/s',mem:36,bw:150e9,fl:7e12},
 {name:'Mac Studio M2 Ultra · 192 GB · 800 GB/s',mem:192,bw:800e9,fl:27e12},
 {name:'A100 · 80 GB · 2,039 GB/s',mem:80,bw:2039e9,fl:312e12},
 {name:'Desktop CPU · 64 GB DDR5 · ≈90 GB/s',mem:64,bw:89.6e9,fl:1.5e12}];
const BASES={
 'qwen3-8b':{name:'Qwen3-8B shape (36 layers, d 4096, 32 heads)',short:'8B',L:36,d:4096,H:32,hd:128,ffn:12288,V:151936,tied:false},
 'qwen3-32b':{name:'Qwen3-32B shape (64 layers, d 5120, 64 heads)',short:'32B',L:64,d:5120,H:64,hd:128,ffn:25600,V:151936,tied:false},
 'llama70':{name:'Llama-3-70B shape (80 layers, d 8192, 64 heads)',short:'70B',L:80,d:8192,H:64,hd:128,ffn:28672,V:128256,tied:false}};
const CTXS=[1024,2048,4096,8192,16384,32768,65536,131072,262144];

/* ================= cost core ================= */
const Hh=(c,b)=>c.H||b.H, HD=(c,b)=>c.hd||(c.H?b.d/c.H:b.hd);
function kinds(c,b){
  if(c.type==='swa')return Array.from({length:b.L},(_,i)=>(i%(c.every+1)===c.every)?'full':'local');
  if(c.type==='lin'){const f=c.frac;return Array.from({length:b.L},(_,i)=>f>=1?'lin':Math.floor((i+1)*(1-f))>Math.floor(i*(1-f))?'full':'lin');}
  return Array(b.L).fill('full');
}
function kvTok(c,b,kvB){const H=Hh(c,b),hd=HD(c,b);
  switch(c.type){case 'mha':return 2*H*hd*kvB;case 'mqa':return 2*hd*kvB;case 'mla':return (c.lat+64)*kvB;case 'dsa':return (c.lat+64)*kvB+128;default:return 2*c.kv*hd*kvB;}}
const stateB=(c,b)=>Hh(c,b)*HD(c,b)*HD(c,b)*4;
function cache(c,b,ctx,kvB){let s=0;kinds(c,b).forEach(k=>{s+=k==='lin'?stateB(c,b):kvTok(c,b,kvB)*(k==='local'?Math.min(ctx,c.win):ctx);});return s;}
function readB(c,b,ctx,kvB){let s=0;kinds(c,b).forEach(k=>{
  if(k==='lin')s+=2*stateB(c,b);
  else if(c.type==='dsa')s+=(c.lat+64)*kvB*Math.min(ctx,c.topk)+128*ctx;
  else s+=kvTok(c,b,kvB)*(k==='local'?Math.min(ctx,c.win):ctx);});return s;}
function attnP(c,b,k){const H=Hh(c,b),hd=HD(c,b),d=b.d;
  if(k==='lin')return 4*d*H*hd+d*H*2;
  const q=d*H*hd,o=H*hd*d;
  switch(c.type){case 'mha':return q+o+2*d*H*hd;case 'mqa':return q+o+2*d*hd;
    case 'mla':return q+o+d*(c.lat+64)+c.lat*H*2*hd;case 'dsa':return q+o+d*(c.lat+64)+c.lat*H*2*hd+d*128+d*64+64*128*16;default:return q+o+2*d*c.kv*hd;}}
function params(c,b){let a=0;kinds(c,b).forEach(k=>a+=attnP(c,b,k));const f=3*b.d*b.ffn*b.L,e=b.V*b.d*(b.tied?1:2);return {attn:a,ffn:f,emb:e,total:a+f+e,active:a+f+b.V*b.d};}
function aFlops(c,b,ctx){const H=Hh(c,b),hd=HD(c,b);let s=0;kinds(c,b).forEach(k=>{
  if(k==='lin')s+=6*H*hd*hd;else if(c.type==='mla')s+=2*H*ctx*(2*c.lat+64);else if(c.type==='dsa')s+=2*H*Math.min(ctx,c.topk)*(2*c.lat+64)+2*64*128*ctx;
  else s+=4*H*hd*(k==='local'?Math.min(ctx,c.win):ctx);});return s;}
function preAttn(c,b,P){const H=Hh(c,b),hd=HD(c,b);let s=0;kinds(c,b).forEach(k=>{
  if(k==='lin')s+=6*H*hd*hd*P;else if(c.type==='mla')s+=H*P*P*(2*c.lat+64);else if(c.type==='dsa')s+=2*H*P*Math.min(P,c.topk)*(2*c.lat+64)+64*128*P*P;
  else if(k==='local')s+=4*H*hd*P*Math.min(P,c.win);else s+=2*H*hd*P*P;});return s;}
function setup(){return {b:BASES[$('#sBase').value],dev:DEVS[+$('#sDev').value],ctx:CTXS[+$('#sCtx').value],bpw:+$('#sFmt').value,kvB:+$('#sKv').value,users:+$('#sUsers').value};}
function stats(c,S,ctxOverride){
  const b=S.b,ctx=ctxOverride||S.ctx,P=params(c,b),W=P.active*S.bpw/8,Wst=P.total*S.bpw/8,C=cache(c,b,ctx,S.kvB),R=readB(c,b,ctx,S.kvB);
  const per=W+R,U=S.users,cap=S.dev.mem*GiB*0.92-0.6*GiB;
  return {P,W,Wst,C,R,per,tps:S.dev.bw/per,agg:U*S.dev.bw/(W+U*R),maxU:Math.max(0,Math.floor((cap-Wst)/C)),fits:Wst+U*C<=cap,
    fl:aFlops(c,b,ctx),pre:(2*P.active*ctx+preAttn(c,b,ctx))/S.dev.fl,kvTokAll:kinds(c,b).reduce((s,k)=>s+(k==='lin'?0:kvTok(c,b,S.kvB)),0),ctx};
}
const QUAL={mha:'Baseline',mqa:'Slightly lower (T5-XXL 46.6 vs 47.2)',gqa:'≈ MHA (GQA-8: 47.1 vs 47.2)',mla:'≈ or above MHA (DeepSeek-V2 ablation)',swa:'≈ baseline (Gemma 3); weaker far-back recall',dsa:'≈ dense (V3.2-Exp vs V3.1); silent misses possible',lin:'Hybrid ≈ full; pure linear weak at exact recall'};
function cname(c,b){switch(c.type){
  case 'mha':return `MHA · ${Hh(c,b)} heads${c.H?` of ${HD(c,b)}`:''}`;case 'mqa':return 'MQA · 1 K/V head';case 'gqa':return `GQA · ${c.kv} K/V heads`;
  case 'mla':return `MLA · latent ${c.lat} + 64`;case 'swa':return `Sliding window ${N(c.win)} · ${c.every}:1 + GQA-${c.kv}`;case 'dsa':return `MLA + DSA · top-${N(c.topk)}`;
  case 'lin':return `Hybrid · ${Math.round(c.frac*100)}% DeltaNet + GQA-${c.kv}`;}}

/* ================= comparison table + "because" ================= */
function compare(A,B,S){
  const a=stats(A,S),b=stats(B,S),u=S.users;
  const row=(name,va,vb,fmt,lowerBetter,raw)=>{const d=raw?'':(vb===va?'<span class="dz">same</span>':`<span class="${(vb<va)===lowerBetter?'up':'dn'}">${(vb<va)?'▼':'▲'} ${pct(va,vb)}</span>`);return `<tr><td>${name}</td><td class="num">${fmt(va)}</td><td class="num">${fmt(vb)}</td><td class="num">${d}</td></tr>`;};
  const t=x=>x.toFixed(1)+' tok/s';
  return {a,b,html:`<tr><th>Meter</th><th>${cname(A,S.b)}</th><th>${cname(B,S.b)}</th><th>Change</th></tr>`+
   row('KV cache per token (all layers)',a.kvTokAll,b.kvTokAll,fB,true)+
   row(`① Cache for one ${fK(S.ctx)} chat`,a.C,b.C,fB,true)+
   row('① Weights stored',a.Wst,b.Wst,fG,true)+
   row(`① Chats that fit in ${S.dev.mem} GB`,a.maxU,b.maxU,v=>N(v),false)+
   row('② Bytes read per new token',a.per,b.per,fG,true)+
   row('② Decode speed, 1 chat (ideal)',a.tps,b.tps,t,false)+
   (u>1?row(`② Total speed, ${u} chats batched`,a.fits?a.agg:0,b.fits?b.agg:0,v=>v?t(v):'does not fit',false):'')+
   row('③ Attention math per token',a.fl,b.fl,v=>(v/1e9).toFixed(2)+' GFLOP',true)+
   row(`③ Prefill of ${fK(S.ctx)} tokens`,a.pre,b.pre,fT,true)+
   `<tr><td>④ Quality evidence</td><td>${QUAL[A.type]}</td><td>${QUAL[B.type]}</td><td></td></tr>`};
}
function because(A,B,S,r){
  const a=r.a,b=r.b,short=Math.min(1024,S.ctx),as=stats(A,S,short),bs=stats(B,S,short);
  const cr=b.C/a.C,sp=b.tps/a.tps,ssp=bs.tps/as.tps;
  if(Math.abs(cr-1)<1e-9&&Math.abs(b.per/a.per-1)<1e-9)return `<b>Same design on both sides.</b> Pick a different type for A or B to see what changes.`;
  let s=`<b>Cause → effect:</b> ${cname(B,S.b)} stores <b>${cr<1?(1/cr).toFixed(1)+'× less':cr>1?cr.toFixed(1)+'× more':'the same'}</b> cache than ${cname(A,S.b)}`;
  s+=` → at ${fK(S.ctx)} tokens that's ${fB(Math.abs(a.C-b.C))} ${cr<1?'saved':'extra'} per chat → bytes per token ${pct(a.per,b.per)} → decode <b>${pct(a.tps,b.tps)}</b> (${a.tps.toFixed(0)} → ${b.tps.toFixed(0)} tok/s on ${S.dev.name.split(' ·')[0]}).`;
  s+=` At a short ${fK(short)} context the speed change is only ${pct(as.tps,bs.tps)}, because the ${fG(a.W)} of weights dominate the bytes.`;
  if(a.maxU!==b.maxU)s+=` Capacity: ${N(a.maxU)} → <b>${N(b.maxU)}</b> chats of ${fK(S.ctx)} fit in memory.`;
  if(Math.abs(b.fl/a.fl-1)>0.05)s+=` Attention math per token ${pct(a.fl,b.fl)}, which shows up in prefill (${fT(a.pre)} → ${fT(b.pre)}).`;
  return s;
}

/* ================= shared renderer: one WebGL context, many views ================= */
const RR={r:null,views:[]};
function initRR(){RR.r=new THREE.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:true});RR.r.setPixelRatio(1);RR.dpr=Math.min(2,window.devicePixelRatio||1);RR.r.setSize(1600*RR.dpr,900*RR.dpr,false);}
const col=n=>new THREE.Color(css(n));
function mat(c,op){return new THREE.MeshLambertMaterial({color:col(c),transparent:op!=null&&op<1,opacity:op==null?1:op});}
function clear(g){while(g.children.length){const c=g.children.pop();if(c.children&&c.children.length)clear(c);if(c.geometry)c.geometry.dispose();if(c.material){if(c.material.dispose)c.material.dispose();}}}
function view(host){
  const cv=host.querySelector('canvas'),ctx=cv.getContext('2d'),scene=new THREE.Scene(),cam=new THREE.PerspectiveCamera(36,1,0.05,400);
  scene.add(new THREE.AmbientLight(0xffffff,0.72));const dl=new THREE.DirectionalLight(0xffffff,0.55);dl.position.set(5,12,9);scene.add(dl);
  const g=new THREE.Group();scene.add(g);
  const v={host,cv,ctx,scene,cam,g,st:{th:0.5,ph:1.15,r:14,lim:null},tg:new THREE.Vector3(),labels:[],vis:false,tick:null};
  let drag=null;
  cv.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,th:v.st.th,ph:v.st.ph};cv.setPointerCapture(e.pointerId);});
  cv.addEventListener('pointermove',e=>{if(!drag)return;v.st.th=drag.th-(e.clientX-drag.x)*0.008;v.st.ph=Math.max(0.5,Math.min(1.5,drag.ph-(e.clientY-drag.y)*0.006));if(v.st.lim)v.st.th=Math.max(v.st.lim[0],Math.min(v.st.lim[1],v.st.th));});
  const end=()=>drag=null;cv.addEventListener('pointerup',end);cv.addEventListener('pointercancel',end);
  if('IntersectionObserver' in window)new IntersectionObserver(es=>es.forEach(x=>v.vis=x.isIntersecting),{threshold:0.02}).observe(host);else v.vis=true;
  RR.views.push(v);return v;
}
function lab(v,html,pos,side,pri,cls){const el=document.createElement('div');el.className='lbl3 s-'+(side||'c')+(cls?' '+cls:'');el.innerHTML=html;v.host.querySelector('.lbls').appendChild(el);const it={el,pos,side:side||'c',pri:pri||0};v.labels.push(it);return it;}
function clearLabels(v){v.host.querySelector('.lbls').innerHTML='';v.labels=[];}
function fitView(v,th,ph,extraX){
  const box=new THREE.Box3();v.g.children.forEach(c=>{if(!c.userData.noFit)box.expandByObject(c);});
  const c=box.getCenter(new THREE.Vector3()),s=box.getSize(new THREE.Vector3()),w=v.cv.clientWidth||600,h=v.cv.clientHeight||360,tn=Math.tan(18*Math.PI/180);
  const dy=(s.y/2+0.5)/tn,dx=((s.x+(extraX||0))/2)/(tn*(w/h));
  v.tg.copy(c);v.st.r=Math.max(dy,dx)*0.92+s.z*0.3;v.st.th=th;v.st.ph=ph;v.st.lim=[th-0.8,th+0.8];
}
function drawView(v,dt){
  const w=v.cv.clientWidth,h=v.cv.clientHeight;if(!w||!h)return;const d=RR.dpr;
  if(v.cv.width!==Math.round(w*d)){v.cv.width=Math.round(w*d);v.cv.height=Math.round(h*d);}
  v.cam.aspect=w/h;v.cam.updateProjectionMatrix();
  const st=v.st;v.cam.position.set(v.tg.x+st.r*Math.sin(st.ph)*Math.sin(st.th),v.tg.y+st.r*Math.cos(st.ph),v.tg.z+st.r*Math.sin(st.ph)*Math.cos(st.th));v.cam.lookAt(v.tg);
  if(v.tick)v.tick(dt);
  const W=Math.round(w*d),H=Math.round(h*d);RR.r.setViewport(0,0,W,H);RR.r.setScissor(0,0,W,H);RR.r.setScissorTest(true);RR.r.clear();RR.r.render(v.scene,v.cam);
  v.ctx.clearRect(0,0,W,H);v.ctx.drawImage(RR.r.domElement,0,RR.r.domElement.height-H,W,H,0,0,W,H);
  // labels: lower priority hides on overlap
  const placed=[],its=v.labels.slice().sort((p,q)=>q.pri-p.pri);
  for(const it of its){const p=it.pos.clone().project(v.cam);if(p.z>1||p.z<-1){it.el.style.visibility='hidden';continue;}
    if(it.w==null){it.w=it.el.offsetWidth;it.h=it.el.offsetHeight;}
    let x=(p.x+1)/2*w,y=(1-p.y)/2*h;if(it.side==='r'){x+=8;y-=it.h/2;}else if(it.side==='l'){x-=8+it.w;y-=it.h/2;}else if(it.side==='t'){x-=it.w/2;y-=it.h+4;}else if(it.side==='b'){x-=it.w/2;y+=4;}else{x-=it.w/2;y-=it.h/2;}
    x=Math.max(2,Math.min(w-it.w-2,x));const r={x0:x,y0:y,x1:x+it.w,y1:y+it.h};
    if((r.y0<0||r.y1>h||placed.some(q=>!(r.x1<q.x0||r.x0>q.x1||r.y1<q.y0||r.y0>q.y1)))&&it.pri<9){it.el.style.visibility='hidden';continue;}
    placed.push(r);it.el.style.visibility='';it.el.style.transform=`translate(${Math.round(x)}px,${Math.round(y)}px)`;}
}
let last=0;function loop(ts){const dt=Math.min(0.05,(ts-last)/1000||0);last=ts;RR.views.forEach(v=>{if(v.vis)drawView(v,dt);});requestAnimationFrame(loop);}

/* ================= attention side: heads, cache lanes, bytes meter ================= */
const NT=20,DZ=0.3,UNIT=0.2/512;           // lane width ∝ bytes per token (0.2 = one head's K+V in BF16)
function side(v,c,S,x0,title){
  const b=S.b,H=Hh(c,b),hd=HD(c,b),kv=c.type==='mha'?H:c.type==='mqa'?1:c.kv||8,kvB=S.kvB,g=new THREE.Group();g.position.x=x0;v.g.add(g);
  const k0=kinds(c,b),isLin=c.type==='lin'&&k0.filter(k=>k==='lin').length>0;
  const Hs=Math.min(H,32),hs=Hs>16?0.26:0.36,qW=(Hs-1)*hs;
  const heads=[];for(let j=0;j<Hs;j++){const m=new THREE.Mesh(new THREE.BoxGeometry(0.16,0.5,0.16),mat('--copper'));m.position.set(-qW/2+j*hs,0.25,1.4);g.add(m);heads.push(m);}
  const lanes=[];const slabs=[];
  const mkLane=(x,parts)=>{ // parts: [{w,color,h}]
    let off=-parts.reduce((s,p)=>s+p.w,0)/2;const ps=parts.map(p=>{const pp=Object.assign({x:x+off+p.w/2},p);off+=p.w;return pp;});
    ps.forEach(p=>{const geo=new THREE.BoxGeometry(Math.max(0.025,p.w*0.92),p.h||0.24,DZ*0.82),im=new THREE.InstancedMesh(geo,new THREE.MeshLambertMaterial({color:0xffffff}),NT);
      const m4=new THREE.Matrix4();for(let i=0;i<NT;i++){m4.makeTranslation(p.x,(p.y||0.12),0.7-i*DZ);im.setMatrixAt(i,m4);im.setColorAt(i,col(p.color));}
      im.instanceColor.needsUpdate=true;im.userData={base:col(p.color),kind:p.kind};g.add(im);slabs.push(im);});
    lanes.push(x);
  };
  let cacheNote='';
  if(isLin){
    const n=Math.min(8,H),sz=0.62;for(let i=0;i<n;i++){const m=new THREE.Mesh(new THREE.BoxGeometry(sz,sz,sz),mat('--good',0.9));m.position.set(-((n-1)*0.85)/2+i*0.85,0.35,-0.6);m.userData.state=true;g.add(m);slabs.push(m);}
    cacheNote=`fixed state ${fB(stateB(c,b))} per layer · ${k0.filter(k=>k==='lin').length} of ${b.L} layers like this`;
  }else if(c.type==='mla'||c.type==='dsa'){
    const parts=[{w:c.lat*kvB*UNIT,color:'--violet',kind:'lat'},{w:64*kvB*UNIT,color:'--gold',kind:'rope'}];mkLane(0,parts);
    if(c.type==='dsa'){const geo=new THREE.BoxGeometry(128*UNIT*0.9,0.24,DZ*0.82),im=new THREE.InstancedMesh(geo,new THREE.MeshLambertMaterial({color:0xffffff}),NT);const m4=new THREE.Matrix4();
      for(let i=0;i<NT;i++){m4.makeTranslation(0.55,0.12,0.7-i*DZ);im.setMatrixAt(i,m4);im.setColorAt(i,col('--steel'));}im.instanceColor.needsUpdate=true;im.userData={base:col('--steel'),kind:'idx'};g.add(im);slabs.push(im);}
    cacheNote=`one latent per token: ${c.lat}+64 numbers${c.type==='dsa'?' · steel = FP8 indexer key':''}`;
  }else{
    const grp=H/kv,w=2*hd*kvB*UNIT;
    for(let gi=0;gi<Math.min(kv,32);gi++){let cx;if(kv>=Hs)cx=-qW/2+gi*hs*(Hs/Math.min(kv,32));else{const per=Hs/kv;cx=-qW/2+(gi*per+(per-1)/2)*hs;}
      mkLane(cx,[{w:w/2,color:'--teal',kind:'k',h:0.24},{w:w/2,color:'--teal2',kind:'v',h:0.24}]);
      if(kv<Hs){const per=Hs/kv;const x1=-qW/2+gi*per*hs,x2=-qW/2+((gi+1)*per-1)*hs;const ln=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x1,0.02,1.25),new THREE.Vector3(x2,0.02,1.25),new THREE.Vector3(cx,0.02,1.25),new THREE.Vector3(cx,0.02,0.85)]),new THREE.LineBasicMaterial({color:col('--violet')}));g.add(ln);}}
    cacheNote=`${kv} K/V lane${kv>1?'s':''} · width = bytes per token`;
  }
  // window / sparse masks
  const local=k0[0]==='local'||(c.type==='swa');
  const wSl=c.type==='swa'?Math.max(1,Math.round(NT*Math.min(1,c.win/S.ctx))):NT;
  const kSl=c.type==='dsa'?Math.max(1,Math.round(NT*Math.min(1,c.topk/S.ctx))):NT;
  if(c.type==='swa')slabs.forEach(im=>{if(!im.isInstancedMesh)return;for(let i=wSl;i<NT;i++)im.setColorAt(i,col('--line'));im.instanceColor.needsUpdate=true;});
  // bytes meter
  const st=stats(c,S),fw=st.W/st.per,MH=3.2,mx=Math.min(-qW/2,-1)-0.9;
  const mw=new THREE.Mesh(new THREE.BoxGeometry(0.34,MH*fw,0.34),mat('--copper',0.35));mw.position.set(mx,MH*fw/2,1.0);g.add(mw);
  const mc=new THREE.Mesh(new THREE.BoxGeometry(0.34,Math.max(0.01,MH*(1-fw)),0.34),mat('--teal',0.35));mc.position.set(mx,MH*fw+MH*(1-fw)/2,1.0);g.add(mc);
  const cur=new THREE.Mesh(new THREE.BoxGeometry(0.5,0.05,0.5),new THREE.MeshBasicMaterial({color:col('--gold')}));cur.position.set(mx,0,1.0);g.add(cur);
  const T=v.labels;
  lab(v,`<b>${title}</b><br>${cname(c,b)}`,new THREE.Vector3(x0,1.25,1.4),'t',8);
  lab(v,cacheNote,new THREE.Vector3(x0,0.1,0.7-NT*DZ),'b',3,'shape');
  if(c.type==='swa')lab(v,`local layer keeps last ${N(c.win)} tokens · 1 in ${c.every+1} layers keeps all`,new THREE.Vector3(x0+0.3,0.35,0.7-(wSl-0.5)*DZ),'r',4,'shape');
  if(c.type==='dsa')lab(v,`reads top-${N(c.topk)} only`,new THREE.Vector3(x0+0.7,0.35,0.7-NT*DZ*0.5),'r',4,'shape');
  lab(v,`bytes / token<br><b>${fG(st.per)}</b>`,new THREE.Vector3(x0+mx,MH+0.15,1.0),'t',6,'shape');
  const sl=lab(v,'',new THREE.Vector3(x0,-0.2,1.6),'b',9);
  return {c,st,g,heads,slabs,fw,MH,cur,wSl,kSl,sl,t:0,n:0,sel:new Set(),isLin};
}
function race(v,sides,S){
  const maxPer=Math.max(...sides.map(s=>s.st.per)),TMAX=3.4;
  sides.forEach(s=>{s.dur=s.st.per/maxPer*TMAX;s.t=0;s.n=0;pickSel(s);});
  const lit=col('--gold');
  v.tick=dt=>{sides.forEach(s=>{
    s.t+=reduce?0:dt/s.dur;if(s.t>=1){s.t=0;s.n++;pickSel(s);}
    const t=s.t,fw=s.fw;s.cur.position.y=s.MH*t;
    const p=t<fw?-1:(t-fw)/(1-fw);
    s.slabs.forEach(im=>{
      if(!im.isInstancedMesh){im.material.emissive=lit;im.material.emissiveIntensity=p>=0?0.5*(1-Math.abs(p-0.5)*2)+0.15:0;return;}
      for(let i=0;i<NT;i++){const readable=(i<s.wSl)&&(im.userData.kind==='idx'||s.c.type!=='dsa'||s.sel.has(i));
        const on=readable&&p>=0&&i/NT<=p&&i/NT>p-0.25;const base=i>=s.wSl?col('--line'):(!readable&&s.c.type==='dsa'?im.userData.base.clone().lerp(col('--sunk'),0.6):im.userData.base);
        im.setColorAt(i,on?lit:(i===0&&t<0.12&&s.n>0?col('--gold'):base));}
      im.instanceColor.needsUpdate=true;});
    s.sl.el.innerHTML=`${s.st.tps.toFixed(0)} tok/s ideal · <b>${s.n}</b> tokens so far`;
  });};
}
function pickSel(s){s.sel=new Set();if(s.c.type!=='dsa')return;const r=Math.random;while(s.sel.size<s.kSl)s.sel.add(r()<0.35?s.sel.size:Math.floor(r()*NT));}
function attnView(v,A,B,S,ta,tb){
  clear(v.g);clearLabels(v);
  const sideW=c=>Math.max(Math.min(Hh(c,S.b),32)*(Math.min(Hh(c,S.b),32)>16?0.26:0.36),2.6)+2.4;
  if(!B){const s=side(v,A,S,0,ta);race(v,[s],S);fitView(v,0.55,0.95,2);return;}
  const wa=sideW(A),wb=sideW(B),gap=1.6;const xa=-(wb+gap)/2,xb=(wa+gap)/2;
  const sa=side(v,A,S,xa,ta),sb=side(v,B,S,xb,tb);race(v,[sa,sb],S);fitView(v,0.3,0.95,1);
}

/* ================= design lab ================= */
const TYPES=[['mha','MHA'],['mqa','MQA'],['gqa','GQA'],['mla','MLA'],['swa','Sliding window'],['dsa','MLA + DSA'],['lin','Hybrid linear']];
function labCol(id,def){
  const el=$(id);el.insertAdjacentHTML('beforeend',`<label class="ctl">Type<select data-k="type">${TYPES.map(([k,n])=>`<option value="${k}"${k===def.type?' selected':''}>${n}</option>`).join('')}</select></label>
   <label class="ctl" data-for="gqa swa lin">K/V heads<select data-k="kv">${[1,2,4,8,16].map(n=>`<option${n===(def.kv||8)?' selected':''}>${n}</option>`).join('')}</select></label>
   <label class="ctl" data-for="mla dsa">Latent<select data-k="lat">${[128,256,512,1024].map(n=>`<option${n===512?' selected':''}>${n}</option>`).join('')}</select></label>
   <label class="ctl" data-for="swa">Window<select data-k="win">${[512,1024,4096,8192].map(n=>`<option${n===1024?' selected':''}>${n}</option>`).join('')}</select></label>
   <label class="ctl" data-for="swa">Local per global<select data-k="every">${[1,3,5,7].map(n=>`<option${n===5?' selected':''}>${n}</option>`).join('')}</select></label>
   <label class="ctl" data-for="dsa">Top-k<select data-k="topk">${[512,1024,2048,4096].map(n=>`<option${n===2048?' selected':''}>${n}</option>`).join('')}</select></label>
   <label class="ctl" data-for="lin">Linear share<select data-k="frac">${[0.5,0.75,0.875,1].map(n=>`<option value="${n}"${n===0.75?' selected':''}>${n*100}%</option>`).join('')}</select></label>`);
  const read=()=>{const o={};el.querySelectorAll('select').forEach(s=>o[s.dataset.k]=s.dataset.k==='type'?s.value:+s.value);el.querySelectorAll('[data-for]').forEach(l=>l.hidden=!l.dataset.for.split(' ').includes(o.type));return o;};
  el.addEventListener('change',renderLab);return read;
}
let labA,labB,VW={};
function renderLab(){const S=setup(),A=labA(),B=labB();const r=compare(A,B,S);$('#labTable').innerHTML=r.html;$('#labWhy').innerHTML=because(A,B,S,r);if(VW.lab)attnView(VW.lab,A,B,S,'Design A','Design B');}

/* ================= attention cards ================= */
function rx(rows){return rows.map(r=>r.sum?`<div class="row sum"><span>${r[0]}</span><span>${r[1]}</span></div>`:`<div class="row"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('');}
function chain(A,B,S,id){const r=compare(A,B,S);$(id).innerHTML=`<div class="tablewrap"><table>${r.html}</table></div><div class="because">${because(A,B,S,r)}</div>`;return r;}
const CARDS={
 mha(S){const Hs=[8,16,32,64],H=Hs[+$('#kMha').value];$('#kMhaO').textContent=`${H} heads × ${S.b.d/H} dims`;
   const A={type:'mha'},B={type:'mha',H};chain(A,B,S,'#chMha');attnView(VW.mha,A,B,S,`${S.b.H} heads (as built)`,`${H} heads`);
   const b=S.b,per=2*b.H*b.hd*S.kvB;
   $('#rxMha').innerHTML=rx([['K + V per token, one layer',`2 × ${b.H} heads × ${b.hd} × ${S.kvB} B = ${fB(per)}`],[`× ${b.L} layers`,fB(per*b.L)+' per token'],[`× ${fK(S.ctx)} tokens`,fB(per*b.L*S.ctx)],{0:'Weights at your format',1:fG(stats(A,S).Wst),sum:1}].map(x=>x.sum?x:x)) + `<div class="row dim"><span>If this model used MHA, one ${fK(S.ctx)} chat would need ${fB(per*b.L*S.ctx)} of cache.</span><span></span></div>`;},
 mqa(S){const A={type:'mha'},B={type:'mqa'};chain(A,B,S,'#chMqa');attnView(VW.mqa,A,B,S,'MHA','MQA');const b=S.b,a=stats(A,S),m=stats(B,S);
   $('#pMqa').innerHTML=`With MHA, a ${fK(S.ctx)}-token chat on the ${S.b.short} shape needs <b>${fB(a.C)}</b> of cache, and each new token reads it all: <b>${fG(a.R)}</b> of the ${fG(a.per)} per token.`;
   $('#rxMqa').innerHTML=rx([['MHA K+V per token, all layers',fB(a.kvTokAll)],['MQA K+V per token, all layers',`2 × 1 × ${b.hd} × ${S.kvB} B × ${b.L} = ${fB(m.kvTokAll)}`],[`Cache for one ${fK(S.ctx)} chat`,`${fB(a.C)} → ${fB(m.C)}`],{0:'Ratio',1:`${b.H}× smaller`,sum:1}]);},
 gqa(S){const ks=[1,2,4,8,16,32],kv=ks[+$('#kGqa').value];$('#kGqaO').textContent=`${kv} (${S.b.H/kv} query heads each)`;
   const A={type:'mha'},B=kv===S.b.H?{type:'mha'}:kv===1?{type:'mqa'}:{type:'gqa',kv};chain(A,B,S,'#chGqa');attnView(VW.gqa,A,B,S,'MHA',`GQA-${kv}`);
   const b=S.b,g=stats({type:'gqa',kv:8},S),a=stats(A,S);
   $('#rxGqa').innerHTML=rx([['Real Qwen3-8B: 8 K/V heads',`2 × 8 × ${b.hd} × ${S.kvB} B × ${b.L} = ${fB(g.kvTokAll)} per token`],[`One ${fK(S.ctx)} chat`,fB(g.C)],['Same model with MHA',fB(a.C)],[`Chats that fit on ${S.dev.name.split(' ·')[0]}`,`${N(a.maxU)} (MHA) → ${N(g.maxU)} (GQA-8)`],{0:`Decode at ${fK(S.ctx)}`,1:`${a.tps.toFixed(0)} → ${g.tps.toFixed(0)} tok/s`,sum:1}]);},
 mla(S){const ls=[128,256,512,768,1024],lat=ls[+$('#kMla').value];$('#kMlaO').textContent=`${lat} + 64 RoPE`;
   const A={type:'gqa',kv:8},B={type:'mla',lat};chain(A,B,S,'#chMla');attnView(VW.mla,A,B,S,'GQA-8 (Qwen3-8B)',`MLA ${lat}`);
   const b=S.b,m=stats(B,S),a=stats({type:'mha'},S);
   $('#rxMla').innerHTML=rx([['MLA cache per token per layer',`(${lat} + 64) × ${S.kvB} B = ${fB((lat+64)*S.kvB)}`],['MHA per token per layer',`2 × ${b.H} × ${b.hd} × ${S.kvB} B = ${fB(2*b.H*b.hd*S.kvB)}`],{0:'Reduction vs MHA',1:`${(2*b.H*b.hd/(lat+64)).toFixed(1)}×`,sum:1},['DeepSeek-V3 real shape (128 heads)',`2·128·128 / 576 = 56.9× vs MHA`],[`Your ${fK(S.ctx)} chat`,`${fB(a.C)} (MHA) → ${fB(m.C)} (MLA)`]]);},
 swa(S){const ws=[512,1024,2048,4096,8192],win=ws[+$('#kSwaW').value],every=+$('#kSwaN').value;$('#kSwaWO').textContent=N(win)+' tokens';$('#kSwaNO').textContent=`${every} : 1`;
   const A={type:'gqa',kv:8},B={type:'swa',kv:8,win,every};chain(A,B,S,'#chSwa');attnView(VW.swa,A,B,S,'All layers global',`Window ${N(win)}, ${every}:1`);
   const b=S.b,gl=kinds(B,b).filter(k=>k==='full').length,lo=b.L-gl,per=2*8*b.hd*S.kvB;
   $('#rxSwa').innerHTML=rx([['Layers',`${lo} local + ${gl} global`],['Local layers',`${lo} × ${fB(per)} × min(${fK(S.ctx)}, ${N(win)}) = ${fB(lo*per*Math.min(S.ctx,win))}`],['Global layers',`${gl} × ${fB(per)} × ${fK(S.ctx)} = ${fB(gl*per*S.ctx)}`],{0:'Total per chat',1:fB(stats(B,S).C),sum:1},['All-global (Qwen3-8B as built)',fB(stats(A,S).C)]]);},
 dsa(S){const ks=[512,1024,2048,4096,8192],topk=ks[+$('#kDsa').value];$('#kDsaO').textContent=N(topk);
   const A={type:'mla',lat:512},B={type:'dsa',lat:512,topk};chain(A,B,S,'#chDsa');attnView(VW.dsa,A,B,S,'MLA, dense',`MLA + DSA top-${N(topk)}`);
   const kvB=S.kvB,ctx=S.ctx,full=576*kvB*ctx,sel=576*kvB*Math.min(ctx,topk)+128*ctx;
   $('#rxDsa').innerHTML=rx([['Dense MLA reads per layer',`576 × ${kvB} B × ${fK(ctx)} = ${fB(full)}`],['DSA: indexer keys (FP8)',`128 B × ${fK(ctx)} = ${fB(128*ctx)}`],['DSA: attention on top-k',`576 × ${kvB} B × ${N(Math.min(ctx,topk))} = ${fB(576*kvB*Math.min(ctx,topk))}`],{0:'Read ratio',1:`${(sel/full*100).toFixed(1)}% of dense`,sum:1},['Memory held','unchanged: every token stays cached']]);},
 lin(S){const fs=[0,0.5,0.75,0.875,1],frac=fs[+$('#kLin').value];$('#kLinO').textContent=frac===0.75?'75% (3:1, Qwen3.5)':Math.round(frac*100)+'%';
   const A={type:'gqa',kv:8},B=frac?{type:'lin',kv:8,frac}:{type:'gqa',kv:8};chain(A,B,S,'#chLin');attnView(VW.lin,A,B,S,'All full attention (GQA-8)',frac?`${Math.round(frac*100)}% DeltaNet`:'0% DeltaNet');
   const b=S.b,st=stateB({},b),per=2*8*b.hd*S.kvB;
   $('#rxLin').innerHTML=rx([['DeltaNet state per layer',`${b.H} heads × ${b.hd}×${b.hd} × 4 B = ${fB(st)}`],['Full layer per token',fB(per)],{0:'Break-even context',1:`${N(st/per)} tokens: beyond this a DeltaNet layer is cheaper`,sum:1},[`Your ${fK(S.ctx)} chat`,`${fB(stats(A,S).C)} → ${fB(stats(B,S).C)}`]]);}
};

/* ================= MoE ================= */
const MOES=[['Qwen3-30B-A3B',30.5,3.3,128,8,0],['Qwen3-235B-A22B',235,22,128,8,0],['DeepSeek-V3 / R1',671,37,256,8,1],['Mixtral 8x7B',46.7,12.9,8,2,0],['gpt-oss-120b',117,5.1,128,4,0]];
const BATCH=[1,2,4,8,16,64,256];
function moeCard(S){
  const m=MOES[+$('#kMoeM').value],B=BATCH[+$('#kMoeB').value];$('#kMoeBO').textContent=B;
  const [name,tot,act,E,k]=m,f=k/E,expert=(tot-act)/(1-f),shared=tot-expert,touch=1-Math.pow(1-f,B),readP=(shared+expert*touch)*1e9;
  const W=S.bpw/8,stored=tot*1e9*W,perTok=readP*W,dense=tot*1e9*W,cap=S.dev.mem*GiB*0.92;
  const tps=S.dev.bw/perTok*B,dt=S.dev.bw/dense*B;
  $('#chMoe').innerHTML=`<div class="tablewrap"><table><tr><th>Meter</th><th>Dense ${tot} B</th><th>${name}</th><th>Change</th></tr>
   <tr><td>① Weights stored</td><td class="num">${fG(dense)}</td><td class="num">${fG(stored)}</td><td class="num"><span class="dz">same</span></td></tr>
   <tr><td>① Fits in ${S.dev.mem} GB?</td><td colspan="2" class="num">${stored<=cap?'<span class="pill good">yes</span>':'<span class="pill bad">no: offload experts to CPU RAM or quantize harder</span>'}</td><td></td></tr>
   <tr><td>② Experts touched per step (batch ${B})</td><td class="num">all</td><td class="num">${(touch*100).toFixed(0)}% of ${E}</td><td></td></tr>
   <tr><td>② Bytes read per step</td><td class="num">${fG(dense)}</td><td class="num">${fG(perTok)}</td><td class="num"><span class="up">▼ ${pct(dense,perTok)}</span></td></tr>
   <tr><td>② Tokens/s (ideal, weights only)</td><td class="num">${dt.toFixed(1)}</td><td class="num">${tps.toFixed(1)}</td><td class="num"><span class="up">▲ ${(tps/dt).toFixed(1)}×</span></td></tr></table></div>
   <div class="because"><b>Cause → effect:</b> each token uses ${k} of ${E} experts (${(f*100).toFixed(1)}%), so one token reads ≈ ${act} B parameters instead of ${tot} B. At batch ${B}, the tokens together touch ${(touch*100).toFixed(0)}% of the experts, so ${B>8?'<b>most of the MoE advantage is gone</b>: the step reads almost every expert':'the advantage holds'}. Memory still has to hold all ${tot} B.</div>`;
  $('#rxMoe').innerHTML=rx([['Stored parameters',`${tot} B → ${fG(stored)} at your format`],['Read per token (batch 1)',`≈ ${act} B → ${fG(act*1e9*W)}`],{0:`Ideal speed on ${S.dev.name.split(' ·')[0]}`,1:`${(S.dev.bw/(act*1e9*W)).toFixed(0)} tok/s vs ${(S.dev.bw/dense).toFixed(1)} for a dense ${tot} B`,sum:1}]);
  // 3D
  const v=VW.moe;clear(v.g);clearLabels(v);
  const dg=new THREE.Group();dg.position.x=-4.2;v.g.add(dg);const dsz=Math.cbrt(tot)/Math.cbrt(671)*3.2+0.6;
  const db=new THREE.Mesh(new THREE.BoxGeometry(dsz,dsz,dsz),mat('--steel',0.85));db.position.y=dsz/2;dg.add(db);
  lab(v,`<b>Dense ${tot} B</b><br>every weight read per token`,new THREE.Vector3(-4.2,dsz+0.2,0),'t',8);
  const cols=Math.min(16,E),rows=Math.ceil(E/cols),ts=Math.min(0.32,4.6/cols),tiles=[];
  for(let e=0;e<E;e++){const m=new THREE.Mesh(new THREE.BoxGeometry(ts*0.85,0.2,ts*0.85),mat('--steel'));m.position.set(1.6+(e%cols-(cols-1)/2)*ts,0.1,(Math.floor(e/cols)-(rows-1)/2)*ts);v.g.add(m);tiles.push(m);}
  let sh=null;if(m[5]){sh=new THREE.Mesh(new THREE.BoxGeometry(0.6,0.6,0.6),mat('--teal'));sh.position.set(1.6-(cols/2)*ts-0.8,0.3,0);v.g.add(sh);lab(v,'shared',new THREE.Vector3(sh.position.x,0.7,0),'t',4,'shape');}
  lab(v,`<b>${name}</b> · ${E} experts<br>gold = read this step (batch ${B})`,new THREE.Vector3(1.6,0.9,-(rows*ts)/2),'t',8);
  const sl=lab(v,'',new THREE.Vector3(1.6,-0.1,(rows*ts)/2+0.3),'b',9),dl=lab(v,'',new THREE.Vector3(-4.2,-0.1,dsz/2+0.3),'b',9);
  const maxB=dense,durD=3.2,durM=Math.max(0.35,perTok/maxB*3.2);let tD=0,tM=0,nD=0,nM=0,lit=new Set();
  const pick=()=>{lit=new Set();for(let b=0;b<B;b++){const s=new Set();while(s.size<k)s.add(Math.floor(Math.random()*E));s.forEach(x=>lit.add(x));}};pick();
  v.tick=dt=>{tD+=reduce?0:dt/durD;tM+=reduce?0:dt/durM;if(tD>=1){tD=0;nD+=B;}if(tM>=1){tM=0;nM+=B;pick();}
    db.material.emissive=col('--gold');db.material.emissiveIntensity=0.35*Math.sin(tD*Math.PI);
    tiles.forEach((t,e)=>{const on=lit.has(e);t.material.color.copy(col(on?'--gold':'--steel'));t.scale.y=on?1+3*Math.sin(tM*Math.PI):1;t.position.y=0.1*t.scale.y;});
    if(sh){sh.material.emissive=col('--gold');sh.material.emissiveIntensity=0.4*Math.sin(tM*Math.PI);}
    sl.el.innerHTML=`<b>${nM}</b> tokens`;dl.el.innerHTML=`<b>${nD}</b> tokens`;};
  fitView(v,0.35,1.05,1);
}

/* ================= activations ================= */
const AF={relu:z=>z>0?z:0,gelu:z=>0.5*z*(1+Math.tanh(0.7978845608*(z+0.044715*z*z*z))),silu:z=>z/(1+Math.exp(-z))};
const AC={relu:'--steel',gelu:'--violet',silu:'--copper'};
function actCards(S){
  const mu=+$('#aMu').value/10,sd=+$('#aSd').value/10;$('#aMuO').textContent=mu.toFixed(1);$('#aSdO').textContent=sd.toFixed(1);
  // curves
  let cv=$('#actCurves'),g=cv.getContext('2d'),W=cv.width,H=cv.height;g.clearRect(0,0,W,H);
  const X=z=>30+(z+4)/8*(W-40),Y=y=>H-24-(y+1)/5*(H-34);
  g.strokeStyle=css('--line');g.beginPath();g.moveTo(30,Y(0));g.lineTo(W-10,Y(0));g.moveTo(X(0),6);g.lineTo(X(0),H-24);g.stroke();
  g.fillStyle=css('--muted');g.font='11px '+css('--mono');g.fillText('z',W-18,Y(0)-4);g.fillText('-4',X(-4)-6,H-8);g.fillText('4',X(4)-4,H-8);
  Object.keys(AF).forEach((k,n)=>{g.strokeStyle=css(AC[k]);g.lineWidth=2.4;g.beginPath();for(let z=-4;z<=4;z+=0.05){const x=X(z),y=Y(AF[k](z));z>-4?g.lineTo(x,y):g.moveTo(x,y);}g.stroke();
    g.setLineDash([4,3]);g.lineWidth=1.2;g.beginPath();for(let z=-4;z<=4;z+=0.05){const d=(AF[k](z+1e-3)-AF[k](z-1e-3))/2e-3,x=X(z),y=Y(d);z>-4?g.lineTo(x,y):g.moveTo(x,y);}g.stroke();g.setLineDash([]);
    g.fillStyle=css(AC[k]);g.fillText(k.toUpperCase()+(k==='silu'?' (inside SwiGLU)':''),36,16+n*14);});
  g.fillStyle=css('--muted');g.fillText('solid = output · dashed = gradient',W-220,16);
  g.fillStyle=css('--copper');g.globalAlpha=0.12;g.fillRect(X(mu-sd),6,X(mu+sd)-X(mu-sd),H-30);g.globalAlpha=1;
  // histogram of outputs for inputs ~ N(mu, sd)
  const r=mulberry(9),n=20000,res={};Object.keys(AF).forEach(k=>res[k]={zero:0,tiny:0,deadg:0,bins:new Array(60).fill(0)});
  for(let i=0;i<n;i++){const z=mu+sd*gauss(r);Object.keys(AF).forEach(k=>{const y=AF[k](z),o=res[k];if(y===0)o.zero++;if(Math.abs(y)<0.01)o.tiny++;const d=(AF[k](z+1e-3)-AF[k](z-1e-3))/2e-3;if(Math.abs(d)<1e-6)o.deadg++;const bi=Math.floor((y+1)/5*60);if(bi>=0&&bi<60)o.bins[bi]++;});}
  cv=$('#actHist');g=cv.getContext('2d');W=cv.width;H=cv.height;g.clearRect(0,0,W,H);
  const mx=Math.max(...Object.values(res).map(o=>Math.max(...o.bins.slice(0))));
  g.fillStyle=css('--muted');g.fillText('distribution of outputs for 20,000 neurons',30,14);
  Object.keys(AF).forEach(k=>{g.strokeStyle=css(AC[k]);g.lineWidth=2;g.beginPath();res[k].bins.forEach((c,i)=>{const x=30+i/59*(W-40),y=H-20-Math.sqrt(c/mx)*(H-40);i?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();});
  g.fillStyle=css('--muted');g.fillText('-1',26,H-6);g.fillText('0',30+(1/5)*(W-40)-3,H-6);g.fillText('4',W-14,H-6);
  // practical effect: skip exact zeros on CPU
  const b=S.b,ffnB=3*b.d*b.ffn*b.L*S.bpw/8,P=params({type:'gqa',kv:8},b),Wb=P.active*S.bpw/8,R=readB({type:'gqa',kv:8},b,S.ctx,S.kvB),s0=res.relu.zero/n;
  const base=S.dev.bw/(Wb+R),skip=S.dev.bw/(Wb-ffnB*s0*(2/3)+R);
  $('#chAct').innerHTML=`<div class="tablewrap"><table><tr><th></th><th>ReLU</th><th>GELU</th><th>SiLU / SwiGLU gate</th></tr>
   <tr><td>Outputs exactly 0</td>${['relu','gelu','silu'].map(k=>`<td class="num">${(res[k].zero/n*100).toFixed(1)}%</td>`).join('')}</tr>
   <tr><td>Outputs near 0 (&lt;0.01)</td>${['relu','gelu','silu'].map(k=>`<td class="num">${(res[k].tiny/n*100).toFixed(1)}%</td>`).join('')}</tr>
   <tr><td>Gradient exactly 0 (can't learn)</td>${['relu','gelu','silu'].map(k=>`<td class="num">${(res[k].deadg/n*100).toFixed(1)}%</td>`).join('')}</tr></table></div>
   <div class="because"><b>Cause → effect on your device:</b> with these inputs ${(s0*100).toFixed(0)}% of ReLU neurons are exactly 0. An engine that skips their rows of W_up and W_down (as PowerInfer does with a predictor) avoids about ${fG(ffnB*s0*2/3)} per token on the ${b.short} shape: ${base.toFixed(0)} → <b>${skip.toFixed(0)} tok/s</b> on ${S.dev.name.split(' ·')[0]} (upper bound). SwiGLU's gate is almost never exactly 0, so this trick doesn't apply. Slide μ negative and watch ReLU's dead-gradient share climb: that's how neurons die during training.</div>`;
}
function gluChart(){
  const d=[['ReLU',1.997],['Swish',1.994],['GELU',1.983],['GLU',1.982],['Bilinear',1.960],['ReGLU',1.953],['SwiGLU',1.944],['GEGLU',1.942]];
  const cv=$('#gluChart'),g=cv.getContext('2d'),W=cv.width,H=cv.height;g.clearRect(0,0,W,H);const lo=1.93,hi=2.0,bh=(H-10)/d.length;
  d.forEach(([n,v],i)=>{const y=5+i*bh,w=(v-lo)/(hi-lo)*(W-170);g.fillStyle=css(n.includes('GLU')&&n!=='GLU'?'--copper':'--steel');g.fillRect(90,y+3,w,bh-7);g.fillStyle=css('--ink');g.font='11px '+css('--mono');g.fillText(n,6,y+bh/2+3);g.fillText(v.toFixed(3),96+w,y+bh/2+3);});
}

/* ================= RoPE ================= */
function rope(S){
  const th=+$('#ropeT').value,d=S.b.hd,cv=$('#ropeCv'),g=cv.getContext('2d'),W=cv.width,H=cv.height;g.clearRect(0,0,W,H);
  const X=t=>30+Math.log10(t)/7*(W-40);const wl=[];for(let i=0;i<d/2;i++)wl.push(2*Math.PI*Math.pow(th,2*i/d));
  g.strokeStyle=css('--line');g.beginPath();g.moveTo(30,H-22);g.lineTo(W-10,H-22);g.stroke();g.fillStyle=css('--muted');g.font='10px '+css('--mono');
  [1,10,100,1e3,1e4,1e5,1e6,1e7].forEach(t=>{g.fillText(t>=1e6?(t/1e6)+'M':t>=1e3?(t/1e3)+'K':String(t),X(t)-8,H-8);});
  wl.forEach((w,i)=>{const x=X(Math.min(w,1e7));g.fillStyle=css(w>S.ctx?'--copper':'--teal');g.fillRect(x-1,H-24-(i+1)*(H-40)/(d/2),3,3);});
  g.strokeStyle=css('--gold');g.lineWidth=2;g.beginPath();g.moveTo(X(S.ctx),6);g.lineTo(X(S.ctx),H-22);g.stroke();g.fillStyle=css('--gold');g.fillText('your context',X(S.ctx)+4,14);
  const longer=wl.filter(w=>w>S.ctx).length;
  $('#ropeOut').innerHTML=`Each dot is one of ${d/2} frequency pairs, placed at its wavelength in tokens. The slowest pair turns once every <b>${N(wl[wl.length-1])}</b> tokens. At ${fK(S.ctx)} context, ${longer} pairs (copper) haven't completed a turn yet, and those are what tell far-apart positions apart. With θ = 10,000 the slowest pair turns every ~54K tokens, one reason Llama 2 struggled past its training length.`;
}

/* ================= abliteration ================= */
function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function gauss(r){let u=0;while(u===0)u=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*r());}
const AB={seed:3,a:0,go:false};
function ablBuild(){
  const v=VW.abl;clear(v.g);clearLabels(v);const r=mulberry(AB.seed);
  const tr=new THREE.Vector3(0.7,0.55,-0.45).normalize();
  const mk=(off)=>Array.from({length:60},()=>new THREE.Vector3(gauss(r)*0.55,gauss(r)*0.55,gauss(r)*0.55).addScaledVector(tr,off));
  AB.ok=mk(-1.0);AB.no=mk(1.3);
  const mean=a=>a.reduce((s,p)=>s.add(p),new THREE.Vector3()).multiplyScalar(1/a.length);
  AB.mOk=mean(AB.ok);AB.mNo=mean(AB.no);AB.rv=AB.mNo.clone().sub(AB.mOk);AB.rh=AB.rv.clone().normalize();
  const sg=new THREE.SphereGeometry(0.07,10,8);
  AB.mOkMesh=new THREE.InstancedMesh(sg,new THREE.MeshLambertMaterial({color:col('--teal')}),60);AB.mNoMesh=new THREE.InstancedMesh(sg,new THREE.MeshLambertMaterial({color:col('--copper')}),60);v.g.add(AB.mOkMesh);v.g.add(AB.mNoMesh);
  // plane perpendicular to r̂ through the origin
  const pl=new THREE.Mesh(new THREE.PlaneGeometry(4.4,4.4),new THREE.MeshBasicMaterial({color:col('--violet'),transparent:true,opacity:0.1,side:THREE.DoubleSide,depthWrite:false}));pl.lookAt(AB.rh);v.g.add(pl);
  const ar=new THREE.ArrowHelper(AB.rh,AB.mOk,AB.rv.length(),col('--gold').getHex(),0.3,0.16);v.g.add(ar);
  const ax=[[1,0,0],[0,1,0],[0,0,1]].map(a=>{const l=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a).multiplyScalar(-2.2),new THREE.Vector3(...a).multiplyScalar(2.2)]),new THREE.LineBasicMaterial({color:col('--line')}));v.g.add(l);});
  lab(v,'<b>r̂</b> = mean(refused) − mean(answered)',AB.mNo.clone().add(new THREE.Vector3(0,0.3,0)),'r',8);
  lab(v,'answered prompts',AB.mOk.clone().add(new THREE.Vector3(0,-0.9,0)),'b',6,'shape');
  lab(v,'refused prompts',AB.mNo.clone().add(new THREE.Vector3(0,0.9,0)),'t',6,'shape');
  lab(v,'plane ⟂ r̂: where everything lands after the edit',new THREE.Vector3(1.8,-1.6,1.2),'r',4,'shape');
  AB.a=0;ablPlace();fitView(v,0.6,1.15,0.5);v.tick=dt=>{if(AB.go){AB.a=Math.min(1,AB.a+dt/2.2);if(AB.a>=1)AB.go=false;ablPlace();}};
}
function ablPlace(){
  const m4=new THREE.Matrix4(),p=new THREE.Vector3(),proj=(x)=>p.copy(x).addScaledVector(AB.rh,-AB.a*x.dot(AB.rh));
  AB.ok.forEach((x,i)=>{proj(x);m4.makeTranslation(p.x,p.y,p.z);AB.mOkMesh.setMatrixAt(i,m4);});AB.mOkMesh.instanceMatrix.needsUpdate=true;
  AB.no.forEach((x,i)=>{proj(x);m4.makeTranslation(p.x,p.y,p.z);AB.mNoMesh.setMatrixAt(i,m4);});AB.mNoMesh.instanceMatrix.needsUpdate=true;
  // histogram of the "refusal score" x·r̂
  const cv=$('#ablHist'),g=cv.getContext('2d'),W=cv.width,H=cv.height;g.clearRect(0,0,W,H);
  const bins=(arr)=>{const b=new Array(40).fill(0);arr.forEach(x=>{const s=x.dot(AB.rh)*(1-AB.a);const i=Math.floor((s+3)/6*40);if(i>=0&&i<40)b[i]++;});return b;};
  const bo=bins(AB.ok),bn=bins(AB.no),mx=Math.max(...bo,...bn,1);
  [[bo,'--teal'],[bn,'--copper']].forEach(([b,c])=>{g.fillStyle=css(c);g.globalAlpha=0.7;b.forEach((n,i)=>{const h=n/mx*(H-40);g.fillRect(20+i*(W-30)/40,H-22-h,(W-30)/40-2,h);});g.globalAlpha=1;});
  g.fillStyle=css('--muted');g.font='11px '+css('--mono');g.fillText('refusal score  x · r̂',20,14);g.fillText('-3',16,H-6);g.fillText('0',20+(W-30)/2-3,H-6);g.fillText('+3',W-22,H-6);
  const s=AB.no[0],sp=s.clone().addScaledVector(AB.rh,-AB.a*s.dot(AB.rh)),f=v=>'['+v.toArray().map(x=>x.toFixed(2)).join(', ')+']';
  const avg=a=>a.reduce((t,x)=>t+x.dot(AB.rh)*(1-AB.a),0)/a.length;
  $('#rxAbl').innerHTML=rx([['r = μ_refused − μ_answered',f(AB.rv)],['r̂ = r / ‖r‖',f(AB.rh)],['one refused prompt x',f(s)],['x · r̂ (its refusal score)',(s.dot(AB.rh)).toFixed(2)],[`x − ${AB.a.toFixed(2)}·(x·r̂)·r̂`,f(sp)],{0:'Mean score: refused / answered',1:`${avg(AB.no).toFixed(2)} / ${avg(AB.ok).toFixed(2)}`,sum:1}]);
}

/* ================= wiring ================= */
const SRC=[['s-vaswani','Vaswani et al., Attention Is All You Need (2017)','https://arxiv.org/abs/1706.03762'],
['s-olsson','Olsson et al., In-context Learning and Induction Heads (2022)','https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html'],
['s-michel','Michel, Levy & Neubig, Are Sixteen Heads Really Better than One? (2019)','https://arxiv.org/abs/1905.10650'],
['s-mqa','Shazeer, Fast Transformer Decoding: One Write-Head Is All You Need (2019)','https://arxiv.org/abs/1911.02150'],
['s-gqa','Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models (2023)','https://arxiv.org/abs/2305.13245'],
['s-dsv2','DeepSeek-AI, DeepSeek-V2 (MLA) (2024)','https://arxiv.org/abs/2405.04434'],
['s-dsv3','DeepSeek-AI, DeepSeek-V3 Technical Report (2024)','https://arxiv.org/abs/2412.19437'],
['s-dsa','DeepSeek-AI, DeepSeek-V3.2-Exp: DeepSeek Sparse Attention','https://github.com/deepseek-ai/DeepSeek-V3.2-Exp'],
['s-gemma3','Gemma Team, Gemma 3 Technical Report (2025)','https://arxiv.org/abs/2503.19786'],
['s-gdn','Yang, Kautz & Hatamizadeh, Gated Delta Networks (2024)','https://arxiv.org/abs/2412.06464'],
['s-q35','Qwen/Qwen3.5-9B model card','https://huggingface.co/Qwen/Qwen3.5-9B'],
['s-zoology','Arora et al., Zoology: Measuring and Improving Recall in Efficient Language Models (2023)','https://arxiv.org/abs/2312.04927'],
['s-glu','Shazeer, GLU Variants Improve Transformer (2020)','https://arxiv.org/abs/2002.05202'],
['s-relu','Mirzadeh et al., ReLU Strikes Back: Exploiting Activation Sparsity in LLMs (2023)','https://arxiv.org/abs/2310.04564'],
['s-powerinfer','Song et al., PowerInfer: Fast LLM Serving on a Consumer-grade GPU (2023)','https://arxiv.org/abs/2312.12456'],
['s-preln','Xiong et al., On Layer Normalization in the Transformer Architecture (2020)','https://arxiv.org/abs/2002.04745'],
['s-rms','Zhang & Sennrich, Root Mean Square Layer Normalization (2019)','https://arxiv.org/abs/1910.07467'],
['s-qkn','Dehghani et al., Scaling Vision Transformers to 22 Billion Parameters (QK-norm) (2023)','https://arxiv.org/abs/2302.05442'],
['s-yarn','Peng et al., YaRN: Efficient Context Window Extension (2023)','https://arxiv.org/abs/2309.00071'],
['s-arditi','Arditi et al., Refusal in Language Models Is Mediated by a Single Direction (2024)','https://arxiv.org/abs/2406.11717']];
function renderAll(){
  const S=setup();$('#sCtxO').textContent=N(S.ctx)+' tokens';$('#sUsersO').textContent=S.users;
  $$('[data-v]').forEach(el=>{const k=el.dataset.v;el.textContent=k==='lat'?'512':N(S.b[k]);});
  renderLab();Object.keys(CARDS).forEach(k=>{try{CARDS[k](S);}catch(e){console.error(k,e);}});
  moeCard(S);actCards(S);rope(S);
}
function init(){
  $('#sBase').innerHTML=Object.entries(BASES).map(([k,b])=>`<option value="${k}">${b.name}</option>`).join('');
  $('#sDev').innerHTML=DEVS.map((d,i)=>`<option value="${i}"${i===2?' selected':''}>${d.name}</option>`).join('');
  $('#kMoeM').innerHTML=MOES.map((m,i)=>`<option value="${i}">${m[0]} · ${m[1]} B total / ${m[2]} B active</option>`).join('');
  labA=labCol('#labA',{type:'mha'});labB=labCol('#labB',{type:'gqa',kv:8});
  $('#srcList').innerHTML=SRC.map(([id,t,u])=>`<li id="${id}"><a href="${u}" rel="noopener">${t}</a></li>`).join('');
  if(window.THREE){initRR();[['lab','#vLab'],['mha','#vMha'],['mqa','#vMqa'],['gqa','#vGqa'],['mla','#vMla'],['swa','#vSwa'],['dsa','#vDsa'],['lin','#vLin'],['moe','#vMoe'],['abl','#vAbl']].forEach(([k,s])=>VW[k]=view($(s)));}
  gluChart();renderAll();if(VW.abl)ablBuild();
  ['#sBase','#sDev','#sCtx','#sFmt','#sKv','#sUsers'].forEach(s=>$(s).addEventListener('input',renderAll));
  [['#kMha','mha'],['#kGqa','gqa'],['#kMla','mla'],['#kSwaW','swa'],['#kSwaN','swa'],['#kDsa','dsa'],['#kLin','lin']].forEach(([s,k])=>$(s).addEventListener('input',()=>CARDS[k](setup())));
  ['#kMoeM','#kMoeB'].forEach(s=>$(s).addEventListener('input',()=>moeCard(setup())));
  ['#aMu','#aSd'].forEach(s=>$(s).addEventListener('input',()=>actCards(setup())));
  $('#ropeT').addEventListener('change',()=>rope(setup()));
  $('#ablGo').onclick=()=>{AB.a=0;AB.go=true;};$('#ablReset').onclick=()=>{AB.go=false;AB.a=0;ablPlace();};$('#ablNew').onclick=()=>{AB.seed++;ablBuild();};
  const links=$$('#rail a[href^="#"]');if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)links.forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+e.target.id));}),{rootMargin:'-35% 0px -60% 0px'});$$('main section').forEach(s=>io.observe(s));}
  const re=()=>{renderAll();gluChart();if(VW.abl)ablBuild();};
  new MutationObserver(re).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  const mq=window.matchMedia('(prefers-color-scheme: dark)');if(mq.addEventListener)mq.addEventListener('change',re);
  window.addEventListener('resize',()=>{Object.keys(CARDS).forEach(k=>CARDS[k](setup()));renderLab();});
  requestAnimationFrame(loop);
}
window.WHY={stats,setup,BASES,DEVS,compare};
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
})();
