/* Part 5 · Kernels & vLLM: roofline, fusion, batching, paged KV, a real scheduler simulation, benchmarks. */
(function(){
"use strict";
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const N=n=>Math.round(n).toLocaleString('en-US');
const GiB=1073741824,MiB=1048576,KiB=1024;
const fB=b=>b>=1e9?(b/1e9).toFixed(2)+' GB':b>=1e6?(b/1e6).toFixed(1)+' MB':b>=1e3?(b/1e3).toFixed(1)+' KB':N(b)+' B';
const fT=s=>s>=1?s.toFixed(2)+' s':s>=1e-3?(s*1e3).toFixed(2)+' ms':(s*1e6).toFixed(1)+' µs';
const fF=f=>f>=1e12?(f/1e12).toFixed(2)+' TFLOP':(f/1e9).toFixed(2)+' GFLOP';
const fK=n=>n>=1024?(n/1024)+'K':String(n);
const rx=rows=>rows.map(r=>`<div class="row${r[2]?' sum':''}"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('');
function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}

/* ================= data ================= */
const DEVS=[
 {name:'RTX 4060 Ti · 16 GB',mem:16,bw:288e9,fl:88e12,sm:34},
 {name:'RTX 3090 · 24 GB',mem:24,bw:936e9,fl:71e12,sm:82},
 {name:'RTX 4090 · 24 GB',mem:24,bw:1008e9,fl:165e12,sm:128},
 {name:'A100 · 80 GB',mem:80,bw:2039e9,fl:312e12,sm:108},
 {name:'H100 SXM · 80 GB',mem:80,bw:3350e9,fl:989e12,sm:132},
 {name:'Mac Studio M2 Ultra · 192 GB',mem:192,bw:800e9,fl:27e12,sm:76}];
const MODELS=[
 {name:'Qwen3-8B (dense, GQA 32/8)',tot:8.19e9,act:8.19e9,L:36,d:4096,H:32,hd:128,kv:8},
 {name:'Qwen3-30B-A3B (MoE)',tot:30.5e9,act:3.3e9,L:48,d:2048,H:32,hd:128,kv:4},
 {name:'Qwen3-32B (dense, GQA 64/8)',tot:32.8e9,act:32.8e9,L:64,d:5120,H:64,hd:128,kv:8},
 {name:'Llama-3-70B (dense, GQA 64/8)',tot:70.6e9,act:70.6e9,L:80,d:8192,H:64,hd:128,kv:8}];
function setup(){const m=MODELS[+$('#sModel').value],bpw=+$('#sFmt').value;const kvTok=2*m.kv*m.hd*2*m.L;
  return {dev:DEVS[+$('#sDev').value],m,bpw,W:m.act*bpw/8,Wst:m.tot*bpw/8,kvTok};}
const LAUNCH=4e-6;

/* ================= shared renderer (one WebGL context, many views) ================= */
const RR={views:[]};
function initRR(){RR.r=new THREE.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:true});RR.r.setPixelRatio(1);RR.dpr=Math.min(2,window.devicePixelRatio||1);RR.r.setSize(1600*RR.dpr,1000*RR.dpr,false);}
const col=n=>new THREE.Color(css(n));
const mat=(c,op)=>new THREE.MeshLambertMaterial({color:col(c),transparent:op!=null&&op<1,opacity:op==null?1:op});
function clear(g){while(g.children.length){const c=g.children.pop();if(c.children&&c.children.length)clear(c);if(c.geometry)c.geometry.dispose();if(c.material&&c.material.dispose)c.material.dispose();}}
function view(host,onPick){
  const cv=host.querySelector('canvas'),ctx=cv.getContext('2d'),scene=new THREE.Scene(),cam=new THREE.PerspectiveCamera(36,1,0.05,400);
  scene.add(new THREE.AmbientLight(0xffffff,0.72));const dl=new THREE.DirectionalLight(0xffffff,0.55);dl.position.set(5,12,9);scene.add(dl);
  const g=new THREE.Group();scene.add(g);
  const v={host,cv,ctx,scene,cam,g,st:{th:0.5,ph:1.15,r:14,lim:null},tg:new THREE.Vector3(),labels:[],vis:false,tick:null,picks:[]};
  let drag=null;
  cv.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,th:v.st.th,ph:v.st.ph,moved:false};cv.setPointerCapture(e.pointerId);});
  cv.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>5)drag.moved=true;if(!drag.moved)return;v.st.th=drag.th-dx*0.008;v.st.ph=Math.max(0.5,Math.min(1.5,drag.ph-dy*0.006));if(v.st.lim)v.st.th=Math.max(v.st.lim[0],Math.min(v.st.lim[1],v.st.th));});
  cv.addEventListener('pointerup',e=>{if(drag&&!drag.moved&&onPick){const r=cv.getBoundingClientRect(),m=new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),ray=new THREE.Raycaster();ray.setFromCamera(m,v.cam);const h=ray.intersectObjects(v.picks);if(h.length)onPick(h[0].object.userData.pick);}drag=null;});
  cv.addEventListener('pointercancel',()=>drag=null);
  if('IntersectionObserver' in window)new IntersectionObserver(es=>es.forEach(x=>v.vis=x.isIntersecting),{threshold:0.02}).observe(host);else v.vis=true;
  RR.views.push(v);return v;
}
function lab(v,html,pos,side,pri,cls){const el=document.createElement('div');el.className='lbl3 s-'+(side||'c')+(cls?' '+cls:'');el.innerHTML=html;v.host.querySelector('.lbls').appendChild(el);const it={el,pos,side:side||'c',pri:pri||0};v.labels.push(it);return it;}
function reset(v){clear(v.g);v.host.querySelector('.lbls').innerHTML='';v.labels=[];v.picks=[];v.tick=null;}
function fit(v,th,ph,k){
  const box=new THREE.Box3();v.g.children.forEach(c=>{if(!c.userData.noFit)box.expandByObject(c);});
  const c=box.getCenter(new THREE.Vector3()),s=box.getSize(new THREE.Vector3()),w=v.cv.clientWidth||600,h=v.cv.clientHeight||400,tn=Math.tan(18*Math.PI/180);
  v.tg.copy(c);v.st.r=Math.max((s.y/2+0.5)/tn,(s.x/2+0.4)/(tn*(w/h)))*(k||0.95)+s.z*0.35;v.st.th=th;v.st.ph=ph;v.st.lim=[th-0.8,th+0.8];
}
function draw(v,dt){
  const w=v.cv.clientWidth,h=v.cv.clientHeight;if(!w||!h)return;const d=RR.dpr,W=Math.round(w*d),H=Math.round(h*d);
  if(v.cv.width!==W){v.cv.width=W;v.cv.height=H;}
  v.cam.aspect=w/h;v.cam.updateProjectionMatrix();const st=v.st;
  v.cam.position.set(v.tg.x+st.r*Math.sin(st.ph)*Math.sin(st.th),v.tg.y+st.r*Math.cos(st.ph),v.tg.z+st.r*Math.sin(st.ph)*Math.cos(st.th));v.cam.lookAt(v.tg);
  if(v.tick)v.tick(dt);
  RR.r.setViewport(0,0,W,H);RR.r.setScissor(0,0,W,H);RR.r.setScissorTest(true);RR.r.clear();RR.r.render(v.scene,v.cam);
  v.ctx.clearRect(0,0,W,H);v.ctx.drawImage(RR.r.domElement,0,RR.r.domElement.height-H,W,H,0,0,W,H);
  const placed=[];for(const it of v.labels.slice().sort((p,q)=>q.pri-p.pri)){const p=it.pos.clone().project(v.cam);if(p.z>1||p.z<-1){it.el.style.visibility='hidden';continue;}
    if(it.w==null||it.dirty){it.w=it.el.offsetWidth;it.h=it.el.offsetHeight;it.dirty=false;}
    let x=(p.x+1)/2*w,y=(1-p.y)/2*h;if(it.side==='r'){x+=8;y-=it.h/2;}else if(it.side==='l'){x-=8+it.w;y-=it.h/2;}else if(it.side==='t'){x-=it.w/2;y-=it.h+4;}else if(it.side==='b'){x-=it.w/2;y+=4;}else{x-=it.w/2;y-=it.h/2;}
    x=Math.max(2,Math.min(w-it.w-2,x));const r={x0:x,y0:y,x1:x+it.w,y1:y+it.h};
    if((r.y0<0||r.y1>h||placed.some(q=>!(r.x1<q.x0||r.x0>q.x1||r.y1<q.y0||r.y0>q.y1)))&&it.pri<9){it.el.style.visibility='hidden';continue;}
    placed.push(r);it.el.style.visibility='';it.el.style.transform=`translate(${Math.round(x)}px,${Math.round(y)}px)`;}
}
let last=0;function loop(ts){const dt=Math.min(0.05,(ts-last)/1000||0);last=ts;RR.views.forEach(v=>{if(v.vis)draw(v,dt);});requestAnimationFrame(loop);}
const VW={};

/* ================= 0 · the stack ================= */
const STACK=[
 {id:'model',c:'--copper',t:'Model architecture',s:'what to compute',p:`<p>The math: layers, attention type, FFN/MoE, shapes. Decided by the model's authors (parts 2–4).</p><p><b>Example:</b> Qwen3-8B = 36 × (RMSNorm → GQA 32Q/8KV → add → RMSNorm → SwiGLU 12,288 → add).</p><p><b>You control:</b> which model and which variant (dense or MoE, attention type).</p>`},
 {id:'graph',c:'--violet',t:'Framework / model code',s:'how the math is written',p:`<p>The model as code: PyTorch modules (vLLM keeps one file per architecture) or llama.cpp's compute graph. This is where ops get fused (<code>torch.compile</code>) and weights get loaded and sharded.</p><p><b>You control:</b> dtype, quantization format, whether to compile, tensor-parallel size.</p>`},
 {id:'engine',c:'--gold',t:'Engine (vLLM, llama.cpp)',s:'what to compute when',p:`<p>The scheduler and memory manager: which requests run this step, how the KV cache is laid out (pages), batching, chunked prefill, prefix caching, sampling, and the HTTP API.</p><p><b>You control:</b> <code>--max-num-seqs</code>, <code>--max-model-len</code>, <code>--gpu-memory-utilization</code>, <code>--max-num-batched-tokens</code>, prefix caching, speculative decoding.</p>`},
 {id:'kernel',c:'--teal',t:'Kernels',s:'how fast each piece runs',p:`<p>GPU programs: GEMM/GEMV for the matrices, FlashAttention/FlashInfer for attention, Marlin for 4-bit weights, fused MoE, fused RMSNorm/RoPE/SiLU, sampling. Captured into CUDA graphs.</p><p><b>You control:</b> the attention backend (e.g. <code>VLLM_ATTENTION_BACKEND</code>), the quantization kernel (implied by the format), CUDA graphs on or off (<code>--enforce-eager</code> turns them off).</p>`},
 {id:'hw',c:'--steel',t:'Hardware',s:'the physical limits',p:`<p>SMs with tensor cores, on-chip SRAM, HBM/GDDR bandwidth and capacity, and the links between GPUs (NVLink, PCIe). These set the roofline every layer above has to live under.</p><p><b>You control:</b> which GPU and how many. Everything else is about getting close to this ceiling.</p>`}];
function stackView(){
  const v=VW.stack;reset(v);const slabs=[];
  STACK.forEach((s,i)=>{const y=(STACK.length-1-i)*1.5;const m=new THREE.Mesh(new THREE.BoxGeometry(6,0.45,3.2),mat(s.c,0.92));m.position.set(0,y,0);m.userData.pick=s.id;v.g.add(m);v.picks.push(m);slabs.push(m);
    lab(v,`<b>${s.t}</b> · ${s.s}`,new THREE.Vector3(3.1,y,1.6),'r',8);});
  const ball=new THREE.Mesh(new THREE.SphereGeometry(0.22,18,12),new THREE.MeshBasicMaterial({color:col('--gold')}));ball.userData.noFit=true;v.g.add(ball);
  lab(v,'a request',new THREE.Vector3(-3.4,STACK.length*1.5,0),'l',3,'shape');
  let t=0;v.tick=dt=>{t=(t+(reduce?0:dt*0.18))%1;const y=(STACK.length-1)*1.5*(1-t)+0.4;ball.position.set(-2.2,y+0.25,1.7);const idx=Math.min(STACK.length-1,Math.floor(t*STACK.length));
    slabs.forEach((m,i)=>{m.material.emissive=col('--gold');m.material.emissiveIntensity=i===idx?0.28:0;});};
  fit(v,0.45,1.15,1.15);
}
function stackPanel(id){const s=STACK.find(x=>x.id===id)||STACK[0];
  $('#stackPanel').innerHTML=`<span class="eyebrow">Layer</span><h3>${s.t}</h3>${s.p}<h4>A request's trip</h4><ol class="steps"><li>The API receives the prompt and the engine tokenizes it.</li><li>The scheduler admits it when KV blocks are free and starts prefill (maybe in chunks).</li><li>Each step, the model code runs one forward pass for every active request together.</li><li>That pass is a few hundred kernel launches (or one CUDA-graph replay).</li><li>The kernels stream weights and KV cache from HBM through the SMs; a token comes out and is streamed back.</li></ol>`;}

/* ================= 1 · GPU chip ================= */
function gpuView(){
  const S=setup(),v=VW.gpu;reset(v);const op=$('#kOp').value,D=S.dev,m=S.m;
  let bytes,fl,name;
  if(op==='gemv'){bytes=S.W;fl=2*m.act;name='decode GEMV, 1 token';}
  else if(op==='gemm'){bytes=S.W+2048*m.d*2*m.L*4;fl=2*m.act*2048;name='prefill GEMM, 2,048 tokens';}
  else{bytes=3*1*m.d*2;fl=4*m.d;name='RMSNorm, 1 token';}
  const tm=bytes/D.bw,tc=fl/D.fl,t=Math.max(tm,tc),bwU=tm/t,flU=tc/t;
  const cols=16,rows=Math.ceil(D.sm/cols),cs=0.42,sms=[];
  for(let i=0;i<D.sm;i++){const x=(i%cols-(cols-1)/2)*cs,z=(Math.floor(i/cols)-(rows-1)/2)*cs;const sm=new THREE.Mesh(new THREE.BoxGeometry(cs*0.86,0.12,cs*0.86),mat('--steel'));sm.position.set(x,0.06,z);v.g.add(sm);
    const tc=new THREE.Mesh(new THREE.BoxGeometry(cs*0.4,0.14,cs*0.4),mat('--copper'));tc.position.set(x,0.19,z);v.g.add(tc);sms.push(tc);}
  const cw=cols*cs,cd=rows*cs,hb=[];
  [-1,1].forEach(sx=>{for(let k=0;k<3;k++){const b=new THREE.Mesh(new THREE.BoxGeometry(0.7,0.5,cd/3*0.85),mat('--violet',0.85));b.position.set(sx*(cw/2+0.8),0.25,(k-1)*cd/3);v.g.add(b);hb.push(b);}});
  const die=new THREE.Mesh(new THREE.BoxGeometry(cw+0.3,0.04,cd+0.3),mat('--sunk'));die.position.y=-0.02;v.g.add(die);
  lab(v,`<b>${D.sm} SMs</b> · copper = tensor cores`,new THREE.Vector3(0,0.4,-cd/2-0.2),'t',8);
  lab(v,`<b>HBM / GDDR</b> · ${N(D.bw/1e9)} GB/s`,new THREE.Vector3(cw/2+0.8,0.6,cd/2),'r',7);
  // particles: weight tiles flowing from memory to SMs, rate ∝ bandwidth use
  const P=[],pg=new THREE.BoxGeometry(0.1,0.1,0.1),pm=new THREE.MeshBasicMaterial({color:col('--gold')});
  for(let i=0;i<70;i++){const p=new THREE.Mesh(pg,pm);p.userData={t:Math.random(),side:i%2?1:-1,z:(Math.random()-0.5)*cd,tx:(Math.random()-0.5)*cw,tz:(Math.random()-0.5)*cd};p.userData.noFit=true;v.g.add(p);P.push(p);}
  const sp=0.25+1.1*bwU;
  v.tick=dt=>{P.forEach(p=>{const u=p.userData;u.t+=reduce?0:dt*sp;if(u.t>1){u.t=0;u.tx=(Math.random()-0.5)*cw;u.tz=(Math.random()-0.5)*cd;}const x0=u.side*(cw/2+0.8);p.position.set(x0+(u.tx-x0)*u.t,0.5+Math.sin(u.t*Math.PI)*0.6,u.z+(u.tz-u.z)*u.t);});
    const ph=performance.now()/300;sms.forEach((s,i)=>{s.material.emissive=col('--gold');s.material.emissiveIntensity=Math.max(0,flU*(0.55+0.45*Math.sin(ph+i*0.7)));});
    hb.forEach(b=>{b.material.emissive=col('--gold');b.material.emissiveIntensity=0.5*bwU;});};
  fit(v,0.35,0.95,0.9);
  const launchShare=LAUNCH/(t+LAUNCH);
  $('#chGpu').innerHTML=`<div class="tablewrap"><table><tr><th>${name}</th><th>${m.name.split(' (')[0]}, ${$('#sFmt').selectedOptions[0].text}</th></tr>
   <tr><td>Bytes from HBM</td><td class="num">${fB(bytes)}</td></tr><tr><td>Math</td><td class="num">${fF(fl)}</td></tr>
   <tr><td>Time if memory-bound</td><td class="num">${fT(tm)}</td></tr><tr><td>Time if compute-bound</td><td class="num">${fT(tc)}</td></tr>
   <tr><td>Memory bus busy / tensor cores busy</td><td class="num">${(bwU*100).toFixed(0)}% / ${(flU*100).toFixed(op==='gemv'?1:0)}%</td></tr>
   <tr><td>Kernel-launch overhead share (≈4 µs per launch)</td><td class="num">${(launchShare*100).toFixed(launchShare<0.01?2:0)}%</td></tr></table></div>
   <div class="because"><b>What you're seeing:</b> ${op==='gemv'?`gold tiles (weights) pour in from memory at full speed, while the tensor cores flicker at about ${(flU*100).toFixed(1)}% busy. Decode is limited by the delivery truck, not the cooks.`:op==='gemm'?`each weight tile is reused for 2,048 tokens, so the tensor cores are now the bottleneck (${(flU*100).toFixed(0)}% busy) while the memory bus has slack. That's why a prompt is processed far faster per token than generation.`:`the useful work takes ${fT(t)}, but launching the kernel takes ≈4 µs, so <b>${(launchShare*100).toFixed(0)}% of the time is overhead</b>. Hundreds of tiny ops like this per token are why engines fuse kernels and use CUDA graphs.`}</div>`;
}

/* ================= 2 · roofline ================= */
const BS=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096];
function roof(){
  const S=setup(),D=S.dev,B=BS[+$('#kB').value],bpb=S.bpw/8;$('#kBO').textContent=B+(B===1?' (decode, 1 user)':B>=512?' (prefill-like)':' users batched');
  const ridge=D.fl/D.bw,I=x=>2*x/bpb;   // weight GEMM: 2B FLOPs per weight, bpb bytes per weight
  const cv=$('#roofCv'),g=cv.getContext('2d'),W=cv.width,H=cv.height;g.clearRect(0,0,W,H);
  const X=i=>50+(Math.log10(i)+1)/5*(W-70),Y=f=>H-30-(Math.log10(f)-9)/(Math.log10(D.fl)+0.3-9)*(H-50);
  g.strokeStyle=css('--line');g.strokeRect(50,10,W-70,H-40);g.fillStyle=css('--muted');g.font='10px '+css('--mono');
  [0.1,1,10,100,1000,10000].forEach(i=>g.fillText(String(i),X(i)-8,H-16));g.fillText('FLOPs per byte (arithmetic intensity) →',W/2-90,H-3);
  [1e9,1e10,1e11,1e12,1e13,1e14,1e15].forEach(f=>{if(f<=D.fl*2){g.fillText(f>=1e12?(f/1e12)+'T':(f/1e9)+'G',10,Y(f)+3);}});
  g.strokeStyle=css('--copper');g.lineWidth=2.5;g.beginPath();for(let i=0.1;i<=10000;i*=1.08){const f=Math.min(D.fl,i*D.bw);const x=X(i),y=Y(f);i===0.1?g.moveTo(x,y):g.lineTo(x,y);}g.stroke();
  g.setLineDash([4,4]);g.strokeStyle=css('--muted');g.beginPath();g.moveTo(X(ridge),10);g.lineTo(X(ridge),H-30);g.stroke();g.setLineDash([]);g.fillStyle=css('--muted');g.fillText('ridge '+ridge.toFixed(0),X(ridge)+4,24);
  const pts=[['decode, 1 user',I(1),'--teal'],['attention decode (GQA 4:1)',4*S.m.H/(2*S.m.kv*2)*2/2,'--violet'],['prefill 2K',I(2048),'--good'],[`you: ${B}`,I(B),'--gold']];
  const dy={'--teal':16,'--violet':-12,'--good':16,'--gold':-26};pts.forEach(([n,i,c])=>{const f=Math.min(D.fl,i*D.bw);g.fillStyle=css(c);g.beginPath();g.arc(X(i),Y(f),c==='--gold'?7:5,0,7);g.fill();const tw=g.measureText(n).width;g.fillText(n,Math.min(X(i)+8,W-24-tw),Y(f)+dy[c]);});
  const ib=I(B),tStep=Math.max(S.W/D.bw,2*S.m.act*B/D.fl);
  $('#rxRoof').innerHTML=rx([['Peak math',(D.fl/1e12).toFixed(0)+' TFLOP/s'],['Bandwidth',N(D.bw/1e9)+' GB/s'],['Ridge point = peak ÷ bandwidth',ridge.toFixed(0)+' FLOPs per byte',1],['Decode (1 user) at your weight format',`2 FLOPs ÷ ${bpb.toFixed(2)} B = ${I(1).toFixed(1)} FLOPs/byte`]]);
  $('#chRoof').innerHTML=`<div class="because"><b>Cause → effect:</b> ${B} token${B>1?'s':''} per weight read gives intensity ${ib.toFixed(0)} FLOPs/byte, ${ib<ridge?`<b>below</b> the ridge (${ridge.toFixed(0)}), so it's memory-bound and a step takes ${fT(S.W/D.bw)} however many tokens ride along`:`<b>above</b> the ridge, so it's compute-bound and a step takes ${fT(tStep)}, growing with every extra token`}. Throughput: ${N(B/tStep)} tokens/s (weights only). To go from memory-bound to compute-bound on ${D.name.split(' ·')[0]} you need about <b>${Math.ceil(ridge*bpb/2)} tokens per weight read</b>. Note the violet dot: GQA's sharing makes attention decode do ${S.m.H/S.m.kv}× more math per cache byte than MHA would.</div>`;
}

/* ================= 3 · FlashAttention ================= */
const NS=[512,1024,2048,4096,8192,16384,32768];
function flash(){
  const S=setup(),D=S.dev,m=S.m,n=NS[+$('#kN').value],H=m.H,hd=m.hd,Br=128;$('#kNO').textContent=N(n)+' tokens';
  const kvh=m.kv,fl=2*n*n*H*hd;                       // QKᵀ + PV over the causal half
  const io=(2*n*H*hd+2*n*kvh*hd)*2;                 // read Q,K,V and write O once
  const naive=io+4*H*n*n*2;                          // + write S, read S, write P, read P
  const flashB=2*n*H*hd*2+0.5*Math.ceil(n/Br)*2*n*kvh*hd*2; // K/V re-read per Q block (causal half)
  const tN=Math.max(naive/D.bw,fl/D.fl),tF=Math.max(flashB/D.bw,fl/D.fl),Sbytes=H*n*n*2;
  $('#rxFlash').innerHTML=rx([['Score matrix S per layer',`${H} heads × ${N(n)}² × 2 B = ${fB(Sbytes)}`],['Naive HBM traffic per layer',`Q,K,V,O + write/read S and P = ${fB(naive)}`],['FlashAttention HBM traffic',`≈ ${fB(flashB)} (K/V re-read once per ${Br}-row block)`],['Math (both)',fF(fl)],[`Time per layer on ${D.name.split(' ·')[0]}`,`${fT(tN)} → ${fT(tF)} (${(tN/tF).toFixed(1)}× faster)`,1]]);
  $('#chFlash').innerHTML=`<div class="because"><b>Cause → effect:</b> at ${N(n)} tokens the naive version must store a ${fB(Sbytes)} score matrix per layer${Sbytes>D.mem*GiB*0.5?` (<b>it doesn't even fit in ${D.mem} GB</b>)`:''} and move ${fB(naive)} through HBM. FlashAttention moves ${fB(flashB)}, the same math with ${(naive/flashB).toFixed(0)}× fewer bytes, so it becomes ${tF===fl/D.fl?'compute-bound (good: the tensor cores are the limit)':'mostly limited by K/V re-reads'}. Double N and the naive traffic quadruples. That's why long-context models could not exist without this kernel.</div>`;
  // 3D: naive (left) with a giant S plate, flash (right) with tiles through an SRAM box
  const v=VW.flash;reset(v);const sc=x=>Math.sqrt(x),Ls=Math.min(5,0.9*sc(n/512)),pw=0.5;
  const mk=(x,y,z,w,h,c,op)=>{const b=new THREE.Mesh(new THREE.BoxGeometry(w,h,0.08),mat(c,op));b.position.set(x,y,z);v.g.add(b);return b;};
  const L=-3.2,R=3.2;
  [['Q','--copper',-0.75],['K','--teal',0],['V','--teal2',0.75]].forEach(([t,c,dx])=>{mk(L+dx,Ls/2,0,pw,Ls,c);mk(R+dx,Ls/2,0,pw,Ls,c);});
  const Sp=mk(L,Ls+Ls/2+0.6,0,Ls,Ls,'--gold',0.55);lab(v,`<b>Naive</b>: S = QKᵀ stored in HBM<br>${fB(Sbytes)} per layer`,new THREE.Vector3(L,Ls*2+0.8,0),'t',8);
  const sram=new THREE.Mesh(new THREE.BoxGeometry(1.6,0.9,0.9),new THREE.MeshLambertMaterial({color:col('--violet'),transparent:true,opacity:0.25}));sram.position.set(R,Ls+1.2,0);v.g.add(sram);
  lab(v,`<b>FlashAttention</b>: tiles in SRAM<br>S never leaves the chip`,new THREE.Vector3(R,Ls+2.2,0),'t',8);
  lab(v,'Q · K · V in HBM',new THREE.Vector3(L,-0.15,0.1),'b',4,'shape');lab(v,'Q · K · V in HBM',new THREE.Vector3(R,-0.15,0.1),'b',4,'shape');
  const tile=new THREE.Mesh(new THREE.BoxGeometry(0.42,0.3,0.3),mat('--gold'));v.g.add(tile);tile.userData.noFit=true;
  const tile2=new THREE.Mesh(new THREE.BoxGeometry(0.42,0.3,0.3),mat('--teal'));v.g.add(tile2);tile2.userData.noFit=true;
  const cN=lab(v,'',new THREE.Vector3(L,-0.6,0.1),'b',9),cF=lab(v,'',new THREE.Vector3(R,-0.6,0.1),'b',9);
  let tn=0,tf=0;const dN=Math.min(6,2.2*tN/tF),dF=2.2;
  v.tick=dt=>{tn=(tn+(reduce?0:dt/dN))%1;tf=(tf+(reduce?0:dt/dF))%1;
    Sp.scale.y=Math.max(0.02,tn<0.5?tn*2:1);Sp.position.y=Ls+0.6+Ls*Sp.scale.y/2;Sp.material.emissive=col('--gold');Sp.material.emissiveIntensity=tn>0.5?0.4*Math.sin((tn-0.5)*2*Math.PI*3):0;
    const k=(tf*8)%1;tile2.position.set(R+0.0,Ls*(1-k)+0.3*k+ (Ls+0.9)*k,0);tile2.position.y=Ls*0.9*(1-k)+(Ls+1.2)*k;tile.position.set(R,Ls+1.2,0);tile.rotation.y+=reduce?0:dt*3;
    cN.el.innerHTML=`HBM traffic ${fB(naive*tn)}`;cF.el.innerHTML=`HBM traffic ${fB(flashB*tf)}`;};
  fit(v,0.3,1.2,1.2);
}

/* ================= 5 · batching curve ================= */
const BCTX=[512,1024,4096,8192,16384,32768,65536];
function stepTime(S,B,ctx){return Math.max((S.W+B*S.kvTok*ctx)/S.dev.bw,2*S.m.act*B/S.dev.fl)+0.0004;}
function batching(){
  const S=setup(),ctx=BCTX[+$('#kCtx').value];$('#kCtxO').textContent=N(ctx)+' tokens';
  const cap=Math.floor((S.dev.mem*GiB*0.9-S.Wst)/(S.kvTok*ctx));
  const cv=$('#batchCv'),g=cv.getContext('2d'),W=cv.width,H=cv.height;g.clearRect(0,0,W,H);
  const Bs=[];for(let b=1;b<=512;b*=1.15)Bs.push(Math.round(b));const pts=[...new Set(Bs)].map(b=>{const t=stepTime(S,b,ctx);return [b,1/t,b/t];});
  const maxA=Math.max(...pts.map(p=>p[2])),maxU=pts[0][1];
  const X=b=>50+Math.log2(b)/9*(W-100),YA=a=>H-30-a/maxA*(H-50),YU=u=>H-30-u/maxU*(H-50);
  g.strokeStyle=css('--line');g.strokeRect(50,10,W-100,H-40);g.fillStyle=css('--muted');g.font='10px '+css('--mono');
  [1,4,16,64,256].forEach(b=>g.fillText(String(b),X(b)-4,H-16));g.fillText('users in the batch →',W/2-50,H-3);
  if(cap<512){g.fillStyle=css('--bad');g.globalAlpha=0.12;g.fillRect(X(Math.max(1,cap)),10,W-50-X(Math.max(1,cap)),H-40);g.globalAlpha=1;g.fillStyle=css('--bad');g.fillText(cap<1?'not even 1 user fits':'KV memory full beyond '+cap,X(Math.max(1,cap))+4,24);}
  g.strokeStyle=css('--copper');g.lineWidth=2.5;g.beginPath();pts.forEach((p,i)=>i?g.lineTo(X(p[0]),YA(p[2])):g.moveTo(X(p[0]),YA(p[2])));g.stroke();
  g.strokeStyle=css('--teal');g.beginPath();pts.forEach((p,i)=>i?g.lineTo(X(p[0]),YU(p[1])):g.moveTo(X(p[0]),YU(p[1])));g.stroke();
  g.fillStyle=css('--copper');g.fillText('total tok/s (max '+N(maxA)+')',56,24);g.fillStyle=css('--teal');g.fillText('tok/s per user (max '+N(maxU)+')',56,38);
  const row=b=>{const t=stepTime(S,b,ctx);return `<tr><td class="num">${b}</td><td class="num">${fT(t)}</td><td class="num">${(1/t).toFixed(1)}</td><td class="num">${N(b/t)}</td><td>${b<=cap?'<span class="pill good">fits</span>':'<span class="pill bad">no memory</span>'}</td></tr>`;};
  $('#chBatch').innerHTML=`<div class="tablewrap"><table><tr><th>Users</th><th>Step time</th><th>tok/s each</th><th>tok/s total</th><th>KV</th></tr>${[1,8,32,Math.max(1,Math.min(cap,256))].filter((x,i,a)=>a.indexOf(x)===i).sort((a,b)=>a-b).map(row).join('')}</table></div>`;
  const t1=stepTime(S,1,ctx),t8=stepTime(S,8,ctx);
  $('#rxBatch').innerHTML=rx([['Weights read per step (shared)',fB(S.W)],[`KV read per user at ${fK(ctx)}`,fB(S.kvTok*ctx)],['1 user: step',fT(t1)],['8 users: step',`${fT(t8)} (only ${((t8/t1-1)*100).toFixed(0)}% longer)`],['Total throughput gain at 8 users',`${(8*t1/t8).toFixed(1)}×`,1],[`Users that fit (${S.dev.mem} GB)`,N(Math.max(0,cap))]]);
}

/* ================= 6 · paged KV ================= */
const PG={};
function pagedInit(){
  const r=mulberry(11);PG.reqs=Array.from({length:18},(_,i)=>({id:i,len:32+Math.floor(r()*150),tok:32,hue:(i*47)%360}));
  PG.cols=12;PG.rows=8;PG.nb=PG.cols*PG.rows;PG.blockT=16;PG.max=192;PG.t=0;PG.play=false;
  PG.share=$('#pgShare').checked;
  // naive: each request reserves max/16 contiguous blocks; 96/12 = 8 slots
  PG.nv={slots:Array(Math.floor(PG.nb/(PG.max/PG.blockT))).fill(null),queue:PG.reqs.map(q=>Object.assign({},q)),done:0};
  PG.pg={blocks:Array(PG.nb).fill(null),run:[],queue:PG.reqs.map(q=>Object.assign({},q,{bl:[]})),done:0,preempt:0};
  if(PG.share){PG.pg.blocks[0]='S';PG.pg.blocks[1]='S';}
  pagedAdmit();pagedDraw(true);
}
function pagedAdmit(){
  const nv=PG.nv;nv.slots.forEach((s,i)=>{if(!s&&nv.queue.length)nv.slots[i]=nv.queue.shift();});
  const pg=PG.pg,free=()=>pg.blocks.reduce((a,b,i)=>b===null?a.concat(i):a,[]);
  while(pg.queue.length){const q=pg.queue[0],need=PG.share?1:Math.ceil(q.tok/PG.blockT)+0;const f=free();if(f.length<need+1)break;pg.queue.shift();q.bl=[];const own=PG.share?Math.ceil((q.tok-32)/PG.blockT)||1:Math.ceil(q.tok/PG.blockT);for(let k=0;k<own;k++){const b=f.shift();pg.blocks[b]=q.id;q.bl.push(b);}pg.run.push(q);}
}
function pagedStep(){
  const nv=PG.nv;nv.slots.forEach((s,i)=>{if(!s)return;s.tok=Math.min(s.len,s.tok+6);if(s.tok>=s.len){nv.slots[i]=null;nv.done++;}});
  const pg=PG.pg;
  pg.run.forEach(q=>{q.tok=Math.min(q.len,q.tok+6);const own=PG.share?Math.max(1,Math.ceil((q.tok-32)/PG.blockT)):Math.ceil(q.tok/PG.blockT);
    while(q.bl.length<own){const b=pg.blocks.indexOf(null);if(b<0)break;pg.blocks[b]=q.id;q.bl.push(b);}});
  pg.run=pg.run.filter(q=>{if(q.tok>=q.len){q.bl.forEach(b=>pg.blocks[b]=null);pg.done++;return false;}return true;});
  pagedAdmit();PG.t++;pagedDraw(false);
}
function pagedDraw(rebuild){
  const v=VW.paged;
  if(rebuild){reset(v);PG.m=[];const sc=0.36;
    [-1,1].forEach((sx,si)=>{const arr=[];for(let i=0;i<PG.nb;i++){const b=new THREE.Mesh(new THREE.BoxGeometry(sc*0.86,0.2,sc*0.86),new THREE.MeshLambertMaterial({color:col('--sunk'),transparent:true,opacity:1}));b.position.set(sx*2.6+(i%PG.cols-(PG.cols-1)/2)*sc,0.1,(Math.floor(i/PG.cols)-(PG.rows-1)/2)*sc);v.g.add(b);arr.push(b);}PG.m.push(arr);});
    lab(v,'<b>Before vLLM</b>: reserve max length, contiguous',new THREE.Vector3(-2.6,0.3,-1.7),'t',8);
    lab(v,'<b>vLLM</b>: 16-token blocks on demand',new THREE.Vector3(2.6,0.3,-1.7),'t',8);
    PG.lN=lab(v,'',new THREE.Vector3(-2.6,0,1.7),'b',9);PG.lP=lab(v,'',new THREE.Vector3(2.6,0,1.7),'b',9);
    fit(v,0.25,0.9,0.92);
    v.tick=dt=>{if(PG.play){PG.acc=(PG.acc||0)+dt;if(PG.acc>0.35){PG.acc=0;pagedStep();if(!PG.nv.queue.length&&!PG.nv.slots.some(x=>x)&&!PG.pg.run.length&&!PG.pg.queue.length)PG.play=false;}}};}
  const c=h=>new THREE.Color().setHSL(h/360,0.55,0.55);
  // naive: slot i owns blocks [i*12, i*12+12)
  const per=PG.max/PG.blockT;let usedN=0,resN=0;
  PG.m[0].forEach((b,i)=>{const s=PG.nv.slots[Math.floor(i/per)],k=i%per;if(!s){b.material.color.copy(col('--sunk'));b.material.opacity=1;return;}resN++;const filled=k<Math.ceil(s.tok/PG.blockT);if(filled)usedN++;b.material.color.copy(filled?c(s.hue):col('--line'));b.material.opacity=filled?1:0.55;});
  let usedP=0,sh=0;PG.m[1].forEach((b,i)=>{const o=PG.pg.blocks[i];if(o===null){b.material.color.copy(col('--sunk'));return;}if(o==='S'){sh++;b.material.color.copy(col('--gold'));return;}usedP++;const q=PG.reqs[o];b.material.color.copy(c(q.hue));});
  const runN=PG.nv.slots.filter(x=>x).length,runP=PG.pg.run.length;
  PG.lN.el.innerHTML=`running <b>${runN}</b> · waiting ${PG.nv.queue.length} · done ${PG.nv.done}`;PG.lP.el.innerHTML=`running <b>${runP}</b> · waiting ${PG.pg.queue.length} · done ${PG.pg.done}`;
  const tokN=PG.nv.slots.reduce((a,s)=>a+(s?s.tok:0),0),tokP=PG.pg.run.reduce((a,q)=>a+q.tok,0);
  $('#chPaged').innerHTML=`<div class="tablewrap"><table><tr><th></th><th>Contiguous reservation</th><th>Paged (vLLM)</th></tr>
   <tr><td>Requests running now</td><td class="num">${runN}</td><td class="num"><b>${runP}</b></td></tr>
   <tr><td>Blocks holding real tokens</td><td class="num">${usedN} of ${resN} reserved</td><td class="num">${usedP}${PG.share?` + ${sh} shared`:''} in use</td></tr>
   <tr><td>Reserved but empty (waste)</td><td class="num">${resN?((1-usedN/resN)*100).toFixed(0):0}%</td><td class="num">${'< 1 block per request'}</td></tr>
   <tr><td>Finished so far (step ${PG.t})</td><td class="num">${PG.nv.done}</td><td class="num"><b>${PG.pg.done}</b></td></tr></table></div>
   <div class="because"><b>Cause → effect:</b> the left side reserves ${PG.max} tokens per request although answers average ~${N(PG.reqs.reduce((a,q)=>a+q.len,0)/PG.reqs.length)}, so the striped grey blocks are memory nobody can use, and only ${PG.nv.slots.length} requests fit. Paging hands out blocks only as tokens arrive, so ${runP>runN?`<b>${runP} requests run at once</b> in the same memory. More requests per step = bigger batch = the free lunch from section 5`:'more requests can run in the same memory'}.${PG.share?' Gold blocks are the shared system prompt, stored once for everyone (prefix caching).':''}</div>`;
}

/* ================= 7 · scheduler simulation ================= */
const RATES=[0.25,0.5,1,2,3,4,6,8,12,16,24,32],INS=[128,512,1024,2048,4096,8192],OUTS=[64,128,256,512,1024];
function simulate(S,policy,rate,inLen,outLen,nReq){
  const r=mulberry(7),D=S.dev,kvT=S.kvTok,cap=Math.max(1,Math.floor((D.mem*GiB*0.9-S.Wst)/kvT));
  let t=0;const reqs=[];for(let i=0;i<nReq;i++){t+=-Math.log(1-r())/rate;reqs.push({id:i,arr:t,inp:inLen,out:Math.max(4,Math.round(outLen*(0.4+1.2*r()))),pre:0,gen:0,ttft:null,fin:null,ps:null});}
  const preT=(n,ctx)=>(2*S.m.act*n+2*S.m.H*S.m.hd*n*ctx*S.m.L)/D.fl;
  let now=0,qi=0;const wait=[],run=[];let maxRun=0;
  const arrive=()=>{while(qi<reqs.length&&reqs[qi].arr<=now)wait.push(reqs[qi++]);};
  let guard=0;
  if(policy==='cont'){
    const budget=2048;let used=0;
    while((qi<reqs.length||wait.length||run.length)&&guard++<200000){
      arrive();
      // admit while KV fits (paged: count real tokens)
      while(wait.length&&used+wait[0].inp+16<=cap){const q=wait.shift();q.ps=now;used+=q.inp;run.push(q);}
      if(!run.length){if(qi>=reqs.length)break;now=Math.max(now,reqs[qi].arr);continue;}
      let nd=0,np=0,kvb=0,pc=0;const dec=run.filter(q=>q.pre>=q.inp);
      dec.forEach(q=>{nd++;kvb+=kvT*(q.inp+q.gen);});
      let left=budget-nd;for(const q of run){if(q.pre<q.inp&&left>0){const c=Math.min(left,q.inp-q.pre);q._c=c;np+=c;pc+=preT(c,q.pre+c/2)-2*S.m.act*c/D.fl;left-=c;}else q._c=0;}
      const dt=Math.max((S.W+kvb)/D.bw,(2*S.m.act*(nd+np))/D.fl+pc)+0.0004;now+=dt;maxRun=Math.max(maxRun,run.length);
      dec.forEach(q=>{q.gen++;used++;if(q.gen>=q.out){q.fin=now;}});
      run.forEach(q=>{if(q._c){q.pre+=q._c;if(q.pre>=q.inp&&q.ttft===null){q.ttft=now;q.gen=1;used++;}}});
      for(let i=run.length-1;i>=0;i--)if(run[i].fin!==null){used-=run[i].inp+run[i].gen;run.splice(i,1);}
    }
  }else{
    const Bmax=Math.max(1,Math.min(64,Math.floor(cap/(inLen+outLen*1.6))));
    while((qi<reqs.length||wait.length)&&guard++<200000){
      arrive();if(!wait.length){if(qi>=reqs.length)break;now=reqs[qi].arr;continue;}
      const batch=wait.splice(0,Bmax);maxRun=Math.max(maxRun,batch.length);batch.forEach(q=>q.ps=now);
      now+=preT(inLen*batch.length,inLen/2)+0.0004;batch.forEach(q=>{q.ttft=now;q.gen=1;q.pre=q.inp;});
      while(batch.some(q=>q.gen<q.out)){const act=batch.filter(q=>q.gen<q.out);const kvb=batch.reduce((a,q)=>a+kvT*(q.inp+q.gen),0);
        now+=Math.max((S.W+kvb)/D.bw,2*S.m.act*act.length/D.fl)+0.0004;act.forEach(q=>{q.gen++;if(q.gen>=q.out)q.fin=now;});}
      batch.forEach(q=>{if(q.fin===null)q.fin=now;});
    }
  }
  const done=reqs.filter(q=>q.fin!==null&&q.ttft!==null),pc=(a,p)=>{const s=a.slice().sort((x,y)=>x-y);return s[Math.min(s.length-1,Math.floor(p*s.length))]||0;};
  const ttft=done.map(q=>q.ttft-q.arr),tpot=done.map(q=>(q.fin-q.ttft)/Math.max(1,q.out-1)),outTok=done.reduce((a,q)=>a+q.out,0);
  const span=Math.max(...done.map(q=>q.fin))-reqs[0].arr;
  return {reqs,ttft50:pc(ttft,0.5),ttft99:pc(ttft,0.99),tpot50:pc(tpot,0.5),tpot99:pc(tpot,0.99),thr:outTok/span,rps:done.length/span,maxRun,cap,done:done.length,
    good:(sT,sP)=>done.filter(q=>q.ttft-q.arr<=sT&&(q.fin-q.ttft)/Math.max(1,q.out-1)<=sP).length/span};
}
function sched(){
  const S=setup(),rate=RATES[+$('#qRate').value],inL=INS[+$('#qIn').value],outL=OUTS[+$('#qOut').value];
  $('#qRateO').textContent=rate+' /s';$('#qInO').textContent=N(inL);$('#qOutO').textContent='~'+N(outL)+' (varies 0.4–1.6×)';
  const A=simulate(S,'static',rate,inL,outL,160),B=simulate(S,'cont',rate,inL,outL,160);
  const r=(n,a,b,f,lower)=>`<tr><td>${n}</td><td class="num">${f(a)}</td><td class="num">${f(b)}</td><td class="num"><span class="${(b<a)===lower?'up':'dn'}">${a?((b/a-1)*100>0?'+':'')+((b/a-1)*100).toFixed(0)+'%':''}</span></td></tr>`;
  $('#chSched').innerHTML=`<div class="tablewrap"><table><tr><th>Metric (160 requests)</th><th>Static batching</th><th>Continuous + paged (vLLM)</th><th>Change</th></tr>
   ${r('TTFT p50',A.ttft50,B.ttft50,fT,true)}${r('TTFT p99',A.ttft99,B.ttft99,fT,true)}${r('TPOT p50',A.tpot50,B.tpot50,fT,true)}${r('TPOT p99',A.tpot99,B.tpot99,fT,true)}
   ${r('Throughput (output tok/s)',A.thr,B.thr,x=>N(x),false)}${r('Most requests running at once',A.maxRun,B.maxRun,x=>N(x),false)}</table></div>
   <div class="because"><b>Cause → effect:</b> static batches wait for their longest answer (outputs vary 0.4–1.6×) and newcomers wait for the whole batch, so TTFT balloons. Continuous batching admits requests every step and frees memory the moment a request finishes, so the batch stays full: throughput ${(B.thr/A.thr).toFixed(1)}×. ${B.ttft99>10?'<b>At this rate the GPU is past capacity:</b> the queue grows without bound, so TTFT explodes for any scheduler. Lower the rate or see section 9.':''} KV capacity on ${S.dev.name.split(' ·')[0]}: ${N(B.cap)} tokens.</div>`;
  schedView(A,B);curves();
}
function schedView(A,B){
  const v=VW.sched;reset(v);const n=18,tmax=Math.max(A.reqs[n-1].fin,B.reqs[n-1].fin)||1,Wd=9,rh=0.22;
  const bar=(x0,x1,y,z,c,h)=>{if(x1<=x0)return;const m=new THREE.Mesh(new THREE.BoxGeometry(Math.max(0.02,x1-x0),h,0.12),mat(c));m.position.set((x0+x1)/2,y+h/2,z);v.g.add(m);return m;};
  const X=t=>-Wd/2+t/tmax*Wd;
  [[A,-3.5,'Static batching'],[B,0.5,'Continuous + paged (vLLM)']].forEach(([R,z0,name])=>{
    for(let i=0;i<n;i++){const q=R.reqs[i];const zz=z0+i*0.17;
      bar(X(q.arr),X(q.ps),0,zz,'--line',0.05);bar(X(q.ps),X(q.ttft),0,zz,'--violet',0.18);bar(X(q.ttft),X(q.fin),0,zz,'--copper',0.18);}
    lab(v,`<b>${name}</b>`,new THREE.Vector3(-Wd/2-0.2,0.2,z0+1.5),'l',8);});
  const cur=new THREE.Mesh(new THREE.BoxGeometry(0.03,0.6,7.2),new THREE.MeshBasicMaterial({color:col('--gold')}));cur.position.set(-Wd/2,0.3,-0.1);cur.userData.noFit=true;v.g.add(cur);
  lab(v,'time →',new THREE.Vector3(Wd/2,0,3.6),'r',4,'shape');
  lab(v,`first ${n} requests · same arrivals`,new THREE.Vector3(0,0.4,-3.7),'t',6,'shape');
  let t=0;v.tick=dt=>{t=(t+(reduce?0:dt/6))%1;cur.position.x=-Wd/2+t*Wd;};
  fit(v,0.25,0.85,0.9);
}

/* ================= 9 · benchmark curves ================= */
function curves(){
  const S=setup(),inL=INS[+$('#qIn').value],outL=OUTS[+$('#qOut').value],sT=+$('#sloT').value,sP=+$('#sloP').value/1000;
  $('#sloTO').textContent=sT+' s';$('#sloPO').textContent=(sP*1000)+' ms';
  const pts=RATES.map(rt=>{const R=simulate(S,'cont',rt,inL,outL,120);return {rt,R,good:R.good(sT,sP)};});
  const cv=$('#curveCv'),g=cv.getContext('2d'),W=cv.width,H=cv.height;g.clearRect(0,0,W,H);
  const X=r=>50+Math.log2(r/0.25)/7*(W-110),Yl=s=>H-30-Math.min(1,Math.log10(Math.max(s,0.01)/0.01)/4)*(H-50),maxT=Math.max(...pts.map(p=>p.R.thr)),Yr=v=>H-30-v/maxT*(H-50);
  g.strokeStyle=css('--line');g.strokeRect(50,10,W-110,H-40);g.fillStyle=css('--muted');g.font='10px '+css('--mono');
  RATES.forEach(r=>g.fillText(String(r),X(r)-6,H-16));g.fillText('request rate (req/s) →',W/2-60,H-3);
  [0.01,0.1,1,10,100].forEach(s=>g.fillText(s+'s',8,Yl(s)+3));
  g.setLineDash([4,4]);g.strokeStyle=css('--bad');g.beginPath();g.moveTo(50,Yl(sT));g.lineTo(W-60,Yl(sT));g.stroke();g.setLineDash([]);g.fillStyle=css('--bad');g.fillText('TTFT SLO',54,Yl(sT)-4);
  const line=(k,c,Y)=>{g.strokeStyle=css(c);g.lineWidth=2.4;g.beginPath();pts.forEach((p,i)=>{const y=Y(k(p));i?g.lineTo(X(p.rt),y):g.moveTo(X(p.rt),y);});g.stroke();};
  line(p=>p.R.ttft50,'--teal',Yl);line(p=>p.R.ttft99,'--violet',Yl);line(p=>p.R.thr,'--copper',Yr);line(p=>p.good*outL,'--gold',Yr);g.fillStyle=css('--copper');[0.25,0.5,0.75,1].forEach(f=>g.fillText(N(maxT*f),W-56,Yr(maxT*f)+3));
  g.fillStyle=css('--teal');g.fillText('TTFT p50',W-150,24);g.fillStyle=css('--violet');g.fillText('TTFT p99',W-150,38);g.fillStyle=css('--copper');g.fillText('throughput tok/s',W-150,52);g.fillStyle=css('--gold');g.fillText('goodput (meets SLO)',W-150,66);
  const ok=pts.filter(p=>p.R.ttft99<=sT&&p.R.tpot99<=sP),best=ok.length?ok[ok.length-1]:null,peak=pts.reduce((a,p)=>p.R.thr>a.R.thr?p:a,pts[0]);
  if(best){g.fillStyle=css('--gold');g.beginPath();g.arc(X(best.rt),Yl(best.R.ttft99),6,0,7);g.fill();}
  $('#chCurve').innerHTML=`<div class="because"><b>Reading the curve:</b> at low load, latency is just prefill plus decode. As the rate approaches what the GPU can serve, the queue starts to grow and p99 TTFT shoots up. That's the <b>knee</b>. Throughput climbs until the GPU saturates at about ${N(peak.R.thr)} tok/s, then flattens. ${best?`<b>Capacity at your SLO</b> (p99 TTFT ≤ ${sT} s, p99 TPOT ≤ ${sP*1000} ms): about <b>${best.rt} requests/s</b> of ${N(inL)}-in / ~${N(outL)}-out on ${S.dev.name.split(' ·')[0]} with ${S.m.name.split(' (')[0]}.`:`<b>No load level meets this SLO</b> on this setup. Relax the SLO, shorten prompts, or use a smaller or quantized model.`} Always quote capacity <i>at an SLO</i>. The peak-throughput number alone hides that users would be waiting ${fT(peak.R.ttft99)} for their first token.</div>`;
}

/* ================= 10 · efficiency ================= */
function eff(){
  const S=setup(),tps=+$('#eTps').value||1,ctx=+$('#eCtx').value,bytes=S.W+S.kvTok*ctx,ach=tps*bytes,p=ach/S.dev.bw,ceil=S.dev.bw/bytes;
  const verdict=p>=0.7?['good','Excellent: the kernels are near the roofline. Only fewer bytes (quantization, MLA/MoE, smaller KV) will make it faster.']:p>=0.4?['warn','Normal to fair. Check CUDA graphs (not --enforce-eager), the attention backend, and whether everything is on the GPU.']:p<=1.05?['bad','Low. Likely CPU launch overhead, layers offloaded over PCIe, a dequantize-then-GEMM path, or debug settings. Profile it with nsys.']:['bad','Above 100%: impossible. Check the context, the format, or whether speculative decoding or batching inflated the number.'];
  $('#chEff').innerHTML=`<div class="receipt">${rx([['Bytes per token (weights + KV at '+fK(ctx)+')',fB(bytes)],['Roofline ceiling',ceil.toFixed(1)+' tok/s'],['Achieved bandwidth = tok/s × bytes',`${(ach/1e9).toFixed(0)} GB/s`],[`Share of ${N(S.dev.bw/1e9)} GB/s peak`,(p*100).toFixed(0)+'%',1]])}</div><div class="because"><span class="pill ${verdict[0]}">${(p*100).toFixed(0)}%</span> ${verdict[1]}</div>`;
}

/* ================= 8 · small calculators ================= */
function tricks(){
  const S=setup(),launches=S.m.L*14+20,ov=launches*LAUNCH,step=S.W/S.dev.bw;
  $('#rxGraph').innerHTML=rx([['Kernel launches per decode step',`≈ ${S.m.L} layers × 14 + 20 = ${N(launches)}`],['Launch cost at ~4 µs each',fT(ov)],['Memory time for the weights',fT(step)],['Without graphs, if launches don\'t overlap',`${(step/(step+ov)*100).toFixed(0)}% of peak speed`,1]]);
  const a=+$('#spA').value/100,k=+$('#spK').value;$('#spAO').textContent=a.toFixed(2);$('#spKO').textContent=k;
  const E=(1-Math.pow(a,k+1))/(1-a),c=0.05,sp=E/(1+k*c);
  $('#rxSpec').innerHTML=rx([['Tokens per verify step',E.toFixed(2)],['Drafting cost (≈5% of a big step per token)',`${(k*c*100).toFixed(0)}%`],['Expected speedup',sp.toFixed(2)+'×',1],[`On ${S.dev.name.split(' ·')[0]}`,`${(S.dev.bw/S.W).toFixed(0)} → ${(S.dev.bw/S.W*sp).toFixed(0)} tok/s (1 user)`]]);
}

/* ================= wiring ================= */
const SRC=[['s-vllm','Kwon et al., Efficient Memory Management for LLM Serving with PagedAttention (vLLM, SOSP 2023)','https://arxiv.org/abs/2309.06180'],
['s-orca','Yu et al., Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022)','https://www.usenix.org/conference/osdi22/presentation/yu'],
['s-fa1','Dao et al., FlashAttention (2022)','https://arxiv.org/abs/2205.14135'],
['s-fa2','Dao, FlashAttention-2 (2023)','https://arxiv.org/abs/2307.08691'],
['s-fa3','Shah et al., FlashAttention-3 (2024)','https://arxiv.org/abs/2407.08608'],
['s-fd','Dao et al., Flash-Decoding for long-context inference (2023)','https://crfm.stanford.edu/2023/10/12/flashdecoding.html'],
['s-sarathi','Agrawal et al., Sarathi-Serve: chunked prefills (2024)','https://arxiv.org/abs/2403.02310'],
['s-dist','Zhong et al., DistServe: goodput-optimized serving (2024)','https://arxiv.org/abs/2401.09670'],
['s-spec','Leviathan et al., Fast Inference via Speculative Decoding (2023)','https://arxiv.org/abs/2211.17192'],
['s-eagle','Li et al., EAGLE: speculative sampling (2024)','https://arxiv.org/abs/2401.15077'],
['s-sglang','Zheng et al., SGLang / RadixAttention (2023)','https://arxiv.org/abs/2312.07104'],
['s-roof','Williams, Waterman & Patterson, Roofline model (CACM 2009)','https://doi.org/10.1145/1498765.1498785'],
['s-marlin','Frantar et al., MARLIN: mixed-precision kernels (2024)','https://arxiv.org/abs/2408.11743'],
['s-flashinfer','Ye et al., FlashInfer (2025)','https://arxiv.org/abs/2501.01005'],
['s-flashmla','DeepSeek, FlashMLA','https://github.com/deepseek-ai/FlashMLA'],
['s-deepgemm','DeepSeek, DeepGEMM and DeepEP','https://github.com/deepseek-ai/DeepGEMM'],
['s-fla','flash-linear-attention (Triton kernels for linear attention)','https://github.com/fla-org/flash-linear-attention'],
['s-vbench','vLLM docs: benchmarking (vllm bench serve / throughput / latency)','https://docs.vllm.ai/en/latest/contributing/benchmarks.html'],
['s-lbench','llama.cpp: llama-bench','https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench'],
['s-nsys','NVIDIA Nsight Systems and Nsight Compute','https://developer.nvidia.com/nsight-systems']];
function all(){gpuView();roof();flash();batching();tricks();eff();pagedDraw(false);sched();}
function init(){
  $('#sDev').innerHTML=DEVS.map((d,i)=>`<option value="${i}"${i===2?' selected':''}>${d.name} · ${N(d.bw/1e9)} GB/s</option>`).join('');
  $('#sModel').innerHTML=MODELS.map((m,i)=>`<option value="${i}">${m.name}</option>`).join('');
  $('#srcList').innerHTML=SRC.map(([id,t,u])=>`<li id="${id}"><a href="${u}" rel="noopener">${t}</a></li>`).join('');
  if(window.THREE){initRR();VW.stack=view($('#vStack'),id=>stackPanel(id));VW.gpu=view($('#vGpu'));VW.flash=view($('#vFlash'));VW.paged=view($('#vPaged'));VW.sched=view($('#vSched'));stackView();}
  stackPanel('engine');pagedInit();all();
  ['#sDev','#sModel','#sFmt'].forEach(s=>$(s).addEventListener('input',all));
  $('#kOp').addEventListener('input',gpuView);$('#kB').addEventListener('input',roof);$('#kN').addEventListener('input',flash);$('#kCtx').addEventListener('input',batching);
  ['#qRate','#qIn','#qOut'].forEach(s=>$(s).addEventListener('input',sched));['#sloT','#sloP'].forEach(s=>$(s).addEventListener('input',curves));
  ['#spA','#spK'].forEach(s=>$(s).addEventListener('input',tricks));['#eTps','#eCtx'].forEach(s=>$(s).addEventListener('input',eff));
  $('#pgPlay').onclick=()=>{PG.play=!PG.play;$('#pgPlay').textContent=PG.play?'❚❚ Pause':'▶ Generate tokens';};$('#pgReset').onclick=()=>{pagedInit();$('#pgPlay').textContent='▶ Generate tokens';};$('#pgShare').onchange=()=>pagedInit();
  const links=$$('#rail a[href^="#"]');if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)links.forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+e.target.id));}),{rootMargin:'-35% 0px -60% 0px'});$$('main section').forEach(s=>io.observe(s));}
  const re=()=>{if(window.THREE){stackView();pagedDraw(true);}all();};new MutationObserver(re).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  requestAnimationFrame(loop);
}
window.KV={simulate,setup};
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
})();
