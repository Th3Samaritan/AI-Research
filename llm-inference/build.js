/* Part 6 · Build it: kernel process, anatomy, tiling, three kernels, vLLM structure, a mini-engine scheduler trace, extension paths. */
(function(){
"use strict";
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const N=n=>Math.round(n).toLocaleString('en-US');
const fB=b=>b>=1e9?(b/1e9).toFixed(2)+' GB':b>=1e6?(b/1e6).toFixed(1)+' MB':b>=1e3?(b/1e3).toFixed(1)+' KB':N(b)+' B';
const fT=s=>s>=1?s.toFixed(2)+' s':s>=1e-3?(s*1e3).toFixed(2)+' ms':(s*1e6).toFixed(1)+' µs';
const rx=rows=>rows.map(r=>`<div class="row${r[2]?' sum':''}"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('');
const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const LAUNCH=4e-6;

/* ================= data ================= */
// smem = max shared memory one block can use (KB); smSM = per SM (KB)
const DEVS=[
 {name:'RTX 4060 Ti · 16 GB',bw:288e9,fl:88e12,sm:34,smem:99,smSM:100},
 {name:'RTX 3090 · 24 GB',bw:936e9,fl:71e12,sm:82,smem:99,smSM:100},
 {name:'RTX 4090 · 24 GB',bw:1008e9,fl:165e12,sm:128,smem:99,smSM:100},
 {name:'A100 · 80 GB',bw:2039e9,fl:312e12,sm:108,smem:163,smSM:164},
 {name:'H100 SXM · 80 GB',bw:3350e9,fl:989e12,sm:132,smem:227,smSM:228},
 {name:'Mac Studio M2 Ultra · 192 GB',bw:800e9,fl:27e12,sm:76,smem:32,smSM:32}];
const dev=()=>DEVS[+$('#sDev').value];

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
function relabel(it,html){it.el.innerHTML=html;it.dirty=true;}
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
    x=Math.max(2,Math.min(w-it.w-2,x));y=Math.max(2,y);const r={x0:x,y0:y,x1:x+it.w,y1:y+it.h};
    if((r.y1>h||placed.some(q=>!(r.x1<q.x0||r.x0>q.x1||r.y1<q.y0||r.y0>q.y1)))&&it.pri<9){it.el.style.visibility='hidden';continue;}
    placed.push(r);it.el.style.visibility='';it.el.style.transform=`translate(${Math.round(x)}px,${Math.round(y)}px)`;}
}
let last=0;function loop(ts){const dt=Math.min(0.05,(ts-last)/1000||0);last=ts;RR.views.forEach(v=>{if(v.vis)draw(v,dt);});requestAnimationFrame(loop);}
const VW={};

/* ================= 1 · the process ladder ================= */
const LADDER=[
 {t:'Theory',q:'Is it memory- or compute-bound, and what is the best possible time?',d:'Count the bytes moved and the FLOPs, and place the op on the roofline (part 5, section 2).',c:()=>{const D=dev();return `You have a target number. W4 GEMV for gate_up on the ${D.name.split(' ·')[0]}: 51.9 MB ÷ ${N(D.bw/1e9)} GB/s ≈ <b>${fT(51.9e6/D.bw)}</b>.`;},f:'If the best case isn\'t clearly better than what already exists, stop here. Most ideas end at this step, and that\'s the point of it.'},
 {t:'Spec',q:'Which shapes, dtypes and memory layouts must it handle?',d:'List every call site from the real model. Qwen3-8B: qkv 4,096→6,144, o 4,096→4,096, gate_up 4,096→24,576, down 12,288→4,096; token counts 1–256 in decode, up to 8,192 in prefill.',c:()=>'A written table of shapes and edge cases: M = 1, odd M, N not a multiple of the tile, the largest batch.',f:'A shape you didn\'t list becomes a crash or a silent wrong answer in production.'},
 {t:'Reference',q:'What exactly is the right answer?',d:'Write it in plain PyTorch, slow and obvious. For a quantization format, write both quantize and dequantize.',c:()=>'It matches Hugging Face Transformers\' module on random inputs.',f:'A bug here poisons every later test, so keep it boring. No cleverness in the reference.'},
 {t:'Map to GPU',q:'What does one program compute, and does its data fit in SRAM?',d:'Choose the parallelization and tile sizes (sections 2–3). Aim for 2–4 programs per SM, with each byte read from HBM once.',c:()=>'The tile fits, programs ≥ number of SMs, and padding waste is small. Check it in the tiling lab.',f:'Too few programs: use split-K or smaller tiles. Doesn\'t fit: fewer pipeline stages or a smaller BLOCK_K.'},
 {t:'Write',q:'What\'s the simplest kernel that could hit the target?',d:'Write it in Triton first. Skip clever tricks until the tests pass.',c:()=>'It compiles and runs on every shape in the spec.',f:'"out of resources: shared memory" means the tile is too big for this GPU.'},
 {t:'Test',q:'Does it match the reference on every shape and dtype?',d:'pytest over the spec table, using torch.testing.assert_close with tolerances that fit the dtype (section 12).',c:()=>'Everything passes with fixed seeds and outlier-heavy inputs.',f:'Fails only at large K: accumulation order or precision. Fails at random positions: an indexing or mask bug.'},
 {t:'Microbench',q:'How close is it to the target from step 1?',d:'Time it with triton.testing.do_bench, then convert to GB/s or TFLOP/s (section 7).',c:()=>'At least 70–80% of peak bandwidth for a memory-bound kernel.',f:'Below 50%: go to step 8.'},
 {t:'Profile',q:'Why isn\'t it faster?',d:'Run ncu and read memory throughput, occupancy and stall reasons. Fixes: coalesce, vectorize, autotune tiles, split-K, load each word only once.',c:()=>'After every change, re-run steps 6 and 7.',f:'Never keep an optimization that you haven\'t re-tested for correctness.'},
 {t:'Integrate',q:'Can the engine call it safely?',d:'Wrap it as a torch.library custom op with a fake implementation, then as a vLLM CustomOp or quantization method (section 10). Avoid host syncs and data-dependent allocation so CUDA graphs can capture it.',c:()=>'Eager and CUDA-graph outputs are identical, and greedy text matches.',f:'A hang or garbage output under graphs usually means a hidden .item() call or an allocation.'},
 {t:'End to end',q:'Did serving actually get better?',d:'Run a vllm bench serve rate sweep against the baseline, and perplexity / lm-eval for quality.',c:()=>'Better TPOT and goodput at equal quality, close to the Amdahl prediction below.',f:'A smaller gain than predicted means the op was a smaller share of time than you thought, or another bottleneck took over.'}];
let ladderI=0;
function ladder(){
  $('#ladder').innerHTML=LADDER.map((s,i)=>`<button type="button" data-i="${i}" class="${i===ladderI?'on':''}"><small>step ${i+1}</small>${s.t}</button>`).join('');
  const s=LADDER[ladderI];
  $('#ladderOut').innerHTML=`<div class="q"><b>Question</b>${s.q}</div><div class="d"><b>Do</b>${s.d}</div><div class="c"><b>Pass when</b>${s.c()}</div><div class="f"><b>If it fails</b>${s.f}</div>`;
  $$('#ladder button').forEach(b=>b.onclick=()=>{ladderI=+b.dataset.i;ladder();});
}
function amdahl(){
  const p=+$('#amS').value/100,x=+$('#amX').value/10;$('#amSO').textContent=(p*100).toFixed(0)+'%';$('#amXO').textContent=x.toFixed(1)+'×';
  const e=1/((1-p)+p/x),cap=1/(1-p);
  $('#rxAm').innerHTML=rx([['Time outside the kernel',`${((1-p)*100).toFixed(0)}% (unchanged)`],['Kernel time after the speedup',`${(p*100).toFixed(0)}% ÷ ${x.toFixed(1)} = ${(p/x*100).toFixed(1)}%`],['New step time',`${(((1-p)+p/x)*100).toFixed(1)}% of before`],['End-to-end speedup',e.toFixed(2)+'×',1],['Ceiling (kernel takes zero time)',cap.toFixed(2)+'×']]);
}

/* ================= 2 · grid → block → warp → thread ================= */
const HI={play:false,t:0,phase:-1};
const PHASES=[
 ['Load + square, pass 1 of 4','Threads 0–1,023 each read one element: x[0…1023]. Neighbours read neighbouring addresses, so each warp\'s 32 reads merge into one transaction.'],
 ['Load + square, pass 2 of 4','x[1024…2047]. Each thread keeps a running sum of squares in its own register.'],
 ['Load + square, pass 3 of 4','x[2048…3071].'],
 ['Load + square, pass 4 of 4','x[3072…4095]. The row has now been read from HBM exactly once.'],
 ['Warp reduce: 5 shuffles','Inside each warp, threads swap partial sums with __shfl_xor_sync at distances 16, 8, 4, 2, 1. After 5 steps, lane 0 holds the warp\'s total. No memory is touched.'],
 ['32 partials → shared memory','Lane 0 of each warp writes its total to __shared__ part[32] on the SM\'s SRAM. __syncthreads() waits for all 32 warps.'],
 ['Warp 0 finishes the sum → rstd','Warp 0 loads the 32 partials, shuffles again, and computes rstd = 1/√(mean + ε). It\'s written back to part[0] for everyone.'],
 ['Scale and write y, 4 passes','Each thread re-reads its 4 elements (now from L1/L2), multiplies by rstd and the weight, and writes y. Done: 8 KB in, 8 KB out.']];
function hierView(){
  const v=VW.hier;reset(v);
  // grid of blocks
  const gb=[];for(let r=0;r<6;r++)for(let c=0;c<8;c++){const sel=r===2&&c===5;const b=new THREE.Mesh(new THREE.BoxGeometry(0.42,0.18,0.42),mat(sel?'--copper':'--steel',sel?1:0.75));b.position.set(-6.2+c*0.52,0,-1.3+r*0.52);v.g.add(b);gb.push(b);}
  lab(v,'<b>Grid</b> · 1 block per token (48 of 2,048 shown)',new THREE.Vector3(-4.4,0.2,-1.7),'t',8);
  // link line
  const lg=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-3.6,0.1,-0.26),new THREE.Vector3(-1.9,0.1,-0.26)]);v.g.add(new THREE.Line(lg,new THREE.LineBasicMaterial({color:col('--copper')})));
  // block: 32 warps as bars (4 rows × 8)
  const warps=[];for(let i=0;i<32;i++){const w=new THREE.Mesh(new THREE.BoxGeometry(0.5,0.16,0.14),mat('--teal'));w.position.set(-1.4+(i%8)*0.6,0,-1.0+Math.floor(i/8)*0.3);v.g.add(w);warps.push(w);}
  const floor=new THREE.Mesh(new THREE.BoxGeometry(5.2,0.04,1.6),mat('--sunk'));floor.position.set(0.7,-0.1,-0.55);v.g.add(floor);
  lab(v,'<b>Block</b> = 1,024 threads = 32 warps (bars), on one SM',new THREE.Vector3(0.7,0.2,-1.3),'t',8);
  // one warp expanded into 32 threads
  const th=[];for(let i=0;i<32;i++){const t=new THREE.Mesh(new THREE.BoxGeometry(0.12,0.12,0.12),mat('--gold'));t.position.set(-1.4+i*0.165,0,0.9);v.g.add(t);th.push(t);}
  lab(v,'<b>Warp 0</b> = 32 threads in lockstep',new THREE.Vector3(1.2,0.1,1.05),'b',7);
  // shared memory
  const sram=[];for(let i=0;i<32;i++){const s=new THREE.Mesh(new THREE.BoxGeometry(0.13,0.06,0.2),mat('--violet',0.6));s.position.set(-1.4+i*0.165,-0.05,1.55);v.g.add(s);sram.push(s);}
  lab(v,'<b>__shared__ part[32]</b> · SRAM',new THREE.Vector3(3.9,0,1.55),'r',6);
  // the row of 4096 elements, 64 segments
  const seg=[];for(let i=0;i<64;i++){const s=new THREE.Mesh(new THREE.BoxGeometry(0.075,0.18,0.18),mat('--steel',0.55));s.position.set(-1.4+i*0.083,1.2,-0.55);v.g.add(s);seg.push(s);}
  lab(v,'<b>row x</b> · 4,096 bf16 in HBM',new THREE.Vector3(-1.5,1.2,-0.55),'l',7);
  const phl=lab(v,'',new THREE.Vector3(0.7,1.75,-0.55),'t',9);
  const C={cop:col('--copper'),gold:col('--gold'),teal:col('--teal'),steel:col('--steel'),vio:col('--violet'),good:col('--good')};
  function paint(ph,u){
    seg.forEach((s,i)=>{const pass=Math.floor(i/16);let c=C.steel,op=0.4;
      if(ph>=0&&ph<=3){if(pass<ph){c=C.cop;op=0.55;}if(pass===ph){c=C.cop;op=1;}}
      else if(ph>=4&&ph<=6){c=C.cop;op=0.55;}
      else if(ph===7){if(pass<Math.floor(u*4)){c=C.good;op=1;}else{c=C.cop;op=0.55;}}
      s.material.color.copy(c);s.material.opacity=op;s.material.transparent=op<1;});
    warps.forEach((w,i)=>{w.material.emissive=C.gold;w.material.emissiveIntensity=(ph<=3&&ph>=0)?0.25:ph===4?0.2+0.35*Math.abs(Math.sin(u*Math.PI*5)):ph===6&&i===0?0.6:0;});
    th.forEach((t,i)=>{t.position.y=ph===4?0.12*Math.sin(u*Math.PI*10+i*0.4):0;t.material.emissive=C.gold;t.material.emissiveIntensity=ph===4?0.4:0;});
    sram.forEach((s,i)=>{const on=ph===5?i<Math.ceil(u*32):ph===6||ph===7;s.material.opacity=on?1:0.35;s.material.transparent=!on;s.material.emissive=C.vio;s.material.emissiveIntensity=ph===6&&i===0?0.8:on?0.25:0;});
    relabel(phl,ph<0?'<b>Press ▶ to run one block</b>':`<b>${PHASES[ph][0]}</b>`);
  }
  paint(HI.phase,0);
  v.tick=dt=>{if(!HI.play)return;HI.t+=dt*(reduce?0.3:0.75);const ph=Math.floor(HI.t)%PHASES.length,u=HI.t%1;if(ph!==HI.phase){HI.phase=ph;$('#hiPhase').textContent=PHASES[ph][0];$('#hiText').innerHTML=`<b>${PHASES[ph][0]}.</b> ${PHASES[ph][1]}`;}paint(ph,u);};
  fit(v,0.15,0.86,0.74);
}

/* ================= 3 · tiling lab ================= */
const LAYERS=[{t:'qkv_proj · fused Q, K, V',N:6144,K:4096},{t:'o_proj',N:4096,K:4096},{t:'gate_up_proj · fused',N:24576,K:4096},{t:'down_proj',N:4096,K:12288},{t:'lm_head',N:151936,K:4096}];
const MS=[1,4,16,64,256,1024,4096,8192];
function tileCalc(){
  const D=dev(),L=LAYERS[+$('#tLayer').value],M=MS[+$('#tM').value],BM=+$('#tBM').value,BN=+$('#tBN').value,BK=+$('#tBK').value,ST=+$('#tST').value,SK=+$('#tSK').value;
  const tm=Math.ceil(M/BM),tn=Math.ceil(L.N/BN),progs=tm*tn*SK,smem=ST*(BM*BK+BK*BN)*2,fits=smem<=D.smem*1024;
  const perSM=fits?Math.max(1,Math.min(4,Math.floor(D.smSM*1024/smem))):0,slots=D.sm*perSM,waves=fits?progs/slots:Infinity,wc=Math.ceil(waves);
  const smUtil=!fits?0:wc===1?Math.min(1,progs/D.sm):progs/(wc*slots);
  const padM=M/(tm*BM),part=SK>1?SK*M*L.N*4*2:0,traffic=M*L.K*2*Math.ceil(tn/8)+L.K*L.N*2*Math.ceil(tm/8)+M*L.N*2+part,worst=M*L.K*2*tn+L.K*L.N*2*tm+M*L.N*2+part;
  const bwEff=D.bw*Math.min(1,Math.min(progs,D.sm)/(0.6*D.sm));
  const tMem=traffic/bwEff,tComp=fits?2*tm*BM*L.N*L.K/(D.fl*Math.max(smUtil,1e-3)):Infinity,t=Math.max(tMem,tComp);
  const ideal=Math.max((M*L.K+L.K*L.N+M*L.N)*2/D.bw,2*M*L.N*L.K/D.fl);
  return {D,L,M,BM,BN,BK,ST,SK,tm,tn,progs,smem,fits,perSM,slots,waves,wc,smUtil,padM,traffic,worst,tMem,tComp,t,ideal,intensity:2*M*L.N*L.K/traffic};
}
function tiling(){
  const T=tileCalc();$('#tMO').textContent=N(T.M);
  const issues=[];
  if(!T.fits)issues.push(['bad',`The tile needs <b>${(T.smem/1024).toFixed(0)} KB</b> of SRAM but one block can use at most ${T.D.smem} KB on this GPU. It won\'t launch (Triton: "out of resources: shared memory"). Lower the stages or BLOCK_K.`]);
  if(T.fits&&T.padM<0.5)issues.push(T.tMem>=T.tComp?['warn',`M = ${T.M} but BLOCK_M = ${T.BM}: <b>${((1-T.padM)*100).toFixed(0)}% of each tile\'s rows are padding</b>. Here it barely costs time, because the op is memory-bound and the tensor cores have capacity to spare. It still wastes SRAM and registers that could hold a wider BLOCK_N. A GEMV-style kernel (BLOCK_M = 1, section 6) avoids it entirely; 16 is the smallest tensor-core tile.`]:['bad',`M = ${T.M} but BLOCK_M = ${T.BM}: <b>${((1-T.padM)*100).toFixed(0)}% of each tile\'s rows are padding</b>, and this op is compute-bound, so that waste is real time. Use a smaller BLOCK_M.`]);
  if(T.fits&&T.progs<0.6*T.D.sm)issues.push(['warn',`Only <b>${N(T.progs)} programs for ${T.D.sm} SMs</b>. Too few SMs are issuing loads to saturate memory. Use a smaller BLOCK_N, or split-K.`]);
  if(T.fits&&T.wc>1&&T.wc<=4&&T.waves%1>0&&T.waves%1<0.5)issues.push(['warn',`Tail effect: ${T.waves.toFixed(2)} waves, so the last wave is only ${((T.waves%1)*100).toFixed(0)}% full and most SMs idle during it.`]);
  if(T.fits&&T.SK>1&&T.M>=256)issues.push(['warn','Split-K at large M adds fp32 partial-sum traffic for no gain. There are already plenty of programs.']);
  if(T.fits&&!issues.length)issues.push(['good','Reasonable config: the tile fits, every SM has work, and padding is small.']);
  const bound=T.tMem>=T.tComp?'memory':'compute';
  $('#tOut').innerHTML=`<div class="tablewrap"><table>
   <tr><th>${T.L.t}: M×N×K = ${N(T.M)}×${N(T.L.N)}×${N(T.L.K)}</th><th>value</th></tr>
   <tr><td>Programs = ⌈M/BM⌉ × ⌈N/BN⌉ × split-K</td><td class="num">${T.tm} × ${N(T.tn)} × ${T.SK} = ${N(T.progs)}</td></tr>
   <tr><td>SRAM per program (stages × tiles × 2 B)</td><td class="num">${(T.smem/1024).toFixed(1)} KB ${T.fits?'✓':'✗'} (limit ${T.D.smem} KB)</td></tr>
   <tr><td>Programs resident per SM · waves</td><td class="num">${T.fits?T.perSM+' · '+T.waves.toFixed(2):'n/a'}</td></tr>
   <tr><td>SM utilization · useful rows in M tiles</td><td class="num">${(T.smUtil*100).toFixed(0)}% · ${(T.padM*100).toFixed(T.padM<0.1?1:0)}%</td></tr>
   <tr><td>HBM traffic, grouped tile order (L2 reuse) · worst case</td><td class="num">${fB(T.traffic)} · ${fB(T.worst)}</td></tr>
   <tr><td>Arithmetic intensity</td><td class="num">${T.intensity.toFixed(1)} FLOP/B</td></tr>
   <tr><td>Estimated time (${bound}-bound) vs ideal</td><td class="num">${T.fits?fT(T.t):'—'} vs ${fT(T.ideal)}</td></tr></table></div>
   ${issues.map(([k,s])=>`<div class="because"><span class="pill ${k}">${k==='good'?'ok':k==='warn'?'tune':'problem'}</span> ${s}</div>`).join('')}
   <p class="note">Estimates: SRAM limits occupancy (capped at 4 programs per SM); memory saturates once about 60% of SMs are loading; tiles are visited in groups of 8 so re-reads mostly hit L2, as in Triton\'s matmul tutorial (worst case: every re-read goes to HBM). Real kernels also depend on registers and L2 hits, which is why autotune measures instead of guessing.</p>`;
  if(VW.tile)tileView(T);
}
function tileView(T){
  const v=VW.tile;reset(v);
  const dn=Math.min(T.tn,32),dm=Math.min(T.tm,16),gw=7.2,cw=gw/dn,rowsPer=T.tm/dm,colsPer=T.tn/dn;
  const cd=Math.min(4.2/dm,Math.max(dm===1?0.9:0.3,Math.min(0.6,cw*T.BM/T.BN*Math.max(1,rowsPer))));
  const waveCols=['--copper','--teal','--violet','--gold','--teal2'],cells=[];
  const lastFrac=(T.M-(T.tm-1)*T.BM)/T.BM;
  for(let r=0;r<dm;r++)for(let c=0;c<dn;c++){
    const pi=Math.floor(r*rowsPer)*T.tn+Math.floor(c*colsPer),wave=T.fits?Math.floor(pi*T.SK/Math.max(1,T.slots)):0;
    const isLast=r===dm-1,frac=isLast&&rowsPer<=1?lastFrac:1;
    const m=new THREE.Mesh(new THREE.BoxGeometry(cw*0.9,0.14,cd*0.9),mat(T.fits?waveCols[wave%5]:'--bad',frac<1?0.3:0.95));
    m.position.set(-gw/2+cw*(c+0.5),0,r*cd);v.g.add(m);cells.push({m,wave});
    if(frac<1){const u=new THREE.Mesh(new THREE.BoxGeometry(cw*0.9,0.16,Math.max(0.02,cd*0.9*frac)),mat(T.fits?waveCols[wave%5]:'--bad'));u.position.set(m.position.x,0.01,r*cd-cd*0.45+cd*0.45*frac);v.g.add(u);}
  }
  const depth=dm*cd;
  // A strip (M×K) on the left, B strip (K×N) behind
  const kd=1.6,A=new THREE.Mesh(new THREE.BoxGeometry(kd,0.08,depth),mat('--steel',0.5));A.position.set(-gw/2-kd/2-0.35,0,depth/2-cd/2);v.g.add(A);
  const B=new THREE.Mesh(new THREE.BoxGeometry(gw,0.08,kd),mat('--steel',0.5));B.position.set(0,0,-kd/2-0.35-cd/2);v.g.add(B);
  const steps=Math.ceil(T.L.K/T.SK/T.BK),kw=Math.max(0.04,kd/Math.min(steps,40));
  const ka=new THREE.Mesh(new THREE.BoxGeometry(kw,0.12,depth),mat('--gold'));ka.userData.noFit=true;v.g.add(ka);
  const kb=new THREE.Mesh(new THREE.BoxGeometry(gw,0.12,kw),mat('--gold'));kb.userData.noFit=true;v.g.add(kb);
  lab(v,`<b>C</b> = ${N(T.M)} × ${N(T.L.N)} · ${N(T.progs)} programs${dn<T.tn||dm<T.tm?' (each cell = several tiles)':''}`,new THREE.Vector3(0,0.15,depth+0.1),'b',9);
  lab(v,`<b>A</b> · ${N(T.M)}×${N(T.L.K)}`,new THREE.Vector3(-gw/2-kd/2-0.35,0.1,depth/2),'t',7);
  lab(v,`<b>B</b> (weights) · ${N(T.L.K)}×${N(T.L.N)}`,new THREE.Vector3(gw/2,0.1,-kd/2-0.35-cd/2),'r',7);
  lab(v,`gold = BLOCK_K chunk · ${steps} steps per program`,new THREE.Vector3(-gw/4,0.1,-kd-0.35-cd/2),'t',5,'shape');
  if(T.fits)lab(v,`${T.waves.toFixed(2)} waves of ${N(T.slots)} slots`,new THREE.Vector3(gw/2,0.1,depth/2),'r',6);
  let t=0;const nW=Math.max(1,T.wc);
  v.tick=dt=>{t+=reduce?0:dt;const cur=Math.floor(t/1.4)%nW,u=(t%1.4)/1.4;
    cells.forEach(o=>{o.m.position.y=T.fits&&o.wave===cur?0.18:0;});
    const kx=-gw/2-0.35-kd+kd*u;ka.position.set(kx,0.06,depth/2-cd/2);kb.position.set(0,0.06,-0.35-cd/2-kd+kd*u);};
  fit(v,0.2,0.72,0.82);
}

/* ================= annotated code ================= */
const ANN={
rms:{lang:'py',lines:[
['@triton.jit','Compiles this Python function into a GPU kernel the first time it\'s called with a given set of argument types and constexpr values. You write code for one <b>program</b> (one block); Triton chooses the threads and warps.'],
['def rmsnorm_kernel(X, W, Y, stride, N, eps, BLOCK: tl.constexpr):','X, W, Y arrive as raw pointers. <code>BLOCK</code> is a compile-time constant: Triton builds a specialised kernel for each value, and it must be a power of two.'],
['    row = tl.program_id(0)                  # one program per token','The launch grid is <code>(num_tokens,)</code>, so program i normalises token i. A 2,048-token prefill is 2,048 programs dealt out across the SMs. A single decode token is just one, which is why tiny ops like this are launch-bound in decode.'],
['    cols = tl.arange(0, BLOCK)','A vector of offsets 0 … BLOCK−1. Every operation on it runs across all lanes at once. This vector is what Triton gives you instead of per-thread code.'],
['    mask = cols < N','BLOCK is the next power of 2 ≥ N. For Qwen3-8B, N = 4,096, so no lane is masked. For Qwen3-32B, N = 5,120, so BLOCK = 8,192 and 3,072 lanes are switched off.'],
['    x = tl.load(X + row * stride + cols, mask=mask, other=0.).to(tl.float32)','The only HBM read of x: 4,096 × 2 bytes = 8 KB into registers. Upcasting to fp32 keeps the sum of squares from losing precision.'],
['    var = tl.sum(x * x, axis=0) / N','A reduction across the row. The compiler emits the warp shuffles and shared memory for you. In CUDA you\'d write them by hand (see below).'],
['    rstd = 1 / tl.sqrt(var + eps)','One scalar per row. eps = 1e-6 for Qwen3 (<code>rms_norm_eps</code> in config.json).'],
['    w = tl.load(W + cols, mask=mask)','The weight vector γ, 8 KB, shared by every row. After the first program reads it, the rest find it in L2 cache.'],
['    y = x * rstd * w','Still in registers. Nothing has gone back to HBM yet. This is where fusion pays: eager PyTorch writes x², the mean and each product to memory as separate kernels.'],
['    tl.store(Y + row * stride + cols, y.to(Y.dtype.element_ty), mask=mask)','The only HBM write: 8 KB. Total traffic is about 16 KB per token, the minimum possible, so the benchmark target is 16 KB × tokens ÷ bandwidth.'],
[''],
['def rmsnorm(x, w, eps=1e-6):','The Python wrapper: allocate the output, choose the grid, launch.'],
['    x2 = x.reshape(-1, x.shape[-1]); y = torch.empty_like(x2)','Flatten [batch, seq, hidden] to [tokens, hidden]. The output comes from PyTorch\'s caching allocator, so there\'s no cudaMalloc per call.'],
['    N = x2.shape[1]; BLOCK = triton.next_power_of_2(N)'],
['    rmsnorm_kernel[(x2.shape[0],)](x2, w, y, x2.stride(0), N, eps, BLOCK=BLOCK)','<code>kernel[grid](args)</code>: grid = number of programs. Tensors are passed as pointers. The first call compiles (about a second); later calls take a few µs to launch.'],
['    return y.view_as(x)']]},
cuda:{lang:'c',lines:[
['#include <torch/extension.h>'],['#include <ATen/cuda/CUDAContext.h>'],['#include <cuda_bf16.h>'],[''],
['__global__ void rmsnorm_kernel(const __nv_bfloat16* x, const __nv_bfloat16* w,','<code>__global__</code> marks a kernel: called from the CPU, runs on the GPU. Unlike Triton, this code is written for <b>one thread</b>.'],
['                               __nv_bfloat16* y, int N, float eps) {'],
['  const __nv_bfloat16* xr = x + (size_t)blockIdx.x * N;   // this block\'s row','<code>blockIdx.x</code> = which block, the same as <code>tl.program_id(0)</code>. The size_t cast avoids int overflow past 2³¹ elements.'],
['  float ss = 0.f;'],
['  for (int i = threadIdx.x; i < N; i += blockDim.x) {     // strided: coalesced','Thread t handles t, t+1024, t+2048, t+3072. In each pass the 32 threads of a warp read 32 consecutive bf16s, 64 bytes, as one memory transaction. Giving each thread a contiguous chunk instead would scatter the warp\'s reads.'],
['    float v = __bfloat162float(xr[i]); ss += v * v;'],['  }'],
['  for (int o = 16; o > 0; o >>= 1)                        // warp reduce: 5 shuffles','Each step adds the value from the lane at distance o (16, 8, 4, 2, 1) using register-to-register shuffles. After 5 steps every lane holds the warp total. This is what <code>tl.sum</code> generated for you.'],
['    ss += __shfl_xor_sync(0xffffffff, ss, o);'],
['  __shared__ float part[32];','32 floats in the SM\'s shared memory (SRAM), visible to every thread in this block. 1,024 threads = 32 warps, so 32 slots.'],
['  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;'],
['  if (lane == 0) part[warp] = ss;                         // one partial per warp'],
['  __syncthreads();','A barrier: no thread continues until all 1,024 arrive. Leave it out and warp 0 may read partials that haven\'t been written yet, a race that only shows up sometimes.'],
['  if (warp == 0) {                                        // warp 0 adds the partials'],
['    ss = lane < blockDim.x / 32 ? part[lane] : 0.f;'],
['    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);'],
['    if (lane == 0) part[0] = rsqrtf(ss / N + eps);','<code>rsqrtf</code> is a single hardware instruction. The result goes back into shared memory so every thread can read it.'],
['  }'],['  __syncthreads();'],['  float r = part[0];'],
['  for (int i = threadIdx.x; i < N; i += blockDim.x)','The second pass re-reads x. It\'s 8 KB per block and was just read, so it comes from L1/L2 rather than HBM.'],
['    y[(size_t)blockIdx.x * N + i] ='],
['        __float2bfloat16(__bfloat162float(xr[i]) * r * __bfloat162float(w[i]));'],['}'],[''],
['torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, double eps) {','The host function PyTorch calls: check inputs, allocate the output, launch.'],
['  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16);','Validate here, not in the kernel. A non-contiguous input would silently compute garbage.'],
['  auto y = torch::empty_like(x);'],
['  int N = x.size(-1), rows = x.numel() / N;'],
['  rmsnorm_kernel<<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(','<code>&lt;&lt;&lt;blocks, threads, shared bytes, stream&gt;&gt;&gt;</code>: one block per row, 1,024 threads each. Launch on PyTorch\'s current stream so the kernel is ordered with other ops and CUDA graphs can capture it.'],
['      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),'],
['      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),'],
['      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), N, (float)eps);'],
['  return y;'],['}'],
['TORCH_LIBRARY(my_kernels, m) { m.def("rmsnorm(Tensor x, Tensor w, float eps) -> Tensor"); }','Registers the op\'s schema, so it becomes <code>torch.ops.my_kernels.rmsnorm</code> and works with torch.compile.'],
['TORCH_LIBRARY_IMPL(my_kernels, CUDA, m) { m.impl("rmsnorm", &rmsnorm); }','Binds the CUDA implementation. You could add a CPU implementation under the same name.']]},
w4:{lang:'py',lines:[
['@triton.jit'],
['def w4a16_gemv_kernel(X, Wq, S, Y, N, K,','X: [tokens, K] fp16 activations. Wq: [K/8, N] int32, holding 8 weights per word along K. S: [K/128, N] fp16 scales. Y: [tokens, N].'],
['                      GROUP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):'],
['    pid_n = tl.program_id(0)                       # which block of BN outputs','Grid axis 0 splits the N outputs into blocks of BN = 64. For gate_up (N = 24,576) that\'s 384 programs, about 3 per SM on a 4090, which is enough to saturate memory.'],
['    row = tl.program_id(1)                         # which token (decode: 1–4)','Grid axis 1 = token. For more than ~4 tokens the op switches to dequantize + cuBLAS, because it becomes compute-bound (section 10a).'],
['    n = pid_n * BN + tl.arange(0, BN)              # output columns this program owns'],
['    acc = tl.zeros([BN], dtype=tl.float32)         # accumulate in fp32 registers','64 running sums held in registers for the whole loop. fp32, because summing 4,096 products in fp16 loses about 3 significant digits.'],
['    for k0 in range(0, K, BK):                     # walk down K, BK rows at a time','K / BK = 32 iterations for K = 4,096. Each iteration handles a BK × BN = 128 × 64 tile of weights.'],
['        k = k0 + tl.arange(0, BK)'],
['        x = tl.load(X + row * K + k).to(tl.float32)                 # [BK] activations','128 activations. They are reused by all 384 programs, so after the first read they come from L2.'],
['        packed = tl.load(Wq + (k // 8)[:, None] * N + n[None, :])   # [BK, BN] int32','The HBM read that matters: packed 4-bit weights. Row k lives in word k // 8. Consecutive n are consecutive addresses, so loads coalesce. Simplification: each word is fetched 8 times (once per k in the group of 8); the 7 repeats hit L1. A tuned kernel loads each word once.'],
['        q = (packed >> ((k % 8) * 4)[:, None]) & 0xF                # pick this row\'s nibble','Shift right by 4 × (k mod 8), then keep the low 4 bits. Integer ops in registers, nearly free compared with the memory read. Try it in the nibble widget above.'],
['        s = tl.load(S + (k // GROUP)[:, None] * N + n[None, :]).to(tl.float32)','One scale per 128 rows per column. That\'s 2 bytes per 128 weights, or 0.125 extra bits per weight.'],
['        w = (q.to(tl.float32) - 8.0) * s                            # dequantize in registers','Stored 0…15 → signed −8…7 → × scale. The bf16 weight exists only in registers, never in memory. That is the whole trick behind 4-bit speed.'],
['        acc += tl.sum(w * x[:, None], axis=0)                       # partial dot products','Multiply each weight by its activation and sum down the BK rows. The result is 64 partial outputs per iteration.'],
['    tl.store(Y + row * N + n, acc.to(Y.dtype.element_ty))','64 outputs written once at the end. Output traffic is negligible: 128 bytes per program.'],
[''],
['def w4a16_gemv(x, qw, s, group=128, BN=64, BK=128):'],
['    M, K = x.shape; N = qw.shape[1]'],
['    assert K % BK == 0 and N % BN == 0 and x.is_contiguous()','The spec from step 2, enforced. Qwen3-8B\'s K (4,096 and 12,288) and N (6,144, 4,096, 24,576) all divide evenly. Another model might not, and then you\'d add masks.'],
['    y = torch.empty(M, N, device=x.device, dtype=x.dtype)'],
['    w4a16_gemv_kernel[(N // BN, M)](x, qw, s, y, N, K, GROUP=group, BN=BN, BK=BK)'],
['    return y']]}};
function annot(el){
  const A=ANN[el.dataset.ann];
  const fmt=s=>{let h=esc(s);const re=A.lang==='py'?/(\s#\s.*)$/:/(\s\/\/.*)$/;return h.replace(re,'<span class="cm">$1</span>');};
  el.innerHTML=`<div class="code">${A.lines.map((l,i)=>`<span class="ln${l[1]?' has':''}" data-i="${i}"><i>${i+1}</i>${fmt(l[0])||' '}</span>`).join('')}</div><div class="note-p"><span class="k">line —</span><div class="nt">Click a line whose number is copper to see what it does on the GPU.</div><div class="btnrow"><button type="button" class="pv">◀ prev note</button><button type="button" class="nx">next note ▶</button></div></div>`;
  const lines=[...el.querySelectorAll('.ln')],noted=A.lines.map((l,i)=>l[1]?i:-1).filter(i=>i>=0);let cur=-1;
  const show=i=>{cur=i;lines.forEach(l=>l.classList.toggle('on',+l.dataset.i===i));el.querySelector('.k').textContent='line '+(i+1);el.querySelector('.nt').innerHTML=A.lines[i][1];};
  lines.forEach(l=>{if(A.lines[+l.dataset.i][1])l.onclick=()=>show(+l.dataset.i);});
  el.querySelector('.nx').onclick=()=>{const n=noted.find(i=>i>cur);show(n==null?noted[0]:n);};
  el.querySelector('.pv').onclick=()=>{const p=noted.slice().reverse().find(i=>i<cur);show(p==null?noted[noted.length-1]:p);};
}

/* ================= 4–6 · kernel numbers ================= */
function rms(){
  const D=dev(),T=+$('#rmT').value,b=2*T*4096*2+8192,ideal=b/D.bw,good=ideal/0.85,eager=10*2*T*4096*2/D.bw+9*LAUNCH;
  $('#rxRms').innerHTML=rx([['Bytes: read x + write y (+ 8 KB weight)',fB(b)],[`Ideal at ${N(D.bw/1e9)} GB/s`,fT(ideal)],['A good Triton kernel (≈85% of peak) + launch',fT(good+LAUNCH),1],['Eager PyTorch reference: ~9 kernels, ~40 B per element',fT(eager)],['Fused speedup over eager',(eager/(good+LAUNCH)).toFixed(1)+'×']])+`<div class="dim" style="white-space:normal">${T===1?'At 1 token the launch (≈4 µs) is ~'+Math.round(LAUNCH/(good+LAUNCH)*100)+'% of the time, so CUDA graphs matter more than the kernel.':'At this size the kernel is bandwidth-bound: judge it in GB/s.'}</div>`;
}
const FUT=[1,8,64,256,2048,8192,32768];
function fusion(){
  const D=dev(),T=FUT[+$('#fuT').value],Nn=12288,g=$('#fuG').checked;$('#fuTO').textContent=N(T);
  const bu=T*Nn*10,bf=T*Nn*6,tu=bu/D.bw+(g?0:2*LAUNCH),tf=bf/D.bw+(g?0:LAUNCH);
  $('#rxFu').innerHTML=rx([['Unfused: silu writes a temp, mul reads it back','10 B/elem · 2 launches'],['Unfused bytes · time',`${fB(bu)} · ${fT(tu)}`],['Fused: read gate + up, write y','6 B/elem · 1 launch'],['Fused bytes · time',`${fB(bf)} · ${fT(tf)}`],['Speedup',(tu/tf).toFixed(2)+'×',1],['× 36 layers per forward pass',`saves ${fT((tu-tf)*36)}`]]);
  const launchShare=g?0:LAUNCH/tf;
  $('#fuWhy').innerHTML=launchShare>0.5?`<b>Launch-bound:</b> at ${N(T)} token${T>1?'s':''} the data is tiny (${fB(bf)}), so ${Math.round(launchShare*100)}% of the fused kernel\'s time is the launch. Fusion helps by halving launches. CUDA graphs (tick the box) remove launch cost altogether, which is why vLLM captures decode in graphs.`:`<b>Bandwidth-bound:</b> the speedup approaches 10/6 = 1.67×, the byte ratio. This is the regime of prefill and large batches.`;
}
let nibJ=5;const NIB_WORD=0x9C3A5F17|0,NIB_SCALE=0.0123;
function nibble(){
  $('#nib').innerHTML=[...Array(8)].map((_,j)=>`<button type="button" class="${j===nibJ?'on':''}" data-j="${j}">row ${j}</button>`).join('');
  $$('#nib button').forEach(b=>b.onclick=()=>{nibJ=+b.dataset.j;nibble();});
  const bits=[];for(let b=31;b>=0;b--){const v=(NIB_WORD>>>b)&1,on=Math.floor(b/4)===nibJ;bits.push(`<span class="${on?'on':''}${b%4===3&&b!==31?' sep':''}">${v}</span>`);}
  $('#bits').innerHTML=bits.join('');
  const q=(NIB_WORD>>(nibJ*4))&0xF;
  $('#rxNib').innerHTML=rx([['word (int32)','0x'+(NIB_WORD>>>0).toString(16).toUpperCase()],[`shift = (k % 8) × 4`,`${nibJ} × 4 = ${nibJ*4} bits`],['(word >> shift) & 0xF',`${q} (0x${q.toString(16).toUpperCase()})`],['q − 8 (signed)',String(q-8)],[`× group scale ${NIB_SCALE}`,((q-8)*NIB_SCALE).toFixed(4),1]]);
}
function w4(){
  const D=dev(),K=4096,Nn=24576,bf=K*Nn*2,q=K*Nn/2,sc=K/128*Nn*2,w=q+sc;
  $('#rxW4').innerHTML=rx([['BF16 weights: 4,096 × 24,576 × 2 B',fB(bf)],['4-bit packed: × 0.5 B',fB(q)],['Scales: 32 groups × 24,576 × 2 B',fB(sc)],['4-bit total',fB(w)+` (${(bf/w).toFixed(2)}× smaller)`,1],[`Ideal time on ${D.name.split(' ·')[0]}: BF16 → 4-bit`,`${fT(bf/D.bw)} → ${fT(w/D.bw)}`],['× 36 layers (gate_up only, per decode step)',`${fT(bf/D.bw*36)} → ${fT(w/D.bw*36)}`]]);
  $('#w4Why').innerHTML=`Decode is memory-bound (part 5), so time ≈ bytes ÷ bandwidth, and reading <b>${(bf/w).toFixed(2)}× fewer bytes</b> makes it up to that much faster. The cost is a few integer ops per weight in registers (shift, mask, subtract, multiply), which the idle tensor cores and ALUs absorb for free. In prefill the matmul is compute-bound and the bytes no longer matter, so the op dequantizes once and calls cuBLAS.`;
}
const KB={rms:{b:2*8192*4096*2,f:0,lbl:'RMSNorm'},silu:{b:2048*12288*6,f:0,lbl:'SiLU×mul'},w4:{b:4096*24576/2+32*24576*2+4096*2+24576*2,f:0,lbl:'W4A16 GEMV'},gemm:{b:(2048*4096+4096*24576+2048*24576)*2,f:2*2048*24576*4096,lbl:'BF16 GEMM'}};
function kbench(setDefault){
  const D=dev(),k=KB[$('#kbK').value];
  if(setDefault){const ideal=Math.max(k.b/D.bw,k.f/D.fl);$('#kbT').value=Math.max(0.5,Math.round(ideal/0.7*1e6*2)/2);}
  const t=Math.max(0.5,+$('#kbT').value)*1e-6,gbs=k.b/t,tf=k.f/t,memB=k.f===0||k.b/D.bw>k.f/D.fl;
  const pct=memB?gbs/D.bw:tf/D.fl,lim=memB?'memory':'compute';
  $('#rxKb').innerHTML=rx([['Bytes moved (minimum)',fB(k.b)],k.f?['FLOPs',(k.f/1e9).toFixed(1)+' GFLOP']:['FLOPs','negligible'],['Bound by',lim],['Achieved',memB?`${N(gbs/1e9)} GB/s of ${N(D.bw/1e9)}`:`${(tf/1e12).toFixed(1)} TFLOP/s of ${(D.fl/1e12).toFixed(0)}`],['Share of peak',(pct*100).toFixed(0)+'%',1],['Ideal time',fT(Math.max(k.b/D.bw,k.f/D.fl))]]);
  const v=pct>1.02?['bad','Faster than the hardware allows. Check the byte count (did L2 cache serve repeated reads?), that the timer waited for the GPU (do_bench does), and that the output is correct.']:pct>=0.8?['good','Done. This kernel is near the roofline; the next speedup has to come from somewhere else in the profile.']:pct>=0.5?['warn','Tune: autotune tile sizes, vectorize loads, and use split-K if there are fewer programs than SMs.']:['bad',`Structural problem. Check coalescing (memory layout), the program count vs ${D.sm} SMs, register spills in ncu, or whether the input is so small you\'re timing the launch.`];
  $('#kbWhy').innerHTML=`<span class="pill ${v[0]}">${(pct*100).toFixed(0)}%</span> ${v[1]}`;
}

/* ================= 8 · vLLM map ================= */
const MAP=[
 {id:'api',b:0,x:-2.6,t:'API server',role:'Speaks HTTP: the OpenAI-compatible endpoints, request validation, chat templates, streaming responses back as server-sent events.',files:'vllm/entrypoints/openai/api_server.py · serving_chat.py',key:'FastAPI app, OpenAIServingChat',chg:'Add an endpoint or a request field; change the auth or logging.',q:'Is the API process a bottleneck? At high request rates, tokenization and detokenization can be: that is why they live in a separate process from the GPU loop.'},
 {id:'llm',b:0,x:2.6,t:'AsyncLLM',role:'The front-end engine. Its Processor tokenizes and validates requests into EngineCoreRequests; its OutputProcessor detokenizes token ids incrementally and routes them to the right stream.',files:'vllm/v1/engine/async_llm.py · processor.py · output_processor.py',key:'AsyncLLM, Processor, OutputProcessor',chg:'Usually nothing. This is the glue between HTTP and the engine core.',q:'How does a request get from Python asyncio to the GPU loop without blocking it? Over a ZeroMQ socket to another process.'},
 {id:'core',b:1,x:-3.6,t:'EngineCore loop',role:'A busy loop in its own process: pull new requests, call the scheduler, send the plan to the executor, collect sampled tokens, update requests, repeat. It runs as fast as the GPU allows.',files:'vllm/v1/engine/core.py · core_client.py',key:'EngineCore.step(), EngineCoreProc',chg:'Rarely. Overlapping CPU scheduling with GPU execution is done here.',q:'Why a separate process? Python\'s GIL: any CPU work in the API process (JSON, tokenizing) would otherwise stall GPU launches.'},
 {id:'sched',b:1,x:0,t:'Scheduler',role:'Decides who runs this step and how many tokens each: running decodes first, then waiting prefills within the token budget, chunking long prompts, preempting when KV blocks run out. The trace in section 9 is a small version of this.',files:'vllm/v1/core/sched/scheduler.py',key:'Scheduler.schedule() → SchedulerOutput',chg:'Policy (FCFS or priority), budgets. Mostly through flags: --max-num-seqs, --max-num-batched-tokens, --scheduling-policy.',q:'What does the GPU run next step? Every request gets "N tokens this step", which unifies prefill, chunked prefill and decode.'},
 {id:'kvm',b:1,x:3.6,t:'KV cache manager',role:'Owns the block pool: allocates and frees 16-token blocks, keeps block tables per request, and hashes full blocks so a repeated prefix reuses cached blocks (prefix caching).',files:'vllm/v1/core/kv_cache_manager.py · block_pool.py',key:'KVCacheManager.allocate_slots(), BlockPool',chg:'Block size (--block-size), prefix caching on or off, eviction behaviour.',q:'How many requests fit? (GPU memory × utilization − weights − activations) ÷ KV bytes per block. That number is printed at startup as the "KV cache size".'},
 {id:'worker',b:2,x:-4.4,t:'Worker (one per GPU)',role:'One process per GPU. Initializes the device and distributed groups (tensor/pipeline parallel), loads its shard of the weights, profiles memory to size the KV cache, and runs the model runner.',files:'vllm/v1/worker/gpu_worker.py · vllm/v1/executor/',key:'Worker, MultiprocExecutor',chg:'Device support or distributed setup. New hardware backends plug in here as platform plugins.',q:'With tensor parallel = 2, each worker holds half of every weight matrix and they all-reduce after attention and the FFN: two all-reduces per layer.'},
 {id:'runner',b:2,x:-1.5,t:'GPU model runner',role:'Turns the scheduler\'s plan into tensors: a persistent batch, flat input_ids and positions, slot_mapping (where each new K/V goes) and block tables (attention metadata). It captures and replays CUDA graphs for decode batch sizes.',files:'vllm/v1/worker/gpu_model_runner.py',key:'GPUModelRunner.execute_model(), capture_model()',chg:'Input preparation, graph capture sizes (--cuda-graph-sizes in recent versions), speculative-decoding hooks.',q:'Why flatten all requests into one long token list? So prefill and decode tokens run in the same kernels, with no padding.'},
 {id:'model',b:2,x:1.5,t:'Model · qwen3.py',role:'The architecture as PyTorch modules built from vLLM layers: QKVParallelLinear, RowParallelLinear, MergedColumnParallelLinear (gate_up), RMSNorm, get_rope, Attention, SiluAndMul. load_weights maps checkpoint names onto the fused parameters.',files:'vllm/model_executor/models/qwen3.py · registry.py · layers/',key:'Qwen3ForCausalLM, Qwen3DecoderLayer, load_weights()',chg:'Add an architecture, or modify one (an extra norm, a different MLP), by writing or subclassing a model file.',q:'Which quantization kernel runs? Every linear layer asks the QuantizationConfig for its method, so Qwen3 with GPTQ automatically gets Marlin GEMMs.'},
 {id:'attn',b:2,x:4.4,t:'Attention backend',role:'The Attention layer writes the new K/V into the paged cache (the reshape_and_cache kernel) and calls a backend that reads through block tables: FlashAttention, FlashInfer, FlashMLA for MLA, or Triton.',files:'vllm/attention/layer.py · vllm/v1/attention/backends/',key:'Attention, FlashAttentionBackend, metadata builders',chg:'Pick the backend with VLLM_ATTENTION_BACKEND. A new attention variant needs a backend here.',q:'Why separate metadata builders? Each backend wants the batch described differently, and building it once per step on the CPU is cheaper than per layer.'},
 {id:'kern',b:3,x:0,t:'Kernels',role:'Where the time goes. csrc/ has hand-written CUDA (layernorm, activation, cache ops, Marlin, MoE, all-reduce); Triton kernels live in Python (fused MoE, some attention); FlashAttention ships as vllm-flash-attn; GEMMs go to cuBLAS or CUTLASS.',files:'csrc/*.cu · csrc/quantization/ · vllm/model_executor/layers/fused_moe/',key:'torch.ops._C.*, Triton @jit functions',chg:'Add a kernel: a CUDA op in csrc (rebuild vLLM) or a Triton kernel in Python (no build). Plugins can ship their own.',q:'Is this kernel on the roofline? Profile with nsys and ncu (sections 7 and 1).'},
 {id:'samp',b:2,x:6.8,t:'Sampler',role:'Takes the last position\'s logits after lm_head, applies penalties, temperature, top-k and top-p, and picks token ids on the GPU. Also used to verify draft tokens in speculative decoding.',files:'vllm/v1/sample/sampler.py · logits_processor/',key:'Sampler.forward()',chg:'Custom logits processors (e.g. constrained decoding), sampling defaults.',q:'Why only the last position? lm_head is a 4,096 × 151,936 matrix; running it for every prompt token would waste about 1.2 GFLOP per token.'}];
const BY=[6.2,3.3,0.4,-1.9],BANDS=[{t:'Process 1 · API server',y:6.2,c:'--violet'},{t:'Process 2 · EngineCore',y:3.3,c:'--gold'},{t:'Processes 3+ · one per GPU',y:0.4,c:'--teal'},{t:'GPU',y:-1.9,c:'--steel'}];
const LIFE=[
 {n:['api'],t:'Request arrives',x:'<code>POST /v1/chat/completions</code> reaches the API server. FastAPI validates it and applies Qwen3\'s chat template to the messages.'},
 {n:['llm'],t:'Tokenize',x:'The Processor tokenizes the prompt (for example 1,200 tokens) and builds an EngineCoreRequest with token ids and sampling parameters. It goes over a ZeroMQ socket to the engine-core process, so tokenization never blocks the GPU loop.'},
 {n:['core'],t:'Queue',x:'EngineCore\'s busy loop picks the request off its input queue and adds it to the scheduler\'s waiting queue.'},
 {n:['sched','kvm'],t:'Schedule + allocate',x:'Next step, the scheduler checks the token budget and asks the KV cache manager for 75 blocks (1,200 ÷ 16). If the first 512 tokens match a cached prefix, 32 of those blocks are reused and only 688 tokens need prefill.'},
 {n:['worker'],t:'Broadcast the plan',x:'The SchedulerOutput (request ids, tokens per request, new block ids) goes to every worker. With tensor parallel = 2, both GPUs receive the same plan and each runs half of every matrix.'},
 {n:['runner'],t:'Build tensors',x:'The model runner updates its persistent batch and builds flat <code>input_ids</code>, <code>positions</code>, <code>slot_mapping</code> and block tables. A decode-only step replays a captured CUDA graph instead of launching kernels one by one.'},
 {n:['model'],t:'Forward pass',x:'Qwen3ForCausalLM runs 36 layers: fused residual + RMSNorm → QKV GEMM → QK-norm + RoPE → attention → o_proj → RMSNorm → gate_up GEMM → SiluAndMul → down_proj.'},
 {n:['attn','kern'],t:'Attention + kernels',x:'Attention writes the new K/V into the paged cache and runs FlashAttention through the block table. Every step above is a kernel: cuBLAS or Marlin GEMMs, vLLM\'s CUDA ops, Triton fused ops. About 400–600 per forward pass.'},
 {n:['samp'],t:'Sample',x:'Only the last position of each request goes through lm_head. The sampler applies temperature and top-p and picks a token id on the GPU.'},
 {n:['core'],t:'Update',x:'Token ids return to EngineCore: append to each request, check stop strings and max_tokens, free the blocks of finished requests. The next step starts immediately.'},
 {n:['llm','api'],t:'Stream back',x:'The OutputProcessor detokenizes incrementally (some tokens are half a character), and the API server streams the text to the client. Then it repeats from "Schedule" for every new token.'}];
const MP={sel:'sched',life:-1,play:false,t:0,ball:null,from:null,to:null};
function nodePos(id){const n=MAP.find(m=>m.id===id);return new THREE.Vector3(n.x,BY[n.b]+0.45,0);}
function mapView(){
  const v=VW.map;reset(v);MP.meshes={};
  BANDS.forEach((b,i)=>{const w=i===3?16:i===2?15.6:9.6,s=new THREE.Mesh(new THREE.BoxGeometry(w,0.08,2.4),mat(b.c,0.22));s.position.set(i===2?1.2:0,b.y-0.1,0);s.userData.noFit=i<3?false:false;v.g.add(s);
    lab(v,`<b>${b.t}</b>`,new THREE.Vector3((i===2?1.2:0)-w/2,b.y,1.2),'r',5);});
  MAP.forEach(n=>{const w=n.id==='kern'?12:2.2,m=new THREE.Mesh(new THREE.BoxGeometry(w,0.7,1.2),mat(n.id==='kern'?'--steel':BANDS[n.b].c,0.95));m.position.set(n.x,BY[n.b]+0.3,0);m.userData.pick=n.id;v.g.add(m);v.picks.push(m);MP.meshes[n.id]=m;
    const low=n.id==='runner'||n.id==='attn';lab(v,`<b>${n.t}</b>`,new THREE.Vector3(n.x,BY[n.b]+(low?-0.05:0.7),0.6),low?'b':'t',8);});
  // connections
  const pairs=[['api','llm'],['llm','core'],['core','sched'],['sched','kvm'],['core','worker'],['worker','runner'],['runner','model'],['model','attn'],['attn','samp'],['model','kern'],['attn','kern']];
  pairs.forEach(([a,b])=>{const g=new THREE.BufferGeometry().setFromPoints([nodePos(a),nodePos(b)]);v.g.add(new THREE.Line(g,new THREE.LineBasicMaterial({color:col('--muted'),transparent:true,opacity:0.55})));});
  lab(v,'ZeroMQ',nodePos('llm').clone().lerp(nodePos('core'),0.5),'r',3,'shape');
  lab(v,'SchedulerOutput',nodePos('core').clone().lerp(nodePos('worker'),0.5),'l',3,'shape');
  const ball=new THREE.Mesh(new THREE.SphereGeometry(0.22,18,12),new THREE.MeshBasicMaterial({color:col('--gold')}));ball.userData.noFit=true;ball.visible=false;v.g.add(ball);MP.ball=ball;
  v.tick=dt=>{
    if(MP.play){MP.t+=dt;if(MP.t>2.2){MP.t=0;lifeGo((MP.life+1)%LIFE.length);}}
    if(MP.to){MP.u=Math.min(1,(MP.u||0)+dt*1.6);const p=MP.from.clone().lerp(MP.to,MP.u);p.y+=Math.sin(MP.u*Math.PI)*0.8;ball.position.copy(p);ball.visible=true;}
    const hot=MP.life>=0?LIFE[MP.life].n:[MP.sel];
    Object.entries(MP.meshes).forEach(([id,m])=>{m.material.emissive=col('--gold');m.material.emissiveIntensity=hot.includes(id)?0.35:0;});};
  fit(v,0.12,1.05,0.88);
}
function mapPanel(id){MP.sel=id;const n=MAP.find(m=>m.id===id);
  $('#mapPanel').innerHTML=`<span class="eyebrow">Component</span><h3>${n.t}</h3><p>${n.role}</p><h4>Files</h4><p><code>${n.files}</code></p><h4>Key pieces</h4><p>${n.key}</p><h4>What you\'d change here</h4><p>${n.chg}</p><h4>Question it answers</h4><p>${n.q}</p>`;}
function lifeGo(i){
  const prev=MP.life;MP.life=i;const L=LIFE[i];
  $$('#lcList button').forEach((b,j)=>b.classList.toggle('on',j===i));
  $('#lcN').textContent=`step ${i+1} of ${LIFE.length}`;
  $('#lcText').innerHTML=`<b>${i+1} · ${L.t}.</b> ${L.x}`;
  if(MP.ball){MP.from=prev>=0?nodePos(LIFE[prev].n[0]):nodePos(L.n[0]).add(new THREE.Vector3(-2,1.5,0));MP.to=nodePos(L.n[0]);MP.u=0;}
  mapPanel(L.n[0]);
}

/* ================= 9 · scheduler trace ================= */
const TR=[
 ['def schedule(self):'],
 ['    batch, budget = [], self.max_tokens','s1'],
 ['    for seq in list(self.running):          # ① decode: 1 new token each','s2'],
 ['        if seq not in self.running: continue  # preempted this step','s3'],
 ['        if seq.n_cached % BS == 0:          # next token opens a new block','s4'],
 ['            if not self.free:               # out of KV memory','s5'],
 ['                victim = self.running.pop() # preempt the newest request','s6'],
 ['                self.free += victim.blocks; victim.reset()','s7'],
 ['                self.waiting.appendleft(victim)  # recompute it later','s8'],
 ['                if victim is seq: continue','s9'],
 ['            seq.blocks.append(self.free.pop())','s10'],
 ['        batch.append((seq, 1)); budget -= 1','s11'],
 ['    while self.waiting and budget > 0:      # ② admit new prefills','s12'],
 ['        seq = self.waiting[0]','s13'],
 ['        need = ceil(seq.n_prefill / BS)     # prompt + tokens so far','s14'],
 ['        if need > len(self.free) or seq.n_prefill > budget: break','s15'],
 ['        self.waiting.popleft()','s16'],
 ['        seq.blocks = [self.free.pop() for _ in range(need)]','s17'],
 ['        self.running.append(seq); batch.append((seq, seq.n_prefill))','s18'],
 ['        budget -= seq.n_prefill','s19'],
 ['    return batch','s20'],
 [''],
 ['def step(self):'],
 ['    batch = self.schedule()','p1'],
 ['    tokens = self.model.forward(batch)      # ONE GPU pass for everyone','p2'],
 ['    for (seq, n), tok in zip(batch, tokens):','p3'],
 ['        seq.n_cached += n; seq.out.append(tok)','p4'],
 ['        if seq.finished():                  # EOS or max_tokens','p5'],
 ['            self.running.remove(seq)','p6'],
 ['            self.free += seq.blocks         # blocks back to the pool','p7']];
const BS=16,NBLK=14,MAXTOK=96;
const SEQ0=[['A',40,26],['B',22,10],['C',35,34],['D',18,14],['E',50,20],['F',28,30]];
const SCOL=['--copper','--teal','--violet','--gold','--good','--bad'];
let E=null,G=null,trTimer=null;
function trNew(){E={waiting:SEQ0.map(([n,p,m],i)=>({n,p,m,i,out:0,cached:0,blocks:[],done:false,pre:0})),running:[],free:[...Array(NBLK).keys()].reverse(),step:0,gen:0,preempt:0,batch:[],v:{},line:null,log:'Press <b>Next line</b> to execute one line, or <b>Next engine step</b> to run a whole step.'};G=null;}
const nPre=s=>s.p+s.out;
function* schedG(batch){
  let budget=MAXTOK;E.v={budget};yield ['s1',`Fresh step: token budget = ${MAXTOK}.`];
  for(const s of E.running.slice()){
    E.v.seq=s.n;yield ['s2',`Looking at running request <b>${s.n}</b>.`];
    if(!E.running.includes(s)){yield ['s3',`${s.n} was preempted earlier in this step, so skip it.`];continue;}
    yield ['s3',`${s.n} is still running.`];
    const nb=s.cached%BS===0;yield ['s4',`${s.n} has ${s.cached} cached tokens. ${nb?`${s.cached} is a multiple of 16, so the next token needs a <b>new block</b>.`:`Its last block still has room (${BS-s.cached%BS} slots).`}`];
    if(nb){
      const out=!E.free.length;yield ['s5',out?'<b>No free blocks left.</b> Something has to give.':`${E.free.length} free blocks, so no problem.`];
      if(out){
        const v=E.running.pop();yield ['s6',`Preempt <b>${v.n}</b>, the newest request.`];
        E.free.push(...v.blocks);v.blocks=[];v.cached=0;v.pre++;E.preempt++;yield ['s7',`${v.n}'s blocks go back to the pool. Its KV cache is discarded and will be recomputed.`];
        E.waiting.unshift(v);yield ['s8',`${v.n} goes to the front of the waiting queue with ${nPre(v)} tokens (prompt + ${v.out} generated) to re-prefill.`];
        if(v===s){yield ['s9',`The victim was ${s.n} itself, so it gets no token this step.`];continue;}
        yield ['s9',`The victim wasn\'t ${s.n}, so carry on.`];
      }
      s.blocks.push(E.free.pop());yield ['s10',`${s.n} takes block ${s.blocks[s.blocks.length-1]}.`];
    }
    batch.push([s,1]);budget-=1;E.v.budget=budget;yield ['s11',`${s.n} decodes 1 token. Budget left: ${budget}.`];
  }
  while(true){
    const go=E.waiting.length>0&&budget>0;yield ['s12',go?`${E.waiting.length} waiting, budget ${budget}: try to admit.`:E.waiting.length?'Budget used up.':'Nobody is waiting.'];
    if(!go)break;
    const s=E.waiting[0];E.v.seq=s.n;yield ['s13',`Head of the queue: <b>${s.n}</b>.`];
    const need=Math.ceil(nPre(s)/BS);E.v.need=need;yield ['s14',`${s.n} needs ⌈${nPre(s)}/16⌉ = <b>${need} blocks</b>.`];
    const no=need>E.free.length||nPre(s)>budget;yield ['s15',no?`Can\'t admit: ${need>E.free.length?`only ${E.free.length} free blocks`:`${nPre(s)} tokens > budget ${budget}`}. Stop admitting (first come, first served).`:`Fits: ${E.free.length} free blocks, budget ${budget}.`];
    if(no)break;
    E.waiting.shift();yield ['s16',`${s.n} leaves the waiting queue.`];
    s.blocks=E.free.splice(-need).reverse();yield ['s17',`${s.n} gets blocks ${s.blocks.join(', ')}.`];
    E.running.push(s);batch.push([s,nPre(s)]);yield ['s18',`${s.n} joins running and is prefilled this step (${nPre(s)} tokens).`];
    budget-=nPre(s);E.v.budget=budget;yield ['s19',`Budget left: ${budget}.`];
  }
  yield ['s20',`Batch: ${batch.map(([s,n])=>`${s.n}×${n}`).join(', ')||'empty'} = ${batch.reduce((a,b)=>a+b[1],0)} tokens.`];
}
function* stepG(){
  const batch=[];E.batch=batch;
  yield* schedG(batch);
  yield ['p1','schedule() returned the batch.'];
  E.step++;yield ['p2',`<b>One forward pass</b> computes ${batch.reduce((a,b)=>a+b[1],0)} tokens for ${batch.length} requests: prefill and decode mixed together. Weights are read from HBM once for all of them.`];
  for(const [s,n] of batch){
    E.v.seq=s.n;yield ['p3',`Result for ${s.n}.`];
    s.cached+=n;s.out++;E.gen++;yield ['p4',`${s.n}: ${s.cached} tokens cached, ${s.out}/${s.m} generated.`];
    const fin=s.out>=s.m;yield ['p5',fin?`${s.n} reached max_tokens.`:`${s.n} continues.`];
    if(fin){E.running.splice(E.running.indexOf(s),1);s.done=true;yield ['p6',`${s.n} finished.`];
      E.free.push(...s.blocks);const fb=s.blocks.length;s.blocks=[];yield ['p7',`${fb} blocks return to the pool. Room for waiting requests next step.`];}
  }
  yield ['end',`<b>Step ${E.step} done.</b> ${batch.length} requests advanced, ${E.free.length} blocks free, ${E.waiting.length} waiting.`];
}
function trAdvance(){
  if(E.waiting.length===0&&E.running.length===0){E.line=null;E.log=`<b>All six requests finished</b> in ${E.step} engine steps, ${E.gen} tokens generated, ${E.preempt} preemption${E.preempt===1?'':'s'}. Press Reset to run it again.`;trRender();return false;}
  if(!G)G=stepG();
  const r=G.next();
  if(r.done){G=null;return trAdvance();}
  E.line=r.value[0];E.log=r.value[1];trRender();return true;
}
function trRender(){
  const ids=TR.map(l=>l[1]);
  $$('#trCode .ln').forEach((el,i)=>el.classList.toggle('on',!!ids[i]&&ids[i]===E.line));
  const on=$('#trCode .ln.on');if(on&&trTimer){const c=$('#trCode');const top=on.offsetTop-c.offsetTop;if(top<c.scrollTop||top>c.scrollTop+c.clientHeight-30)c.scrollTop=top-60;}
  const chip=s=>`<span class="sq" style="background:${css(SCOL[s.i])}">${s.n} ${s.cached}+${s.out}/${s.m}${s.pre?' ↺'+s.pre:''}</span>`;
  $('#trWait').innerHTML=E.waiting.map(s=>`<span class="sq" style="background:${css(SCOL[s.i])}">${s.n} · ${nPre(s)} tok${s.pre?' ↺':''}</span>`).join('')||'<span class="sq ghost">empty</span>';
  $('#trRun').innerHTML=E.running.map(chip).join('')||'<span class="sq ghost">empty</span>';
  const owner={};[...E.running,...E.waiting].forEach(s=>s.blocks.forEach(b=>owner[b]=s));
  $('#trBlocks').innerHTML=[...Array(NBLK).keys()].map(b=>{const s=owner[b];if(!s)return `<div title="free">${b}</div>`;return `<div class="u" style="background:${css(SCOL[s.i])}" title="${s.n}">${s.n}${b}</div>`;}).join('');
  $('#trBatch').innerHTML=E.batch.map(([s,n])=>`<span class="sq" style="background:${css(SCOL[s.i])}">${s.n} × ${n} ${n>1?'prefill':'decode'}</span>`).join('')||'<span class="sq ghost">not built yet</span>';
  $('#trVars').innerHTML=`<span>step <b>${E.step}</b></span><span>budget <b>${E.v.budget??MAXTOK}</b></span><span>free blocks <b>${E.free.length}</b>/${NBLK}</span><span>seq <b>${E.v.seq??'—'}</b></span><span>generated <b>${E.gen}</b></span><span>preemptions <b>${E.preempt}</b></span>`;
  $('#trLog').innerHTML=E.log;
}
function trInit(){
  $('#trCode').innerHTML=TR.map((l,i)=>{let h=esc(l[0]).replace(/(\s#\s.*)$/,'<span class="cm">$1</span>');return `<span class="ln" data-i="${i}"><i>${i+1}</i>${h||' '}</span>`;}).join('');
  $('#trCode').style.maxHeight='620px';
  trNew();trRender();
  const stop=()=>{if(trTimer){clearInterval(trTimer);trTimer=null;$('#trPlay').textContent='▶ Play';}};
  $('#trLine').onclick=()=>{stop();trAdvance();};
  $('#trStep').onclick=()=>{stop();let n=0;while(n++<900){if(!trAdvance())break;if(E.line==='end')break;}};
  $('#trPlay').onclick=()=>{if(trTimer){stop();return;}$('#trPlay').textContent='❚❚ Pause';trTimer=setInterval(()=>{if(!trAdvance())stop();},reduce?900:320);};
  $('#trReset').onclick=()=>{stop();trNew();trRender();};
}

/* ================= 11 · paths ================= */
const GOALS=[
 {id:'serve',t:'Serve Qwen3-8B in 4-bit, now',s:[
  ['Quantize with an existing tool','Which format already has a fast kernel on our GPUs?','llm-compressor (GPTQ W4A16, group 128) or AutoAWQ; save the checkpoint.','vLLM loads it and the log shows the Marlin kernel being used.'],
  ['Check quality','Did 4-bit hurt the model?','Perplexity and lm-eval, BF16 vs 4-bit, same tasks and settings.','Within your tolerance (for example ≤ 1–2 points on the tasks you care about).'],
  ['Benchmark serving','How many users can we serve at our SLO?','vllm bench serve rate sweep, BF16 vs 4-bit (part 5, section 11).','A goodput table the team can decide from. No kernel work needed.']]},
 {id:'fmt',t:'Try a new quantization idea',s:[
  ['Fake-quantize first','Is the quality worth it before writing any kernel?','Quantize → dequantize the weights in PyTorch, then run the normal BF16 model on them.','Perplexity is acceptable. Quality never depends on the kernel, so measure it with none.'],
  ['Cost model','What would a perfect kernel give us?','Bytes per token at the new bit width ÷ bandwidth (part 5 roofline).','Clearly faster than 4-bit Marlin. If not, stop here.'],
  ['Reference + Triton kernel','Does the kernel compute what the format defines?','Pack/unpack in PyTorch, then a GEMV like section 6.','Matches the dequantized reference on every Qwen3-8B shape.'],
  ['Microbenchmark','How close is it to the roofline?','do_bench at M = 1, 4, 16.','≥ 70% of peak bandwidth at M = 1.'],
  ['Plug into vLLM','Can the whole team serve it?','Custom op + quantization-method plugin (section 10).','Greedy outputs match the fake-quant model; eager and graphs agree.'],
  ['Serve and compare','Is it better than what we had?','vllm bench serve against Marlin 4-bit.','Better TPOT or goodput at equal quality.']]},
 {id:'arch',t:'Serve an architecture vLLM lacks',s:[
  ['Read the HF modeling file','Which ops and shapes does it use, and which already exist as vLLM layers?','List norms, attention type, MLP/MoE, position encoding.','Every op is mapped to a vLLM layer or marked "new".'],
  ['Write the vLLM model file','Do the weights land in the right fused parameters?','Copy the closest existing model (e.g. qwen3.py), adapt it, and write the load_weights mapping.','Loads with no missing or unexpected keys.'],
  ['Compare with HF','Is it the same model?','Same prompts at temperature 0 through both.','First 32 tokens match; logits within bf16 noise.'],
  ['Register and scale','Does it work with tensor parallel?','ModelRegistry.register_model in a plugin; test TP = 1 and 2.','Same outputs with TP 1 and 2.'],
  ['Benchmark','Is any new op slow?','nsys profile one step.','If one op dominates, follow "Speed up a slow op".']]},
 {id:'op',t:'Speed up a slow op',s:[
  ['Profile','What share of step time is it?','nsys on a real serving step; read the kernel table.','Amdahl (section 1) says the end-to-end gain is worth the work.'],
  ['Isolate','How fast is it now, versus the roofline?','Microbenchmark the op alone with the real shapes.','A gap of more than 2× from the roofline, or a fusion opportunity.'],
  ['Write a fused Triton kernel','Can several ops become one pass over memory?','Fuse neighbours (e.g. residual add + RMSNorm), as in sections 4–5.','Tests pass; bytes and launches go down.'],
  ['Swap it in','Does the engine use it, and is it graph-safe?','A CustomOp forward_cuda in your model file or plugin.','Eager = graphs; the end-to-end gain is close to the Amdahl prediction.']]},
 {id:'cpu',t:'Run on CPU, Mac or edge',s:[
  ['Convert and quantize','Which quant fits the memory and keeps quality?','convert_hf_to_gguf.py, then llama-quantize to Q4_K_M or Q5_K_M.','llama-perplexity within tolerance of BF16.'],
  ['Measure','Is tok/s close to bandwidth ÷ bytes?','llama-bench -p 512 -n 128 (part 5, section 10).','≥ 60–70% of the bandwidth bound.'],
  ['Add a missing op','Does every backend compute it the same?','Add the op to ggml and its CPU, Metal and CUDA backends.','test-backend-ops passes: it checks every backend against the CPU reference.']]}];
let goal='fmt';
function paths(){
  $('#goals').innerHTML=GOALS.map(g=>`<button type="button" class="${g.id===goal?'on':''}" data-g="${g.id}">${g.t}</button>`).join('');
  $$('#goals button').forEach(b=>b.onclick=()=>{goal=b.dataset.g;paths();});
  const G0=GOALS.find(g=>g.id===goal);
  $('#pathOut').innerHTML=`<ol>${G0.s.map(([t,q,d,c])=>`<li><b class="s">${t}</b><span class="m"><b>Question:</b> ${q}</span><span class="m"><b>Do:</b> ${d}</span><span class="m"><b>Pass when:</b> ${c}</span></li>`).join('')}</ol>`;
}

/* ================= wiring ================= */
const SRC=[
['Triton language and tutorials (vector add, softmax, matmul, layer norm)','https://triton-lang.org/main/getting-started/tutorials/index.html'],
['PyTorch: custom Python operators (torch.library.custom_op, register_fake)','https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html'],
['PyTorch: custom C++ and CUDA operators (TORCH_LIBRARY)','https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html'],
['vLLM docs: architecture overview','https://docs.vllm.ai/en/latest/design/arch_overview.html'],
['vLLM docs: adding a new model','https://docs.vllm.ai/en/latest/contributing/model/index.html'],
['vLLM docs: plugin system (vllm.general_plugins)','https://docs.vllm.ai/en/latest/design/plugin_system.html'],
['vLLM V1 alpha release: the multi-process engine-core design (2025)','https://blog.vllm.ai/2025/01/27/v1-alpha-release.html'],
['vLLM source code','https://github.com/vllm-project/vllm'],
['nano-vllm: a vLLM-style engine in ~1,200 lines','https://github.com/GeeeekExplorer/nano-vllm'],
['llm-compressor: GPTQ/AWQ/FP8 checkpoints for vLLM','https://github.com/vllm-project/llm-compressor'],
['Frantar et al., MARLIN: mixed-precision kernels (2024)','https://arxiv.org/abs/2408.11743'],
['Hwu, Kirk & El Hajj, Programming Massively Parallel Processors, 4th ed.','https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0'],
['Simon Boehm, How to optimize a CUDA matmul kernel','https://siboehm.com/articles/22/CUDA-MMM'],
['GPU MODE lectures','https://github.com/gpu-mode/lectures'],
['NVIDIA CUTLASS / CuTe','https://github.com/NVIDIA/cutlass'],
['HazyResearch ThunderKittens','https://github.com/HazyResearch/ThunderKittens'],
['Liger Kernel: Triton kernels for LLMs (RMSNorm, SwiGLU, RoPE…)','https://github.com/linkedin/Liger-Kernel'],
['NVIDIA Nsight Compute','https://developer.nvidia.com/nsight-compute'],
['llama.cpp (GGUF, ggml backends, test-backend-ops)','https://github.com/ggml-org/llama.cpp']];
function all(){
  const D=dev();$('#devNote').textContent=`${D.sm} SMs · ${N(D.bw/1e9)} GB/s · ${(D.fl/1e12).toFixed(0)} TFLOP/s dense BF16 · ${D.smem} KB shared memory per block${D.name.includes('Mac')?' (Metal threadgroup memory)':''}`;
  ladder();amdahl();tiling();rms();fusion();w4();kbench(false);
}
function init(){
  $('#sDev').innerHTML=DEVS.map((d,i)=>`<option value="${i}"${i===2?' selected':''}>${d.name}</option>`).join('');
  $('#tLayer').innerHTML=LAYERS.map((l,i)=>`<option value="${i}"${i===2?' selected':''}>${l.t} · ${N(l.K)}→${N(l.N)}</option>`).join('');
  $('#srcList').innerHTML=SRC.map(([t,u])=>`<li><a href="${u}" rel="noopener">${t}</a></li>`).join('');
  $('#lcList').innerHTML=LIFE.map((l,i)=>`<button type="button" data-i="${i}">${i+1} · ${l.t}</button>`).join('');
  $$('#lcList button').forEach(b=>b.onclick=()=>{MP.play=false;$('#lcPlay').textContent='▶ Play';lifeGo(+b.dataset.i);});
  $$('.annot').forEach(annot);
  if(window.THREE){initRR();VW.hier=view($('#vHier'));VW.tile=view($('#vTile'));VW.map=view($('#vMap'),id=>{MP.life=-1;MP.play=false;$('#lcPlay').textContent='▶ Play';$$('#lcList button').forEach(b=>b.classList.remove('on'));mapPanel(id);});hierView();mapView();}
  mapPanel('sched');$('#lcText').innerHTML='Press <b>Play</b> or pick a step to follow one chat request through every component.';
  nibble();paths();trInit();all();kbench(true);
  $('#sDev').addEventListener('input',()=>{all();kbench(true);});
  ['#amS','#amX'].forEach(s=>$(s).addEventListener('input',amdahl));
  ['#tLayer','#tM','#tBM','#tBN','#tBK','#tST','#tSK'].forEach(s=>$(s).addEventListener('input',tiling));
  $('#rmT').addEventListener('input',rms);['#fuT','#fuG'].forEach(s=>$(s).addEventListener('input',fusion));
  $('#kbK').addEventListener('input',()=>kbench(true));$('#kbT').addEventListener('input',()=>kbench(false));
  $('#hiPlay').onclick=()=>{HI.play=!HI.play;$('#hiPlay').textContent=HI.play?'❚❚ Pause':'▶ Run one block';if(HI.play&&HI.phase<0){HI.t=0;}};
  $('#hiText').innerHTML='One block normalizes one token\'s 4,096 numbers. Press <b>Run one block</b> to watch the CUDA kernel from section 4 execute: four strided load passes, a warp-shuffle reduction, a shared-memory combine, then the scaled write.';
  $('#lcPrev').onclick=()=>{MP.play=false;$('#lcPlay').textContent='▶ Play';lifeGo((Math.max(0,MP.life)-1+LIFE.length)%LIFE.length);};
  $('#lcNext').onclick=()=>{MP.play=false;$('#lcPlay').textContent='▶ Play';lifeGo((MP.life+1)%LIFE.length);};
  $('#lcPlay').onclick=()=>{MP.play=!MP.play;$('#lcPlay').textContent=MP.play?'❚❚ Pause':'▶ Play';if(MP.play){MP.t=0;lifeGo((MP.life+1)%LIFE.length);}};
  const links=$$('#rail a[href^="#"]');if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)links.forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+e.target.id));}),{rootMargin:'-35% 0px -60% 0px'});$$('main section').forEach(s=>io.observe(s));}
  const re=()=>{if(window.THREE){hierView();mapView();}all();trRender();};new MutationObserver(re).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  requestAnimationFrame(loop);
}
window.B6={tileCalc,E:()=>E,trAdvance};
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
})();
