/* Token Journey · model data, a tiny real transformer, and the cost model.
   Real-model numbers come from each model's official config.json; the toy keeps the same
   wiring (head ratios, layer pattern, norms, router) at 16 dimensions so every number is visible. */
(function(){
"use strict";
const TJ=window.TJ=window.TJ||{};

/* ---------- random numbers (seeded, so a prompt always gives the same run) ---------- */
function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function gauss(r){let u=0;while(u===0)u=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*r());}
function hash(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619);}return h>>>0;}
TJ.mulberry=mulberry;TJ.gauss=gauss;TJ.hash=hash;

/* ---------- architectures ---------- */
const ARCH=[
 {id:'qwen1-7b',name:'Qwen-7B (2023) · multi-head',short:'Qwen-7B',L:32,h:4096,H:32,kv:32,hd:128,ffn:11008,V:151936,tied:false,attn:'mha',act:'swiglu',
  blurb:'The first Qwen. Classic multi-head attention: each of the 32 query heads has its own key and value head, so the cache is as big as it gets.'},
 {id:'gemma-2b',name:'Gemma 2B (2024) · multi-query',short:'Gemma 2B',L:18,h:2048,H:8,kv:1,hd:256,ffn:16384,V:256000,tied:true,attn:'mqa',act:'geglu',embScale:true,
  blurb:'All 8 query heads share one key/value head (MQA). Tied embeddings: the LM head reuses the embedding table.'},
 {id:'qwen3-8b',name:'Qwen3-8B (2025) · grouped-query',short:'Qwen3-8B',L:36,h:4096,H:32,kv:8,hd:128,ffn:12288,V:151936,tied:false,attn:'gqa',act:'swiglu',qkNorm:true,
  blurb:'Your team baseline. 32 query heads share 8 key/value heads (4 per group). QK-Norm keeps attention scores stable.'},
 {id:'gemma3-4b',name:'Gemma 3 4B (2025) · sliding window 5:1',short:'Gemma 3 4B',L:34,h:2560,H:8,kv:4,hd:256,ffn:10240,V:262144,tied:true,attn:'swa',window:1024,act:'geglu',qkNorm:true,sandwich:true,embScale:true,
  blurb:'Five local layers (each sees only the last 1,024 tokens) for every global layer. Norms before and after each block ("sandwich").'},
 {id:'qwen3-30b-a3b',name:'Qwen3-30B-A3B (2025) · GQA + MoE',short:'Qwen3-30B-A3B',L:48,h:2048,H:32,kv:4,hd:128,ffn:768,V:151936,tied:false,attn:'gqa',act:'swiglu',qkNorm:true,
  moe:{E:128,k:8,s:0,score:'softmax'},
  blurb:'Mixture of experts: 128 small FFNs per layer, a router picks 8 per token. Stores 30.5 B weights, reads about 3.3 B per token.'},
 {id:'qwen35-9b',name:'Qwen3.5-9B (2026) · hybrid linear 3:1',short:'Qwen3.5-9B',L:32,h:4096,H:16,kv:4,hd:256,ffn:12288,V:248320,tied:false,attn:'hybrid',act:'swiglu',qkNorm:true,
  lin:{kh:16,vh:32,dk:128,dv:128},
  blurb:'Three Gated DeltaNet layers (a fixed-size memory, no growing cache) for every gated full-attention layer.'},
 {id:'deepseek-v3',name:'DeepSeek-V3 (2024) · MLA + MoE',short:'DeepSeek-V3',L:61,h:7168,H:128,kv:null,hd:128,ffn:2048,V:129280,tied:false,attn:'mla',act:'swiglu',
  mla:{qr:1536,kvr:512,nope:128,rope:64,v:128},moe:{E:256,k:8,s:1,score:'sigmoid',dense:3,denseFfn:18432},
  blurb:'Multi-head latent attention caches one 576-number vector per token per layer. 256 routed experts + 1 shared; the first 3 layers are dense.'},
 {id:'deepseek-v32',name:'DeepSeek-V3.2 (2025) · MLA + DSA + MoE',short:'DeepSeek-V3.2',L:61,h:7168,H:128,kv:null,hd:128,ffn:2048,V:129280,tied:false,attn:'dsa',act:'swiglu',
  mla:{qr:1536,kvr:512,nope:128,rope:64,v:128},moe:{E:256,k:8,s:1,score:'sigmoid',dense:3,denseFfn:18432},dsa:{heads:64,dim:128,topk:2048},
  blurb:'V3 plus DeepSeek Sparse Attention: a cheap "lightning indexer" scores every past token, then full attention runs on only the top 2,048.'}
];
TJ.ARCH=ARCH;
TJ.arch=id=>ARCH.find(a=>a.id===id);

/* which kind of block each layer is */
function kind(a,i){
  let mix='full';
  if(a.attn==='swa')mix=(i%6===5)?'full':'local';
  if(a.attn==='hybrid')mix=(i%4===3)?'full':'linear';
  let ffn='dense';
  if(a.moe)ffn=(a.moe.dense&&i<a.moe.dense)?'dense':'moe';
  return {mix,ffn};
}
TJ.kind=kind;

/* ---------- hardware & formats ---------- */
TJ.HW=[
 {id:'4090',name:'RTX 4090 · 24 GB · 1,008 GB/s',bw:1008e9,fl:165e12,mem:24},
 {id:'3090',name:'RTX 3090 · 24 GB · 936 GB/s',bw:936e9,fl:71e12,mem:24},
 {id:'a100',name:'A100 80 GB · 2,039 GB/s',bw:2039e9,fl:312e12,mem:80},
 {id:'h100',name:'H100 SXM · 80 GB · 3,350 GB/s',bw:3350e9,fl:989e12,mem:80},
 {id:'m2u',name:'Apple M2 Ultra · 192 GB · 800 GB/s',bw:800e9,fl:27e12,mem:192},
 {id:'cpu',name:'Desktop CPU · DDR5 dual-channel · ≈90 GB/s',bw:89.6e9,fl:1.5e12,mem:64}
];
TJ.FMT=[
 {id:'bf16',name:'BF16 (as released) · 16 bits',bpw:16},
 {id:'q8',name:'Q8_0 · 8.5 bits',bpw:8.5},
 {id:'q6',name:'Q6_K · 6.56 bits',bpw:6.5625},
 {id:'q4',name:'Q4_K_M · ≈4.89 bits',bpw:4.89},
 {id:'q3',name:'Q3_K_M · ≈3.9 bits',bpw:3.9}
];

/* ---------- cost model: parameters, bytes and FLOPs for every stage ---------- */
function mixParams(a,i){
  const k=kind(a,i).mix,h=a.h;
  if(k==='linear'){const l=a.lin;return {qkv:h*(l.kh*l.dk*2+l.vh*l.dv*2)+h*l.vh*2,o:l.vh*l.dv*h};}
  if(a.mla){const m=a.mla,H=a.H;const p={qkv:h*m.qr+m.qr*H*(m.nope+m.rope)+h*(m.kvr+m.rope),up:m.kvr*H*(m.nope+m.v),o:H*m.v*h};
    if(a.dsa){const d=a.dsa;p.idx=m.qr*d.heads*d.dim+h*d.dim+h*d.heads;}return p;}
  const gate=(a.attn==='hybrid')?2:1;
  return {qkv:h*a.H*a.hd*gate+2*h*a.kv*a.hd,o:a.H*a.hd*h};
}
function ffnParams(a,i){
  const k=kind(a,i).ffn,h=a.h;
  if(k==='moe'){const m=a.moe;return {router:h*m.E,active:(m.k+m.s)*3*h*a.ffn,stored:(m.E+m.s)*3*h*a.ffn};}
  const F=(a.moe&&a.moe.denseFfn)?a.moe.denseFfn:a.ffn;
  return {gu:2*h*F,down:F*h,stored:3*h*F};
}
TJ.mixParams=mixParams;TJ.ffnParams=ffnParams;
TJ.totals=function(a){
  let stored=0,active=0;
  for(let i=0;i<a.L;i++){const m=mixParams(a,i),f=ffnParams(a,i);const mp=(m.qkv+(m.up||0)+(m.idx||0)+m.o);
    stored+=mp+f.stored;active+=mp+(f.active!=null?f.active+f.router:f.stored);}
  const emb=a.V*a.h*(a.tied?1:2);
  return {stored:stored+emb,active:active+a.V*a.h,emb};
};
/* per-layer cache read for one decode step, in bytes */
function cacheRead(a,i,ctx,kvB){
  const k=kind(a,i).mix;
  if(k==='linear'){const l=a.lin;return 2*l.vh*l.dk*l.dv*4;}         // read + write the fp32 state
  if(a.mla){const m=a.mla;if(a.dsa)return Math.min(ctx,a.dsa.topk)*(m.kvr+m.rope)*kvB+ctx*a.dsa.dim;return ctx*(m.kvr+m.rope)*kvB;}
  const c=(k==='local')?Math.min(ctx,a.window):ctx;
  return 2*a.kv*a.hd*kvB*c;
}
function attnFlops(a,i,ctx){
  const k=kind(a,i).mix;
  if(k==='linear'){const l=a.lin;return 6*l.vh*l.dk*l.dv;}
  if(a.mla){const m=a.mla,c=a.dsa?Math.min(ctx,a.dsa.topk):ctx;let f=2*a.H*c*(m.kvr+m.rope)+2*a.H*c*m.kvr;if(a.dsa)f+=2*ctx*a.dsa.heads*a.dsa.dim;return f;}
  const c=(k==='local')?Math.min(ctx,a.window):ctx;
  return 4*a.H*a.hd*c;
}
TJ.cacheRead=cacheRead;
/* cost of one stage: {bytes, flops} for the real model, decode of one token at context ctx */
TJ.stageCost=function(a,st,o){
  const W=o.bpw/8,i=st.layer,h=a.h;
  const w=p=>({bytes:p*W,flops:2*p});
  const add=(x,y)=>({bytes:x.bytes+y.bytes,flops:x.flops+y.flops});
  const z={bytes:h*2,flops:4*h};
  switch(st.kind){
    case 'tok':return {bytes:0,flops:0};
    case 'emb':return {bytes:h*W,flops:h};
    case 'in':case 'add1':case 'add2':return {bytes:h*4,flops:h};
    case 'norm1':case 'norm2':case 'fnorm':return z;
    case 'qkv':{const m=mixParams(a,i);return w(m.qkv);}
    case 'rope':return {bytes:h*2,flops:6*a.H*(a.hd||128)};
    case 'compress':{return {bytes:(a.mla.kvr+a.mla.rope)*o.kvB,flops:0};}
    case 'up':{const m=mixParams(a,i);return {bytes:m.up*W,flops:2*m.up};}
    case 'index':{const m=mixParams(a,i);return {bytes:m.idx*W+o.ctx*a.dsa.dim,flops:2*m.idx+2*o.ctx*a.dsa.heads*a.dsa.dim};}
    case 'score':return {bytes:cacheRead(a,i,o.ctx,o.kvB)/2,flops:attnFlops(a,i,o.ctx)/2};
    case 'softmax':{const c=Math.min(o.ctx,kind(a,i).mix==='local'?a.window:o.ctx);return {bytes:0,flops:5*a.H*c};}
    case 'mix':return {bytes:cacheRead(a,i,o.ctx,o.kvB)/2,flops:attnFlops(a,i,o.ctx)/2};
    case 'decay':case 'write':case 'read':{const l=a.lin;return {bytes:l.vh*l.dk*l.dv*4*(st.kind==='read'?0:1),flops:2*l.vh*l.dk*l.dv};}
    case 'oproj':{const m=mixParams(a,i);return w(m.o);}
    case 'gu':{const f=ffnParams(a,i);return w(f.gu);}
    case 'act':{const F=(a.moe&&a.moe.denseFfn)?a.moe.denseFfn:a.ffn;return {bytes:0,flops:8*F};}
    case 'down':{const f=ffnParams(a,i);return w(f.down);}
    case 'route':{const f=ffnParams(a,i);return w(f.router);}
    case 'experts':{const f=ffnParams(a,i);return w(f.active);}
    case 'combine':return {bytes:h*4*(a.moe.k+a.moe.s),flops:2*h*(a.moe.k+a.moe.s)};
    case 'layer':{const m=mixParams(a,i),f=ffnParams(a,i);
      let c=w(m.qkv+(m.up||0)+(m.idx||0)+m.o);c=add(c,{bytes:cacheRead(a,i,o.ctx,o.kvB),flops:attnFlops(a,i,o.ctx)});
      c=add(c,w(f.active!=null?f.active+f.router:f.stored));return c;}
    case 'lmhead':return w(a.V*h);
    case 'sample':return {bytes:a.V*4,flops:3*a.V};
  }
  return {bytes:0,flops:0};
};
TJ.time=function(c,hw,phase,P){
  // decode: weights are read once per token. prefill of P tokens: same bytes, P× the FLOPs.
  const fl=phase==='prefill'?c.flops*P:c.flops;
  const tm=c.bytes/hw.bw,tc=fl/hw.fl;
  return {t:Math.max(tm,tc),tm,tc,bound:tm>=tc?'memory':'compute'};
};

/* ---------- toy vocabulary & tokenizer ---------- */
const WORDS=('the a of to in on and is are it this model token word next text qwen gemma deepseek gpu cpu memory cache layer attention head key value query weight weights reads writes runs learns predicts fast slow big small new old cat sat mat dog ran home sun rain day night i you we they think know see go . , ?').split(' ');
TJ.WORDS=WORDS;
const PIECES=12; // extra rows for sub-word pieces of unknown words
TJ.tokenize=function(text,a){
  const out=[];const raw=text.toLowerCase().match(/[a-z0-9]+|[.,?!]/g)||[];
  for(const w of raw){
    const ix=WORDS.indexOf(w);
    if(ix>=0){out.push({text:w,id:ix,rid:(hash(a.id+w)%a.V),piece:false});continue;}
    for(let s=0;s<w.length;s+=4){const p=w.slice(s,s+4);out.push({text:(s?'##':'')+p,id:WORDS.length+(hash(p)%PIECES),rid:(hash(a.id+p)%a.V),piece:true});}
  }
  return out.slice(-14);
};

/* ---------- activations ---------- */
const sig=z=>1/(1+Math.exp(-z));
const silu=z=>z*sig(z);
const gelu=z=>0.5*z*(1+Math.tanh(0.7978845608*(z+0.044715*z*z*z)));
const relu=z=>z>0?z:0;
const d=(f,z)=>(f(z+1e-3)-f(z-1e-3))/2e-3;
TJ.ACTS={
 swiglu:{name:'SwiGLU · SiLU(gate) × up',f:silu,glu:true,who:'Qwen, Llama, DeepSeek'},
 geglu:{name:'GeGLU · GELU(gate) × up',f:gelu,glu:true,who:'Gemma 1–4'},
 reglu:{name:'ReGLU · ReLU(gate) × up',f:relu,glu:true,who:'tested in Shazeer 2020'},
 gelu:{name:'GELU, no gate',f:gelu,glu:false,who:'GPT-2, BERT'},
 relu:{name:'ReLU, no gate',f:relu,glu:false,who:'Original Transformer (2017)'},
 silu:{name:'SiLU, no gate',f:silu,glu:false,who:'—'}
};
TJ.dAct=d;TJ.sig=sig;

/* ---------- the toy transformer ---------- */
const D=16,NQ=8,HD=2,F=32,E=8,KT=2,EH=8,LC=4,WIN=4,TOPK=4,LH=4,LD=4;
TJ.toyDims={D,NQ,HD,F,E,KT,EH,LC,WIN,TOPK,LH,LD};
function toyKV(a){if(a.attn==='mha')return NQ;if(a.attn==='mqa')return 1;return Math.max(1,Math.round(NQ*a.kv/a.H));}
TJ.toyKV=toyKV;
function mat(r,rows,cols,scale){const m=new Float32Array(rows*cols),s=scale/Math.sqrt(cols);for(let i=0;i<m.length;i++)m[i]=gauss(r)*s;m.rows=rows;m.cols=cols;return m;}
function mv(m,x){const o=new Float32Array(m.rows);for(let i=0;i<m.rows;i++){let s=0;const b=i*m.cols;for(let j=0;j<m.cols;j++)s+=m[b+j]*x[j];o[i]=s;}return o;}
function rms(x){let s=0;for(const v of x)s+=v*v;const r=1/Math.sqrt(s/x.length+1e-6);return Float32Array.from(x,v=>v*r);}
function norm2(x){let s=0;for(const v of x)s+=v*v;return Math.sqrt(s);}
function addv(x,y){return Float32Array.from(x,(v,i)=>v+y[i]);}
function softmax(a){const m=Math.max(...a.filter(v=>v!=null));const e=a.map(v=>v==null?0:Math.exp(v-m));const s=e.reduce((p,q)=>p+q,0);return e.map(v=>v/s);}
function rot(v,ang){const c=Math.cos(ang),s=Math.sin(ang);return [v[0]*c-v[1]*s,v[0]*s+v[1]*c];}
TJ.rms=rms;TJ.norm2=norm2;

TJ.buildToy=function(a){
  const r=mulberry(hash(a.id)),kvT=toyKV(a);
  const V=WORDS.length+PIECES;
  const emb=mat(r,V,D,(a.embScale?0.15:0.6)*Math.sqrt(D));
  const W={emb,V,kvT,layers:[]};
  W.unemb=a.tied?null:mat(r,WORDS.length,D,1);
  for(let i=0;i<a.L;i++){
    const k=kind(a,i),L={};
    if(k.mix==='linear'){L.q=mat(r,LH*LD,D,1);L.k=mat(r,LH*LD,D,1);L.v=mat(r,LH*LD,D,1);L.a=mat(r,LH,D,0.5);L.b=mat(r,LH,D,0.5);L.z=mat(r,D,D,1);L.o=mat(r,D,D,0.6);}
    else if(a.mla){L.dq=mat(r,8,D,1);L.uqn=mat(r,NQ*HD,8,1);L.uqr=mat(r,NQ*HD,8,1);L.dkv=mat(r,LC,D,1);L.kr=mat(r,HD,D,1);L.uk=mat(r,NQ*HD,LC,1);L.uv=mat(r,NQ*HD,LC,1);L.o=mat(r,D,NQ*HD,0.6);
      if(a.dsa){L.iq=mat(r,2*4,8,1);L.iw=mat(r,2,D,1);L.ik=mat(r,4,D,1);}}
    else{L.q=mat(r,NQ*HD,D,1);L.k=mat(r,kvT*HD,D,1);L.v=mat(r,kvT*HD,D,1);L.o=mat(r,D,NQ*HD,0.6);if(a.attn==='hybrid')L.g=mat(r,NQ*HD,D,1);}
    if(k.ffn==='moe'){L.r=mat(r,E,D,1.5);L.ex=[];for(let e=0;e<E;e++)L.ex.push({g:mat(r,EH,D,1),u:mat(r,EH,D,1),d:mat(r,D,EH,0.6)});
      if(a.moe.s)L.sh={g:mat(r,EH,D,1),u:mat(r,EH,D,1),d:mat(r,D,EH,0.6)};}
    else{L.g2=mat(r,F,D,1);L.u2=mat(r,F,D,1);L.d2=mat(r,D,F,0.6);}
    W.layers.push(L);
  }
  return W;
};

function ffnBlock(L,x,act){
  const g=mv(L.g,x),u=mv(L.u,x);
  const av=Float32Array.from(act.glu?g:u,act.f);
  const h=act.glu?Float32Array.from(av,(v,i)=>v*u[i]):av;
  return {g,u,av,h,out:mv(L.d,h)};
}

/* forward pass over the whole sequence; records every intermediate for the LAST token */
TJ.forward=function(a,W,toks,actId){
  const act=TJ.ACTS[actId],T=toks.length,kvT=W.kvT;
  let X=toks.map(t=>{const v=W.emb.subarray(t.id*D,t.id*D+D);return Float32Array.from(v,z=>a.embScale?z*Math.sqrt(D):z);});
  const rec={T,embRaw:Float32Array.from(W.emb.subarray(toks[T-1].id*D,toks[T-1].id*D+D)),layers:[]};
  rec.emb=Float32Array.from(X[T-1]);
  for(let li=0;li<a.L;li++){
    const k=kind(a,li),L=W.layers[li],R={kind:k,xin:Float32Array.from(X[T-1])};
    const N1=X.map(rms);R.n1=N1[T-1];
    let A;  // attention output for every position
    if(k.mix==='linear'){
      const S=[];for(let h=0;h<LH;h++)S.push(new Float32Array(LD*LD));
      A=[];
      for(let t=0;t<T;t++){
        const n=N1[t],q=mv(L.q,n),kk=mv(L.k,n),v=mv(L.v,n),al=mv(L.a,n),be=mv(L.b,n),zg=mv(L.z,n);
        const last=t===T-1;if(last){R.mix={type:'lin',Sprev:S.map(s=>Float32Array.from(s)),alpha:[],beta:[],q:[],k:[],v:[]};}
        const o=new Float32Array(D);
        for(let h=0;h<LH;h++){
          const qh=q.slice(h*LD,h*LD+LD),kh=kk.slice(h*LD,h*LD+LD),vh=v.slice(h*LD,h*LD+LD);
          const nq=norm2(qh)||1,nk=norm2(kh)||1;for(let j=0;j<LD;j++){qh[j]/=nq;kh[j]/=nk;}
          const alpha=TJ.sig(al[h]+2.2),beta=TJ.sig(be[h]);const s=S[h];
          for(let j=0;j<s.length;j++)s[j]*=alpha;
          if(last){R.mix.Sdecay=R.mix.Sdecay||[];R.mix.Sdecay[h]=Float32Array.from(s);}
          const sk=new Float32Array(LD);for(let r2=0;r2<LD;r2++){let z=0;for(let c=0;c<LD;c++)z+=s[r2*LD+c]*kh[c];sk[r2]=z;}
          for(let r2=0;r2<LD;r2++)for(let c=0;c<LD;c++)s[r2*LD+c]+=beta*(vh[r2]-sk[r2])*kh[c];
          for(let r2=0;r2<LD;r2++){let z=0;for(let c=0;c<LD;c++)z+=s[r2*LD+c]*qh[c];o[h*LD+r2]=z;}
          if(last){R.mix.alpha.push(alpha);R.mix.beta.push(beta);R.mix.q.push(qh);R.mix.k.push(kh);R.mix.v.push(vh);}
        }
        const gated=Float32Array.from(o,(z,j)=>z*TJ.ACTS.silu.f(zg[j]));
        if(last){R.mix.Snew=S.map(s=>Float32Array.from(s));R.mix.heads=o;}
        A.push(mv(L.o,gated));
      }
    }else if(a.mla){
      const C=[],KR=[],IK=[];
      for(let t=0;t<T;t++){C.push(mv(L.dkv,N1[t]));KR.push(rot(mv(L.kr,N1[t]),t*0.6));if(a.dsa)IK.push(mv(L.ik,N1[t]));}
      A=[];
      for(let t=0;t<T;t++){
        const last=t===T-1,n=N1[t],cq=mv(L.dq,n),qn=mv(L.uqn,cq),qr=mv(L.uqr,cq);
        let allow=null,idx=null;
        if(a.dsa){const iq=mv(L.iq,cq),iw=mv(L.iw,n);idx=[];
          for(let s=0;s<=t;s++){let sc=0;for(let h=0;h<2;h++){let dd=0;for(let j=0;j<4;j++)dd+=iq[h*4+j]*IK[s][j];sc+=iw[h]*Math.max(0,dd);}idx.push(sc);}
          const order=idx.map((v,s)=>[v,s]).sort((p,q2)=>q2[0]-p[0]).slice(0,TOPK).map(p=>p[1]);allow=new Set(order);}
        const heads=new Float32Array(NQ*HD),sc=[],wt=[];
        for(let j=0;j<NQ;j++){
          const qj=[qn[j*2],qn[j*2+1]],qrj=rot([qr[j*2],qr[j*2+1]],t*0.6);const row=[];
          for(let s=0;s<=t;s++){
            if(allow&&!allow.has(s)){row.push(null);continue;}
            const kn=[0,0];for(let c=0;c<LC;c++){kn[0]+=L.uk[(j*2)*LC+c]*C[s][c];kn[1]+=L.uk[(j*2+1)*LC+c]*C[s][c];}
            row.push((qj[0]*kn[0]+qj[1]*kn[1]+qrj[0]*KR[s][0]+qrj[1]*KR[s][1])/2);
          }
          const w=softmax(row);
          for(let s=0;s<=t;s++){if(!w[s])continue;for(let dd=0;dd<2;dd++){let vv=0;for(let c=0;c<LC;c++)vv+=L.uv[(j*2+dd)*LC+c]*C[s][c];heads[j*2+dd]+=w[s]*vv;}}
          sc.push(row);wt.push(w);
        }
        if(last)R.mix={type:'mla',c:C.map(x=>Float32Array.from(x)),kr:KR,scores:sc,w:wt,heads,idx,sel:allow?[...allow]:null};
        A.push(mv(L.o,heads));
      }
    }else{
      const local=k.mix==='local';
      const Q=[],K=[],Vv=[];
      for(let t=0;t<T;t++){
        const n=N1[t];let q=mv(L.q,n),kk=mv(L.k,n);const v=mv(L.v,n);
        if(a.qkNorm){q=normHeads(q);kk=normHeads(kk);}
        const qr=[],kr=[];for(let j=0;j<NQ;j++)qr.push(rot([q[j*2],q[j*2+1]],t*0.6));for(let g=0;g<kvT;g++)kr.push(rot([kk[g*2],kk[g*2+1]],t*0.6));
        Q.push(qr);K.push(kr);const vr=[];for(let g=0;g<kvT;g++)vr.push([v[g*2],v[g*2+1]]);Vv.push(vr);
      }
      A=[];
      for(let t=0;t<T;t++){
        const last=t===T-1,heads=new Float32Array(NQ*HD),sc=[],wt=[];
        for(let j=0;j<NQ;j++){
          const g=Math.floor(j*kvT/NQ),row=[];
          for(let s=0;s<=t;s++){if(local&&t-s>=WIN){row.push(null);continue;}row.push((Q[t][j][0]*K[s][g][0]+Q[t][j][1]*K[s][g][1])/Math.SQRT2*(a.qkNorm?2.5:1));}
          const w=softmax(row);
          for(let s=0;s<=t;s++){heads[j*2]+=w[s]*Vv[s][g][0];heads[j*2+1]+=w[s]*Vv[s][g][1];}
          sc.push(row);wt.push(w);
        }
        let hv=heads;
        if(a.attn==='hybrid'){const gt=mv(L.g,N1[t]);hv=Float32Array.from(heads,(z,j)=>z*TJ.sig(gt[j]));if(last)R.gate=gt;}
        if(last)R.mix={type:'std',kvT,local,q:Q[t],K:K.map(x=>x),V:Vv,scores:sc,w:wt,heads};
        A.push(mv(L.o,hv));
      }
    }
    if(a.sandwich)A=A.map(rms).map(v=>Float32Array.from(v,z=>z*0.5));
    R.attn=A[T-1];
    X=X.map((x,t)=>addv(x,A[t]));
    R.xmid=Float32Array.from(X[T-1]);
    const N2=X.map(rms);R.n2=N2[T-1];
    const Fo=[];
    for(let t=0;t<T;t++){
      const n=N2[t],last=t===T-1;
      if(k.ffn==='moe'){
        const lg=mv(L.r,n);let probs,sel,w;
        if(a.moe.score==='softmax'){probs=softmax([...lg]);}else{probs=[...lg].map(TJ.sig);}
        sel=probs.map((p,e)=>[p,e]).sort((p,q)=>q[0]-p[0]).slice(0,KT).map(p=>p[1]);
        const tot=sel.reduce((s,e)=>s+probs[e],0);w=sel.map(e=>probs[e]/tot*(a.moe.score==='sigmoid'?2.5:1));
        let out=new Float32Array(D);const eh={};
        sel.forEach((e,ii)=>{const r2=ffnBlock(L.ex[e],n,act);eh[e]=r2;for(let j=0;j<D;j++)out[j]+=w[ii]*r2.out[j];});
        let shr=null;if(L.sh){shr=ffnBlock(L.sh,n,act);out=addv(out,shr.out);}
        if(last)R.ffn={type:'moe',logits:lg,probs,sel,w,eh,shared:shr};
        Fo.push(out);
      }else{
        const r2=ffnBlock({g:L.g2,u:L.u2,d:L.d2},n,act);
        if(last)R.ffn={type:'dense',g:r2.g,u:r2.u,av:r2.av,h:r2.h};
        Fo.push(r2.out);
      }
    }
    let Fx=Fo;if(a.sandwich)Fx=Fo.map(rms).map(v=>Float32Array.from(v,z=>z*0.5));
    R.ffnOut=Fx[T-1];
    X=X.map((x,t)=>addv(x,Fx[t]));
    R.xout=Float32Array.from(X[T-1]);
    R.norms={in:norm2(R.xin),attn:norm2(R.attn),ffn:norm2(R.ffnOut),out:norm2(R.xout)};
    rec.layers.push(R);
  }
  rec.final=rms(X[T-1]);
  const nW=WORDS.length,logits=new Float32Array(nW);
  for(let v=0;v<nW;v++){let s=0;const M=W.unemb||W.emb;for(let j=0;j<D;j++)s+=M[v*D+j]*rec.final[j];logits[v]=s*(W.unemb?1:0.6);}
  rec.logits=logits;
  return rec;
};
function normHeads(x){const o=new Float32Array(x.length);for(let i=0;i<x.length;i+=2){const n=Math.sqrt((x[i]*x[i]+x[i+1]*x[i+1])/2)+1e-6;o[i]=x[i]/n;o[i+1]=x[i+1]/n;}return o;}
TJ.probs=function(logits,temp){const t=Math.max(temp,0.05);return softmax([...logits].map(v=>v/t));};
})();
