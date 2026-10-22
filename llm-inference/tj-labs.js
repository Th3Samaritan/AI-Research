/* Token Journey · labs: activation surfaces, optimizer landscape, a real mini training run. */
(function(){
"use strict";
const TJ=window.TJ,$=s=>document.querySelector(s),css=TJ.css,T3=TJ.three;
const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/* small 3D utilities */
function surfaceGeo(nx,nz,fn){
  const pos=new Float32Array((nx+1)*(nz+1)*3),colr=new Float32Array(pos.length),idx=[];
  for(let j=0;j<=nz;j++)for(let i=0;i<=nx;i++){const k=(j*(nx+1)+i)*3,[x,y,z,c]=fn(i/nx,j/nz);pos[k]=x;pos[k+1]=y;pos[k+2]=z;colr[k]=c.r;colr[k+1]=c.g;colr[k+2]=c.b;}
  for(let j=0;j<nz;j++)for(let i=0;i<nx;i++){const a=j*(nx+1)+i,b=a+1,c=a+nx+1,d=c+1;idx.push(a,c,b,b,c,d);}
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(pos,3));g.setAttribute('color',new THREE.BufferAttribute(colr,3));g.setIndex(idx);g.computeVertexNormals();return g;
}
function labelsFor(stageSel){const box=$(stageSel+' .lbls');const list=[];return {add(t,p,cls){const el=document.createElement('div');el.className='lbl3'+(cls?' '+cls:'');el.textContent=t;box.appendChild(el);list.push({el,p});return el;},
  clear(){box.innerHTML='';list.length=0;},place(S){list.forEach(({el,p})=>{const q=T3.project(S,p);el.style.visibility=q.ok?'':'hidden';el.style.transform=`translate(${q.x}px,${q.y}px) translate(-50%,-50%)`;});}};}
function axisLines(g,len){[[[-len[0],0,0],[len[0],0,0]],[[0,0,-len[2]],[0,0,len[2]]],[[0,-len[1],0],[0,len[1],0]]].forEach(p=>{const geo=new THREE.BufferGeometry().setFromPoints(p.map(v=>new THREE.Vector3(...v)));g.add(new THREE.Line(geo,new THREE.LineBasicMaterial({color:T3.col('--muted'),transparent:true,opacity:0.6})));});}

/* ================= 2 · activation surfaces ================= */
const AS={};
function actInit(){
  $('#aFn').innerHTML=Object.entries(TJ.ACTS).map(([k,v])=>`<option value="${k}">${v.name}</option>`).join('');
  syncActSelect();
  if(!window.THREE)return;
  const st={theta:-0.7,phi:1.05,r:15},tg=new THREE.Vector3(0,0.6,0);
  AS.S=T3.makeScene($('#aCanvas'),$('#aStage'),st,tg);AS.g=new THREE.Group();AS.S.scene.add(AS.g);AS.L=labelsFor('#aStage');AS.S.place();
  const spin=()=>{if(AS.S.vis){if(!reduce&&$('#aSpin').checked){st.theta+=0.002;AS.S.place();AS.S.dirty=true;}if(AS.S.dirty){AS.S.render();AS.L.place(AS.S);AS.S.dirty=false;}}requestAnimationFrame(spin);};
  requestAnimationFrame(spin);
  actBuild();AS.S.resize();
  $('#aFn').onchange=()=>{TJ.setAct($('#aFn').value);actBuild();};
  ['#aDeriv','#aDots'].forEach(s=>$(s).addEventListener('change',actBuild));
  document.addEventListener('tj-act',()=>{syncActSelect();actBuild();});
  document.addEventListener('tj-run',actBuild);
}
function syncActSelect(){const a=TJ.journeyArch();$('#aFn').value=$('#jAct').value||(a?a.act:'swiglu');}
function actBuild(){
  const key=$('#aFn').value,A=TJ.ACTS[key],der=$('#aDeriv').checked;
  const out=(g,u)=>{if(der)return A.glu?TJ.dAct(A.f,g)*u:TJ.dAct(A.f,g);return A.glu?A.f(g)*u:A.f(g);};
  const SY=0.55,clampY=y=>Math.max(-3.2,Math.min(3.6,y*SY));
  if(AS.S){
    T3.clear(AS.g);AS.L.clear();
    const geo=surfaceGeo(64,48,(fx,fz)=>{const g=-4+8*fx,u=-3+6*fz,y=out(g,u);const c=new THREE.Color(css('--sunk')).lerp(new THREE.Color(css(y>=0?'--copper':'--teal')),Math.min(1,0.2+Math.abs(y)/4));return [g,clampY(y),u,c];});
    AS.g.add(new THREE.Mesh(geo,new THREE.MeshLambertMaterial({vertexColors:true,side:THREE.DoubleSide,transparent:true,opacity:0.92})));
    AS.g.add(new THREE.Mesh(geo,new THREE.MeshBasicMaterial({color:T3.col('--ink'),wireframe:true,transparent:true,opacity:0.07})));
    axisLines(AS.g,[4.4,3.4,3.4]);
    AS.L.add(A.glu?'gate g →':'input z →',new THREE.Vector3(4.7,0,0),'b');
    AS.L.add(A.glu?'up u →':'(no second input)',new THREE.Vector3(0,0,3.7),'b');
    AS.L.add(der?(A.glu?'∂h/∂g':'act′(z)'):'output h',new THREE.Vector3(0,3.7,0),'b');
    if(!A.glu)AS.L.add('no gate: the surface is the same curve at every depth',new THREE.Vector3(0,-1.2,3.2));
  }
  // neurons of the current token, from the journey's magnified layer
  const rec=TJ.journeyRec(),li=TJ.journeyFocus();let F=null;
  if(rec){const R=rec.layers[li];F=R.ffn.type==='dense'?R.ffn:(()=>{const e=R.ffn.sel[0];const x=R.ffn.eh[e];return {g:x.g,u:x.u,h:x.h,av:x.av};})();}
  let near=0;
  if(F&&AS.S&&$('#aDots').checked){
    const sg=new THREE.SphereGeometry(0.09,12,8),mt=new THREE.MeshBasicMaterial({color:T3.col('--gold')});
    for(let i=0;i<F.h.length;i++){const g=A.glu?F.g[i]:F.u[i],u=A.glu?F.u[i]:0;const y=out(g,u);const m=new THREE.Mesh(sg,mt);m.position.set(Math.max(-4,Math.min(4,g)),clampY(y)+0.05,Math.max(-3,Math.min(3,u)));AS.g.add(m);}
  }
  if(F)near=[...F.h].filter(v=>Math.abs(v)<0.05).length;
  if(AS.S)AS.S.dirty=true;
  const fm={swiglu:'h = SiLU(x·W_g) ⊙ (x·W_u),   SiLU(z) = z·σ(z)',geglu:'h = GELU(x·W_g) ⊙ (x·W_u),   GELU(z) ≈ ½z(1+tanh(√(2/π)(z+0.0447z³)))',reglu:'h = max(0, x·W_g) ⊙ (x·W_u)',gelu:'h = GELU(x·W)',relu:'h = max(0, x·W)',silu:'h = SiLU(x·W)'}[key];
  $('#aFormula').textContent=fm;
  $('#aOut').innerHTML=`<div class="metric"><span class="k">Used by</span><span class="v" style="font-size:1.05rem">${A.who}</span></div>
   <div class="metric"><span class="k">Toy neurons ≈ 0 now</span><span class="v">${F?near:'–'} / ${F?F.h.length:32}</span><span class="s">layer ${li+1} of the journey, current token</span></div>
   <div class="metric"><span class="k">Gradient for z &lt; 0</span><span class="v">${key.includes('relu')||key==='reglu'?'0 (dead)':'small, not 0'}</span><span class="s">${key.includes('relu')||key==='reglu'?'a neuron stuck below 0 stops learning':'negative inputs still learn'}</span></div>`;
}

/* ================= 3 · optimizer landscape ================= */
const SURF={
 ravine:{name:'Long narrow valley (ill-conditioned)',f:(x,y)=>{const a=x-1,b=y-0.5,u=(a+b)/Math.SQRT2,v=(a-b)/Math.SQRT2;return 0.08*u*u+1.5*v*v;},sc:0.6,start:[-2.4,1.4],min:[1,0.5]},
 rosen:{name:'Curved banana valley (Rosenbrock)',f:(x,y)=>0.3*((1-x)*(1-x)+5*(y-x*x)*(y-x*x)),sc:0.4,start:[-2.2,2.6],min:[1,1]},
 bumpy:{name:'Bumpy bowl (local minima)',f:(x,y)=>0.18*((x-1)*(x-1)+(y-1)*(y-1))+0.45*(1-Math.cos(2.6*(x-1))*Math.cos(2.6*(y-1))),sc:1.0,start:[-2.6,-2.4],min:[1,1]},
 saddle:{name:'Saddle point',f:(x,y)=>0.25*x*x-0.5*y*y+0.08*y*y*y*y+1.0,sc:1.0,start:[-2.7,0.03],min:[0,1.77]}
};
const OPTS=[{id:'sgd',name:'SGD',c:'--steel'},{id:'mom',name:'SGD + momentum',c:'--gold'},{id:'rms',name:'RMSProp',c:'--violet'},{id:'adam',name:'Adam (L2 decay)',c:'--teal'},{id:'adamw',name:'AdamW',c:'--copper'}];
const OS={step:0,run:false};
const lrOf=()=>Math.pow(10,-3+(+$('#oLr').value)/100*3);
function sched(step,total,kind,warm,base){
  if(kind==='const')return base;
  if(kind==='step')return base*(step<total*0.5?1:step<total*0.8?0.3:0.09);
  if(kind==='cos')return base*0.5*(1+Math.cos(Math.PI*Math.min(1,step/total)));
  if(kind==='wcos'){if(step<warm)return base*(step+1)/warm;return base*0.5*(1+Math.cos(Math.PI*Math.min(1,(step-warm)/Math.max(1,total-warm))));}
  if(kind==='wsd'){if(step<warm)return base*(step+1)/warm;if(step<total*0.8)return base;return base*Math.max(0,1-(step-total*0.8)/(total*0.2));}
  return base;
}
TJ.sched=sched;
function optInit(){
  $('#oSurf').innerHTML=Object.entries(SURF).map(([k,v])=>`<option value="${k}">${v.name}</option>`).join('');
  $('#oShow').innerHTML=OPTS.map(o=>`<label class="chk"><input type="checkbox" value="${o.id}" checked> <i style="background:var(${o.c})"></i>${o.name}</label>`).join('');
  $('#oExplain').innerHTML=OPTS.map(o=>`<option value="${o.id}"${o.id==='adamw'?' selected':''}>${o.name}</option>`).join('');
  const outs=()=>{$('#oLrO').textContent=lrOf().toPrecision(2);$('#oB1O').textContent=(+$('#oB1').value).toFixed(2);$('#oB2O').textContent=(+$('#oB2').value).toFixed(4);$('#oWdO').textContent=(+$('#oWd').value).toFixed(2);$('#oNoiseO').textContent=(+$('#oNoise').value).toFixed(2);$('#oWarmO').textContent=$('#oWarm').value;};
  ['#oLr','#oB1','#oB2','#oWd','#oNoise','#oWarm'].forEach(s=>$(s).addEventListener('input',()=>{outs();schedDraw();}));
  outs();
  if(window.THREE){
    const st={theta:0.5,phi:0.95,r:16.5},tg=new THREE.Vector3(0,0.6,0);
    OS.S=T3.makeScene($('#oCanvas'),$('#oStage'),st,tg,optClick);OS.g=new THREE.Group();OS.S.scene.add(OS.g);OS.L=labelsFor('#oStage');OS.S.place();
  }
  $('#oSurf').onchange=()=>{OS.start=null;optBuild();optReset();};
  $('#oSched').onchange=()=>{schedDraw();};
  $('#oPlay').onclick=()=>{OS.run=!OS.run;$('#oPlay').textContent=OS.run?'❚❚ Pause':'▶ Run';};
  $('#oReset').onclick=optReset;
  $('#oShow').addEventListener('change',()=>{OS.opts.forEach(o=>{o.on=[...document.querySelectorAll('#oShow input:checked')].some(i=>i.value===o.id);o.ball.visible=o.on;o.trail.visible=o.on;});OS.S&&(OS.S.dirty=true);});
  $('#oExplain').onchange=optReceipt;
  optBuild();optReset();
  let acc=0;
  const tick=()=>{if(OS.S&&OS.S.vis){if(OS.run){const n=+$('#oSpeed').value;for(let i=0;i<n;i++)optStep();optDraw();}if(OS.S.dirty){OS.S.render();OS.L.place(OS.S);OS.S.dirty=false;}}requestAnimationFrame(tick);};
  requestAnimationFrame(tick);
  OS.S&&OS.S.resize();
}
const toW=(x,y,f)=>new THREE.Vector3(x*1.25,3.6*Math.log1p(Math.max(0,f-OS.surf.fmin))/Math.log1p(OS.surf.fmax-OS.surf.fmin),y*1.25);
function optBuild(){
  const s=SURF[$('#oSurf').value];OS.surf=s;
  if(!OS.S)return;T3.clear(OS.g);OS.L.clear();
  let mx=-1e9,mn=1e9;for(let i=0;i<=60;i++)for(let j=0;j<=60;j++){const v=s.f(-3+i*0.1,-3+j*0.1);mx=Math.max(mx,v);mn=Math.min(mn,v);}s.fmax=mx;s.fmin=mn;
  const geo=surfaceGeo(70,70,(fx,fz)=>{const x=-3+6*fx,y=-3+6*fz,f=s.f(x,y),p=toW(x,y,f);const t=p.y/3.6;const c=new THREE.Color(css('--teal')).lerp(new THREE.Color(css('--copper')),t);return [p.x,p.y,p.z,c];});
  OS.mesh=new THREE.Mesh(geo,new THREE.MeshLambertMaterial({vertexColors:true,side:THREE.DoubleSide,transparent:true,opacity:0.85}));OS.g.add(OS.mesh);
  OS.g.add(new THREE.Mesh(geo,new THREE.MeshBasicMaterial({color:T3.col('--ink'),wireframe:true,transparent:true,opacity:0.06})));
  const mp=toW(s.min[0],s.min[1],s.f(...s.min));const mk=new THREE.Mesh(new THREE.ConeGeometry(0.15,0.4,12),new THREE.MeshBasicMaterial({color:T3.col('--good')}));mk.position.copy(mp).add(new THREE.Vector3(0,0.35,0));mk.rotation.x=Math.PI;OS.g.add(mk);
  OS.L.add('minimum',mp.clone().add(new THREE.Vector3(0,0.8,0)),'b');
  const o=new THREE.Mesh(new THREE.SphereGeometry(0.08,10,8),new THREE.MeshBasicMaterial({color:T3.col('--ink')}));o.position.copy(toW(0,0,s.f(0,0)));OS.g.add(o);OS.L.add('origin (where weight decay pulls)',o.position.clone().add(new THREE.Vector3(0,-0.5,1.1)));
  OS.opts=OPTS.map(op=>{
    const ball=new THREE.Mesh(new THREE.SphereGeometry(0.16,16,10),new THREE.MeshLambertMaterial({color:T3.col(op.c)}));OS.g.add(ball);
    const tg=new THREE.BufferGeometry();tg.setAttribute('position',new THREE.BufferAttribute(new Float32Array(3*2000),3));tg.setDrawRange(0,0);
    const trail=new THREE.Line(tg,new THREE.LineBasicMaterial({color:T3.col(op.c)}));OS.g.add(trail);
    const on=[...document.querySelectorAll('#oShow input:checked')].some(i=>i.value===op.id);ball.visible=on;trail.visible=on;
    return Object.assign({},op,{ball,trail,on});
  });
}
function optReset(){
  const s=OS.surf,st=OS.start||s.start;OS.step=0;
  OS.opts.forEach(o=>{o.p=[st[0],st[1]];o.m=[0,0];o.v=[0,0];o.n=0;o.trailN=0;o.last=null;o.dead=false;});
  OS.rng=TJ.mulberry(42);
  optDraw();optReceipt();schedDraw();
}
function grad(f,x,y){const h=1e-4;return [(f(x+h,y)-f(x-h,y))/(2*h),(f(x,y+h)-f(x,y-h))/(2*h)];}
function optStep(){
  const s=OS.surf,total=+$('#oTotal').value;if(OS.step>=total){OS.run=false;$('#oPlay').textContent='▶ Run';return;}
  const base=lrOf(),lr=sched(OS.step,total,$('#oSched').value,+$('#oWarm').value,base);OS.lr=lr;
  const b1=+$('#oB1').value,b2=+$('#oB2').value,wd=+$('#oWd').value,sg=+$('#oNoise').value,eps=1e-8;
  const noise=[TJ.gauss(OS.rng)*sg,TJ.gauss(OS.rng)*sg];
  OS.opts.forEach(o=>{
    if(o.dead)return;
    const g0=grad(s.f,o.p[0],o.p[1]),g=[g0[0]+noise[0],g0[1]+noise[1]];o.n++;
    const L2=[g[0]+wd*o.p[0],g[1]+wd*o.p[1]];const rec={g:g.slice(),lr};
    let d=[0,0];
    if(o.id==='sgd'){d=[-lr*L2[0],-lr*L2[1]];}
    else if(o.id==='mom'){o.m=[b1*o.m[0]+L2[0],b1*o.m[1]+L2[1]];d=[-lr*o.m[0],-lr*o.m[1]];rec.m=o.m.slice();}
    else if(o.id==='rms'){o.v=[b2*o.v[0]+(1-b2)*L2[0]*L2[0],b2*o.v[1]+(1-b2)*L2[1]*L2[1]];d=[-lr*L2[0]/(Math.sqrt(o.v[0])+eps),-lr*L2[1]/(Math.sqrt(o.v[1])+eps)];rec.v=o.v.slice();}
    else{const gg=o.id==='adam'?L2:g;o.m=[b1*o.m[0]+(1-b1)*gg[0],b1*o.m[1]+(1-b1)*gg[1]];o.v=[b2*o.v[0]+(1-b2)*gg[0]*gg[0],b2*o.v[1]+(1-b2)*gg[1]*gg[1]];
      const mh=o.m.map(x=>x/(1-Math.pow(b1,o.n))),vh=o.v.map(x=>x/(1-Math.pow(b2,o.n)));
      d=[-lr*mh[0]/(Math.sqrt(vh[0])+eps),-lr*mh[1]/(Math.sqrt(vh[1])+eps)];
      if(o.id==='adamw'){d[0]-=lr*wd*o.p[0];d[1]-=lr*wd*o.p[1];}
      Object.assign(rec,{m:o.m.slice(),v:o.v.slice(),mh,vh});}
    o.p=[o.p[0]+d[0],o.p[1]+d[1]];rec.d=d;o.last=rec;
    if(!isFinite(o.p[0])||Math.abs(o.p[0])>3.4||Math.abs(o.p[1])>3.4){o.dead=true;o.p=o.p.map(v=>Math.max(-3.4,Math.min(3.4,isFinite(v)?v:3.4)));}
    if(o.trailN<2000){const a=o.trail.geometry.attributes.position,w=toW(o.p[0],o.p[1],s.f(o.p[0],o.p[1]));a.setXYZ(o.trailN,w.x,w.y+0.04,w.z);o.trailN++;a.needsUpdate=true;o.trail.geometry.setDrawRange(0,o.trailN);}
  });
  OS.step++;
}
function optDraw(){
  const s=OS.surf;if(OS.S){OS.opts.forEach(o=>{const w=toW(o.p[0],o.p[1],s.f(o.p[0],o.p[1]));o.ball.position.set(w.x,w.y+0.16,w.z);});OS.S.dirty=true;}
  $('#oTable').innerHTML=`<tr><th>Optimizer</th><th>Loss</th><th>Distance to minimum</th><th>Status</th></tr>`+OS.opts.map(o=>{const L=s.f(o.p[0],o.p[1]),dm=Math.hypot(o.p[0]-s.min[0],o.p[1]-s.min[1]);return `<tr${o.on?'':' style="opacity:.4"'}><td><i class="sw" style="background:var(${o.c})"></i>${o.name}</td><td class="num">${L.toFixed(4)}</td><td class="num">${dm.toFixed(3)}</td><td>${o.dead?'<span class="pill bad">diverged</span>':dm<0.05?'<span class="pill good">arrived</span>':''}</td></tr>`;}).join('');
  $('#oStep').textContent=`step ${OS.step} / ${$('#oTotal').value} · lr now ${(OS.lr||lrOf()).toPrecision(2)}`;
  if(OS.step%5===0)optReceipt();
  if(OS.step%10===0)schedDraw();
}
function optReceipt(){
  const o=OS.opts.find(x=>x.id===$('#oExplain').value),r=o&&o.last,f=v=>v==null?'–':(Array.isArray(v)?'['+v.map(x=>x.toExponential(2)).join(', ')+']':v.toPrecision(3));
  const rule={sgd:'θ ← θ − lr·(g + λθ)',mom:'m ← β₁m + (g + λθ);   θ ← θ − lr·m',rms:'v ← β₂v + (1−β₂)g²;   θ ← θ − lr·g/(√v+ε)',adam:'g ← g + λθ (decay mixed into the gradient)\nm ← β₁m + (1−β₁)g;  v ← β₂v + (1−β₂)g²\nm̂ = m/(1−β₁ᵗ);  v̂ = v/(1−β₂ᵗ)\nθ ← θ − lr·m̂/(√v̂+ε)',adamw:'m ← β₁m + (1−β₁)g;  v ← β₂v + (1−β₂)g²\nm̂ = m/(1−β₁ᵗ);  v̂ = v/(1−β₂ᵗ)\nθ ← θ − lr·( m̂/(√v̂+ε) + λθ )   ← decay applied separately'}[o.id];
  const why={sgd:'Steps straight downhill, scaled by the slope. In a narrow valley it bounces between the steep walls and crawls along the gentle floor.',mom:'Keeps a running velocity, so the zig-zags cancel and progress along the floor builds up. Too much momentum overshoots.',rms:'Divides each coordinate by its recent gradient size, so steep and flat directions take similar-sized steps.',adam:'Momentum + RMSProp + bias correction. Here weight decay is added to the gradient, so it also gets divided by √v̂ and becomes weak where gradients are large.',adamw:'Adam with decoupled weight decay: the pull toward zero is applied directly, at the same strength for every weight. It\'s the default for LLM pre-training (DeepSeek-V3: β₁ 0.9, β₂ 0.95, λ 0.1).'}[o.id];
  $('#oReceipt').innerHTML=`<pre>${rule}</pre><p class="note">${why}</p><div class="receipt">${[['step t',o.n],['lr (after schedule)',f(r&&r.lr)],['gradient g',f(r&&r.g)],['m (first moment)',f(r&&r.m)],['v (second moment)',f(r&&r.v)],['m̂ (bias-corrected)',f(r&&r.mh)],['v̂ (bias-corrected)',f(r&&r.vh)],['update Δθ',f(r&&r.d)],['position θ',f(o.p)]].map(([a,b])=>`<div class="row"><span>${a}</span><span>${b}</span></div>`).join('')}</div>`;
}
function schedDraw(){
  const cv=$('#oSched2'),g=cv.getContext('2d'),W=cv.width,H=cv.height,total=+$('#oTotal').value,base=lrOf(),k=$('#oSched').value,wm=+$('#oWarm').value;
  g.clearRect(0,0,W,H);g.strokeStyle=css('--line');g.beginPath();g.moveTo(30,H-18);g.lineTo(W-6,H-18);g.stroke();
  g.fillStyle=css('--muted');g.font='10px '+css('--mono');g.fillText('lr',4,12);g.fillText('0',30,H-5);g.fillText(String(total),W-30,H-5);
  g.strokeStyle=css('--copper');g.lineWidth=2;g.beginPath();for(let i=0;i<=total;i+=Math.max(1,total/200)){const x=30+i/total*(W-36),y=H-18-sched(i,total,k,wm,base)/base*(H-30);i?g.lineTo(x,y):g.moveTo(x,y);}g.stroke();
  const x=30+Math.min(OS.step,total)/total*(W-36);g.strokeStyle=css('--ink');g.lineWidth=1;g.beginPath();g.moveTo(x,6);g.lineTo(x,H-18);g.stroke();
}
function optClick(e){
  if(!OS.mesh)return;const r=OS.S.cv.getBoundingClientRect(),m=new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);
  const ray=new THREE.Raycaster();ray.setFromCamera(m,OS.S.cam);const h=ray.intersectObject(OS.mesh);if(!h.length)return;
  OS.start=[h[0].point.x/1.25,h[0].point.z/1.25];optReset();OS.run=true;$('#oPlay').textContent='❚❚ Pause';
}

/* ================= 4 · a real mini training run ================= */
const CORPUS=['the cat sat on the mat .','the dog sat on the mat .','the cat ran home .','the dog ran home .','the gpu reads the weights .','the cpu reads the weights .','the model reads the cache .','the model predicts the next token .','the model predicts the next word .','qwen reads the cache fast .','gemma reads the cache fast .','deepseek reads the cache fast .','the gpu is fast .','the cpu is slow .','the cache is big .','the cache is small .','attention reads the cache .','the layer writes the cache .','we think the model learns .','they think the model learns fast .','the sun is big .','i see the dog .','you see the cat .','the dog sat on the rug .'];
const TR={};
function trData(){
  const vocab=['<s>'];const seqs=CORPUS.map(s=>s.split(' '));seqs.flat().forEach(w=>{if(!vocab.includes(w))vocab.push(w);});
  const ex=[];seqs.forEach(s=>{const ids=[0,0,...s.map(w=>vocab.indexOf(w))];for(let i=2;i<ids.length;i++)ex.push([ids[i-2],ids[i-1],ids[i]]);});
  return {vocab,ex};
}
const ACT2={relu:{f:z=>z>0?z:0,d:z=>z>0?1:0},gelu:{f:z=>0.5*z*(1+Math.tanh(0.7978845608*(z+0.044715*z*z*z))),d:z=>TJ.dAct(z2=>0.5*z2*(1+Math.tanh(0.7978845608*(z2+0.044715*z2*z2*z2))),z)},silu:{f:z=>z/(1+Math.exp(-z)),d:z=>{const s=1/(1+Math.exp(-z));return s*(1+z*(1-s));}},tanh:{f:Math.tanh,d:z=>{const t=Math.tanh(z);return 1-t*t;}}};
const BASELR={sgd:0.3,mom:0.05,rms:0.003,adam:0.006,adamw:0.006};
const ED=12,HID=48;
function trModel(V,seed){
  const r=TJ.mulberry(seed),init=(n,s)=>Float32Array.from({length:n},()=>TJ.gauss(r)*s);
  return {E:init(V*ED,0.5),W1:init(HID*2*ED,1/Math.sqrt(2*ED)),b1:new Float32Array(HID),W2:init(V*HID,1/Math.sqrt(HID)),b2:new Float32Array(V)};
}
const DECAY={E:true,W1:true,W2:true,b1:false,b2:false};
function trForward(P,V,a,b,act){
  const x=new Float32Array(2*ED);for(let j=0;j<ED;j++){x[j]=P.E[a*ED+j];x[ED+j]=P.E[b*ED+j];}
  const z1=new Float32Array(HID),h=new Float32Array(HID);
  for(let i=0;i<HID;i++){let s=P.b1[i];for(let j=0;j<2*ED;j++)s+=P.W1[i*2*ED+j]*x[j];z1[i]=s;h[i]=act.f(s);}
  const z2=new Float32Array(V);let mx=-1e9;for(let v=0;v<V;v++){let s=P.b2[v];for(let i=0;i<HID;i++)s+=P.W2[v*HID+i]*h[i];z2[v]=s;if(s>mx)mx=s;}
  let sm=0;for(let v=0;v<V;v++){z2[v]=Math.exp(z2[v]-mx);sm+=z2[v];}for(let v=0;v<V;v++)z2[v]/=sm;
  return {x,z1,h,p:z2};
}
function trStep(o,batch,lr,wd,act,V){
  const P=o.P,G={};for(const k in P)G[k]=new Float32Array(P[k].length);let loss=0;
  for(const [a,b,y] of batch){
    const f=trForward(P,V,a,b,act);loss-=Math.log(f.p[y]+1e-9);
    const dz2=f.p;dz2[y]-=1;const dh=new Float32Array(HID);
    for(let v=0;v<V;v++){const d=dz2[v];if(!d)continue;G.b2[v]+=d;for(let i=0;i<HID;i++){G.W2[v*HID+i]+=d*f.h[i];dh[i]+=d*P.W2[v*HID+i];}}
    const dx=new Float32Array(2*ED);
    for(let i=0;i<HID;i++){const dz=dh[i]*act.d(f.z1[i]);if(!dz)continue;G.b1[i]+=dz;for(let j=0;j<2*ED;j++){G.W1[i*2*ED+j]+=dz*f.x[j];dx[j]+=dz*P.W1[i*2*ED+j];}}
    for(let j=0;j<ED;j++){G.E[a*ED+j]+=dx[j];G.E[b*ED+j]+=dx[ED+j];}
  }
  const n=batch.length,b1=0.9,b2=0.999,eps=1e-8;o.t++;
  for(const k in P){
    const p=P[k],g=G[k],m=o.m[k],v=o.v[k],dec=DECAY[k]?wd:0;
    for(let i=0;i<p.length;i++){
      let gi=g[i]/n;
      if(o.id==='sgd'){p[i]-=lr*(gi+dec*p[i]);}
      else if(o.id==='mom'){m[i]=0.9*m[i]+gi+dec*p[i];p[i]-=lr*m[i];}
      else if(o.id==='rms'){gi+=dec*p[i];v[i]=0.99*v[i]+0.01*gi*gi;p[i]-=lr*gi/(Math.sqrt(v[i])+1e-8);}
      else{if(o.id==='adam')gi+=dec*p[i];m[i]=b1*m[i]+(1-b1)*gi;v[i]=b2*v[i]+(1-b2)*gi*gi;const mh=m[i]/(1-Math.pow(b1,o.t)),vh=v[i]/(1-Math.pow(b2,o.t));p[i]-=lr*mh/(Math.sqrt(vh)+eps);if(o.id==='adamw')p[i]-=lr*dec*p[i];}
    }
  }
  return loss/n;
}
function trEval(o,act){let L=0;const dead=new Uint8Array(HID).fill(1);for(const [a,b,y] of TR.data.ex){const f=trForward(o.P,TR.V,a,b,act);L-=Math.log(f.p[y]+1e-9);for(let i=0;i<HID;i++)if(f.z1[i]>0)dead[i]=0;}return {loss:L/TR.data.ex.length,dead:dead.reduce((s,x)=>s+x,0)};}
function trSample(o,act){const out=[];let a=0,b=0;for(let i=0;i<9;i++){const f=trForward(o.P,TR.V,a,b,act);let best=1;for(let v=1;v<TR.V;v++)if(f.p[v]>f.p[best])best=v;out.push(TR.data.vocab[best]);if(TR.data.vocab[best]==='.')break;a=b;b=best;}return out.join(' ');}
function trReset(){
  TR.data=trData();TR.V=TR.data.vocab.length;TR.step=0;TR.run=false;$('#tStart').textContent='▶ Train';
  const r=TJ.mulberry(7);TR.batches=[];const total=+$('#tSteps').value;for(let s=0;s<total;s++){const b=[];for(let i=0;i<16;i++)b.push(TR.data.ex[Math.floor(r()*TR.data.ex.length)]);TR.batches.push(b);}
  TR.opts=OPTS.map(op=>{const P=trModel(TR.V,123),m={},v={};for(const k in P){m[k]=new Float32Array(P[k].length);v[k]=new Float32Array(P[k].length);}return Object.assign({},op,{P,m,v,t:0,hist:[],evalL:null,dead:0});});
  TR.opts.forEach(o=>{const e=trEval(o,ACT2[$('#tAct').value]);o.evalL=e.loss;o.dead=e.dead;o.hist.push([0,e.loss]);});
  trDraw();
}
function trTick(){
  if(TR.run){
    const total=+$('#tSteps').value,act=ACT2[$('#tAct').value],mult=Math.pow(10,(+$('#tLr').value-50)/50),wd=+$('#tWd').value,k=$('#tSched').value;
    for(let s=0;s<6&&TR.step<total;s++){
      const batch=TR.batches[TR.step];
      TR.opts.forEach(o=>{const lr=sched(TR.step,total,k,Math.round(total*0.05),BASELR[o.id]*mult);trStep(o,batch,lr,wd,act,TR.V);});
      TR.step++;
      if(TR.step%20===0||TR.step===total)TR.opts.forEach(o=>{const e=trEval(o,act);o.evalL=e.loss;o.dead=e.dead;o.hist.push([TR.step,e.loss]);});
    }
    if(TR.step>=total){TR.run=false;$('#tStart').textContent='▶ Train';}
    trDraw();
  }
  requestAnimationFrame(trTick);
}
function trDraw(){
  const cv=$('#tCanvas'),g=cv.getContext('2d'),W=cv.width,H=cv.height,total=+$('#tSteps').value;g.clearRect(0,0,W,H);
  const maxL=Math.log(TR.V)*1.15;
  g.strokeStyle=css('--line');g.lineWidth=1;g.beginPath();g.moveTo(44,10);g.lineTo(44,H-26);g.lineTo(W-10,H-26);g.stroke();
  g.fillStyle=css('--muted');g.font='11px '+css('--mono');g.fillText('loss',4,16);g.fillText(maxL.toFixed(1),8,30);g.fillText('0',30,H-28);g.fillText('step '+total,W-70,H-8);
  const ln=Math.log(TR.V);const yr=H-26-ln/maxL*(H-36);g.setLineDash([4,4]);g.strokeStyle=css('--muted');g.beginPath();g.moveTo(44,yr);g.lineTo(W-10,yr);g.stroke();g.setLineDash([]);g.fillText('random guessing = ln '+TR.V,50,yr-4);
  TR.opts.forEach(o=>{g.strokeStyle=css(o.c);g.lineWidth=2.2;g.beginPath();o.hist.forEach(([s,L],i)=>{const x=44+s/total*(W-54),y=H-26-Math.min(L,maxL)/maxL*(H-36);i?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();});
  const act=ACT2[$('#tAct').value];
  $('#tTable').innerHTML=`<tr><th>Optimizer</th><th>Loss (whole set)</th><th>Dead neurons</th><th>Greedy text from "&lt;s&gt;"</th></tr>`+TR.opts.map(o=>`<tr><td><i class="sw" style="background:var(${o.c})"></i>${o.name}</td><td class="num">${o.evalL.toFixed(3)}</td><td class="num">${$('#tAct').value==='relu'?o.dead+' / '+HID:'n/a'}</td><td class="mono">${TR.step?trSample(o,act):'–'}</td></tr>`).join('');
  $('#tStep').textContent=`step ${TR.step} / ${total} · ${TR.data.ex.length} training examples · vocabulary ${TR.V}`;
}
function trInit(){
  $('#tStart').onclick=()=>{if(TR.step>=+$('#tSteps').value)trReset();TR.run=!TR.run;$('#tStart').textContent=TR.run?'❚❚ Pause':'▶ Train';};
  $('#tReset').onclick=trReset;['#tAct','#tSteps'].forEach(s=>$(s).addEventListener('change',trReset));
  const o=()=>{$('#tLrO').textContent='×'+Math.pow(10,(+$('#tLr').value-50)/50).toPrecision(2);$('#tWdO').textContent=(+$('#tWd').value).toFixed(3);};
  ['#tLr','#tWd'].forEach(s=>$(s).addEventListener('input',o));o();
  trReset();requestAnimationFrame(trTick);
}

TJ.initLabs=function(){try{actInit();}catch(e){console.error(e);}try{optInit();}catch(e){console.error(e);}try{trInit();}catch(e){console.error(e);}};
TJ.relabs=function(){if(AS.S)actBuild();if(OS.S){optBuild();optReset();}trDraw();};
})();
