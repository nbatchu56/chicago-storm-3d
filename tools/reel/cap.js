// Films LakeView: the page's frame loop is driven by hand on a virtual clock (1/FPS s per frame), each frame read
// straight off the WebGL canvas; the UI and the caption are captured once per scene as a transparent overlay.
const {chromium}=require('playwright'); const fs=require('fs');
const FPS=15, W=960, H=540;
const SCENES=[
 {name:'open', dur:3.5, ui:'title', ov:'natural', t:'2026-11-02T20:30:00Z',
  a:[41.9203,-87.6195,1.58,284,34], b:[41.9210,-87.6190,1.30,292,31], cap:"Chicago's lakefront, live — wind, waves and weather for paddlers"},
 {name:'flow', dur:3.0, ui:'none', ov:'flow', t:'2026-11-02T19:00:00Z',
  a:[41.930,-87.600,3.5,200,40], b:[41.930,-87.600,3.5,216,42], cap:"Wave energy flow — SWAN and FUNWAVE over the lake bottom"},
 {name:'wind', dur:3.0, ui:'none', ov:'natural', t:'2026-11-02T21:00:00Z', wind:true,
  a:[41.9230,-87.6150,2.4,300,36], b:[41.9230,-87.6150,2.2,308,38], cap:"Wind and gusts from NOAA HRRR, streaking over the water"},
 {name:'river', dur:3.0, ui:'none', ov:'natural', t:'2026-11-02T19:30:00Z', drone:true,
  a:[41.8890,-87.6010,0.9,270,42], b:[41.8884,-87.6160,0.9,270,42], cap:"The Chicago River — the current, the lock, the overflows"},
 {name:'wildmile', dur:3.0, ui:'none', ov:'natural', t:'2026-11-02T20:00:00Z', drone:true,
  a:[41.9015,-87.6490,0.55,350,40], b:[41.9040,-87.6488,0.5,5,42], cap:"The North Branch and the Wild Mile — every launch on the map"},
 {name:'launch', dur:3.5, ui:'full', ov:'natural', t:'2026-11-02T20:00:00Z',
  a:[41.92654,-87.63285,0.12,51,30], b:[41.92654,-87.63285,0.17,58,30], cap:"A paddling light for every launch, from the buoys and the models", capTop:true},
 {name:'night', dur:3.0, ui:'none', ov:'natural', t:'2026-11-02T23:15:00Z',
  a:[41.9120,-87.6120,1.9,262,15], b:[41.9120,-87.6120,1.7,254,17], cap:"Day and night, past storms, the next ten days"},
];
(async()=>{
const only=process.argv.slice(2);
const b=await chromium.launch({args:['--use-angle=swiftshader','--enable-webgl','--ignore-gpu-blocklist','--enable-unsafe-swiftshader']});
const p=await b.newPage({viewport:{width:W,height:H}});
await p.addInitScript(()=>{try{localStorage.setItem('lakeworld-q','0');localStorage.setItem('lakeworld-hint','1')}catch(e){}});
p.on('pageerror',e=>console.log('ERR',e.message.slice(0,200)));
await p.goto('http://127.0.0.1:8123/world.html',{waitUntil:'load',timeout:120000});
await p.waitForFunction(()=>window.__world&&__world.MAP&&__world.MAP.ok,{timeout:150000}).catch(()=>console.log('map not ok'));
await p.waitForTimeout(6000);
await p.addStyleTag({content:`
 #cap{position:absolute;left:50%;transform:translateX(-50%);bottom:42px;z-index:9;font:500 20px/1.3 var(--font-ui);color:#eef3f7;background:rgba(9,15,21,.72);padding:8px 18px;border-radius:8px;white-space:nowrap;letter-spacing:.01em;box-shadow:0 4px 24px rgba(0,0,0,.35)}
 #cap.top{bottom:auto;top:16px;left:auto;right:16px;transform:none}
 body.film #timeline,body.film #imm,body.film #hint,body.film #nav,body.film #toggle,body.film #legend,body.film #bar,body.film #hud,body.film #labels,body.film #attr,body.film #controls{display:none!important}
 body.film.title #hud{display:grid!important} body.film.title #hud > :not(h1){display:none!important} body.film.title #hud h1{transform:scale(1.2);transform-origin:top left;margin:8px 0 0 8px}
 body.film.full #bar,body.film.full #hud,body.film.full #labels,body.film.full #attr{display:revert!important}
 body.ovl{background:transparent!important} body.ovl canvas#gl,body.ovl #kfade{visibility:hidden!important}`});
// the loop in my hands: rAF callbacks queue up and run when I say; the clock moves 1/FPS s per frame
await p.evaluate((FPS)=>{const W=__world;W.S.follow=false;W.S.playing=false;document.body.classList.add('film');const h=document.querySelector('#hint');if(h)h.hidden=true;
  window.__q=[];window.requestAnimationFrame=(f)=>{window.__q.push(f);return window.__q.length;};window.cancelAnimationFrame=()=>{};
  window.__vt=performance.now();performance.now=()=>window.__vt;
  window.__step=(n)=>{for(let i=0;i<n;i++){window.__vt+=1000/FPS;const q=window.__q;window.__q=[];for(const f of q)f(window.__vt);}};},FPS);
await p.waitForTimeout(500); await p.evaluate(()=>window.__step(3));
for(const sc of SCENES){ if(only.length&&!only.includes(sc.name)) continue;
  const dir=`frames/${sc.name}`; fs.rmSync(dir,{recursive:true,force:true}); fs.mkdirSync(dir,{recursive:true});
  await p.evaluate((sc)=>{const W=__world;document.body.classList.remove('title','full','ovl');if(sc.ui!=='none')document.body.classList.add(sc.ui);
    document.querySelector(`[data-ov="${sc.ov}"]`).click();W.S.t=Date.parse(sc.t);W.DRONE.on=!!sc.drone;W.setWindOn(!!sc.wind);
    let c=document.querySelector('#cap');if(!c){c=document.createElement('div');c.id='cap';document.querySelector('#app').appendChild(c);}c.textContent=sc.cap;c.className=sc.capTop?'top':'';
    const [la,lo,d,h,pp]=sc.a,[x,y]=W.ll(la,lo);W.flyTo(x,y,d,h,pp,true);W.DRONE.placed=true;},sc);
  await p.evaluate((n)=>window.__step(n),sc.drone?30:24);
  const n=process.env.TEST?3:Math.round(sc.dur*FPS), t0=Date.now();
  for(let i=0;i<n;i++){const f=i/(n-1), e=f*f*(3-2*f);
    const d=await p.evaluate(([sc,e])=>{const W=__world,L=(a,b)=>a+(b-a)*e;const [la,lo,d,h,pp]=[0,1,2,3,4].map(k=>L(sc.a[k],sc.b[k]));const [x,y]=W.ll(la,lo);W.flyTo(x,y,d,h,pp,true);W.DRONE.placed=true;
      let img;for(let k=0;k<3;k++){window.__step(1);img=document.querySelector('canvas#gl').toDataURL('image/jpeg',0.92);if(img.length>12000)break;}return img;},[sc,e]);
    fs.writeFileSync(`${dir}/f${String(i).padStart(4,'0')}.jpg`,Buffer.from(d.split(',')[1],'base64'));}
  console.log(sc.name,n,'frames',((Date.now()-t0)/1000/n).toFixed(1)+'s per frame');
  // the overlay: the UI and the caption over a transparent page, at the scene's last camera
  await p.evaluate(()=>document.body.classList.add('ovl'));
  await p.screenshot({path:`${dir}/overlay.png`,omitBackground:true,timeout:240000});
  await p.evaluate(()=>document.body.classList.remove('ovl'));
  console.log(sc.name,'overlay done');}
await b.close();})().catch(e=>{console.log(e);process.exit(1)});
