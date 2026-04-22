

const API_BASE = ""; // ปล่อยว่างถ้า frontend กับ backend โฮสต์โดเมนเดียวกัน
const NGROK_HDR = { "ngrok-skip-browser-warning": "any" };
const NGROK_QS  = "ngrok-skip-browser-warning=1";

/* -------------------- UTILITIES -------------------- */
function $(id){ return document.getElementById(id); }
const sleep = (ms)=>new Promise(r=>setTimeout(r,ms));
const now   = ()=>Date.now();
const addQS = (u)=> u+(u.includes("?")?"&":"?")+NGROK_QS;
const setDis = (el, v)=> el && el.setAttribute("aria-disabled", v ? "true" : "false");
function fmt(t){
  if (!isFinite(t) || t < 0) t = 0;
  const h=Math.floor(t/3600), m=Math.floor((t%3600)/60), s=Math.floor(t%60);
  const z=n=>n.toString().padStart(2,'0');
  return h>0?`${h}:${z(m)}:${z(s)}`:`${z(m)}:${z(s)}`;
}
function parseK(raw, fallback=6){
  const s = String(raw??"").trim();
  const m = s.match(/(\d+)/);
  const k = m ? +m[1] : +s;
  if (!Number.isFinite(k)) return fallback;
  return Math.min(9, Math.max(1, k));
}
function getInitK(){
  try{
    const d = JSON.parse(sessionStorage.getItem("sketch_result")||"{}");
    return parseK(d.k, 6); // <- K แรกที่เลือกจากหน้าแรก
  }catch(_){ return 6; }
}

/* -------------------- INDEX (UPLOAD) -------------------- */
/* ใช้ในหน้า index.html: ปุ่มอัปโหลดควรเรียก window.mockUpload() */
window.mockUpload = async function () {
  const f = document.getElementById("imageInput")?.files?.[0];
  if (!f) return alert("Please select an image first.");

  // เก็บเมตาไว้ให้หน้า result
  sessionStorage.setItem("orig_name", f.name);
  try{
    const objUrl = URL.createObjectURL(f);
    await new Promise((res)=>{
      const im = new Image();
      im.onload = () => {
        sessionStorage.setItem("orig_w", String(im.naturalWidth||im.width||0));
        sessionStorage.setItem("orig_h", String(im.naturalHeight||im.height||0));
        try{ URL.revokeObjectURL(objUrl); }catch(_){}
        res();
      };
      im.onerror = () => { try{ URL.revokeObjectURL(objUrl); }catch(_){ } res(); };
      im.src = objUrl;
    });
  }catch(_){}

  const status = document.getElementById("status");
  if (status) { status.style.display = "inline-block"; status.textContent = "Uploading & Processing…"; }

  // K จากหน้าแรก (เช่น hidden input id="colorCount" ที่ผูกกับ UI เลือกสี)
  const kHidden = document.getElementById("colorCount");
  const k = (kHidden?.value || "6");

  try {
    const fd = new FormData();
    fd.append("file", f);
    fd.append("k", k);
    const res = await fetch(`${API_BASE}/api/upload`, { method:"POST", body:fd, headers:NGROK_HDR });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json(); // { job_id, k, ... }
    sessionStorage.setItem("sketch_result", JSON.stringify(data));

    // ส่งรูปพรีวิวไปหน้า result
    const r = new FileReader();
    r.onload = e => { localStorage.setItem("previewURL", e.target.result); location.href = "result.html?v=2"; };
    r.readAsDataURL(f);
  } catch (e) {
    console.error(e); alert("Upload failed.");
  } finally {
    if (status) status.style.display = "none";
  }
};

/* -------------------- RESULT (PLAYER + REPROCESS) -------------------- */
(function () {
  // ถ้าไม่ใช่หน้า result.html ก็ไม่ต้องรันส่วนนี้
  if (!$("#resultVideo")) return;

  console.log("SketchColor result.js v2025-11-14-2");

  let currentK = null;        // K ที่กำลังแสดงอยู่
  let selectedK = null;       // K ที่เลือกในเมนู แต่ยังไม่ reprocess
  let currentObjectURL = null;

  function revokeCurrentURL(){
    try{ if (currentObjectURL){ URL.revokeObjectURL(currentObjectURL); currentObjectURL=null; } }catch(_){}
  }
  function getActiveK(){
    // ลำดับความสำคัญ: currentK (แสดงอยู่) > selectedK (เลือกค้างในเมนู) > K แรกจากหน้าแรก
    return typeof currentK==="number"&&currentK ? currentK
         : typeof selectedK==="number"&&selectedK ? selectedK
         : getInitK();
  }

  // helper แสดงพาเลตสี
  function hexToRgb(hex){ const m=/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex||""); return m?{r:parseInt(m[1],16),g:parseInt(m[2],16),b:parseInt(m[3],16)}:{r:0,g:0,b:0}; }
  function rgbToHsl(r,g,b){ r/=255;g/=255;b/=255; const max=Math.max(r,g,b),min=Math.min(r,g,b); let h,s,l=(max+min)/2; if(max===min){h=0;s=0}else{const d=max-min; s=l>0.5?d/(2-max-min):d/(max+min); switch(max){case r:h=(g-b)/d+(g<b?6:0);break;case g:h=(b-r)/d+2;break;case b:h=(r-g)/d+4;break} h*=60} return {h,s,l}; }
  function tonePrefix(l){ if(l<=0.20)return"Very Dark "; if(l<=0.28)return"Dark "; if(l>=0.85)return"Very Light "; if(l>=0.72)return"Light "; return""; }
  function prettyColorName(hex){ const {r,g,b}=hexToRgb(hex); const {h,s,l}=rgbToHsl(r,g,b); const v=Math.max(r,g,b)/255;
    if(v<0.10)return"Black"; if(s<0.10){ if(l<0.25)return"Dark Gray"; if(l<0.60)return"Gray"; if(l<0.90)return"Light Gray"; return"White"; }
    if(l>0.93&&s<0.25)return"White";
    if(h>=15&&h<45&&l<0.65)return tonePrefix(l)+"Brown";
    if(h>=45&&h<95&&l<0.60)return tonePrefix(l)+"Olive";
    let base; if(h<12||h>=348)base="Red"; else if(h<28)base="Orange"; else if(h<45)base="Yellow"; else if(h<66)base="Yellow Green"; else if(h<150)base="Green"; else if(h<175)base="Teal"; else if(h<195)base="Cyan"; else if(h<215)base="Sky Blue"; else if(h<235)base="Blue"; else if(h<260)base="Indigo"; else if(h<290)base="Purple"; else if(h<330)base="Magenta"; else base="Pink";
    return tonePrefix(l)+base;
  }
  async function renderPalette(job_id, colorsFromCache){
    const ul = $("palette"); if (!ul) return;
    ul.innerHTML = "";
    const colors = Array.isArray(colorsFromCache) ? colorsFromCache : null;
    if (colors){
      colors.forEach(hex=>{
        const li=document.createElement("li");
        li.className="swatch"; li.style.background=hex;
        const label=`${prettyColorName(hex)} ${hex}`;
        li.title=label; li.setAttribute("aria-label",label);
        ul.appendChild(li);
      });
      return;
    }
    try{
      const res = await fetch(`${API_BASE}/api/colors/${job_id}?_=${now()}`, {headers:NGROK_HDR, cache:"no-store"});
      if (!res.ok) return;
      const data = await res.json();
      (data.colors||[]).forEach(hex=>{
        const li=document.createElement("li");
        li.className="swatch"; li.style.background=hex;
        const label=`${prettyColorName(hex)} ${hex}`;
        li.title=label; li.setAttribute("aria-label",label);
        ul.appendChild(li);
      });
    }catch(e){ console.log("palette err", e); }
  }

  // รอ job พร้อม
  async function waitJobReady(job_id){
    while(true){
      try{
        const s = await fetch(`${API_BASE}/api/job/${job_id}?_=${now()}`, {headers:NGROK_HDR, cache:"no-store"}).then(r=>r.json());
        if (s.ready && s.result_url) return s;
      }catch(_){}
      await sleep(900);
    }
  }

  // ทำ snapshot (ให้มี result_k{K}.mp4 / final_k{K}.jpg) ฝั่งเซิร์ฟเวอร์
  async function cacheNow(job_id, k){
    try{ await fetch(`${API_BASE}/api/cache/${job_id}?k=${k}&_=${now()}`, { headers: NGROK_HDR, cache:"no-store" }); }catch(_){}
  }

  // พยายามโหลดวิดีโอ K ที่เคยสร้างไว้จากเซิร์ฟเวอร์ (โดยไม่ต้อง process ใหม่)
  async function tryLoadCached(job_id, k){
    try{
      const url = `${API_BASE}/api/video_k/${job_id}?k=${k}&${NGROK_QS}&_=${now()}`;
      const r   = await fetch(url, { headers: NGROK_HDR, cache: "no-store" });
      if (!r.ok) return false;
      const blob = await r.blob();
      const newURL = URL.createObjectURL(blob);
      revokeCurrentURL();
      currentObjectURL = newURL;
      $("resultSrc").src = currentObjectURL;
      $("resultVideo").load();

      // palette เฉพาะ K
      let palColors = [];
      try{
        const pc = await fetch(`${API_BASE}/api/colors_k/${job_id}?k=${k}&_=${now()}`, { headers: NGROK_HDR, cache: "no-store" }).then(x=>x.json());
        palColors = pc.colors || [];
      }catch(_){}
      renderPalette(job_id, palColors);

      currentK = k;
      enableAll(true);
      return true;
    }catch(_){
      return false;
    }
  }

  // เรียก reprocess (เมื่อเปลี่ยน K ที่ไม่มี cache)
  async function reprocess(job_id, k, hooks){
    const { onStart, onSwap } = hooks || {};
    try{
      onStart && onStart();
      const fd = new FormData(); fd.append("k", String(k));
      const r = await fetch(`${API_BASE}/api/reprocess/${job_id}`, { method:"POST", body:fd, headers:NGROK_HDR });
      if (!r.ok) throw new Error(await r.text());
      const st = await waitJobReady(job_id);
      onSwap && onSwap(st);
    }catch(e){
      console.error(e); alert("Re-process failed.");
      throw e;
    }
  }

  // UI helpers
  function setKBtn(k){ const reBtn=$("reColorBtn"); if (reBtn) reBtn.textContent = `Number of Colors: ${k} ▾`; }
  function setLoadingUI(on){
    const frame=$("processedFrame");
    const btnVid=$("downloadVideoBtn"), btnImg=$("downloadImageBtn");
    const btnPause=$("toggleBtn"), btnBack=$("skipBackBtn"), btnFwd=$("skipFwdBtn");
    const seekBar=$("seekBar"), speedSel=$("speedSel");
    const reBtn=$("reprocessBtn");
    if (on){
      frame?.classList.add("loading");
      setDis(reBtn,true); setDis(btnVid,true); setDis(btnImg,true);
      setDis(btnPause,true); setDis(btnBack,true); setDis(btnFwd,true);
      if (seekBar) seekBar.disabled = true; if (speedSel) speedSel.disabled = true;
    }else{
      frame?.classList.remove("loading");
      setDis(reBtn,false); setDis(btnVid,false); setDis(btnImg,false);
      setDis(btnPause,false); setDis(btnBack,false); setDis(btnFwd,false);
      if (seekBar) seekBar.disabled = false; if (speedSel) speedSel.disabled = false;
    }
  }
  function enableAll(playable){
    const frame=$("processedFrame");
    const btnVid=$("downloadVideoBtn"), btnImg=$("downloadImageBtn");
    const btnPause=$("toggleBtn"), btnBack=$("skipBackBtn"), btnFwd=$("skipFwdBtn");
    const seekBar=$("seekBar"), speedSel=$("speedSel");
    frame?.classList.remove("loading");
    setDis(btnVid,false); setDis(btnImg,false);
    setDis(btnPause,!playable); setDis(btnBack,!playable); setDis(btnFwd,!playable);
    if (seekBar) seekBar.disabled = !playable;
    if (speedSel) speedSel.disabled = !playable;
  }

  async function initResult(){
    const frame=$("processedFrame"), video=$("resultVideo"), source=$("resultSrc"), orig=$("originalImage");
    const meta=$("origMeta"), btnPause=$("toggleBtn"), btnBack=$("skipBackBtn"), btnFwd=$("skipFwdBtn");
    const seekBar=$("seekBar"), timeLabel=$("timeLabel"), speedSel=$("speedSel");
    const btnVid=$("downloadVideoBtn"), btnImg=$("downloadImageBtn");
    const reWrap=$("reColorWrap"), reBtn=$("reColorBtn"), reMenu=$("reColorMenu"), reProcess=$("reprocessBtn");

    // ปิด autoplay/loop ชัวร์
    if (video){
      video.removeAttribute('autoplay'); video.removeAttribute('loop');
      video.autoplay=false; video.loop=false; video.setAttribute('playsinline',''); video.setAttribute('webkit-playsinline','');
    }

    // ป้องกันเล่นเองหลังจบ
    let hasEnded=false, lastUserGesture=0;
    ['click','keydown','touchend'].forEach(ev=>document.addEventListener(ev,()=>{lastUserGesture=Date.now()},{capture:true}));
    video?.addEventListener('ended',e=>{e.stopImmediatePropagation();hasEnded=true;try{video.pause();const d=video.duration||0;if(d>0)video.currentTime=Math.max(0,d-0.001)}catch(_){} setPlayLabel(true)},{capture:true});
    video?.addEventListener('play',e=>{const g=(Date.now()-lastUserGesture)<=600;if(hasEnded&&!g){e.stopImmediatePropagation();video.pause();}},{capture:true});

    // พรีวิวเดิมฝั่งซ้าย
    const pv = localStorage.getItem("previewURL"); if (pv && orig) orig.src = pv;
    const name = sessionStorage.getItem("orig_name") || "image";
    const w = +(sessionStorage.getItem("orig_w")||0), h=+(sessionStorage.getItem("orig_h")||0);
    const MAX_SIDE = 300;

    if (meta){
      let sz = "";
      let proc = "";

      if (w > 0 && h > 0){
        // ขนาดต้นฉบับ
        sz = ` size ${w}×${h} `;

        // คิดขนาดหลังย่อให้ด้านยาวสุด = MAX_SIDE (แต่ไม่ขยายเกินของเดิม)
        const ratio = Math.min(1, MAX_SIDE / Math.max(w, h));
        const pw = Math.round(w * ratio);
        const ph = Math.round(h * ratio);

        proc = `· processed ${pw}×${ph}px`;
      } else {
        proc = `· processed at max side ${MAX_SIDE}px`;
      }

      meta.textContent = `${name}${sz} ${proc}`;
    }

    // ข้อมูลงาน
    const saved = sessionStorage.getItem("sketch_result");
    if (!saved){ alert("Please upload an image first."); location.href="index.html"; return; }
    const initData = JSON.parse(saved); const { job_id } = initData;

    const setPlayLabel = paused => { if (btnPause) btnPause.textContent = paused ? "▶ Play" : "⏸ Pause"; };

    // ปุ่มดาวน์โหลด (ผูกกับ K ที่ใช้อยู่)
    btnVid?.addEventListener("click", async (ev)=>{
      if (btnVid.getAttribute("aria-disabled")==="true") return ev.preventDefault();
      ev.preventDefault();
      const k = getActiveK();
      try{
        const url = addQS(`${API_BASE}/api/video_k/${job_id}?k=${k}&download=1&_=${now()}`);
        const r   = await fetch(url, { headers: NGROK_HDR, cache:"no-store" });
        if (!r.ok){
          alert(`ยังไม่มีวิดีโอของ K${k} ให้กด Process ให้เสร็จก่อน แล้วค่อยดาวน์โหลด`);
          return;
        }
        const b = await r.blob(); const o = URL.createObjectURL(b);
        const a = document.createElement("a"); a.href = o; a.download = `sketch_${job_id}_K${k}.mp4`;
        document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(o);
      }catch(e){ console.error(e); alert("Cannot download MP4."); }
    });

    btnImg?.addEventListener("click", async (ev)=>{
      if (btnImg.getAttribute("aria-disabled")==="true") return ev.preventDefault();
      ev.preventDefault();
      const k = getActiveK();
      try{
        // พยายามไฟล์เฉพาะ K ก่อน
        let url = addQS(`${API_BASE}/static/${job_id}/final_k${k}.jpg?_=${now()}`);
        let r   = await fetch(url, { headers: NGROK_HDR, cache:"no-store" });
        if (!r.ok){
          // fallback เป็นรูปล่าสุด (ไม่ระบุ K)
          url = addQS(`${API_BASE}/api/image/${job_id}?keep=1&_=${now()}`);
          r   = await fetch(url, { headers: NGROK_HDR, cache:"no-store" });
          if (!r.ok){
            alert(`ยังไม่มีรูปของ K${k} ให้กด Process ให้เสร็จก่อน แล้วค่อยดาวน์โหลด`);
            return;
          }
        }
        const b = await r.blob(); const o = URL.createObjectURL(b);
        const a = document.createElement("a"); a.href = o; a.download = `sketch_${job_id}_K${k}.jpg`;
        document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(o);
      }catch(e){ console.error(e); alert("Cannot download JPG."); }
    });

    // ปุ่มเล่น/หยุด/ข้าม/ความเร็ว
    btnPause && (btnPause.onclick = ()=>{
      if (btnPause.getAttribute("aria-disabled")==="true") return;
      if (video.paused){
        video.play().catch(()=>{});
        setPlayLabel(false);
      } else {
        video.pause();
        setPlayLabel(true);
      }
    });
    const skip = d => { try{ const t=Math.max(0,Math.min((video.duration||0),(video.currentTime||0)+d)); video.currentTime=t; }catch(_){} };
    btnBack && (btnBack.onclick = ()=>{ if (btnBack.getAttribute("aria-disabled")==="true") return; skip(-10); });
    btnFwd  && (btnFwd .onclick = ()=>{ if (btnFwd .getAttribute("aria-disabled")==="true") return; skip(+10); });
    speedSel && (speedSel.onchange=()=>{ const v=parseFloat(speedSel.value||"1"); video.playbackRate=(isFinite(v)&&v>0)?v:1; });

    // เวลา/seekbar
    let scrubbing=false;
    const updateTime = ()=>{ const cur=video.currentTime||0, dur=video.duration||0;
      if (timeLabel) timeLabel.textContent=`${fmt(cur)} / ${fmt(dur)}`;
      if (!scrubbing && dur>0 && seekBar){ seekBar.value=Math.round((cur/dur)*1000); }
    };
    video.addEventListener("timeupdate", updateTime);
    video.addEventListener("durationchange", updateTime);
    seekBar && seekBar.addEventListener("input", ()=>{ const dur=video.duration||0; if (dur>0){ scrubbing=true; const t=(+seekBar.value/1000)*dur; if (timeLabel) timeLabel.textContent=`${fmt(t)} / ${fmt(dur)}`; video.currentTime=t; }});
    seekBar && seekBar.addEventListener("change", ()=>{ scrubbing=false; });

    // คีย์ลัด j(-10) k(play/pause) l(+10)
    window.addEventListener("keydown", e=>{
      if (document.activeElement && ['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)) return;
      const k=e.key.toLowerCase();
      if (k==='j') skip(-10);
      else if (k==='l') skip(+10);
      else if (k==='k'){
        if (!video.paused) video.pause();
        else video.play().catch(()=>{});
        setPlayLabel(video.paused);
      }
    });

    // โหลดผลครั้งแรก → ตั้ง currentK = K แรกจริง และ snapshot ไว้
    (async ()=>{
      const s = await waitJobReady(job_id);
      const resURL = `${API_BASE}${s.result_url}?${NGROK_QS}&_=${now()}`;
      const blob = await fetch(resURL, {headers:NGROK_HDR, cache:"no-store"}).then(r=>r.blob());

      const newURL = URL.createObjectURL(blob);
      revokeCurrentURL();
      currentObjectURL = newURL;
      $("resultSrc").src = currentObjectURL;
      video.load();

      let palColors=[];
      try{
        const pal = await fetch(`${API_BASE}/api/colors/${job_id}?_=${now()}`, {headers:NGROK_HDR, cache:"no-store"}).then(r=>r.json());
        palColors = pal.colors || [];
      }catch(_){}
      renderPalette(job_id, palColors);

      // **จุดสำคัญ**: ตั้ง currentK = K แรก และสแน็ปช็อตไว้ฝั่งเซิร์ฟเวอร์
      currentK = getInitK();
      await cacheNow(job_id, currentK);

      setTimeout(()=>{ enableAll(true); setPlayLabel(true); }, 400);
    })();

    // ----- เมนู Re-Color (เลือก K แล้วกด Process) -----
    selectedK = getInitK(); setKBtn(selectedK);
    reBtn?.addEventListener('click', e=>{
      e.preventDefault();
      const opened=reWrap.classList.toggle('open');
      reBtn.setAttribute('aria-expanded', opened ? 'true' : 'false');
    });
    document.addEventListener('click', e=>{
      if (reWrap && !reWrap.contains(e.target)){
        reWrap.classList.remove('open'); reBtn?.setAttribute('aria-expanded','false');
      }
    });
    [...(reMenu?.querySelectorAll('a[data-k]')||[])].forEach(a=>a.addEventListener('click', e=>{
      e.preventDefault();
      const raw = a.getAttribute('data-k') || a.textContent || '6';
      selectedK = parseK(raw, getInitK());
      setKBtn(selectedK);
      reWrap.classList.remove('open'); reBtn?.setAttribute('aria-expanded','false');
    }));

    // ปุ่ม Process
    $("reprocessBtn")?.addEventListener('click', async ()=>{
      const job_id = JSON.parse(sessionStorage.getItem("sketch_result")||"{}").job_id;
      const targetK = selectedK ?? getInitK();
      if (targetK === currentK) return;

      // 1) ลองดึง cache K จากเซิร์ฟเวอร์ (ถ้ามีจะเร็ว ไม่ต้อง process)
      setLoadingUI(true);
      const ok = await tryLoadCached(job_id, targetK);
      if (ok){ setLoadingUI(false); return; }

      // 2) ไม่มี cache → reprocess
      await reprocess(job_id, targetK, {
        onStart: ()=> setLoadingUI(true),
        onSwap: (st)=>{
          (async ()=>{
            const url = `${API_BASE}${st.result_url}?${NGROK_QS}&_=${now()}`;
            const blob = await fetch(url, {headers:NGROK_HDR, cache:"no-store"}).then(r=>r.blob());
            const obj = URL.createObjectURL(blob);
            revokeCurrentURL();
            currentObjectURL = obj;
            $("resultSrc").src = currentObjectURL;
            $("resultVideo").load();

            const onOk = async ()=>{
              let palColors=[];
              try{
                const pal = await fetch(`${API_BASE}/api/colors/${job_id}?_=${now()}`, {headers:NGROK_HDR, cache:"no-store"}).then(r=>r.json());
                palColors = pal.colors || [];
              }catch(_){}
              renderPalette(job_id, palColors);

              currentK = targetK;              // <- สลับ K สำเร็จ
              await cacheNow(job_id, targetK); // <- เก็บ snapshot final_k{K}.jpg/result_k{K}.mp4
              enableAll(true);
              setLoadingUI(false);
              $("resultVideo").removeEventListener("loadeddata", onOk);
            };
            $("resultVideo").addEventListener("loadeddata", onOk);
          })();
        }
      });
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initResult, { once:true });
  else initResult();
})();
