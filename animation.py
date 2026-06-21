# -*- coding: utf-8 -*-
"""
animation.py
============
브라우저에서 도는 **실시간 3D 비행 애니메이션** 컴포넌트 (Three.js).

- 서버(Streamlit) 재실행 없이 클라이언트(WebGL)에서 60fps 로 렌더 → 부드럽고 랙 없음
- 마우스 드래그=시점 회전, 휠=확대
- 지면 그리드 위에서 pitch·roll·yaw 가 실제 3D 자세로 움직임
- 초기 자세는 반투명(고스트), 현재 자세는 진하게 표시
- 재생/일시정지/정지/속도/반복 컨트롤은 모두 JS 안에서 동작
- **사용자가 STL 모델을 업로드하면 그 모델을 비행기로 사용**하고,
  업로드 후 **크기·정렬(회전)을 컴포넌트 안에서 실시간 조절** 가능
"""

from __future__ import annotations
import json
from simulation import SimResult


def realtime_animation_html(res: SimResult, vtail_count: int = 1,
                            stl_b64: str = "") -> str:
    """pitch/roll/yaw 시계열(+선택적 STL)을 JS 로 넘겨 3D 애니메이션 HTML 생성."""
    t = res.t
    n = len(t)
    step = max(1, n // 1500)
    def arr(a):
        return [round(float(v), 3) for v in a[::step]]
    data = {
        "pitch": arr(res.pitch), "roll": arr(res.roll), "yaw": arr(res.yaw),
        "aoa": arr(res.aoa), "t": arr(t),
        "pitch0": float(res.pitch0), "roll0": float(res.roll0), "yaw0": float(res.yaw0),
        "dt": float(t[1] - t[0]) * step,
        "vtail": int(vtail_count),
    }
    # STL 은 데이터가 클 수 있어 본문과 분리해 치환
    return (_TEMPLATE
            .replace("__DATA__", json.dumps(data))
            .replace("__STL_B64__", json.dumps(stl_b64 or "")))


_TEMPLATE = r"""
<div id="acwrap" style="font-family:'Segoe UI',sans-serif;color:#1a2330;">
  <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:2px 0 6px;">
    <button id="ac_play"  style="padding:6px 14px;border:0;border-radius:8px;background:#1f7ae0;color:#fff;font-weight:600;cursor:pointer;">&#9654; 재생</button>
    <button id="ac_pause" style="padding:6px 14px;border:0;border-radius:8px;background:#6b7785;color:#fff;font-weight:600;cursor:pointer;">&#9208; 일시정지</button>
    <button id="ac_stop"  style="padding:6px 14px;border:0;border-radius:8px;background:#d64545;color:#fff;font-weight:600;cursor:pointer;">&#9209; 정지</button>
    <label style="margin-left:8px;font-size:13px;">속도
      <select id="ac_speed" style="padding:3px;border-radius:6px;">
        <option value="0.25">0.25&times;</option><option value="0.5">0.5&times;</option>
        <option value="1" selected>1&times;</option><option value="2">2&times;</option>
        <option value="4">4&times;</option>
      </select>
    </label>
    <label style="font-size:13px;"><input type="checkbox" id="ac_loop" checked> 반복</label>
    <label style="font-size:13px;"><input type="checkbox" id="ac_ghost" checked> 초기자세</label>
    <span id="ac_hint" style="margin-left:auto;font-size:12px;color:#6b7785;">드래그=회전 · 휠=확대</span>
  </div>
  <div id="ac_modelbar" style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:0 0 8px;padding:6px 8px;background:#eef3f9;border-radius:8px;font-size:12px;">
    <b>모델 조절</b>
    <label>크기 <input id="ac_scale" type="range" min="0.2" max="4" step="0.05" value="1" style="vertical-align:middle;"><span id="ac_scaleval">1.00&times;</span></label>
    <label>회전X <input id="ac_rx" type="range" min="-180" max="180" step="5" value="0"></label>
    <label>회전Y <input id="ac_ry" type="range" min="-180" max="180" step="5" value="0"></label>
    <label>회전Z <input id="ac_rz" type="range" min="-180" max="180" step="5" value="0"></label>
    <button id="ac_zup"  style="padding:3px 8px;border:0;border-radius:6px;background:#cfe0f3;cursor:pointer;">Z-up 보정</button>
    <button id="ac_reset" style="padding:3px 8px;border:0;border-radius:6px;background:#e2e8f0;cursor:pointer;">리셋</button>
  </div>
  <div id="ac_holder" style="position:relative;width:100%;height:460px;border-radius:10px;overflow:hidden;background:linear-gradient(#dfeaf6,#eef2f7);">
    <div id="ac_hud" style="position:absolute;left:10px;top:8px;font-size:13px;font-weight:600;color:#15314f;background:rgba(255,255,255,0.65);padding:4px 8px;border-radius:6px;font-variant-numeric:tabular-nums;"></div>
    <div id="ac_err" style="position:absolute;left:10px;bottom:8px;font-size:12px;color:#b00;"></div>
  </div>
  <input id="ac_scrub" type="range" min="0" value="0" step="1" style="width:100%;margin-top:6px;">
</div>
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
<script>
(function(){
  const DATA = __DATA__;
  const STL_B64 = __STL_B64__;
  const N = DATA.t.length;
  const D2R = Math.PI/180;
  const holder = document.getElementById('ac_holder');
  const hud = document.getElementById('ac_hud');
  const errEl = document.getElementById('ac_err');
  const scrub = document.getElementById('ac_scrub');
  scrub.max = N-1;

  if (typeof THREE === 'undefined') {
    errEl.textContent = '3D 라이브러리를 불러오지 못했습니다(인터넷 연결 확인).';
    return;
  }

  // --- Scene ---
  const scene = new THREE.Scene();
  let W = holder.clientWidth || 700, H = 460;
  const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(W, H);
  holder.appendChild(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(42, W/H, 0.1, 500);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x6b7a90, 0.95));
  const dir = new THREE.DirectionalLight(0xffffff, 0.75);
  dir.position.set(5, 10, 7); scene.add(dir);
  const dir2 = new THREE.DirectionalLight(0xffffff, 0.3);
  dir2.position.set(-6, 4, -5); scene.add(dir2);

  const grid = new THREE.GridHelper(24, 24, 0x9fb3c8, 0xc7d4e2);
  grid.position.y = -1.4; scene.add(grid);

  // 정면에서 불어오는 바람(공기 흐름) 화살표 (+X 쪽에서 -X 로)
  for (let k=-1;k<=1;k++){
    scene.add(new THREE.ArrowHelper(new THREE.Vector3(-1,0,0),
        new THREE.Vector3(3.6, 0.0, k*0.7), 1.6, 0x57b0ff, 0.35, 0.22));
  }

  // --- STL 파싱(있으면) ---
  let STL_GEOM = null, baseFit = 1.0;
  if (STL_B64) {
    try {
      const bin = atob(STL_B64);
      const bytes = new Uint8Array(bin.length);
      for (let i=0;i<bin.length;i++) bytes[i] = bin.charCodeAt(i);
      const geom = new THREE.STLLoader().parse(bytes.buffer);
      geom.center();
      geom.computeVertexNormals();
      geom.computeBoundingBox();
      const size = new THREE.Vector3(); geom.boundingBox.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      baseFit = 3.0 / maxDim;                 // 최대 치수를 약 3 단위로 정규화
      STL_GEOM = geom;
      document.getElementById('ac_hint').textContent =
        'STL 사용 중 · 기수가 파란 바람 화살표 쪽(+X)을 향하도록 회전 조절';
    } catch (e) {
      errEl.textContent = 'STL 을 읽지 못했습니다(' + e.message + '). 기본 모델을 사용합니다.';
      STL_GEOM = null;
    }
  }

  // --- 기본(내장) 항공기 모델 ---
  function buildPrimitive(root, opts){
    const matBody = new THREE.MeshStandardMaterial({color:opts.body, metalness:0.2, roughness:0.6, transparent:opts.alpha<1, opacity:opts.alpha});
    const matWing = new THREE.MeshStandardMaterial({color:opts.wing, metalness:0.2, roughness:0.6, transparent:opts.alpha<1, opacity:opts.alpha});
    const matAcc  = new THREE.MeshStandardMaterial({color:opts.acc,  metalness:0.2, roughness:0.6, transparent:opts.alpha<1, opacity:opts.alpha});
    const fus = new THREE.Mesh(new THREE.CylinderGeometry(0.17,0.15,2.4,20), matBody);
    fus.rotation.z = Math.PI/2; root.add(fus);
    const nose = new THREE.Mesh(new THREE.ConeGeometry(0.17,0.5,20), matBody);
    nose.rotation.z = -Math.PI/2; nose.position.x = 1.45; root.add(nose);
    const tail = new THREE.Mesh(new THREE.ConeGeometry(0.15,0.4,20), matBody);
    tail.rotation.z = Math.PI/2; tail.position.x = -1.4; root.add(tail);
    const wing = new THREE.Mesh(new THREE.BoxGeometry(0.55,0.04,3.0), matWing);
    wing.position.set(0.05,0,0); root.add(wing);
    const htail = new THREE.Mesh(new THREE.BoxGeometry(0.38,0.035,1.15), matWing);
    htail.position.set(-1.25,0.02,0); root.add(htail);
    function fin(zoff, cant){
      const f = new THREE.Mesh(new THREE.BoxGeometry(0.5,0.6,0.04), matAcc);
      f.position.set(-1.2, 0.32, zoff); f.rotation.x = cant; root.add(f);
    }
    if ((DATA.vtail||1) >= 2){ fin(-0.45, 0.20); fin(0.45,-0.20); } else { fin(0,0); }
    for (const z of [-0.95, 0.95]){
      const e = new THREE.Mesh(new THREE.CylinderGeometry(0.12,0.12,0.5,14), matAcc);
      e.rotation.z = Math.PI/2; e.position.set(0.18,-0.16,z); root.add(e);
    }
  }

  // 모델 루트(정렬 회전 + 크기) 생성
  function makeModel(alpha){
    const root = new THREE.Group();
    if (STL_GEOM){
      const col = alpha < 1 ? 0xb9c4d2 : 0x3b82c4;
      const mat = new THREE.MeshStandardMaterial({color:col, metalness:0.25, roughness:0.55,
          transparent:alpha<1, opacity:alpha});
      root.add(new THREE.Mesh(STL_GEOM, mat));
    } else {
      const opts = alpha < 1
        ? {body:0xb9c4d2, wing:0x9fb6d4, acc:0x9fb6d4, alpha:alpha}
        : {body:0xf2f5f9, wing:0x1f7ae0, acc:0x0d3b66, alpha:alpha};
      buildPrimitive(root, opts);
    }
    return root;
  }

  // 외곽 그룹(비행 자세) + 내부 모델 루트(정렬/크기)
  const plane = new THREE.Group();
  const planeModel = makeModel(1.0); plane.add(planeModel); scene.add(plane);
  const ghost = new THREE.Group();
  const ghostModel = makeModel(0.22); ghost.add(ghostModel); scene.add(ghost);

  // 자세 → 쿼터니언 (yaw:Y, pitch:Z, roll:X)
  const AX_Y=new THREE.Vector3(0,1,0), AX_Z=new THREE.Vector3(0,0,1), AX_X=new THREE.Vector3(1,0,0);
  const _qy=new THREE.Quaternion(), _qp=new THREE.Quaternion(), _qr=new THREE.Quaternion();
  function setAttitude(obj, pDeg, rDeg, yDeg){
    _qy.setFromAxisAngle(AX_Y, -yDeg*D2R);
    _qp.setFromAxisAngle(AX_Z,  pDeg*D2R);
    _qr.setFromAxisAngle(AX_X,  rDeg*D2R);
    obj.quaternion.copy(_qy).multiply(_qp).multiply(_qr);
  }
  setAttitude(ghost, DATA.pitch0, DATA.roll0, DATA.yaw0);

  // --- 모델 크기/정렬 컨트롤 ---
  const elScale=document.getElementById('ac_scale'), elScaleV=document.getElementById('ac_scaleval');
  const elRx=document.getElementById('ac_rx'), elRy=document.getElementById('ac_ry'), elRz=document.getElementById('ac_rz');
  function applyModelTransform(){
    const s = parseFloat(elScale.value) * baseFit;
    const rx=parseFloat(elRx.value)*D2R, ry=parseFloat(elRy.value)*D2R, rz=parseFloat(elRz.value)*D2R;
    for (const m of [planeModel, ghostModel]){ m.scale.setScalar(s); m.rotation.set(rx,ry,rz); }
    elScaleV.innerHTML = parseFloat(elScale.value).toFixed(2)+'&times;';
  }
  [elScale, elRx, elRy, elRz].forEach(el => el.addEventListener('input', applyModelTransform));
  document.getElementById('ac_zup').onclick = function(){ elRx.value=-90; applyModelTransform(); };
  document.getElementById('ac_reset').onclick = function(){ elScale.value=1; elRx.value=0; elRy.value=0; elRz.value=0; applyModelTransform(); };
  applyModelTransform();

  // --- 궤도 카메라 ---
  const target = new THREE.Vector3(0,0,0);
  let az = -0.7, el = 0.32, radius = 6.2;
  function updateCam(){
    camera.position.set(
      target.x + radius*Math.cos(el)*Math.cos(az),
      target.y + radius*Math.sin(el),
      target.z + radius*Math.cos(el)*Math.sin(az));
    camera.lookAt(target);
  }
  let dragging=false, px=0, py=0;
  renderer.domElement.addEventListener('mousedown', e=>{dragging=true;px=e.clientX;py=e.clientY;});
  window.addEventListener('mouseup', ()=>{dragging=false;});
  window.addEventListener('mousemove', e=>{
    if(!dragging)return;
    az -= (e.clientX-px)*0.01; el += (e.clientY-py)*0.01;
    el = Math.max(-1.3, Math.min(1.3, el)); px=e.clientX; py=e.clientY; updateCam();
  });
  renderer.domElement.addEventListener('wheel', e=>{
    e.preventDefault(); radius *= (1 + Math.sign(e.deltaY)*0.08);
    radius = Math.max(3, Math.min(20, radius)); updateCam();
  }, {passive:false});
  updateCam();

  // --- 렌더 / 재생 ---
  function show(i){
    if(i<0)i=0; if(i>N-1)i=N-1;
    const p=DATA.pitch[i], r=DATA.roll[i], y=DATA.yaw[i];
    setAttitude(plane, p, r, y);
    hud.innerHTML = 't = '+DATA.t[i].toFixed(2)+' s<br>pitch '+p.toFixed(1)+
      '&deg; &nbsp; roll '+r.toFixed(1)+'&deg;<br>yaw '+y.toFixed(1)+
      '&deg; &nbsp; AoA '+DATA.aoa[i].toFixed(1)+'&deg;';
  }
  let idxF=0, playing=false, last=null, speed=1, loop=true;
  function setIdx(i){idxF=i;scrub.value=Math.round(i);show(Math.round(i));}
  function frame(ts){
    if(holder.clientWidth && holder.clientWidth!==W){
      W=holder.clientWidth; renderer.setSize(W,H); camera.aspect=W/H; camera.updateProjectionMatrix();
    }
    if(playing){
      if(last==null)last=ts;
      const dw=(ts-last)/1000; last=ts;
      idxF += speed*dw/DATA.dt;
      if(idxF>=N-1){ if(loop){idxF=0;} else {idxF=N-1;playing=false;} }
      scrub.value=Math.round(idxF);
    }
    show(Math.round(idxF));
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  document.getElementById('ac_play').onclick =function(){ if(Math.round(idxF)>=N-1)idxF=0; playing=true; last=null; };
  document.getElementById('ac_pause').onclick=function(){ playing=false; };
  document.getElementById('ac_stop').onclick =function(){ playing=false; setIdx(0); };
  document.getElementById('ac_speed').onchange=function(e){ speed=parseFloat(e.target.value); };
  document.getElementById('ac_loop').onchange =function(e){ loop=e.target.checked; };
  document.getElementById('ac_ghost').onchange=function(e){ ghost.visible=e.target.checked; };
  scrub.oninput=function(e){ playing=false; setIdx(parseInt(e.target.value)); };

  setIdx(0);
  requestAnimationFrame(frame);
})();
</script>
"""
