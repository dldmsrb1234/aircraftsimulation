# -*- coding: utf-8 -*-
"""
animation.py
============
브라우저에서 도는 **실시간 3D 비행 애니메이션** 컴포넌트 (Three.js).

- 물리(회전 동역학)를 **브라우저에서 실시간으로 계속 적분**한다.
  → 시간 제한(t_end) 없이 **정지 버튼을 누를 때까지 계속** 시뮬레이션.
- 파이썬 시뮬레이션과 **동일한 모델·반암시적 적분식**을 JS 로 이식 → 그래프와 일치.
- aero_model 이 있으면 STL 표면 패널 공력 표를, 없으면 매개변수 모델을 사용.
- 마우스 드래그=시점 회전, 휠=확대. STL 업로드 시 모델·크기·정렬 조절.
"""

from __future__ import annotations
import json
import math
import numpy as np

import stl_analysis


# ---------------------------------------------------------------------------
# 동역학 파라미터(파이썬 run_simulation 과 동일하게) → JS 로 전달
# ---------------------------------------------------------------------------
def _build_dyn(ac, env, init, sim, aero_model) -> dict:
    q = 0.5 * env.rho * env.V * env.V
    I_pitch, I_roll, I_yaw = ac.Iy, ac.Ix, ac.Iz
    mult = sim.damping_mult

    if aero_model is not None:
        import panel_aero
        cgp = panel_aero.cg_point_from_nose(aero_model, ac.cg)
        k_th = abs(panel_aero.pitch_stiffness(aero_model, q, cgp))
        k_ps = abs(panel_aero.yaw_stiffness(aero_model, q, cgp))
        zeta = 0.4
        cd_p = 2 * zeta * math.sqrt(max(k_th, 1e-12) * I_pitch) * mult
        cd_y = 2 * zeta * math.sqrt(max(k_ps, 1e-12) * I_yaw) * mult
        cd_r = I_roll * 3.0 * mult
        aero = {
            "alphas": [float(a) for a in aero_model["alphas"]],
            "betas": [float(b) for b in aero_model["betas"]],
            "Fg": [float(x) for x in np.asarray(aero_model["Fg"]).ravel()],
            "Mg": [float(x) for x in np.asarray(aero_model["Mg"]).ravel()],
            "cg_point": [float(c) for c in cgp],
        }
        param, mode = None, "aero"
    else:
        import physics
        cd_p, cd_r, cd_y = ac.cd_pitch * mult, ac.cd_roll * mult, ac.cd_yaw * mult
        k_th = abs(physics.pitch_stiffness(ac, env, 0.0))
        k_ps = abs(physics.yaw_stiffness(ac, env))
        aero = None
        param = {
            "wing_aoa": ac.wing.aoa_deg, "cl_alpha": ac.wing.cl_alpha,
            "alpha_stall": ac.wing.alpha_stall_deg, "cl_max": ac.wing.cl_max,
            "S_wing": ac.wing.area, "cp_base": ac.wing.cp_base,
            "cp_auto": bool(ac.wing.cp_auto), "k_cp": ac.wing.k_cp, "cg": ac.cg,
            "htail_aoa": ac.htail.aoa_deg, "htail_cl_alpha": ac.htail.cl_alpha,
            "S_htail": ac.htail.area, "htail_arm": ac.htail.arm, "span": ac.wing.span,
            "asymmetry": ac.asymmetry, "S_vtail": ac.vtail.area,
            "vtail_count": ac.vtail.count, "vtail_cl_alpha": ac.vtail.cl_alpha,
            "vtail_arm": ac.vtail.arm,
        }
        mode = "param"

    w_pitch = math.sqrt(k_th / I_pitch) if I_pitch > 0 else 0.0
    w_yaw = math.sqrt(k_ps / I_yaw) if I_yaw > 0 else 0.0
    scale_guard = 2.0 if aero_model is not None else 1.0  # UI 크기 배율 4x → ω roughly 2x
    n_sub = int(min(300, max(1, math.ceil(sim.dt * max(w_pitch, w_yaw, 1e-9) * scale_guard / 1.5))))

    return {
        "mode": mode, "q": q, "dt": sim.dt, "n_sub": n_sub,
        "Ipitch": I_pitch, "Iroll": I_roll, "Iyaw": I_yaw,
        "cd_p": cd_p, "cd_r": cd_r, "cd_y": cd_y,
        "pitch0": init.pitch0_deg, "roll0": init.roll0_deg, "yaw0": init.yaw0_deg,
        "p0": init.p0_deg, "q0": init.q0_deg, "r0": init.r0_deg,
        "aero": aero, "param": param,
    }


def realtime_animation_html(ac, env, init, sim, aero_model=None,
                            stl_b64: str = "",
                            align_fwd: str = "+X", align_up: str = "+Y",
                            pre_rot: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> str:
    """실시간 연속 3D 비행 애니메이션 HTML 생성."""
    align = stl_analysis.align_matrix(align_fwd, align_up)
    pre = stl_analysis.rotation_matrix_xyz(*pre_rot)
    data = {
        "vtail": int(ac.vtail.count),
        "align": (pre @ align).tolist(),
        "pitch0": float(init.pitch0_deg),
        "roll0": float(init.roll0_deg),
        "yaw0": float(init.yaw0_deg),
        "dyn": _build_dyn(ac, env, init, sim, aero_model),
    }
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
      <input id="ac_speed" type="range" min="0.05" max="10" step="0.05" value="1" style="width:120px;vertical-align:middle;">
      <input id="ac_speed_num" type="number" min="0.01" max="50" step="0.05" value="1" style="width:62px;padding:2px;border-radius:6px;border:1px solid #cbd5e1;">&times;
    </label>
    <label style="font-size:13px;"><input type="checkbox" id="ac_ghost" checked> 초기자세</label>
    <span id="ac_hint" style="margin-left:auto;font-size:12px;color:#6b7785;">정지 전까지 연속 시뮬레이션 · 드래그=회전 · 휠=확대</span>
  </div>
  <div id="ac_modelbar" style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:0 0 8px;padding:6px 8px;background:#eef3f9;border-radius:8px;font-size:12px;">
    <b>모델 조절</b>
    <label>크기 <input id="ac_scale" type="range" min="0.2" max="4" step="0.05" value="1" style="vertical-align:middle;"><span id="ac_scaleval">1.00&times;</span></label>
    <label>미세 Roll <input id="ac_rx" type="range" min="-180" max="180" step="5" value="0"></label>
    <label>미세 Pitch <input id="ac_rz" type="range" min="-180" max="180" step="5" value="0"></label>
    <label>미세 Yaw <input id="ac_ry" type="range" min="-180" max="180" step="5" value="0"></label>
    <button id="ac_reset" style="padding:3px 8px;border:0;border-radius:6px;background:#e2e8f0;cursor:pointer;">리셋</button>
  </div>
  <div id="ac_holder" style="position:relative;width:100%;height:460px;border-radius:10px;overflow:hidden;background:linear-gradient(#dfeaf6,#eef2f7);">
    <div id="ac_hud" style="position:absolute;left:10px;top:8px;font-size:13px;font-weight:600;color:#15314f;background:rgba(255,255,255,0.65);padding:4px 8px;border-radius:6px;font-variant-numeric:tabular-nums;"></div>
    <div id="ac_err" style="position:absolute;left:10px;bottom:8px;font-size:12px;color:#b00;"></div>
  </div>
</div>
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
<script>
(function(){
  const DATA = __DATA__;
  const STL_B64 = __STL_B64__;
  const DYN = DATA.dyn;
  const D2R = Math.PI/180, R2D = 180/Math.PI;
  const holder = document.getElementById('ac_holder');
  const hud = document.getElementById('ac_hud');
  const errEl = document.getElementById('ac_err');

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
  const dir = new THREE.DirectionalLight(0xffffff, 0.75); dir.position.set(5,10,7); scene.add(dir);
  const dir2 = new THREE.DirectionalLight(0xffffff, 0.3); dir2.position.set(-6,4,-5); scene.add(dir2);
  const grid = new THREE.GridHelper(24, 24, 0x9fb3c8, 0xc7d4e2); grid.position.y = -1.4; scene.add(grid);
  for (let k=-1;k<=1;k++)
    scene.add(new THREE.ArrowHelper(new THREE.Vector3(-1,0,0), new THREE.Vector3(3.6,0,k*0.7), 1.6, 0x57b0ff, 0.35, 0.22));

  // --- STL 파싱 ---
  let STL_GEOM = null, baseFit = 1.0;
  if (STL_B64) {
    try {
      const bin = atob(STL_B64), bytes = new Uint8Array(bin.length);
      for (let i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i);
      const g = new THREE.STLLoader().parse(bytes.buffer);
      g.center(); g.computeVertexNormals(); g.computeBoundingBox();
      const s = new THREE.Vector3(); g.boundingBox.getSize(s);
      baseFit = 3.0 / (Math.max(s.x,s.y,s.z) || 1);
      STL_GEOM = g;
      document.getElementById('ac_hint').textContent =
        'STL 연속 시뮬레이션 · 속도 자유 조정 · 크기 조절은 ray 물리에도 즉시 반영';
    } catch(e){ errEl.textContent = 'STL 을 읽지 못했습니다: '+e.message; STL_GEOM=null; }
  }

  function buildPrimitive(root, o){
    const mb=new THREE.MeshStandardMaterial({color:o.body,metalness:0.2,roughness:0.6,transparent:o.alpha<1,opacity:o.alpha});
    const mw=new THREE.MeshStandardMaterial({color:o.wing,metalness:0.2,roughness:0.6,transparent:o.alpha<1,opacity:o.alpha});
    const ma=new THREE.MeshStandardMaterial({color:o.acc,metalness:0.2,roughness:0.6,transparent:o.alpha<1,opacity:o.alpha});
    const fus=new THREE.Mesh(new THREE.CylinderGeometry(0.17,0.15,2.4,20),mb); fus.rotation.z=Math.PI/2; root.add(fus);
    const nose=new THREE.Mesh(new THREE.ConeGeometry(0.17,0.5,20),mb); nose.rotation.z=-Math.PI/2; nose.position.x=1.45; root.add(nose);
    const tail=new THREE.Mesh(new THREE.ConeGeometry(0.15,0.4,20),mb); tail.rotation.z=Math.PI/2; tail.position.x=-1.4; root.add(tail);
    const wing=new THREE.Mesh(new THREE.BoxGeometry(0.55,0.04,3.0),mw); wing.position.set(0.05,0,0); root.add(wing);
    const ht=new THREE.Mesh(new THREE.BoxGeometry(0.38,0.035,1.15),mw); ht.position.set(-1.25,0.02,0); root.add(ht);
    function fin(z,c){ const f=new THREE.Mesh(new THREE.BoxGeometry(0.5,0.6,0.04),ma); f.position.set(-1.2,0.32,z); f.rotation.x=c; root.add(f); }
    if((DATA.vtail||1)>=2){fin(-0.45,0.20);fin(0.45,-0.20);} else {fin(0,0);}
    for(const z of [-0.95,0.95]){ const e=new THREE.Mesh(new THREE.CylinderGeometry(0.12,0.12,0.5,14),ma); e.rotation.z=Math.PI/2; e.position.set(0.18,-0.16,z); root.add(e); }
  }
  function makeModel(alpha){
    const root=new THREE.Group();
    if(STL_GEOM){
      const col=alpha<1?0xb9c4d2:0x3b82c4;
      root.add(new THREE.Mesh(STL_GEOM, new THREE.MeshStandardMaterial({color:col,metalness:0.25,roughness:0.55,transparent:alpha<1,opacity:alpha})));
    } else {
      buildPrimitive(root, alpha<1?{body:0xb9c4d2,wing:0x9fb6d4,acc:0x9fb6d4,alpha:alpha}:{body:0xf2f5f9,wing:0x1f7ae0,acc:0x0d3b66,alpha:alpha});
    }
    return root;
  }

  const plane=new THREE.Group(); const planeModel=makeModel(1.0); plane.add(planeModel); scene.add(plane);
  const ghost=new THREE.Group(); const ghostModel=makeModel(0.22); ghost.add(ghostModel); scene.add(ghost);

  // 자세 쿼터니언 (yaw:Y, pitch:Z, roll:X)
  const AX_Y=new THREE.Vector3(0,1,0), AX_Z=new THREE.Vector3(0,0,1), AX_X=new THREE.Vector3(1,0,0);
  const _qy=new THREE.Quaternion(), _qp=new THREE.Quaternion(), _qr=new THREE.Quaternion();
  function attitudeQuat(pRad,rRad,yRad){
    _qy.setFromAxisAngle(AX_Y,-yRad); _qp.setFromAxisAngle(AX_Z,pRad); _qr.setFromAxisAngle(AX_X,rRad);
    return new THREE.Quaternion().copy(_qy).multiply(_qp).multiply(_qr);
  }
  function setAttitude(obj,pRad,rRad,yRad){ obj.quaternion.copy(attitudeQuat(pRad,rRad,yRad)); }
  setAttitude(ghost, DATA.pitch0*D2R, DATA.roll0*D2R, DATA.yaw0*D2R);

  // 정렬(모델→표준) + 모델 크기/미세회전
  const A=DATA.align, mAlign=new THREE.Matrix4();
  mAlign.set(A[0][0],A[0][1],A[0][2],0, A[1][0],A[1][1],A[1][2],0, A[2][0],A[2][1],A[2][2],0, 0,0,0,1);
  const qAlign=new THREE.Quaternion().setFromRotationMatrix(mAlign);
  const elScale=document.getElementById('ac_scale'), elScaleV=document.getElementById('ac_scaleval');
  const elRx=document.getElementById('ac_rx'), elRy=document.getElementById('ac_ry'), elRz=document.getElementById('ac_rz');
  const _fine=new THREE.Quaternion(), _qm=new THREE.Quaternion();
  let physScale = 1.0;
  function applyModelTransform(){
    physScale = parseFloat(elScale.value) || 1.0;
    const s=physScale*baseFit;
    _fine.setFromEuler(new THREE.Euler(parseFloat(elRx.value)*D2R,parseFloat(elRy.value)*D2R,parseFloat(elRz.value)*D2R,'XYZ'));
    _qm.copy(qAlign).multiply(_fine);
    for(const m of [planeModel,ghostModel]){ m.scale.setScalar(s); m.quaternion.copy(_qm); }
    elScaleV.innerHTML=parseFloat(elScale.value).toFixed(2)+'&times;';
  }
  [elScale,elRx,elRy,elRz].forEach(el=>el.addEventListener('input',applyModelTransform));
  document.getElementById('ac_reset').onclick=function(){ elScale.value=1; elRx.value=0; elRy.value=0; elRz.value=0; applyModelTransform(); };
  applyModelTransform();

  // 궤도 카메라
  const target=new THREE.Vector3(0,0,0); let az=-0.7, el=0.32, radius=6.2;
  function updateCam(){ camera.position.set(target.x+radius*Math.cos(el)*Math.cos(az), target.y+radius*Math.sin(el), target.z+radius*Math.cos(el)*Math.sin(az)); camera.lookAt(target); }
  let drag=false,px=0,py=0;
  renderer.domElement.addEventListener('mousedown',e=>{drag=true;px=e.clientX;py=e.clientY;});
  window.addEventListener('mouseup',()=>{drag=false;});
  window.addEventListener('mousemove',e=>{ if(!drag)return; az-=(e.clientX-px)*0.01; el+=(e.clientY-py)*0.01; el=Math.max(-1.3,Math.min(1.3,el)); px=e.clientX; py=e.clientY; updateCam(); });
  renderer.domElement.addEventListener('wheel',e=>{ e.preventDefault(); radius*=(1+Math.sign(e.deltaY)*0.08); radius=Math.max(3,Math.min(20,radius)); updateCam(); }, {passive:false});
  updateCam();

  // ===================== 실시간 연속 동역학 =====================
  function clBody(alpha, p){
    const sgn=alpha>=0?1:-1, a=Math.abs(alpha), ast=p.alpha_stall*D2R, lin=p.cl_alpha*a;
    let peak=p.cl_alpha*ast; if(p.cl_max>0 && p.cl_max<peak) peak=p.cl_max;
    if(a<=ast) return sgn*lin;
    let post=peak-0.7*p.cl_alpha*(a-ast); post=Math.max(post,0.35*peak); return sgn*post;
  }
  function momentsParam(th,ph,ps){
    const p=DYN.param, q=DYN.q;
    const aw=p.wing_aoa*D2R+th, clw=clBody(aw,p), Lw=q*p.S_wing*clw;
    const xcp=p.cp_base+(p.cp_auto? p.k_cp*aw:0);
    const at=p.htail_aoa*D2R+th, Lt=q*p.S_htail*p.htail_cl_alpha*at;
    const Mp=Lw*(p.cg-xcp)-Lt*p.htail_arm;
    const Lwb=q*p.S_wing*clBody(p.wing_aoa*D2R,p);
    const Mr=p.asymmetry*Lwb*(p.span/4);
    const My=-(q*p.S_vtail*p.vtail_count*p.vtail_cl_alpha*ps)*p.vtail_arm + p.asymmetry*0.05*q*p.S_wing*p.span;
    return [Mp,Mr,My, aw*R2D, Lw];
  }
  function windBody(th,ph,ps){
    const qf=attitudeQuat(th,ph,ps).invert();
    const w=new THREE.Vector3(-1,0,0).applyQuaternion(qf);
    return [w.x,w.y,w.z];
  }
  function bilin(arr, al, be){
    const A=DYN.aero.alphas, B=DYN.aero.betas, na=A.length, nb=B.length;
    let a=Math.min(Math.max(al,A[0]),A[na-1]), b=Math.min(Math.max(be,B[0]),B[nb-1]);
    let ia=0; while(ia<na-2 && A[ia+1]<a) ia++;
    let ib=0; while(ib<nb-2 && B[ib+1]<b) ib++;
    const ta=(a-A[ia])/(A[ia+1]-A[ia]), tb=(b-B[ib])/(B[ib+1]-B[ib]);
    const g=(i,j,k)=>arr[((i*nb+j)*3)+k];
    const out=[0,0,0];
    for(let k=0;k<3;k++) out[k]=g(ia,ib,k)*(1-ta)*(1-tb)+g(ia+1,ib,k)*ta*(1-tb)+g(ia,ib+1,k)*(1-ta)*tb+g(ia+1,ib+1,k)*ta*tb;
    return out;
  }
  function momentsAero(th,ph,ps){
    const w=windBody(th,ph,ps);
    const al=Math.atan2(w[1],-w[0]), be=Math.atan2(w[2],-w[0]);
    const Fb=bilin(DYN.aero.Fg,al,be), Mob=bilin(DYN.aero.Mg,al,be), cg0=DYN.aero.cg_point, q=DYN.q;
    const ss=(DYN.mode==='aero'?physScale:1), s2=ss*ss, s3=s2*ss;
    const F=[Fb[0]*q*s2,Fb[1]*q*s2,Fb[2]*q*s2];
    const Mo=[Mob[0]*q*s3,Mob[1]*q*s3,Mob[2]*q*s3];
    const cg=[cg0[0]*ss,cg0[1]*ss,cg0[2]*ss];
    // F ∝ scale², origin moment and cg×F moment ∝ scale³
    const Mx=Mo[0]-(cg[1]*F[2]-cg[2]*F[1]);
    const My=Mo[1]-(cg[2]*F[0]-cg[0]*F[2]);
    const Mz=Mo[2]-(cg[0]*F[1]-cg[1]*F[0]);
    // body_moments: roll=Mx, pitch=Mz, yaw=-My
    return [Mz, Mx, -My, al*R2D, F[1]];
  }
  function moments(th,ph,ps){ return DYN.mode==='aero'? momentsAero(th,ph,ps):momentsParam(th,ph,ps); }

  const CLP=89*D2R;
  function wrap(a){ while(a>Math.PI)a-=2*Math.PI; while(a<-Math.PI)a+=2*Math.PI; return a; }
  let st = initState();
  function initState(){
    return {th:DATA.pitch0*D2R, ph:DATA.roll0*D2R, ps:DATA.yaw0*D2R,
            q:DYN.q0*D2R, p:DYN.p0*D2R, r:DYN.r0*D2R, T:0, aoa:0, lift:0};
  }
  function stepDt(){
    const h=DYN.dt/DYN.n_sub;
    for(let s=0;s<DYN.n_sub;s++){
      const m=moments(st.th,st.ph,st.ps);
      const ss=(DYN.mode==='aero'?physScale:1), si=ss*ss, sc=Math.pow(ss,2.5);
      const Ip=DYN.Ipitch*si, Ir=DYN.Iroll*si, Iy=DYN.Iyaw*si;
      const cdp=DYN.cd_p*sc, cdr=DYN.cd_r*sc, cdy=DYN.cd_y*sc;
      st.q=(st.q+(m[0]/Ip)*h)/(1+(cdp/Ip)*h);
      st.p=(st.p+(m[1]/Ir)*h)/(1+(cdr/Ir)*h);
      st.r=(st.r+(m[2]/Iy)*h)/(1+(cdy/Iy)*h);
      st.th+=st.q*h; st.ph+=st.p*h; st.ps+=st.r*h;
      st.th=Math.max(-CLP,Math.min(CLP,st.th)); st.ph=wrap(st.ph); st.ps=wrap(st.ps);
      st.aoa=m[3]; st.lift=m[4];
    }
    st.T+=DYN.dt;
  }

  // ===================== 재생 루프 =====================
  let playing=false, speed=1, lastTs=null, accum=0;
  const speedRange=document.getElementById('ac_speed'), speedNum=document.getElementById('ac_speed_num');
  function setSpeed(v){
    speed=Math.max(0.01,Math.min(50,parseFloat(v)||1));
    speedRange.value=Math.max(0.05,Math.min(10,speed));
    speedNum.value=speed.toFixed(2).replace(/\.00$/,'');
  }
  speedRange.addEventListener('input',e=>setSpeed(e.target.value));
  speedNum.addEventListener('change',e=>setSpeed(e.target.value));
  setSpeed(1);
  function frame(ts){
    if(holder.clientWidth && holder.clientWidth!==W){ W=holder.clientWidth; renderer.setSize(W,H); camera.aspect=W/H; camera.updateProjectionMatrix(); }
    if(playing){
      if(lastTs==null) lastTs=ts;
      const rdt=Math.min((ts-lastTs)/1000, 0.1); lastTs=ts;
      accum+=rdt*speed;
      let steps=Math.floor(accum/DYN.dt); if(steps>300)steps=300; accum-=steps*DYN.dt;
      for(let s=0;s<steps;s++) stepDt();
    } else { lastTs=null; }
    setAttitude(plane, st.th, st.ph, st.ps);
    hud.innerHTML = 't = '+st.T.toFixed(1)+' s'+(playing?' ▶':' ⏸')+'<br>pitch '+(st.th*R2D).toFixed(1)+
      '&deg; &nbsp; roll '+(st.ph*R2D).toFixed(1)+'&deg;<br>yaw '+(st.ps*R2D).toFixed(1)+
      '&deg; &nbsp; AoA '+st.aoa.toFixed(1)+'&deg;<br>속도 '+speed.toFixed(2)+
      '&times;'+(DYN.mode==='aero'?' &nbsp; 크기 '+physScale.toFixed(2)+'&times;':'');
    renderer.render(scene,camera);
    requestAnimationFrame(frame);
  }
  document.getElementById('ac_play').onclick =function(){ playing=true; lastTs=null; };
  document.getElementById('ac_pause').onclick=function(){ playing=false; };
  document.getElementById('ac_stop').onclick =function(){ playing=false; st=initState(); };
  document.getElementById('ac_ghost').onchange=function(e){ ghost.visible=e.target.checked; };

  requestAnimationFrame(frame);
})();
</script>
"""
