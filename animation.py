# -*- coding: utf-8 -*-
"""Unlimited browser-side 3D animation for the ray-only STL engine."""

from __future__ import annotations

import json
import math

import numpy as np

import panel_aero
import stl_analysis


def _build_dyn(ac, env, init, sim, aero_model) -> dict:
    if aero_model is None:
        raise ValueError("ray-only animation requires an STL aero_model")

    q_dyn = 0.5 * float(env.rho) * float(env.V) * float(env.V)
    i_pitch = max(float(ac.Iy), 1e-12)
    i_roll = max(float(ac.Ix), 1e-12)
    i_yaw = max(float(ac.Iz), 1e-12)
    damping_mult = float(sim.damping_mult)

    cgp = panel_aero.cg_point_from_nose(aero_model, float(ac.cg))
    k_pitch_raw = float(panel_aero.pitch_stiffness(aero_model, q_dyn, cgp))
    k_yaw_raw = float(panel_aero.yaw_stiffness(aero_model, q_dyn, cgp))
    zeta = 0.4
    cd_pitch = 2.0 * zeta * math.sqrt(max(abs(k_pitch_raw), 1e-12) * i_pitch) * damping_mult
    cd_yaw = 2.0 * zeta * math.sqrt(max(abs(k_yaw_raw), 1e-12) * i_yaw) * damping_mult
    cd_roll = i_roll * 3.0 * damping_mult

    w_pitch = math.sqrt(abs(k_pitch_raw) / i_pitch) if i_pitch > 0 else 0.0
    w_yaw = math.sqrt(abs(k_yaw_raw) / i_yaw) if i_yaw > 0 else 0.0
    n_sub = int(min(300, max(1, math.ceil(float(sim.dt) * max(w_pitch, w_yaw, 1e-9) * 2.0 / 1.5))))

    aero = {
        "alphas": [float(a) for a in aero_model["alphas"]],
        "betas": [float(b) for b in aero_model["betas"]],
        "Fg": [float(x) for x in np.asarray(aero_model["Fg"]).ravel()],
        "Mg": [float(x) for x in np.asarray(aero_model["Mg"]).ravel()],
        "Ag": [float(x) for x in np.asarray(aero_model["Ag"]).ravel()],
        "Dg": [float(x) for x in np.asarray(aero_model["Dg"]).ravel()],
        "cg_point": [float(c) for c in cgp],
    }

    return {
        "q": q_dyn,
        "dt": float(sim.dt),
        "n_sub": n_sub,
        "Ipitch": i_pitch,
        "Iroll": i_roll,
        "Iyaw": i_yaw,
        "cd_p": cd_pitch,
        "cd_r": cd_roll,
        "cd_y": cd_yaw,
        "pitch0": float(init.pitch0_deg),
        "roll0": float(init.roll0_deg),
        "yaw0": float(init.yaw0_deg),
        "p0": float(init.p0_deg),
        "q0": float(init.q0_deg),
        "r0": float(init.r0_deg),
        "aero": aero,
    }


def realtime_animation_html(
    ac,
    env,
    init,
    sim,
    aero_model=None,
    stl_b64: str = "",
    align_fwd: str = "+X",
    align_up: str = "+Y",
    pre_rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    if aero_model is None:
        raise ValueError("ray-only animation requires an STL aero_model")
    if not stl_b64:
        raise ValueError("ray-only animation requires STL geometry")

    align = stl_analysis.align_matrix(align_fwd, align_up)
    pre = stl_analysis.rotation_matrix_xyz(*pre_rot)
    data = {
        "align": (pre @ align).tolist(),
        "pitch0": float(init.pitch0_deg),
        "roll0": float(init.roll0_deg),
        "yaw0": float(init.yaw0_deg),
        "dyn": _build_dyn(ac, env, init, sim, aero_model),
    }
    return (
        _TEMPLATE
        .replace("__DATA__", json.dumps(data))
        .replace("__STL_B64__", json.dumps(stl_b64))
    )


_TEMPLATE = r"""
<div id="acwrap" style="font-family:'Segoe UI',Arial,sans-serif;color:#1a2330;">
  <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:2px 0 6px;">
    <button id="ac_play" style="padding:6px 14px;border:0;border-radius:8px;background:#1f7ae0;color:#fff;font-weight:600;cursor:pointer;">Play</button>
    <button id="ac_pause" style="padding:6px 14px;border:0;border-radius:8px;background:#6b7785;color:#fff;font-weight:600;cursor:pointer;">Pause</button>
    <button id="ac_stop" style="padding:6px 14px;border:0;border-radius:8px;background:#d64545;color:#fff;font-weight:600;cursor:pointer;">Stop</button>
    <label style="margin-left:8px;font-size:13px;">Speed
      <input id="ac_speed" type="range" min="0.05" max="10" step="0.05" value="1" style="width:120px;vertical-align:middle;">
      <input id="ac_speed_num" type="number" min="0.01" max="50" step="0.05" value="1" style="width:62px;padding:2px;border-radius:6px;border:1px solid #cbd5e1;">&times;
    </label>
    <label style="font-size:13px;"><input type="checkbox" id="ac_ghost" checked> initial pose</label>
    <span id="ac_hint" style="margin-left:auto;font-size:12px;color:#6b7785;">ray-only STL physics, unlimited runtime</span>
  </div>
  <div id="ac_modelbar" style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:0 0 8px;padding:6px 8px;background:#eef3f9;border-radius:8px;font-size:12px;">
    <b>Model</b>
    <label>scale <input id="ac_scale" type="range" min="0.2" max="4" step="0.05" value="1" style="vertical-align:middle;"><span id="ac_scaleval">1.00&times;</span></label>
    <label>roll <input id="ac_rx" type="range" min="-180" max="180" step="5" value="0"></label>
    <label>pitch <input id="ac_rz" type="range" min="-180" max="180" step="5" value="0"></label>
    <label>yaw <input id="ac_ry" type="range" min="-180" max="180" step="5" value="0"></label>
    <button id="ac_reset" style="padding:3px 8px;border:0;border-radius:6px;background:#e2e8f0;cursor:pointer;">reset</button>
  </div>
  <div id="ac_holder" style="position:relative;width:100%;height:460px;border-radius:10px;overflow:hidden;background:linear-gradient(#dfeaf6,#eef2f7);">
    <div id="ac_hud" style="position:absolute;left:10px;top:8px;font-size:13px;font-weight:600;color:#15314f;background:rgba(255,255,255,0.7);padding:4px 8px;border-radius:6px;font-variant-numeric:tabular-nums;"></div>
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
  const D2R = Math.PI / 180, R2D = 180 / Math.PI;
  const holder = document.getElementById('ac_holder');
  const hud = document.getElementById('ac_hud');
  const errEl = document.getElementById('ac_err');

  if (typeof THREE === 'undefined' || typeof THREE.STLLoader === 'undefined') {
    errEl.textContent = 'Three.js/STLLoader failed to load.';
    return;
  }

  const scene = new THREE.Scene();
  let W = holder.clientWidth || 700, H = 460;
  const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(W, H);
  holder.appendChild(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(42, W / H, 0.1, 500);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x6b7a90, 0.95));
  const dir = new THREE.DirectionalLight(0xffffff, 0.75);
  dir.position.set(5, 10, 7);
  scene.add(dir);
  const dir2 = new THREE.DirectionalLight(0xffffff, 0.3);
  dir2.position.set(-6, 4, -5);
  scene.add(dir2);
  const grid = new THREE.GridHelper(24, 24, 0x9fb3c8, 0xc7d4e2);
  grid.position.y = -1.4;
  scene.add(grid);
  for (let k = -1; k <= 1; k++) {
    scene.add(new THREE.ArrowHelper(
      new THREE.Vector3(-1, 0, 0),
      new THREE.Vector3(3.6, 0, k * 0.7),
      1.6, 0x57b0ff, 0.35, 0.22
    ));
  }

  let STL_GEOM = null, baseFit = 1.0;
  try {
    const bin = atob(STL_B64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const g = new THREE.STLLoader().parse(bytes.buffer);
    g.center();
    g.computeVertexNormals();
    g.computeBoundingBox();
    const s = new THREE.Vector3();
    g.boundingBox.getSize(s);
    baseFit = 3.0 / (Math.max(s.x, s.y, s.z) || 1.0);
    STL_GEOM = g;
  } catch (e) {
    errEl.textContent = 'Could not parse STL: ' + e.message;
    return;
  }

  function makeModel(alpha){
    const root = new THREE.Group();
    const col = alpha < 1 ? 0xb9c4d2 : 0x3b82c4;
    const mat = new THREE.MeshStandardMaterial({
      color: col,
      metalness: 0.25,
      roughness: 0.55,
      transparent: alpha < 1,
      opacity: alpha
    });
    root.add(new THREE.Mesh(STL_GEOM, mat));
    return root;
  }

  const plane = new THREE.Group();
  const planeModel = makeModel(1.0);
  plane.add(planeModel);
  scene.add(plane);

  const ghost = new THREE.Group();
  const ghostModel = makeModel(0.22);
  ghost.add(ghostModel);
  scene.add(ghost);

  const AX_Y = new THREE.Vector3(0, 1, 0);
  const AX_Z = new THREE.Vector3(0, 0, 1);
  const AX_X = new THREE.Vector3(1, 0, 0);
  const _qy = new THREE.Quaternion(), _qp = new THREE.Quaternion(), _qr = new THREE.Quaternion();
  function attitudeQuat(pRad, rRad, yRad){
    _qy.setFromAxisAngle(AX_Y, -yRad);
    _qp.setFromAxisAngle(AX_Z, pRad);
    _qr.setFromAxisAngle(AX_X, rRad);
    return new THREE.Quaternion().copy(_qy).multiply(_qp).multiply(_qr);
  }
  function setAttitude(obj, pRad, rRad, yRad){
    obj.quaternion.copy(attitudeQuat(pRad, rRad, yRad));
  }
  setAttitude(ghost, DATA.pitch0 * D2R, DATA.roll0 * D2R, DATA.yaw0 * D2R);

  const A = DATA.align, mAlign = new THREE.Matrix4();
  mAlign.set(
    A[0][0], A[0][1], A[0][2], 0,
    A[1][0], A[1][1], A[1][2], 0,
    A[2][0], A[2][1], A[2][2], 0,
    0, 0, 0, 1
  );
  const qAlign = new THREE.Quaternion().setFromRotationMatrix(mAlign);
  const elScale = document.getElementById('ac_scale'), elScaleV = document.getElementById('ac_scaleval');
  const elRx = document.getElementById('ac_rx'), elRy = document.getElementById('ac_ry'), elRz = document.getElementById('ac_rz');
  const _fine = new THREE.Quaternion(), _qm = new THREE.Quaternion();
  let physScale = 1.0;
  function applyModelTransform(){
    physScale = Math.max(0.2, Math.min(4.0, parseFloat(elScale.value) || 1.0));
    const s = physScale * baseFit;
    _fine.setFromEuler(new THREE.Euler(
      (parseFloat(elRx.value) || 0) * D2R,
      (parseFloat(elRy.value) || 0) * D2R,
      (parseFloat(elRz.value) || 0) * D2R,
      'XYZ'
    ));
    _qm.copy(qAlign).multiply(_fine);
    for (const m of [planeModel, ghostModel]) {
      m.scale.setScalar(s);
      m.quaternion.copy(_qm);
    }
    elScaleV.innerHTML = physScale.toFixed(2) + '&times;';
  }
  [elScale, elRx, elRy, elRz].forEach(el => el.addEventListener('input', applyModelTransform));
  document.getElementById('ac_reset').onclick = function(){
    elScale.value = 1;
    elRx.value = 0;
    elRy.value = 0;
    elRz.value = 0;
    applyModelTransform();
  };
  applyModelTransform();

  const target = new THREE.Vector3(0, 0, 0);
  let az = -0.7, el = 0.32, radius = 6.2;
  function updateCam(){
    camera.position.set(
      target.x + radius * Math.cos(el) * Math.cos(az),
      target.y + radius * Math.sin(el),
      target.z + radius * Math.cos(el) * Math.sin(az)
    );
    camera.lookAt(target);
  }
  let drag = false, px = 0, py = 0;
  renderer.domElement.addEventListener('mousedown', e => { drag = true; px = e.clientX; py = e.clientY; });
  window.addEventListener('mouseup', () => { drag = false; });
  window.addEventListener('mousemove', e => {
    if (!drag) return;
    az -= (e.clientX - px) * 0.01;
    el += (e.clientY - py) * 0.01;
    el = Math.max(-1.3, Math.min(1.3, el));
    px = e.clientX;
    py = e.clientY;
    updateCam();
  });
  renderer.domElement.addEventListener('wheel', e => {
    e.preventDefault();
    radius *= (1 + Math.sign(e.deltaY) * 0.08);
    radius = Math.max(3, Math.min(20, radius));
    updateCam();
  }, {passive:false});
  updateCam();

  function windBody(th, ph, ps){
    const qf = attitudeQuat(th, ph, ps).invert();
    const w = new THREE.Vector3(-1, 0, 0).applyQuaternion(qf);
    return [w.x, w.y, w.z];
  }
  function bilin(arr, al, be){
    const A = DYN.aero.alphas, B = DYN.aero.betas;
    const na = A.length, nb = B.length;
    let a = Math.min(Math.max(al, A[0]), A[na - 1]);
    let b = Math.min(Math.max(be, B[0]), B[nb - 1]);
    let ia = 0;
    while (ia < na - 2 && A[ia + 1] < a) ia++;
    let ib = 0;
    while (ib < nb - 2 && B[ib + 1] < b) ib++;
    const ta = (a - A[ia]) / (A[ia + 1] - A[ia]);
    const tb = (b - B[ib]) / (B[ib + 1] - B[ib]);
    const g = (i, j, k) => arr[((i * nb + j) * 3) + k];
    const out = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      out[k] =
        g(ia, ib, k) * (1 - ta) * (1 - tb) +
        g(ia + 1, ib, k) * ta * (1 - tb) +
        g(ia, ib + 1, k) * (1 - ta) * tb +
        g(ia + 1, ib + 1, k) * ta * tb;
    }
    return out;
  }
  function bilin1(arr, al, be){
    const A = DYN.aero.alphas, B = DYN.aero.betas;
    const na = A.length, nb = B.length;
    let a = Math.min(Math.max(al, A[0]), A[na - 1]);
    let b = Math.min(Math.max(be, B[0]), B[nb - 1]);
    let ia = 0;
    while (ia < na - 2 && A[ia + 1] < a) ia++;
    let ib = 0;
    while (ib < nb - 2 && B[ib + 1] < b) ib++;
    const ta = (a - A[ia]) / (A[ia + 1] - A[ia]);
    const tb = (b - B[ib]) / (B[ib + 1] - B[ib]);
    const g = (i, j) => arr[i * nb + j];
    return (
      g(ia, ib) * (1 - ta) * (1 - tb) +
      g(ia + 1, ib) * ta * (1 - tb) +
      g(ia, ib + 1) * (1 - ta) * tb +
      g(ia + 1, ib + 1) * ta * tb
    );
  }
  function momentsAero(th, ph, ps){
    const w = windBody(th, ph, ps);
    const al = Math.atan2(w[1], -w[0]);
    const be = Math.atan2(w[2], -w[0]);
    const Fb = bilin(DYN.aero.Fg, al, be);
    const Mob = bilin(DYN.aero.Mg, al, be);
    const cg0 = DYN.aero.cg_point, q = DYN.q;
    const ss = physScale, s2 = ss * ss, s3 = s2 * ss;
    const F = [Fb[0] * q * s2, Fb[1] * q * s2, Fb[2] * q * s2];
    const Mo = [Mob[0] * q * s3, Mob[1] * q * s3, Mob[2] * q * s3];
    const cg = [cg0[0] * ss, cg0[1] * ss, cg0[2] * ss];
    const Mx = Mo[0] - (cg[1] * F[2] - cg[2] * F[1]);
    const My = Mo[1] - (cg[2] * F[0] - cg[0] * F[2]);
    const Mz = Mo[2] - (cg[0] * F[1] - cg[1] * F[0]);
    const hitArea = bilin1(DYN.aero.Ag, al, be) * s2;
    const wakeDistance = bilin1(DYN.aero.Dg, al, be) * ss;
    return [Mz, Mx, -My, al * R2D, F[1], hitArea, wakeDistance];
  }

  function wrap(a){
    while (a > Math.PI) a -= 2 * Math.PI;
    while (a < -Math.PI) a += 2 * Math.PI;
    return a;
  }
  function initState(){
    return {
      th: DATA.pitch0 * D2R,
      ph: DATA.roll0 * D2R,
      ps: DATA.yaw0 * D2R,
      q: DYN.q0 * D2R,
      p: DYN.p0 * D2R,
      r: DYN.r0 * D2R,
      T: 0,
      aoa: 0,
      lift: 0,
      hitArea: 0,
      wakeDistance: 0
    };
  }
  let st = initState();
  function stepDt(){
    const h = DYN.dt / DYN.n_sub;
    for (let s = 0; s < DYN.n_sub; s++) {
      const m = momentsAero(st.th, st.ph, st.ps);
      const ss = physScale, si = ss * ss, sc = Math.pow(ss, 2.5);
      const Ip = Math.max(DYN.Ipitch * si, 1e-9);
      const Ir = Math.max(DYN.Iroll * si, 1e-9);
      const Iy = Math.max(DYN.Iyaw * si, 1e-9);
      const cdp = DYN.cd_p * sc;
      const cdr = DYN.cd_r * sc;
      const cdy = DYN.cd_y * sc;
      st.q = (st.q + (m[0] / Ip) * h) / (1 + (cdp / Ip) * h);
      st.p = (st.p + (m[1] / Ir) * h) / (1 + (cdr / Ir) * h);
      st.r = (st.r + (m[2] / Iy) * h) / (1 + (cdy / Iy) * h);
      st.th = wrap(st.th + st.q * h);
      st.ph = wrap(st.ph + st.p * h);
      st.ps = wrap(st.ps + st.r * h);
      st.aoa = m[3];
      st.lift = m[4];
      st.hitArea = m[5];
      st.wakeDistance = m[6];
    }
    st.T += DYN.dt;
  }

  let playing = false, speed = 1, lastTs = null, accum = 0;
  const speedRange = document.getElementById('ac_speed'), speedNum = document.getElementById('ac_speed_num');
  function setSpeed(v){
    speed = Math.max(0.01, Math.min(50, parseFloat(v) || 1));
    speedRange.value = Math.max(0.05, Math.min(10, speed));
    speedNum.value = speed.toFixed(2).replace(/\.00$/, '');
  }
  speedRange.addEventListener('input', e => setSpeed(e.target.value));
  speedNum.addEventListener('change', e => setSpeed(e.target.value));
  setSpeed(1);

  function frame(ts){
    if (holder.clientWidth && holder.clientWidth !== W) {
      W = holder.clientWidth;
      renderer.setSize(W, H);
      camera.aspect = W / H;
      camera.updateProjectionMatrix();
    }
    if (playing) {
      if (lastTs == null) lastTs = ts;
      const rdt = Math.min((ts - lastTs) / 1000, 0.1);
      lastTs = ts;
      accum += rdt * speed;
      let steps = Math.floor(accum / DYN.dt);
      if (steps > 500) steps = 500;
      accum -= steps * DYN.dt;
      for (let s = 0; s < steps; s++) stepDt();
    } else {
      lastTs = null;
    }
    setAttitude(plane, st.th, st.ph, st.ps);
    hud.innerHTML =
      't = ' + st.T.toFixed(1) + ' s' + (playing ? ' running' : '') +
      '<br>pitch ' + (st.th * R2D).toFixed(1) + '&deg; &nbsp; roll ' + (st.ph * R2D).toFixed(1) + '&deg;' +
      '<br>yaw ' + (st.ps * R2D).toFixed(1) + '&deg; &nbsp; AoA ' + st.aoa.toFixed(1) + '&deg;' +
      '<br>hit area ' + st.hitArea.toFixed(3) + ' m² &nbsp; wake ' + st.wakeDistance.toFixed(3) + ' m' +
      '<br>speed ' + speed.toFixed(2) + '&times; &nbsp; scale ' + physScale.toFixed(2) + '&times;';
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }

  document.getElementById('ac_play').onclick = function(){ playing = true; lastTs = null; };
  document.getElementById('ac_pause').onclick = function(){ playing = false; };
  document.getElementById('ac_stop').onclick = function(){ playing = false; st = initState(); accum = 0; };
  document.getElementById('ac_ghost').onchange = function(e){ ghost.visible = e.target.checked; };

  requestAnimationFrame(frame);
})();
</script>
"""
