# -*- coding: utf-8 -*-
"""
app.py
======
항공기 모형 비행 양상 실시간 시각화 시뮬레이터 (Streamlit).

실행:  streamlit run app.py
"""

from __future__ import annotations
import base64
import copy
import hashlib
import math
import numpy as np
import streamlit as st

import presets
from aircraft import (aircraft_from_dict, environment_from_dict,
                      initial_from_dict, sim_from_dict)
import simulation
import analysis
import physics
import visualization as viz
import animation
import stl_analysis
import panel_aero
import streamlit.components.v1 as components

st.set_page_config(page_title="항공기 비행 양상 시뮬레이터",
                   page_icon="✈️", layout="wide")

# 입력 key 목록 (프리셋 dict 에서 메타키 제외)
_META = {"name", "_preset", "V_default_fast"}
INPUT_KEYS = [k for k in presets.CUSTOM.keys() if k not in _META]

LEVEL_COLOR = {"ok": "#2e7d32", "warn": "#ef6c00", "danger": "#c62828"}
STL_QUALITY = {
    "빠름": dict(n_alpha=27, n_beta=11, max_tris=30000, occlusion=False, shadow_bins=0),
    "기본": dict(n_alpha=41, n_beta=17, max_tris=60000, occlusion=True, shadow_bins=80),
    "정밀": dict(n_alpha=61, n_beta=25, max_tris=100000, occlusion=True, shadow_bins=120),
}
STL_PREVIEW_QUALITY = dict(n_alpha=21, n_beta=9, max_tris=18000,
                           occlusion=True, shadow_bins=50)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def load_preset_into_state(name: str):
    """프리셋 값을 session_state 의 위젯 key 로 주입."""
    d = presets.get_preset(name)
    for k, v in d.items():
        if k not in _META:
            st.session_state[k] = v
    st.session_state["_preset"] = name


def badge(text: str, level: str) -> str:
    c = LEVEL_COLOR.get(level, "#555")
    return (f"<span style='background:{c};color:#fff;padding:3px 10px;"
            f"border-radius:12px;font-size:0.85em;white-space:nowrap'>{text}</span>")


def stl_signature(raw: bytes, unit_label: str, physics_scale: float,
                  quality: str, pre_rot: tuple[float, float, float],
                  mass: float, cg_mode: str, cg_ratio: float | None,
                  cg_m: float | None) -> str:
    h = hashlib.sha1(raw).hexdigest()[:16]
    rx, ry, rz = pre_rot
    cg_ratio_s = "" if cg_ratio is None else f"{float(cg_ratio):.6g}"
    cg_m_s = "" if cg_m is None else f"{float(cg_m):.6g}"
    return (f"{h}:{unit_label}:{physics_scale:.6g}:{quality}:{float(mass):.6g}:"
            f"{cg_mode}:{cg_ratio_s}:{cg_m_s}:{rx:.3f}:{ry:.3f}:{rz:.3f}")


def stl_preview_chart_key(stl_sig: str, cur_values: dict) -> str:
    live_keys = ("rho", "V", "pitch0", "roll0", "yaw0", "cg")
    live = ":".join(f"{k}={float(cur_values[k]):.6g}" for k in live_keys)
    return "stl_preview_" + hashlib.sha1(f"{stl_sig}:{live}".encode()).hexdigest()[:16]


def inertia_with_cg_override(props: dict, mass: float, cg_from_nose: float) -> tuple[float, float, float]:
    """균일밀도 CM 기준 관성을 사용자가 고른 CG 기준으로 이동."""
    dx = float(cg_from_nose) - float(props["cg"])
    return (
        max(float(props["Ix"]), 1e-9),
        max(float(props["Iy"]) + mass * dx * dx, 1e-9),
        max(float(props["Iz"]) + mass * dx * dx, 1e-9),
    )


def cg_from_stl_settings(props: dict, mode: str, current_cg: float,
                         ratio: float | None, cg_m: float | None) -> float:
    if mode == "현재 입력 CG 사용":
        cg_from_nose = float(current_cg)
    elif mode == "기수 기준 비율":
        cg_from_nose = float(props["length"]) * float(ratio) / 100.0
    elif mode == "기수 기준 거리(m)":
        cg_from_nose = float(cg_m)
    else:
        cg_from_nose = float(props["cg"])
    if not (0.0 <= cg_from_nose <= float(props["length"])):
        raise ValueError(f"CG는 0~{props['length']:.3g} m 범위 안이어야 합니다.")
    return cg_from_nose


def stiffness_status(k: float, tol: float) -> tuple[str, str]:
    if k > tol:
        return "복원 안정", "ok"
    if k < -tol:
        return "발산 위험", "danger"
    return "중립/약함", "warn"


def tendency_status(angle_deg: float, moment: float,
                    stiffness_label: str, stiffness_level: str) -> tuple[str, str]:
    if abs(angle_deg) < 0.5:
        return stiffness_label, stiffness_level
    if math.radians(angle_deg) * moment > 0:
        return "현재 자세에서 발산 방향", "danger"
    return "현재 자세에서 복원 방향", "ok"


def moment_direction(axis: str, moment: float) -> str:
    if axis == "pitch":
        return "기수 들림" if moment >= 0 else "기수 숙임"
    if axis == "roll":
        return "우측 날개 내려감" if moment >= 0 else "좌측 날개 내려감"
    if axis == "yaw":
        return "기수 우향" if moment >= 0 else "기수 좌향"
    return "중립"


def make_stl_preview(raw: bytes, unit_factor: float, mass: float,
                     cg_mode: str, cg_ratio: float | None,
                     cg_m: float | None, cur_values: dict,
                     pre_rot: tuple[float, float, float]) -> dict:
    tris = stl_analysis.parse_stl(raw) * unit_factor
    rotated = stl_analysis.rotate_mesh(tris, *pre_rot)
    props = stl_analysis.analyze(rotated, "+X", "+Y", mass)
    cg_from_nose = cg_from_stl_settings(
        props, cg_mode, float(cur_values["cg"]), cg_ratio, cg_m)
    Ix, Iy, Iz = inertia_with_cg_override(props, mass, cg_from_nose)
    model = panel_aero.build_aero_model(
        rotated, np.asarray(props["cm"]), **STL_PREVIEW_QUALITY)
    q_dyn = physics.dynamic_pressure(float(cur_values["rho"]), float(cur_values["V"]))
    cgp = panel_aero.cg_point_from_nose(model, cg_from_nose)
    k_pitch = panel_aero.pitch_stiffness(model, q_dyn, cgp)
    k_roll = panel_aero.roll_stiffness(model, q_dyn, cgp)
    k_yaw = panel_aero.yaw_stiffness(model, q_dyn, cgp)
    w = panel_aero.relative_wind_body(
        math.radians(float(cur_values["pitch0"])),
        math.radians(float(cur_values["roll0"])),
        math.radians(float(cur_values["yaw0"])))
    F, M, alpha, beta = panel_aero.aero(model, w, q_dyn, cgp)
    m_roll, m_pitch, m_yaw = panel_aero.body_moments(M)
    ref = max(q_dyn * float(model["ref_area"]) * max(float(props["length"]), 1e-6), 1e-6)
    tol = ref * 1e-4
    p_stiff = stiffness_status(k_pitch, tol)
    r_stiff = stiffness_status(k_roll, tol)
    y_stiff = stiffness_status(k_yaw, tol)
    stability = {}
    for axis, angle, moment, stiff in (
        ("pitch", float(cur_values["pitch0"]), m_pitch, p_stiff),
        ("roll", float(cur_values["roll0"]), m_roll, r_stiff),
        ("yaw", float(cur_values["yaw0"]), m_yaw, y_stiff),
    ):
        label, level = tendency_status(angle, moment, *stiff)
        stability[axis] = {
            "label": label,
            "level": level,
            "k": {"pitch": k_pitch, "roll": k_roll, "yaw": k_yaw}[axis],
            "moment": float(moment),
            "direction": moment_direction(axis, float(moment)),
        }
    props = dict(props)
    props.update({
        "cg_applied": cg_from_nose, "cg_auto": float(props["cg"]),
        "Ix": Ix, "Iy": Iy, "Iz": Iz, "alpha_preview": math.degrees(float(alpha)),
        "beta_preview": math.degrees(float(beta)), "L_preview": float(F[1]),
        "D_preview": float(-F[0]), "n_tri_preview": int(model["n_tri"]),
    })
    return {"tris": rotated, "props": props, "cg": cg_from_nose,
            "stability": stability}


def compute(cur: dict, aero_model=None):
    ac = aircraft_from_dict(cur)
    env = environment_from_dict(cur)
    init = initial_from_dict(cur)
    sim = sim_from_dict(cur)
    aero_err = None
    res = None
    if aero_model is not None:
        try:
            res = simulation.run_simulation(ac, env, init, sim, aero_model=aero_model)
        except Exception as e:           # 패널 공력 실패 시 매개변수 모델로 안전 대체
            aero_err = f"{type(e).__name__}: {e}"
    if res is None:
        res = simulation.run_simulation(ac, env, init, sim, aero_model=None)
    used_aero = (aero_model is not None and aero_err is None)
    k_theta_override = None
    k_psi_override = None
    if used_aero:
        try:
            q_dyn = physics.dynamic_pressure(env.rho, env.V)
            cgp = panel_aero.cg_point_from_nose(aero_model, ac.cg)
            k_theta_override = panel_aero.pitch_stiffness(aero_model, q_dyn, cgp)
            k_psi_override = panel_aero.yaw_stiffness(aero_model, q_dyn, cgp)
        except Exception:
            k_theta_override = None
            k_psi_override = None
    assess = analysis.overall_assessment(ac, env, res, k_theta_override, k_psi_override)
    return {"ac": ac, "env": env, "res": res, "assess": assess,
            "cur": copy.deepcopy(cur),
            "aero": used_aero, "aero_err": aero_err,
            "aero_model": aero_model if used_aero else None}


# ---------------------------------------------------------------------------
# 사이드바 — 입력
# ---------------------------------------------------------------------------
st.sidebar.title("✈️ 입력 패널")

preset_name = st.sidebar.selectbox("1) 항공기 타입", presets.PRESET_NAMES,
                                   key="preset_select")
if st.session_state.get("_preset") != preset_name:
    load_preset_into_state(preset_name)

if st.sidebar.button("↺ 이 타입의 기본값으로 초기화"):
    load_preset_into_state(preset_name)

# STL 형상 분석값을 입력 위젯에 반영(위젯 생성 전에 적용)
if st.session_state.pop("_apply_stl", False):
    for k, v in st.session_state.get("_stl_props", {}).items():
        st.session_state[k] = v

st.sidebar.caption("프리셋은 실제 제조사 데이터가 아닌 **교육용 근사값**입니다.")

with st.sidebar.expander("🌬️ 환경 / 비행 조건", expanded=True):
    st.slider("풍속 V (m/s)", 0.0, 200.0, step=1.0, key="V")
    st.slider("공기밀도 ρ (kg/m³)", 0.5, 1.5, step=0.025, key="rho")

with st.sidebar.expander("📐 기체 제원"):
    st.number_input("전체 질량 mass (kg)", min_value=0.001, key="mass", format="%.3f")
    st.number_input("기체 길이 length (m)", min_value=0.01, key="length", format="%.3f")
    st.number_input("날개폭 span (m)", min_value=0.01, key="span", format="%.3f")
    st.number_input("높이 height (m)", min_value=0.01, key="height", format="%.3f")

with st.sidebar.expander("🛩️ 주날개"):
    st.number_input("주날개 면적 S_wing (m²)", min_value=0.0001, key="S_wing", format="%.4f")
    st.number_input("주날개 위치 (기수 기준, m)", key="wing_pos", format="%.3f")
    st.slider("주날개 AoA (deg)", -5.0, 20.0, step=0.5, key="wing_aoa")
    st.slider("양력기울기 CL_α (1/rad)", 3.0, 7.0, step=0.1, key="cl_alpha")
    st.slider("실속각 (deg)", 5.0, 25.0, step=1.0, key="alpha_stall")
    st.slider("최대 양력계수 CL_max", 0.8, 2.0, step=0.05, key="cl_max")
    st.number_input("양력중심 CP 기준위치 (기수 기준, m)", key="cp_base", format="%.3f")
    st.checkbox("AoA에 따른 CP 자동 이동", key="cp_auto")
    st.slider("CP 이동 계수 k_cp (m/rad)", 0.0, 2.0, step=0.01, key="k_cp")

with st.sidebar.expander("⚖️ 무게중심 / 좌우 비대칭"):
    st.number_input("무게중심 CG (기수 기준, m)", key="cg", format="%.3f")
    st.slider("좌우 비대칭 정도", -0.5, 0.5, step=0.01, key="asymmetry")

with st.sidebar.expander("🪶 수평꼬리날개"):
    st.number_input("수평꼬리 면적 S_htail (m²)", min_value=0.0, key="S_htail", format="%.4f")
    st.slider("수평꼬리 AoA (deg)", -10.0, 10.0, step=0.5, key="htail_aoa")
    st.number_input("수평꼬리 거리 (CG 기준, m)", min_value=0.0, key="htail_arm", format="%.3f")
    st.number_input("수평꼬리 높이 (m)", key="htail_height", format="%.3f")
    st.number_input("수평꼬리 CL_α (1/rad)", min_value=0.0, key="htail_cl_alpha", format="%.2f")

with st.sidebar.expander("🪁 수직꼬리날개"):
    st.number_input("수직꼬리 면적 S_vtail (m²)", min_value=0.0, key="S_vtail", format="%.4f")
    st.number_input("수직꼬리 개수", min_value=0, max_value=4, step=1, key="vtail_count")
    st.number_input("수직꼬리 거리 (CG 기준, m)", min_value=0.0, key="vtail_arm", format="%.3f")
    st.number_input("수직꼬리 CL_α (1/rad)", min_value=0.0, key="vtail_cl_alpha", format="%.2f")

with st.sidebar.expander("🎯 초기 자세 / 각속도"):
    st.slider("초기 pitch (deg)", -30.0, 30.0, step=1.0, key="pitch0")
    st.slider("초기 roll (deg)", -30.0, 30.0, step=1.0, key="roll0")
    st.slider("초기 yaw (deg)", -30.0, 30.0, step=1.0, key="yaw0")
    st.slider("초기 pitch 각속도 (deg/s)", -60.0, 60.0, step=1.0, key="q0")
    st.slider("초기 roll 각속도 (deg/s)", -60.0, 60.0, step=1.0, key="p0")
    st.slider("초기 yaw 각속도 (deg/s)", -60.0, 60.0, step=1.0, key="r0")

with st.sidebar.expander("🔧 관성 / 감쇠 / 시간 (심화)"):
    st.number_input("Ix (roll 관성)", min_value=1e-9, key="Ix", format="%.4g")
    st.number_input("Iy (pitch 관성)", min_value=1e-9, key="Iy", format="%.4g")
    st.number_input("Iz (yaw 관성)", min_value=1e-9, key="Iz", format="%.4g")
    st.number_input("pitch 감쇠 기준 cd_pitch", min_value=0.0, key="cd_pitch", format="%.4g")
    st.number_input("roll 감쇠 기준 cd_roll", min_value=0.0, key="cd_roll", format="%.4g")
    st.number_input("yaw 감쇠 기준 cd_yaw", min_value=0.0, key="cd_yaw", format="%.4g")
    st.slider("회전 감쇠 배율", 0.0, 3.0, step=0.1, key="damping_mult")
    st.slider("시뮬레이션 시간 (s)", 2.0, 60.0, step=1.0, key="t_end")
    st.slider("시간 간격 dt (s)", 0.005, 0.1, step=0.005, key="dt")

stl_b64 = ""
stl_fwd, stl_up = "+X", "+Y"
stl_sig_current = None
stl_preview = None
stl_preview_error = None
stl_preview_key = None
stl_pre_rot = (0.0, 0.0, 0.0)
_UNIT = {"mm (밀리미터)": 0.001, "cm (센티미터)": 0.01, "m (미터)": 1.0}
with st.sidebar.expander("🛩️ 3D 모델 (STL 업로드 + 형상 분석)"):
    up = st.file_uploader("STL 파일 (.stl) 업로드", type=["stl"], key="stl_upload")
    unit_label = st.selectbox("STL 단위", list(_UNIT.keys()), index=0, key="stl_unit")
    stl_mass = st.number_input(
        "적용 전 질량 (kg)", min_value=0.001,
        value=float(st.session_state.get("stl_mass_input", st.session_state["mass"])),
        key="stl_mass_input", format="%.3f")
    stl_physics_scale = st.slider(
        "물리 크기 배율 (ray/면적/관성에 반영)", 0.10, 5.00, step=0.05,
        value=float(st.session_state.get("stl_physics_scale", 1.0)),
        key="stl_physics_scale")
    st.caption("초기 STL 방향 설정: 각도 숫자를 바꾸면 적용 전 3D 진단이 즉시 갱신됩니다.")
    r_roll, r_pitch, r_yaw = st.columns(3)
    stl_init_roll = r_roll.number_input(
        "Roll (deg)", value=float(st.session_state.get("stl_init_roll", 0.0)),
        step=1.0, key="stl_init_roll", format="%.1f")
    stl_init_pitch = r_pitch.number_input(
        "Pitch (deg)", value=float(st.session_state.get("stl_init_pitch", 0.0)),
        step=1.0, key="stl_init_pitch", format="%.1f")
    stl_init_yaw = r_yaw.number_input(
        "Yaw (deg)", value=float(st.session_state.get("stl_init_yaw", 0.0)),
        step=1.0, key="stl_init_yaw", format="%.1f")
    # 내부 회전축: roll=x, yaw=y, pitch=z.
    stl_pre_rot = (float(stl_init_roll), float(stl_init_yaw), float(stl_init_pitch))
    stl_cg_mode = st.selectbox(
        "무게중심(CG) 지정", ["STL 균일밀도 자동", "현재 입력 CG 사용", "기수 기준 비율", "기수 기준 거리(m)"],
        key="stl_cg_mode")
    stl_cg_ratio = st.slider("CG 위치 (% 기체 길이, 기수=0)", 0.0, 100.0, 35.0, step=1.0,
                             key="stl_cg_ratio") if stl_cg_mode == "기수 기준 비율" else None
    stl_cg_m = st.number_input("CG 위치 (기수 기준, m)", min_value=0.0,
                               value=float(st.session_state.get("stl_cg_m", st.session_state["cg"])),
                               key="stl_cg_m", format="%.4f") if stl_cg_mode == "기수 기준 거리(m)" else None
    stl_quality = st.selectbox("ray 물리 정밀도", list(STL_QUALITY.keys()), index=1, key="stl_quality")
    stl_show_preview = st.checkbox("적용 전 3D 진단 표시", value=True, key="stl_preview_on")

    if up is not None:
        raw = up.getvalue()
        mb = len(raw) / 1e6
        if st.checkbox("업로드한 STL 모델 사용", value=True, key="use_stl"):
            stl_b64 = base64.b64encode(raw).decode()
        st.caption(f"파일: {up.name} · {mb:.1f} MB")
        if mb > 6:
            st.warning("파일이 큽니다(>6MB). 렌더가 느릴 수 있어요. "
                       "이진(binary) STL / 폴리곤 수 줄인 모델 권장.")
        stl_sig_current = stl_signature(
            raw, unit_label, stl_physics_scale, stl_quality, stl_pre_rot,
            float(stl_mass), stl_cg_mode, stl_cg_ratio, stl_cg_m)
        applied_sig = st.session_state.get("_aero_signature")
        if applied_sig and applied_sig != stl_sig_current:
            st.info("STL 파일/단위/질량/물리 크기/초기 방향 각도/CG/정밀도 설정이 바뀌었습니다. "
                    "ray 물리에 반영하려면 아래 적용 버튼을 다시 누르세요.")
        if stl_show_preview:
            try:
                preview_cur = {k: st.session_state[k] for k in INPUT_KEYS}
                stl_preview_key = stl_preview_chart_key(stl_sig_current, preview_cur)
                stl_preview = make_stl_preview(
                    raw, _UNIT[unit_label] * float(stl_physics_scale),
                    float(stl_mass), stl_cg_mode, stl_cg_ratio, stl_cg_m,
                    preview_cur, stl_pre_rot)
            except Exception as e:
                stl_preview_error = f"{type(e).__name__}: {e}"
                st.warning(f"적용 전 3D 진단 생성 실패: {stl_preview_error}")

        if st.button("📐 STL 형상의 항공역학 특성 적용", width='stretch'):
            try:
                factor = _UNIT[unit_label] * float(stl_physics_scale)
                tris = stl_analysis.parse_stl(raw) * factor
                V = stl_analysis.rotate_mesh(tris, *stl_pre_rot)
                props = stl_analysis.analyze(V, "+X", "+Y", float(stl_mass))
                cg_from_nose = cg_from_stl_settings(
                    props, stl_cg_mode, float(st.session_state["cg"]),
                    stl_cg_ratio, stl_cg_m)

                Ix, Iy, Iz = inertia_with_cg_override(props, float(stl_mass), cg_from_nose)
                keys = ["length", "span", "height", "S_wing", "wing_pos", "cp_base",
                        "S_htail", "htail_arm", "S_vtail", "vtail_arm"]
                geo = {k: float(props[k]) for k in keys}
                geo["mass"] = float(stl_mass)
                geo["cg"] = cg_from_nose
                geo["Ix"], geo["Iy"], geo["Iz"] = Ix, Iy, Iz
                # STL 적용 시 CP 고정(자동이동 OFF): AoA 의존 CP 이동이
                # 영양력각 부근에서 강성 부호를 바꿔 한쪽으로 발산시키는 것을 방지
                geo["cp_auto"] = False

                # 표면 패널 공력 모델 생성(정렬 메시 + 무게중심점)
                aero_model = panel_aero.build_aero_model(
                    V, np.asarray(props["cm"]), **STL_QUALITY[stl_quality])
                q0 = 0.5 * float(st.session_state["rho"]) * float(st.session_state["V"]) ** 2
                cgp = panel_aero.cg_point_from_nose(aero_model, geo["cg"])
                props["k_theta_panel"] = panel_aero.pitch_stiffness(aero_model, q0, cgp)
                props["k_psi_panel"] = panel_aero.yaw_stiffness(aero_model, q0, cgp)
                k_th = max(abs(props["k_theta_panel"]), 1e-12)
                k_ps = max(abs(props["k_psi_panel"]), 1e-12)
                zeta = 0.38   # 목표 감쇠비(가볍게 진동하며 수렴)
                geo["cd_pitch"] = 2 * zeta * math.sqrt(k_th * geo["Iy"])
                geo["cd_yaw"] = 2 * zeta * math.sqrt(k_ps * geo["Iz"])
                geo["cd_roll"] = geo["Ix"] * 3.0   # roll 복원 없음 → 시간상수 기반

                props["cg_auto"] = float(props["cg"])
                props["cg_applied"] = float(cg_from_nose)
                props["mass"] = float(stl_mass)
                props["Ix"], props["Iy"], props["Iz"] = geo["Ix"], geo["Iy"], geo["Iz"]
                props["static_margin"] = geo["cp_base"] - geo["cg"]
                props["physics_scale"] = float(stl_physics_scale)
                props["quality"] = stl_quality
                props["pre_rot"] = [float(v) for v in stl_pre_rot]
                st.session_state["_aero_model"] = aero_model
                st.session_state["_aero_signature"] = stl_sig_current
                st.session_state["_stl_props"] = geo
                st.session_state["_apply_stl"] = True
                st.session_state["_stl_report"] = props
                st.session_state["sim"] = None         # 즉시 재계산되도록
                st.rerun()
            except Exception as e:
                st.error(f"STL 분석 실패: {e}")

        rp = st.session_state.get("_stl_report")
        if rp:
            st.success(
                f"추출됨(형상 기반 추정) · 삼각형 {rp['n_tri']:,}개 · ray {rp.get('quality', '기본')}\n\n"
                f"- 길이 {rp['length']:.3g} m · 날개폭 {rp['span']:.3g} m · 높이 {rp['height']:.3g} m\n"
                f"- 주날개 면적 {rp['S_wing']:.3g} m² · 수평꼬리 {rp['S_htail']:.2g} m²\n"
                f"- CG {rp.get('cg_applied', rp['cg']):.3g} m "
                f"(자동 추정 {rp.get('cg_auto', rp['cg']):.3g} m) · CP {rp['cp_base']:.3g} m\n"
                f"- 물리 크기 배율 {rp.get('physics_scale', 1.0):.2f}×\n"
                f"- 초기 방향 Roll/Pitch/Yaw = {rp.get('pre_rot', [0, 0, 0])[0]:.0f}° / "
                f"{rp.get('pre_rot', [0, 0, 0])[2]:.0f}° / {rp.get('pre_rot', [0, 0, 0])[1]:.0f}°\n"
                f"- Ix {rp['Ix']:.2e} · Iy {rp['Iy']:.2e} · Iz {rp['Iz']:.2e} kg·m²\n"
                f"- 표면 패널 공력 적용(세로강성 k_θ={rp.get('k_theta_panel', 0):.2f}, "
                f"방향강성 k_ψ={rp.get('k_psi_panel', 0):.2f})")
            if rp.get("k_theta_panel", 1) <= 0:
                st.warning(
                    "⚠️ 이 형상은 **세로 정적 불안정**(무게중심이 양력중심보다 뒤)하여 "
                    "기수가 점점 들리거나 숙여지며 **발산**합니다. 실제 모형도 이대로면 뒤집힙니다.\n\n"
                    "→ **무게중심(CG)을 앞으로** 옮기세요(기수에 무게추). 사이드바 ‘무게중심 CG’ 값을 "
                    f"현재 CP({rp['cp_base']:.3g} m)보다 **작게**(앞으로) 조정한 뒤 다시 시작하면 안정화됩니다. "
                    "(균일밀도 가정이라 실제 무게추 위치와 다를 수 있어요.)")
        st.caption("① 단위·질량·물리 크기·초기 방향 각도·CG 를 먼저 정하고 → "
                   "② **‘적용’** 을 누르면 길이·면적·CG·CP·관성·ray 공력표가 함께 반영됩니다. "
                   "이후 **▶ 시뮬레이션 시작/재시작**.")
    else:
        st.caption("업로드하지 않으면 기본 내장 항공기 모델을 사용합니다. "
                   "형상 분석을 쓰려면 모델의 기수/위 축을 고른 뒤 파일을 올리세요.")

st.sidebar.markdown("---")
auto = st.sidebar.checkbox("슬라이더 변경 시 자동 재계산", value=False, key="auto_run")
run_clicked = st.sidebar.button("▶ 시뮬레이션 시작 / 재시작", type="primary",
                                width='stretch')

# 현재 입력 스냅샷
cur = {k: st.session_state[k] for k in INPUT_KEYS}
cur["name"] = preset_name

# ---------------------------------------------------------------------------
# 시뮬레이션 실행 결정
# ---------------------------------------------------------------------------
# STL 을 사용 중이고 패널 공력 모델이 있으면 그것으로 동역학 계산
aero_ready = (
    bool(stl_b64)
    and st.session_state.get("_aero_model") is not None
    and st.session_state.get("_aero_signature") == stl_sig_current
)
stl_settings_pending = (
    bool(stl_b64)
    and st.session_state.get("_aero_signature") is not None
    and st.session_state.get("_aero_signature") != stl_sig_current
)
aero_model = st.session_state.get("_aero_model") if aero_ready else None
need = (st.session_state.get("sim") is None) or run_clicked or auto or stl_settings_pending
if need:
    st.session_state["sim"] = compute(cur, aero_model)

sim_data = st.session_state["sim"]
ac, env, res, assess = sim_data["ac"], sim_data["env"], sim_data["res"], sim_data["assess"]
pending_changed = (not auto) and (sim_data["cur"] != {**cur})

# ---------------------------------------------------------------------------
# 메인 — 헤더 / 상태
# ---------------------------------------------------------------------------
st.title("✈️ 항공기 비행 양상 시뮬레이터")
st.caption("받음각·무게중심·양력중심·꼬리날개 조건에 따른 pitch·roll·yaw 변화를 "
           "근사 계산하여 보여주는 **교육용** 도구입니다. 실제 비행 성능 예측이 아닙니다.")

if pending_changed:
    st.info("입력값이 바뀌었습니다. 좌측 **▶ 시뮬레이션 시작/재시작** 을 눌러 반영하세요. "
            "(현재 화면은 직전에 계산된 결과입니다.)")

if stl_settings_pending:
    st.info("STL 설정이 마지막 적용값과 달라 현재 ray 패널 공력은 비활성화했습니다. "
            "사이드바에서 **STL 형상의 항공역학 특성 적용** 을 다시 누르면 새 설정으로 계산합니다.")

if sim_data.get("aero_err"):
    st.error("⚠️ STL 표면 패널 공력 계산 중 오류가 발생해 매개변수 모델로 대체했습니다. "
             f"원인: `{sim_data['aero_err']}` — 이 메시지를 알려주시면 정확히 고치겠습니다.")

# 경고
aoa_max = float(max(abs(res.aoa.min()), abs(res.aoa.max())))
if aoa_max >= 18.0:
    st.error(f"⚠️ 실속 위험: 최대 받음각 {aoa_max:.1f}° (18° 이상). "
             "선형 양력 근사가 무너지고 양력이 급감하는 영역입니다.")
elif aoa_max >= 12.0:
    st.warning(f"주의: 최대 받음각 {aoa_max:.1f}° (12° 이상). 실속에 가까워지고 있습니다.")

# 상태 배지
c1, c2, c3, c4, c5, c6 = st.columns(6)
for col, title, key in [(c1, "전체 안정성", "overall"), (c2, "AoA", "aoa"),
                        (c3, "CP–CG", "cp_cg"), (c4, "Pitch", "pitch"),
                        (c5, "Roll", "roll"), (c6, "Yaw", "yaw")]:
    label, lvl = assess[key]
    col.markdown(f"**{title}**<br>{badge(label, lvl)}", unsafe_allow_html=True)

st.markdown("##### 🏗️ 제작 적합성 점수")
sc = assess["score"]
pcol1, pcol2 = st.columns([1, 4])
pcol1.metric("점수", f"{sc} / 100")
pcol2.progress(sc / 100.0)

# ---------------------------------------------------------------------------
# STL 적용 전 3D 진단
# ---------------------------------------------------------------------------
if stl_preview is not None:
    pp = stl_preview["props"]
    st.markdown("### 🧭 STL 적용 전 3D 진단")
    st.caption("아래 내용은 **적용 버튼을 누르기 전** 현재 STL 설정(단위·질량·물리 크기·초기 방향 각도·CG)과 "
               "초기 pitch/roll/yaw, 풍속 변경을 즉시 반영해 다시 계산한 미리보기입니다. "
               "검은 점=현재 CG, 보라색 X=CP, 굵은 화살표=현재 모멘트/발산 방향입니다.")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("CG / CP", f"{pp['cg_applied']:.3g} / {pp['cp_base']:.3g} m",
              f"CP-CG {pp['cp_base'] - pp['cg_applied']:+.3g} m")
    d2.metric("Preview AoA", f"{pp['alpha_preview']:.1f}°",
              f"β {pp['beta_preview']:.1f}°")
    d3.metric("미리보기 양력", f"{pp['L_preview']:.3g} N",
              f"Drag {pp['D_preview']:.3g} N")
    d4.metric("ray 패널", f"{pp['n_tri_preview']:,}개",
              f"원본 {pp['n_tri']:,}개")

    s1, s2, s3 = st.columns(3)
    for col, name, key in [(s1, "Pitch 발산 여부", "pitch"),
                           (s2, "Roll 발산 여부", "roll"),
                           (s3, "Yaw 발산 여부", "yaw")]:
        info = stl_preview["stability"][key]
        label, lvl = info["label"], info["level"]
        k_val, m_val = info["k"], info["moment"]
        col.markdown(f"**{name}**<br>{badge(label, lvl)}", unsafe_allow_html=True)
        col.caption(f"방향: {info['direction']} · k={k_val:.3g} N·m/rad · 현재 모멘트={m_val:.3g} N·m")

    st.plotly_chart(
        viz.stl_diagnostic_figure(
            stl_preview["tris"], pp, pp["cg_applied"], stl_preview["stability"]),
        width='stretch', key=stl_preview_key)

    if stl_preview["stability"]["pitch"]["level"] == "danger":
        st.warning(f"Pitch가 **{stl_preview['stability']['pitch']['direction']}** 쪽으로 발산합니다. "
                   "CG를 앞으로 옮기거나 수평꼬리/질량/크기를 조정해 보세요.")
    if stl_preview["stability"]["roll"]["level"] == "danger":
        st.warning(f"Roll이 **{stl_preview['stability']['roll']['direction']}** 쪽으로 발산합니다. "
                   "좌우 형상 비대칭 또는 기울어진 축 정렬을 확인해 보세요.")
    if stl_preview["stability"]["yaw"]["level"] == "danger":
        st.warning(f"Yaw가 **{stl_preview['stability']['yaw']['direction']}** 쪽으로 발산합니다. "
                   "수직꼬리 형상/축 정렬/CG 위치를 확인해 보세요.")
elif stl_preview_error:
    st.warning(f"STL 적용 전 3D 진단을 만들지 못했습니다: `{stl_preview_error}`")

# ---------------------------------------------------------------------------
# 🎬 실시간 3D 비행 애니메이션 (브라우저 WebGL/Three.js — 부드럽고 랙 없음)
# ---------------------------------------------------------------------------
st.markdown("### 🎬 실시간 3D 비행 애니메이션")
st.caption("**▶ 재생** 을 누르면 3D 항공기가 사이드바의 **시뮬레이션 시간**까지만 "
           "pitch·roll·yaw 자세를 시뮬레이션합니다(브라우저에서 실시간 적분). "
           "**반투명=초기 자세, 진한 색=현재 자세**. **마우스 드래그=시점 회전, 휠=확대**. "
           "사이드바 **‘🛩️ 3D 모델 (STL 업로드)’** 로 모델을 바꾸고 **‘모델 조절’** 로 크기·회전을 맞춥니다. "
           "(아래 그래프·분석과 같은 시간 구간 기준입니다)")
_anim_init = initial_from_dict(sim_data["cur"])
_anim_sim = sim_from_dict(sim_data["cur"])
components.html(
    animation.realtime_animation_html(ac, env, _anim_init, _anim_sim,
                                      aero_model=sim_data.get("aero_model"),
                                      stl_b64=stl_b64,
                                      align_fwd=stl_fwd, align_up=stl_up,
                                      pre_rot=stl_pre_rot),
    height=640, scrolling=False)

# ---------------------------------------------------------------------------
# 시간 슬라이더 (아래 정적 그림/그래프 전용)
# ---------------------------------------------------------------------------
t_max = float(res.t[-1])
spacing = float(res.t[1] - res.t[0])
if "t_now" in st.session_state and st.session_state["t_now"] > t_max:
    st.session_state["t_now"] = t_max
st.markdown("##### 🔎 정적 분석용 시점 선택")
t_now = st.slider("⏱️ 시각 t (s) — 아래 측면도/그래프를 이 시점으로 고정해서 자세히 봅니다",
                  0.0, t_max, step=spacing, key="t_now")
idx = res.index_at(t_now)

# ---------------------------------------------------------------------------
# 탭
# ---------------------------------------------------------------------------
tab_view, tab_graph, tab_ana, tab_in = st.tabs(
    ["🛩️ 비행 자세", "📈 그래프", "🧪 분석 / 해석", "📋 입력 요약"])

with tab_view:
    st.caption(f"현재 표시 시점: t = {res.t[idx]:.2f}s (위 ‘정적 분석용 시점 선택’ 슬라이더로 변경)")
    # 발표용 3D 자세(선택 시점 스냅샷)
    st.plotly_chart(viz.attitude_3d_figure(ac, res, idx), width='stretch')

    # 힘·모멘트 벡터가 필요한 분석용 단면도는 접어 둠(기본 닫힘)
    with st.expander("📐 보조 단면도 (힘·모멘트 벡터 확인용 · 측면/정면/상면)"):
        r1c1, r1c2 = st.columns(2)
        r1c1.plotly_chart(viz.side_view_figure(ac, env, res, idx), width='stretch')
        r1c2.plotly_chart(viz.front_view_figure(ac, env, res, idx), width='stretch')
        st.plotly_chart(viz.top_view_figure(ac, env, res, idx), width='stretch')

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pitch (deg)", f"{res.pitch[idx]:.1f}", f"초기 {res.pitch0:.1f}")
    m2.metric("Roll (deg)", f"{res.roll[idx]:.1f}", f"초기 {res.roll0:.1f}")
    m3.metric("Yaw (deg)", f"{res.yaw[idx]:.1f}", f"초기 {res.yaw0:.1f}")
    m4.metric("AoA (deg)", f"{res.aoa[idx]:.1f}")

with tab_graph:
    g1, g2 = st.columns(2)
    g1.plotly_chart(viz.time_series_figure(
        res.t, [(res.pitch, "pitch", "#1f77b4"), (res.roll, "roll", "#2ca02c"),
                (res.yaw, "yaw", "#d62728")],
        "자세각 변화", "각도 (deg)", t_now), width='stretch')
    g2.plotly_chart(viz.time_series_figure(
        res.t, [(res.aoa, "AoA", "#9467bd")], "받음각(AoA)", "AoA (deg)", t_now),
        width='stretch')
    g3, g4 = st.columns(2)
    g3.plotly_chart(viz.time_series_figure(
        res.t, [(res.cp, "CP 위치", "#8c564b")], "양력중심(CP) 위치", "x_cp (m)", t_now),
        width='stretch')
    g4.plotly_chart(viz.time_series_figure(
        res.t, [(res.L_wing, "주날개", "#2ca02c"), (res.L_tail, "꼬리날개", "#ff7f0e")],
        "양력", "양력 (N)", t_now), width='stretch')
    g5, g6 = st.columns(2)
    g5.plotly_chart(viz.time_series_figure(
        res.t, [(res.M_pitch, "M_pitch", "#1f77b4")], "Pitch 모멘트", "N·m", t_now),
        width='stretch')
    g6.plotly_chart(viz.time_series_figure(
        res.t, [(res.M_roll, "M_roll", "#2ca02c"), (res.M_yaw, "M_yaw", "#d62728")],
        "Roll / Yaw 모멘트", "N·m", t_now), width='stretch')

with tab_ana:
    st.subheader("🧭 상태 판정")
    rows = [("전체 안정성", "overall"), ("AoA 상태", "aoa"), ("CP–CG 관계", "cp_cg"),
            ("Pitch 경향", "pitch"), ("Roll 경향", "roll"), ("Yaw 경향", "yaw")]
    for name, key in rows:
        label, lvl = assess[key]
        st.markdown(f"- **{name}**: {badge(label, lvl)}", unsafe_allow_html=True)

    st.markdown(
        f"- **세로 정적안정 k_θ** = {assess['k_theta']:.3g} N·m/rad "
        f"({'양(+) → 안정' if assess['k_theta'] > 0 else '음(−) → 불안정'})")
    st.markdown(
        f"- **방향 안정 k_ψ** = {assess['k_psi']:.3g} N·m/rad "
        f"({'양(+) → 안정' if assess['k_psi'] > 0 else '음(−) → 불안정'})")

    st.subheader("📝 자동 해석 (과학탐구 보고서용)")
    for s in analysis.interpretation_sentences(ac, env, res, assess):
        st.markdown(f"- {s}")

    st.subheader("⚠️ 모델의 한계")
    st.markdown(
        "- 선형 양력계수 근사는 **작은 받음각**에서만 잘 맞습니다.\n"
        "- 양력중심 이동은 실제 익형 해석이 아니라 **단순 1차 근사**입니다.\n"
        "- roll 에는 상반각(dihedral) 복원이 포함되지 않아 비대칭이 있으면 계속 구릅니다.\n"
        "- 본 도구는 **설계 비교용**이며 실제 비행 성공을 보장하지 않습니다.")

with tab_in:
    st.subheader("📋 현재 적용된 입력값")
    show = dict(sim_data["cur"])
    st.json(show, expanded=False)
    st.caption("이 값은 마지막으로 ‘시작/재시작’ 버튼을 눌렀을 때(또는 자동 재계산 시) "
               "적용된 값입니다.")
