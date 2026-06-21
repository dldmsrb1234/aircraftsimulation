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
    assess = analysis.overall_assessment(ac, env, res)
    used_aero = (aero_model is not None and aero_err is None)
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
    st.number_input("Ix (roll 관성)", min_value=0.0, key="Ix", format="%.4g")
    st.number_input("Iy (pitch 관성)", min_value=0.0, key="Iy", format="%.4g")
    st.number_input("Iz (yaw 관성)", min_value=0.0, key="Iz", format="%.4g")
    st.number_input("pitch 감쇠 기준 cd_pitch", min_value=0.0, key="cd_pitch", format="%.4g")
    st.number_input("roll 감쇠 기준 cd_roll", min_value=0.0, key="cd_roll", format="%.4g")
    st.number_input("yaw 감쇠 기준 cd_yaw", min_value=0.0, key="cd_yaw", format="%.4g")
    st.slider("회전 감쇠 배율", 0.0, 3.0, step=0.1, key="damping_mult")
    st.slider("시뮬레이션 시간 (s)", 2.0, 60.0, step=1.0, key="t_end")
    st.slider("시간 간격 dt (s)", 0.005, 0.1, step=0.005, key="dt")

stl_b64 = ""
stl_fwd, stl_up = "+X", "+Y"
_UNIT = {"mm (밀리미터)": 0.001, "cm (센티미터)": 0.01, "m (미터)": 1.0}
with st.sidebar.expander("🛩️ 3D 모델 (STL 업로드 + 형상 분석)"):
    up = st.file_uploader("STL 파일 (.stl) 업로드", type=["stl"], key="stl_upload")
    c_f, c_u = st.columns(2)
    stl_fwd = c_f.selectbox("기수(앞) 축", stl_analysis.AXIS_NAMES, index=0, key="stl_fwd")
    stl_up = c_u.selectbox("위(상단) 축", stl_analysis.AXIS_NAMES, index=2, key="stl_up")
    unit_label = st.selectbox("STL 단위", list(_UNIT.keys()), index=0, key="stl_unit")

    if up is not None:
        raw = up.getvalue()
        mb = len(raw) / 1e6
        if st.checkbox("업로드한 STL 모델 사용", value=True, key="use_stl"):
            stl_b64 = base64.b64encode(raw).decode()
        st.caption(f"파일: {up.name} · {mb:.1f} MB")
        if mb > 6:
            st.warning("파일이 큽니다(>6MB). 렌더가 느릴 수 있어요. "
                       "이진(binary) STL / 폴리곤 수 줄인 모델 권장.")

        if st.button("📐 STL 형상의 항공역학 특성 적용", width='stretch'):
            try:
                factor = _UNIT[unit_label]
                tris = stl_analysis.parse_stl(raw) * factor
                props = stl_analysis.analyze(tris, stl_fwd, stl_up,
                                             float(st.session_state["mass"]))
                keys = ["length", "span", "height", "S_wing", "wing_pos", "cp_base",
                        "cg", "S_htail", "htail_arm", "S_vtail", "vtail_arm",
                        "Ix", "Iy", "Iz"]
                geo = {k: float(props[k]) for k in keys}
                # STL 적용 시 CP 고정(자동이동 OFF): AoA 의존 CP 이동이
                # 영양력각 부근에서 강성 부호를 바꿔 한쪽으로 발산시키는 것을 방지
                geo["cp_auto"] = False

                # 새 형상(관성·강성)에 맞는 감쇠 자동 산출 → 발산/과감쇠 방지
                merged = {k: st.session_state[k] for k in INPUT_KEYS}
                merged.update(geo)
                merged["name"] = "stl"
                ac_t = aircraft_from_dict(merged)
                env_t = environment_from_dict(merged)
                k_th_signed = physics.pitch_stiffness(ac_t, env_t, 0.0)
                k_th = max(abs(k_th_signed), 1e-12)
                k_ps = max(abs(physics.yaw_stiffness(ac_t, env_t)), 1e-12)
                zeta = 0.35   # 목표 감쇠비(가볍게 진동하며 수렴)
                geo["cd_pitch"] = 2 * zeta * math.sqrt(k_th * geo["Iy"])
                geo["cd_yaw"] = 2 * zeta * math.sqrt(k_ps * geo["Iz"])
                geo["cd_roll"] = geo["Ix"] * 3.0   # roll 복원 없음 → 시간상수 기반

                # 표면 패널 공력 모델 생성(정렬 메시 + 무게중심점)
                V = stl_analysis.align_mesh(tris, stl_fwd, stl_up)
                aero_model = panel_aero.build_aero_model(V, np.asarray(props["cm"]))
                q0 = 0.5 * float(st.session_state["rho"]) * float(st.session_state["V"]) ** 2
                cgp = panel_aero.cg_point_from_nose(aero_model, geo["cg"])
                props["k_theta_panel"] = panel_aero.pitch_stiffness(aero_model, q0, cgp)
                props["k_psi_panel"] = panel_aero.yaw_stiffness(aero_model, q0, cgp)

                props["k_theta"] = k_th_signed
                props["static_margin"] = geo["cp_base"] - geo["cg"]
                st.session_state["_aero_model"] = aero_model
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
                f"추출됨(형상 기반 추정) · 삼각형 {rp['n_tri']:,}개\n\n"
                f"- 길이 {rp['length']:.3g} m · 날개폭 {rp['span']:.3g} m · 높이 {rp['height']:.3g} m\n"
                f"- 주날개 면적 {rp['S_wing']:.3g} m² · 수평꼬리 {rp['S_htail']:.2g} m²\n"
                f"- CG {rp['cg']:.3g} m · CP {rp['cp_base']:.3g} m (기수 기준)\n"
                f"- Ix {rp['Ix']:.2e} · Iy {rp['Iy']:.2e} · Iz {rp['Iz']:.2e} kg·m²\n"
                f"- 표면 패널 공력 적용(세로강성 k_θ={rp.get('k_theta_panel', 0):.2f}, "
                f"방향강성 k_ψ={rp.get('k_psi_panel', 0):.2f})")
            if rp.get("k_theta_panel", rp.get("k_theta", 1)) <= 0:
                st.warning(
                    "⚠️ 이 형상은 **세로 정적 불안정**(무게중심이 양력중심보다 뒤)하여 "
                    "기수가 점점 들리거나 숙여지며 **발산**합니다. 실제 모형도 이대로면 뒤집힙니다.\n\n"
                    "→ **무게중심(CG)을 앞으로** 옮기세요(기수에 무게추). 사이드바 ‘무게중심 CG’ 값을 "
                    f"현재 CP({rp['cp_base']:.3g} m)보다 **작게**(앞으로) 조정한 뒤 다시 시작하면 안정화됩니다. "
                    "(균일밀도 가정이라 실제 무게추 위치와 다를 수 있어요.)")
        st.caption("① 기수/위 축을 모델에 맞게 고르고 → ② **‘질량(kg)’** 을 먼저 입력한 뒤 → "
                   "③ **‘적용’** 을 누르면 길이·면적·CG·CP·관성이 자동 반영됩니다. "
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
aero_model = st.session_state.get("_aero_model") if stl_b64 else None
need = (st.session_state.get("sim") is None) or run_clicked or auto
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
# 🎬 실시간 3D 비행 애니메이션 (브라우저 WebGL/Three.js — 부드럽고 랙 없음)
# ---------------------------------------------------------------------------
st.markdown("### 🎬 실시간 3D 비행 애니메이션 (연속 시뮬레이션)")
st.caption("**▶ 재생** 을 누르면 3D 항공기가 **시간 제한 없이 ⏹ 정지를 누를 때까지 계속** "
           "pitch·roll·yaw 자세를 시뮬레이션합니다(브라우저에서 실시간 적분). "
           "**반투명=초기 자세, 진한 색=현재 자세**. **마우스 드래그=시점 회전, 휠=확대**. "
           "사이드바 **‘🛩️ 3D 모델 (STL 업로드)’** 로 모델을 바꾸고 **‘모델 조절’** 로 크기·회전을 맞춥니다. "
           "(아래 그래프·분석은 입력한 시뮬레이션 시간 구간 기준입니다)")
_anim_init = initial_from_dict(sim_data["cur"])
_anim_sim = sim_from_dict(sim_data["cur"])
components.html(
    animation.realtime_animation_html(ac, env, _anim_init, _anim_sim,
                                      aero_model=sim_data.get("aero_model"),
                                      stl_b64=stl_b64, align_fwd=stl_fwd, align_up=stl_up),
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
