# -*- coding: utf-8 -*-
"""
physics.py
==========
근사 공력(空力) 물리 모델.

이 모듈은 "고등학교 과학탐구 수준"으로 단순화된 식만 사용한다.
실제 익형(airfoil) 해석이나 CFD 가 아니라, 받음각·면적·거리로부터
양력과 회전 모멘트를 *근사*하는 교육용 모델이다.

부호 약속
---------
* pitch 모멘트 M_pitch : 기수 들림(+)
* roll  모멘트 M_roll  : 우측 날개 내려감(+)
* yaw   모멘트 M_yaw   : 기수 우향(+)
* 위치 x : 기수=0, 꼬리쪽 +  (m)
"""

from __future__ import annotations
import math
from aircraft import Aircraft, Environment


# ---------------------------------------------------------------------------
# 1) 동압
# ---------------------------------------------------------------------------
def dynamic_pressure(rho: float, V: float) -> float:
    """동압  q = 1/2 · ρ · V²  [Pa]"""
    return 0.5 * rho * V * V


# ---------------------------------------------------------------------------
# 2) AoA 기반 양력계수 (선형 + 실속 근사)
# ---------------------------------------------------------------------------
def cl_from_aoa(alpha_rad: float, cl_alpha: float,
                alpha_stall_rad: float, cl_max: float) -> float:
    """
    받음각 α(라디안)에 대한 양력계수 CL.

    * |α| ≤ 실속각  : CL = CL_α · α            (선형 구간)
    * |α| > 실속각  : 실속 후 CL 이 점차 감소   (단순 선형 감쇠 근사)

    실제 익형의 비선형 거동을 정밀하게 재현하지 않는 *근사*임에 유의.
    """
    sign = 1.0 if alpha_rad >= 0.0 else -1.0
    a = abs(alpha_rad)

    cl_linear = cl_alpha * a
    cl_peak = cl_alpha * alpha_stall_rad          # 실속각에서의 값(정점 근사)
    if cl_max > 0:
        cl_peak = min(cl_peak, cl_max) if cl_max < cl_peak else cl_peak

    if a <= alpha_stall_rad:
        return sign * cl_linear

    # 실속 이후: 정점에서 출발해 기울기 0.7·CL_α 로 감소, 정점의 35%까지만 하강
    excess = a - alpha_stall_rad
    cl_post = cl_peak - 0.7 * cl_alpha * excess
    cl_post = max(cl_post, 0.35 * cl_peak)
    return sign * cl_post


def cl_alpha_effective(alpha_rad: float, cl_alpha: float,
                       alpha_stall_rad: float, cl_max: float,
                       eps: float = 1e-4) -> float:
    """현재 α 근방의 국소 양력기울기(수치미분). 실속 후 음(-)이 될 수 있음."""
    f1 = cl_from_aoa(alpha_rad + eps, cl_alpha, alpha_stall_rad, cl_max)
    f0 = cl_from_aoa(alpha_rad - eps, cl_alpha, alpha_stall_rad, cl_max)
    return (f1 - f0) / (2 * eps)


# ---------------------------------------------------------------------------
# 3) 양력중심(CP) 위치
# ---------------------------------------------------------------------------
def cp_position(wing, alpha_rad: float) -> float:
    """
    양력중심 위치 x_cp [m] (기수 기준).
      x_cp = x_cp_base + k_cp · α      (자동 이동 ON 일 때)
    실제 공력중심 이동을 단순 1차식으로 근사한 것.
    """
    if wing.cp_auto:
        return wing.cp_base + wing.k_cp * alpha_rad
    return wing.cp_base


# ---------------------------------------------------------------------------
# 4) 양력
# ---------------------------------------------------------------------------
def wing_lift(q_dyn: float, wing, alpha_rad: float):
    """주날개 양력 L_wing = q · S_wing · CL_wing.  (L, CL) 반환"""
    cl = cl_from_aoa(alpha_rad, wing.cl_alpha,
                     math.radians(wing.alpha_stall_deg), wing.cl_max)
    return q_dyn * wing.area * cl, cl


def tail_lift(q_dyn: float, htail, alpha_tail_rad: float):
    """수평꼬리날개 양력 L_tail = q · S_tail · CL_tail.  (L, CL) 반환"""
    cl = htail.cl_alpha * alpha_tail_rad
    return q_dyn * htail.area * cl, cl


# ---------------------------------------------------------------------------
# 5~8) 모멘트
# ---------------------------------------------------------------------------
def pitch_state(ac: Aircraft, env: Environment, theta: float) -> dict:
    """
    현재 pitch 자세 θ(라디안)에서의 공력 상태와 pitch 모멘트.

    유효 받음각 = 설치각 + 자세각
      α_wing = i_wing + θ
      α_tail = i_tail + θ

    pitch 모멘트(기수 들림 +):
      M_pitch = L_wing·(x_cg − x_cp) − L_tail·l_tail
        · L_wing 이 CG 앞(x_cp < x_cg)에 작용 → 기수 들림(+)
        · 꼬리날개 양력(위) 은 CG 뒤에 작용 → 기수 숙임(−)
    """
    q_dyn = dynamic_pressure(env.rho, env.V)

    alpha_w = math.radians(ac.wing.aoa_deg) + theta
    L_w, cl_w = wing_lift(q_dyn, ac.wing, alpha_w)
    x_cp = cp_position(ac.wing, alpha_w)

    alpha_t = math.radians(ac.htail.aoa_deg) + theta
    L_t, cl_t = tail_lift(q_dyn, ac.htail, alpha_t)

    M = L_w * (ac.cg - x_cp) - L_t * ac.htail.arm

    return {
        "q_dyn": q_dyn,
        "alpha_wing_deg": math.degrees(alpha_w),
        "alpha_tail_deg": math.degrees(alpha_t),
        "L_wing": L_w, "CL_wing": cl_w,
        "L_tail": L_t, "CL_tail": cl_t,
        "x_cp": x_cp,
        "M_pitch": M,
    }


def roll_moment(ac: Aircraft, env: Environment, phi: float) -> dict:
    """
    roll 모멘트(우측 날개 내려감 +).
      M_roll = (비대칭 양력) · (가로 거리)
      비대칭 양력 = asymmetry · L_wing,  가로 거리 = b/4
    좌우 비대칭이 클수록 한쪽 양력이 커져 계속 구르는 경향.
    (※ 본 모델은 상반각(dihedral)에 의한 roll 복원은 포함하지 않음 → 한계 참고)
    """
    q_dyn = dynamic_pressure(env.rho, env.V)
    alpha_w = math.radians(ac.wing.aoa_deg)
    L_w, _ = wing_lift(q_dyn, ac.wing, alpha_w)
    asym_lift = ac.asymmetry * L_w
    lateral = ac.wing.span / 4.0
    M = asym_lift * lateral
    return {"M_roll": M, "L_wing": L_w}


def yaw_moment(ac: Aircraft, env: Environment, psi: float) -> dict:
    """
    yaw 모멘트(기수 우향 +).

    수직꼬리날개의 풍향계(weathercock) 안정:
      옆미끄럼각 β ≈ ψ 가 생기면 수직꼬리날개가 측력을 만들어 ψ 를 줄임(복원).
      측력 = q · S_v · n · CL_α_v · β
      M_restore = − 측력 · arm           (ψ 를 0 으로 되돌림)
    좌우 비대칭은 작은 yaw 외란을 추가한다.
    """
    q_dyn = dynamic_pressure(env.rho, env.V)
    beta = psi
    side = q_dyn * ac.vtail.area * ac.vtail.count * ac.vtail.cl_alpha * beta
    M_restore = -side * ac.vtail.arm
    # 비대칭에 의한 작은 yaw 외란
    M_asym = ac.asymmetry * 0.05 * q_dyn * ac.wing.area * ac.wing.span
    return {"M_yaw": M_restore + M_asym, "side_force": side}


def yaw_stiffness(ac: Aircraft, env: Environment) -> float:
    """yaw 복원 강성 k_ψ = q · S_v · n · CL_α_v · arm  [N·m/rad] (방향안정 지표)."""
    q_dyn = dynamic_pressure(env.rho, env.V)
    return q_dyn * ac.vtail.area * ac.vtail.count * ac.vtail.cl_alpha * ac.vtail.arm


def pitch_stiffness(ac: Aircraft, env: Environment, theta: float = 0.0,
                    eps: float = 1e-3) -> float:
    """
    pitch 복원 강성 k_θ = −dM_pitch/dθ  [N·m/rad].
    k_θ > 0 이면 정적 세로 안정(자세를 되돌리려는 경향).
    """
    m1 = pitch_state(ac, env, theta + eps)["M_pitch"]
    m0 = pitch_state(ac, env, theta - eps)["M_pitch"]
    return -(m1 - m0) / (2 * eps)
