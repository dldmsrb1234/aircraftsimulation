# -*- coding: utf-8 -*-
"""
simulation.py
=============
시간에 따른 pitch / roll / yaw 자세 변화를 적분으로 계산한다.

회전 운동(각 축 독립, 감쇠 포함):
    각가속도 = (모멘트 − 감쇠계수 · 각속도) / 관성모멘트
    각속도  += 각가속도 · dt          (전진 오일러 적분)
    각도    += 각속도   · dt
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

from aircraft import Aircraft, Environment, InitialState, SimConfig
import physics


@dataclass
class SimResult:
    """시뮬레이션 결과 시계열(모두 numpy 배열, 각도는 deg)."""
    t: np.ndarray
    pitch: np.ndarray
    roll: np.ndarray
    yaw: np.ndarray
    aoa: np.ndarray          # 주날개 유효 받음각
    cp: np.ndarray           # 양력중심 위치
    L_wing: np.ndarray
    L_tail: np.ndarray
    M_pitch: np.ndarray
    M_roll: np.ndarray
    M_yaw: np.ndarray
    pitch_rate: np.ndarray   # deg/s
    roll_rate: np.ndarray
    yaw_rate: np.ndarray

    # 초기 자세(비교용, deg)
    pitch0: float = 0.0
    roll0: float = 0.0
    yaw0: float = 0.0

    def index_at(self, t_query: float) -> int:
        """주어진 시각에 가장 가까운 배열 인덱스."""
        return int(np.clip(np.searchsorted(self.t, t_query), 0, len(self.t) - 1))


def run_simulation(ac: Aircraft, env: Environment,
                   init: InitialState, sim: SimConfig,
                   aero_model=None) -> SimResult:
    """전체 시뮬레이션 실행 후 SimResult 반환.

    aero_model 이 주어지면 STL 표면 패널 공력으로 모멘트를 계산하고,
    없으면 매개변수(면적·CP·CG) 모델을 사용한다.
    """
    n = int(round(sim.t_end / sim.dt)) + 1
    n = max(n, 2)

    # 상태 변수 (라디안)
    theta = math.radians(init.pitch0_deg)   # pitch
    phi = math.radians(init.roll0_deg)       # roll
    psi = math.radians(init.yaw0_deg)        # yaw
    q_rate = math.radians(init.q0_deg)       # pitch 각속도
    p_rate = math.radians(init.p0_deg)       # roll  각속도
    r_rate = math.radians(init.r0_deg)       # yaw   각속도

    q_dyn = physics.dynamic_pressure(env.rho, env.V)
    I_pitch, I_roll, I_yaw = ac.Iy, ac.Ix, ac.Iz

    # --- 모멘트 제공자: STL 패널 공력(aero_model) 또는 매개변수 모델 ---
    if aero_model is not None:
        import panel_aero
        cg_point = panel_aero.cg_point_from_nose(aero_model, ac.cg)   # 슬라이더 CG 반영
        k_th = abs(panel_aero.pitch_stiffness(aero_model, q_dyn, cg_point))
        k_ps = abs(panel_aero.yaw_stiffness(aero_model, q_dyn, cg_point))
        zeta = 0.4
        cd_p = 2 * zeta * math.sqrt(max(k_th, 1e-12) * I_pitch) * sim.damping_mult
        cd_y = 2 * zeta * math.sqrt(max(k_ps, 1e-12) * I_yaw) * sim.damping_mult
        cd_r = I_roll * 3.0 * sim.damping_mult

        def moments(th, ph, ps_):
            w = panel_aero.relative_wind_body(th, ph, ps_)
            F, M, al, _ = panel_aero.aero(aero_model, w, q_dyn, cg_point)
            mr, mp, my = panel_aero.body_moments(M)
            mr, mp, my = float(mr), float(mp), float(my)     # 순수 float 로 고정
            fy = float(F[1]) if abs(float(F[1])) > 1e-6 else 1e-6
            cp = float(np.clip(ac.cg - mp / fy, 0.0, max(ac.length, 1e-3)))
            return mp, mr, my, (math.degrees(float(al)), cp, float(F[1]), float(-F[0]))
    else:
        cd_p = ac.cd_pitch * sim.damping_mult
        cd_r = ac.cd_roll * sim.damping_mult
        cd_y = ac.cd_yaw * sim.damping_mult
        k_th = abs(physics.pitch_stiffness(ac, env, 0.0))
        k_ps = abs(physics.yaw_stiffness(ac, env))

        def moments(th, ph, ps_):
            ps = physics.pitch_state(ac, env, th)
            rs = physics.roll_moment(ac, env, ph)
            ys = physics.yaw_moment(ac, env, ps_)
            return (ps["M_pitch"], rs["M_roll"], ys["M_yaw"],
                    (ps["alpha_wing_deg"], ps["x_cp"], ps["L_wing"], ps["L_tail"]))

    # 수치 안정성: ω·h<1.5 가 되도록 dt 를 내부 서브스텝으로 분할
    w_pitch = math.sqrt(k_th / I_pitch) if I_pitch > 0 else 0.0
    w_yaw = math.sqrt(k_ps / I_yaw) if I_yaw > 0 else 0.0
    w_max = max(w_pitch, w_yaw, 1e-9)
    n_sub = int(min(200, max(1, math.ceil(sim.dt * w_max / 1.5))))
    h = sim.dt / n_sub

    rec = {k: np.zeros(n) for k in (
        "t", "pitch", "roll", "yaw", "aoa", "cp",
        "L_wing", "L_tail", "M_pitch", "M_roll", "M_yaw",
        "pitch_rate", "roll_rate", "yaw_rate")}

    LO_T, HI_T, LIM = math.radians(-90), math.radians(90), math.radians(180)

    for i in range(n):
        mp, mr, my, aux = moments(theta, phi, psi)
        rec["t"][i] = i * sim.dt
        rec["pitch"][i] = math.degrees(theta)
        rec["roll"][i] = math.degrees(phi)
        rec["yaw"][i] = math.degrees(psi)
        rec["aoa"][i] = aux[0]
        rec["cp"][i] = aux[1]
        rec["L_wing"][i] = aux[2]
        rec["L_tail"][i] = aux[3]
        rec["M_pitch"][i] = mp
        rec["M_roll"][i] = mr
        rec["M_yaw"][i] = my
        rec["pitch_rate"][i] = math.degrees(q_rate)
        rec["roll_rate"][i] = math.degrees(p_rate)
        rec["yaw_rate"][i] = math.degrees(r_rate)

        # 반암시적 적분: rate=(rate+(M/I)·h)/(1+(c/I)·h)  (감쇠 무조건 안정)
        for _ in range(n_sub):
            mp, mr, my, _ = moments(theta, phi, psi)
            q_rate = (q_rate + (mp / I_pitch) * h) / (1.0 + (cd_p / I_pitch) * h)
            p_rate = (p_rate + (mr / I_roll) * h) / (1.0 + (cd_r / I_roll) * h)
            r_rate = (r_rate + (my / I_yaw) * h) / (1.0 + (cd_y / I_yaw) * h)
            theta += q_rate * h
            phi += p_rate * h
            psi += r_rate * h
            theta = LO_T if theta < LO_T else (HI_T if theta > HI_T else theta)
            phi = -LIM if phi < -LIM else (LIM if phi > LIM else phi)
            psi = -LIM if psi < -LIM else (LIM if psi > LIM else psi)

    return SimResult(
        t=rec["t"], pitch=rec["pitch"], roll=rec["roll"], yaw=rec["yaw"],
        aoa=rec["aoa"], cp=rec["cp"],
        L_wing=rec["L_wing"], L_tail=rec["L_tail"],
        M_pitch=rec["M_pitch"], M_roll=rec["M_roll"], M_yaw=rec["M_yaw"],
        pitch_rate=rec["pitch_rate"], roll_rate=rec["roll_rate"],
        yaw_rate=rec["yaw_rate"],
        pitch0=init.pitch0_deg, roll0=init.roll0_deg, yaw0=init.yaw0_deg,
    )
