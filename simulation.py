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
                   init: InitialState, sim: SimConfig) -> SimResult:
    """전체 시뮬레이션 실행 후 SimResult 반환."""
    n = int(round(sim.t_end / sim.dt)) + 1
    n = max(n, 2)

    # 상태 변수 (라디안)
    theta = math.radians(init.pitch0_deg)   # pitch
    phi = math.radians(init.roll0_deg)       # roll
    psi = math.radians(init.yaw0_deg)        # yaw
    q_rate = math.radians(init.q0_deg)       # pitch 각속도
    p_rate = math.radians(init.p0_deg)       # roll  각속도
    r_rate = math.radians(init.r0_deg)       # yaw   각속도

    # 감쇠계수 (사용자 배율 적용)
    cd_p = ac.cd_pitch * sim.damping_mult
    cd_r = ac.cd_roll * sim.damping_mult
    cd_y = ac.cd_yaw * sim.damping_mult

    # 수치 안정성: 내부 서브스텝
    #   고유진동수 ω=√(k/I) 가 크면(작은 모델 등) ω·h<1.5 가 되도록 dt 를 잘게 쪼갠다.
    #   감쇠는 아래 적분에서 '암시적'으로 처리하므로 감쇠가 커도 발산하지 않는다.
    k_th = abs(physics.pitch_stiffness(ac, env, 0.0))
    k_ps = abs(physics.yaw_stiffness(ac, env))
    w_pitch = math.sqrt(k_th / ac.Iy) if ac.Iy > 0 else 0.0
    w_yaw = math.sqrt(k_ps / ac.Iz) if ac.Iz > 0 else 0.0
    w_max = max(w_pitch, w_yaw, 1e-9)
    n_sub = int(min(200, max(1, math.ceil(sim.dt * w_max / 1.5))))
    h = sim.dt / n_sub

    # 기록 버퍼
    rec = {k: np.zeros(n) for k in (
        "t", "pitch", "roll", "yaw", "aoa", "cp",
        "L_wing", "L_tail", "M_pitch", "M_roll", "M_yaw",
        "pitch_rate", "roll_rate", "yaw_rate")}

    LO_T, HI_T, LIM = math.radians(-90), math.radians(90), math.radians(180)

    for i in range(n):
        # --- 현재 상태에서의 공력/모멘트 (기록용) ---
        ps = physics.pitch_state(ac, env, theta)
        rs = physics.roll_moment(ac, env, phi)
        ys = physics.yaw_moment(ac, env, psi)

        rec["t"][i] = i * sim.dt
        rec["pitch"][i] = math.degrees(theta)
        rec["roll"][i] = math.degrees(phi)
        rec["yaw"][i] = math.degrees(psi)
        rec["aoa"][i] = ps["alpha_wing_deg"]
        rec["cp"][i] = ps["x_cp"]
        rec["L_wing"][i] = ps["L_wing"]
        rec["L_tail"][i] = ps["L_tail"]
        rec["M_pitch"][i] = ps["M_pitch"]
        rec["M_roll"][i] = rs["M_roll"]
        rec["M_yaw"][i] = ys["M_yaw"]
        rec["pitch_rate"][i] = math.degrees(q_rate)
        rec["roll_rate"][i] = math.degrees(p_rate)
        rec["yaw_rate"][i] = math.degrees(r_rate)

        # --- dt 를 n_sub 개의 반암시적 스텝으로 적분 ---
        #   각속도: rate_new = (rate + (M/I)·h) / (1 + (c/I)·h)
        #   → 감쇠 항이 분모로 들어가 c 가 아무리 커도 |계수|<1 (무조건 안정)
        for _ in range(n_sub):
            mp = physics.pitch_state(ac, env, theta)["M_pitch"]
            mr = physics.roll_moment(ac, env, phi)["M_roll"]
            my = physics.yaw_moment(ac, env, psi)["M_yaw"]
            q_rate = (q_rate + (mp / ac.Iy) * h) / (1.0 + (cd_p / ac.Iy) * h)
            p_rate = (p_rate + (mr / ac.Ix) * h) / (1.0 + (cd_r / ac.Ix) * h)
            r_rate = (r_rate + (my / ac.Iz) * h) / (1.0 + (cd_y / ac.Iz) * h)
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
