# -*- coding: utf-8 -*-
"""
aircraft.py
============
항공기 / 날개 / 꼬리날개를 표현하는 dataclass 정의와,
UI(딕셔너리) 값으로부터 dataclass 객체를 만들어 주는 헬퍼 함수들.

좌표 약속 (물리/시각화 공통)
---------------------------------
* x : 기체 길이 방향. 기수(nose)=0, 꼬리쪽이 +  [m]
* y : 우측 날개 방향이 +  [m]
* z : 위쪽이 +  [m]  (시각화 기준. 물리 모멘트 계산은 부호를 명시해서 사용)
* pitch(θ) : 기수가 위로 들리면 +
* roll(φ)  : 우측 날개가 아래로 내려가면 +
* yaw(ψ)   : 기수가 우측으로 돌면 +
"""

from __future__ import annotations
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 구성 요소 dataclass
# ---------------------------------------------------------------------------
@dataclass
class MainWing:
    """주(主)날개."""
    area: float          # 주날개 면적 S_wing [m^2]
    position: float      # 주날개 위치 (기수 기준) [m]
    aoa_deg: float       # 주날개 설치 받음각(AoA) [deg]
    cp_base: float       # 양력중심(CP) 기준 위치 (기수 기준) [m]
    cp_auto: bool        # AoA 변화에 따른 CP 자동 이동 사용 여부
    k_cp: float          # CP 이동 계수 [m / rad]
    cl_alpha: float      # 양력기울기 dCL/dα [1/rad]
    alpha_stall_deg: float  # 실속 시작 받음각 [deg]
    cl_max: float        # 최대 양력계수(참고/표시용)
    span: float          # 날개폭 b [m]


@dataclass
class HorizontalTail:
    """수평꼬리날개(승강타 역할)."""
    area: float          # 면적 [m^2]
    aoa_deg: float       # 설치 받음각 [deg]
    arm: float           # 무게중심(CG)으로부터의 거리 l_tail [m]
    height: float        # 수직 위치(시각화용) [m]
    cl_alpha: float      # 양력기울기 [1/rad]


@dataclass
class VerticalTail:
    """수직꼬리날개(방향 안정)."""
    area: float          # 1장당 면적 [m^2]
    count: int           # 수직꼬리날개 개수
    arm: float           # CG으로부터의 거리 [m]
    cl_alpha: float      # 측력 기울기 [1/rad]


@dataclass
class Aircraft:
    """항공기 전체."""
    name: str
    mass: float          # 전체 질량 [kg]
    length: float        # 기체 길이 [m]
    span: float          # 전체 날개폭 [m]
    height: float        # 높이 [m]
    cg: float            # 무게중심 위치 (기수 기준) [m]
    asymmetry: float     # 좌우 비대칭 정도 [-1 ~ 1]
    Ix: float            # roll 관성모멘트 [kg·m^2]
    Iy: float            # pitch 관성모멘트
    Iz: float            # yaw 관성모멘트
    cd_pitch: float      # pitch 회전 감쇠 기준값
    cd_roll: float       # roll  회전 감쇠 기준값
    cd_yaw: float        # yaw   회전 감쇠 기준값
    wing: MainWing
    htail: HorizontalTail
    vtail: VerticalTail


# ---------------------------------------------------------------------------
# 환경 / 초기상태 / 시뮬레이션 설정
# ---------------------------------------------------------------------------
@dataclass
class Environment:
    V: float             # 풍속 [m/s]
    rho: float           # 공기밀도 [kg/m^3]


@dataclass
class InitialState:
    pitch0_deg: float
    roll0_deg: float
    yaw0_deg: float
    q0_deg: float        # 초기 pitch 각속도 [deg/s]
    p0_deg: float        # 초기 roll  각속도 [deg/s]
    r0_deg: float        # 초기 yaw   각속도 [deg/s]


@dataclass
class SimConfig:
    t_end: float         # 그래프 기록 시간 [s]
    dt: float            # 시간 간격 [s]
    damping_mult: float  # 감쇠 배율(사용자 조절용 단일 노브)


# ---------------------------------------------------------------------------
# 딕셔너리 -> dataclass 변환 헬퍼 (Streamlit session_state 의 flat dict 사용)
# ---------------------------------------------------------------------------
def aircraft_from_dict(d: dict) -> Aircraft:
    wing = MainWing(
        area=d["S_wing"],
        position=d["wing_pos"],
        aoa_deg=d["wing_aoa"],
        cp_base=d["cp_base"],
        cp_auto=d["cp_auto"],
        k_cp=d["k_cp"],
        cl_alpha=d["cl_alpha"],
        alpha_stall_deg=d["alpha_stall"],
        cl_max=d["cl_max"],
        span=d["span"],
    )
    htail = HorizontalTail(
        area=d["S_htail"],
        aoa_deg=d["htail_aoa"],
        arm=d["htail_arm"],
        height=d["htail_height"],
        cl_alpha=d["htail_cl_alpha"],
    )
    vtail = VerticalTail(
        area=d["S_vtail"],
        count=int(d["vtail_count"]),
        arm=d["vtail_arm"],
        cl_alpha=d["vtail_cl_alpha"],
    )
    return Aircraft(
        name=d.get("name", d.get("_preset", "사용자 정의형")),
        mass=d["mass"],
        length=d["length"],
        span=d["span"],
        height=d["height"],
        cg=d["cg"],
        asymmetry=d["asymmetry"],
        Ix=d["Ix"], Iy=d["Iy"], Iz=d["Iz"],
        cd_pitch=d["cd_pitch"], cd_roll=d["cd_roll"], cd_yaw=d["cd_yaw"],
        wing=wing, htail=htail, vtail=vtail,
    )


def environment_from_dict(d: dict) -> Environment:
    return Environment(V=d["V"], rho=d["rho"])


def initial_from_dict(d: dict) -> InitialState:
    return InitialState(
        pitch0_deg=d["pitch0"], roll0_deg=d["roll0"], yaw0_deg=d["yaw0"],
        q0_deg=d["q0"], p0_deg=d["p0"], r0_deg=d["r0"],
    )


def sim_from_dict(d: dict) -> SimConfig:
    return SimConfig(t_end=d["t_end"], dt=d["dt"], damping_mult=d["damping_mult"])
