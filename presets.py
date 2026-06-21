# -*- coding: utf-8 -*-
"""
presets.py
==========
항공기 유형별 기본 입력값(프리셋).

⚠️ 중요: 아래 수치는 실제 제조사의 공력 데이터가 아니다.
각 항공기 "유형"의 일반적인 형상 특징(날개 면적, 꼬리 크기, 무게중심 위치 등)을
반영한 **교육용 근사값**이다. 관성모멘트·감쇠계수는 시뮬레이션에서
자세 변화가 눈에 잘 보이도록 조정한 값이며 실제 기체값과 다르다.

각 프리셋은 app.py 의 슬라이더 key 와 1:1 대응하는 flat dict 이다.
"""

from __future__ import annotations
import copy


# 모든 항공기에 공통으로 쓰이는 초기상태 / 시뮬레이션 기본값
_COMMON = {
    # 초기 자세
    "pitch0": 5.0, "roll0": 0.0, "yaw0": 0.0,
    # 초기 각속도 [deg/s]
    "q0": 0.0, "p0": 0.0, "r0": 0.0,
    # 시뮬레이션
    "t_end": 15.0, "dt": 0.02, "damping_mult": 1.0,
}


def _merge(d: dict) -> dict:
    out = copy.deepcopy(_COMMON)
    out.update(d)
    return out


# ---------------------------------------------------------------------------
# B737형 여객기 : 큰 후퇴익, 일반형 꼬리, 단일 수직미익, 안정 지향
# ---------------------------------------------------------------------------
B737 = _merge({
    "name": "B737형 여객기",
    "V": 60.0, "rho": 1.225,
    "mass": 60000.0, "length": 33.0, "span": 34.0, "height": 12.0,
    # 주날개
    "S_wing": 120.0, "wing_pos": 15.0, "wing_aoa": 3.0,
    "cp_base": 16.8, "cp_auto": True, "k_cp": 1.2,
    "cl_alpha": 5.5, "alpha_stall": 14.0, "cl_max": 1.4,
    # 무게중심
    "cg": 16.5, "asymmetry": 0.0,
    # 수평꼬리날개
    "S_htail": 32.0, "htail_aoa": -1.5, "htail_arm": 15.0,
    "htail_height": 0.5, "htail_cl_alpha": 4.0,
    # 수직꼬리날개
    "S_vtail": 21.0, "vtail_count": 1, "vtail_arm": 16.0, "vtail_cl_alpha": 3.0,
    # 관성 / 감쇠 (교육용 조정값)
    "Ix": 1.3e6, "Iy": 1.6e6, "Iz": 2.6e6,
    "cd_pitch": 6.0e5, "cd_roll": 3.0e5, "cd_yaw": 8.0e5,
})


# ---------------------------------------------------------------------------
# Cessna형 경비행기 : 작은 직선익, 가벼움, 매우 안정적
# ---------------------------------------------------------------------------
CESSNA = _merge({
    "name": "Cessna형 경비행기",
    "V": 35.0, "rho": 1.225,
    "mass": 1100.0, "length": 8.3, "span": 11.0, "height": 2.7,
    "S_wing": 16.2, "wing_pos": 3.4, "wing_aoa": 4.0,
    "cp_base": 3.7, "cp_auto": True, "k_cp": 0.4,
    "cl_alpha": 5.7, "alpha_stall": 15.0, "cl_max": 1.5,
    "cg": 3.55, "asymmetry": 0.0,
    "S_htail": 3.0, "htail_aoa": -1.0, "htail_arm": 4.2,
    "htail_height": 0.2, "htail_cl_alpha": 4.0,
    "S_vtail": 1.4, "vtail_count": 1, "vtail_arm": 4.4, "vtail_cl_alpha": 3.0,
    "Ix": 1.3e3, "Iy": 1.8e3, "Iz": 2.8e3,
    "cd_pitch": 6.0e2, "cd_roll": 3.5e2, "cd_yaw": 8.0e2,
})


# ---------------------------------------------------------------------------
# F-15형 전투기 : 큰 양력면, 빠름, 쌍수직미익, 기동성↑ 안정여유는 작게
# ---------------------------------------------------------------------------
F15 = _merge({
    "name": "F-15형 전투기",
    "V": 120.0, "rho": 1.225,
    "mass": 20000.0, "length": 19.4, "span": 13.0, "height": 5.6,
    "S_wing": 56.5, "wing_pos": 9.0, "wing_aoa": 2.0,
    "cp_base": 9.6, "cp_auto": True, "k_cp": 0.8,
    "cl_alpha": 4.5, "alpha_stall": 22.0, "cl_max": 1.6,
    "cg": 9.5, "asymmetry": 0.0,
    "S_htail": 9.0, "htail_aoa": 0.0, "htail_arm": 7.5,
    "htail_height": 0.0, "htail_cl_alpha": 3.6,
    "S_vtail": 5.0, "vtail_count": 2, "vtail_arm": 7.8, "vtail_cl_alpha": 2.8,
    "Ix": 1.6e5, "Iy": 2.4e5, "Iz": 3.6e5,
    "cd_pitch": 5.0e4, "cd_roll": 4.0e4, "cd_yaw": 6.0e4,
    "V_default_fast": True,
    "pitch0": 5.0,
})


# ---------------------------------------------------------------------------
# 글라이더형 : 매우 큰 가로세로비, 가벼움, 저속, 매우 안정
# ---------------------------------------------------------------------------
GLIDER = _merge({
    "name": "글라이더형",
    "V": 20.0, "rho": 1.225,
    "mass": 400.0, "length": 7.0, "span": 18.0, "height": 1.6,
    "S_wing": 14.0, "wing_pos": 2.9, "wing_aoa": 4.5,
    "cp_base": 3.2, "cp_auto": True, "k_cp": 0.3,
    "cl_alpha": 6.0, "alpha_stall": 13.0, "cl_max": 1.3,
    "cg": 3.05, "asymmetry": 0.0,
    "S_htail": 1.8, "htail_aoa": -1.5, "htail_arm": 3.8,
    "htail_height": 0.1, "htail_cl_alpha": 4.2,
    "S_vtail": 0.9, "vtail_count": 1, "vtail_arm": 3.9, "vtail_cl_alpha": 3.2,
    "Ix": 9.0e2, "Iy": 7.0e2, "Iz": 1.4e3,
    "cd_pitch": 3.0e2, "cd_roll": 2.5e2, "cd_yaw": 4.0e2,
})


# ---------------------------------------------------------------------------
# 사용자 정의형 : 중립에 가까운 일반 모형값 (자유 편집 시작점)
# ---------------------------------------------------------------------------
CUSTOM = _merge({
    "name": "사용자 정의형",
    "V": 15.0, "rho": 1.225,
    "mass": 1.5, "length": 0.26, "span": 0.26, "height": 0.05,
    "S_wing": 0.03, "wing_pos": 0.12, "wing_aoa": 4.0,
    "cp_base": 0.135, "cp_auto": True, "k_cp": 0.02,
    "cl_alpha": 5.0, "alpha_stall": 12.0, "cl_max": 1.2,
    "cg": 0.13, "asymmetry": 0.0,
    "S_htail": 0.006, "htail_aoa": -1.0, "htail_arm": 0.12,
    "htail_height": 0.0, "htail_cl_alpha": 3.5,
    "S_vtail": 0.004, "vtail_count": 1, "vtail_arm": 0.13, "vtail_cl_alpha": 3.0,
    "Ix": 0.010, "Iy": 0.014, "Iz": 0.022,
    "cd_pitch": 0.010, "cd_roll": 0.006, "cd_yaw": 0.014,
})


PRESETS = {
    "B737형 여객기": B737,
    "Cessna형 경비행기": CESSNA,
    "F-15형 전투기": F15,
    "글라이더형": GLIDER,
    "사용자 정의형": CUSTOM,
}

PRESET_NAMES = list(PRESETS.keys())


def get_preset(name: str) -> dict:
    """프리셋 flat dict 의 복사본 반환."""
    return copy.deepcopy(PRESETS.get(name, CUSTOM))
