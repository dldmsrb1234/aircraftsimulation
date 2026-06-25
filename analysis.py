# -*- coding: utf-8 -*-
"""
analysis.py
===========
시뮬레이션 결과를 해석하여
  · AoA 위험도
  · CP–CG 관계
  · pitch / roll / yaw 경향
  · 전체 안정성
  · 제작 적합성 점수 (0~100)
  · 과학탐구 보고서용 자동 해석 문장
을 만들어 낸다.

판정 기준은 교육용 근사이며, 실제 비행 성공을 보장하지 않는다.
"""

from __future__ import annotations
import numpy as np

from aircraft import Aircraft, Environment
from simulation import SimResult


# 신호등 레벨
OK, WARN, DANGER = "ok", "warn", "danger"


# ---------------------------------------------------------------------------
# 개별 판정
# ---------------------------------------------------------------------------
def aoa_status(aoa_deg: float) -> tuple[str, str]:
    """주날개 받음각 위험도. (라벨, 레벨)"""
    a = abs(aoa_deg)
    if a >= 18.0:
        return "실속 위험", DANGER
    if a >= 12.0:
        return "주의", WARN
    return "정상", OK


def cp_cg_relation(x_cp: float, x_cg: float, length: float) -> tuple[str, str]:
    """양력중심(CP) 과 무게중심(CG) 의 전후 관계."""
    if not np.isfinite(x_cp):
        return "CP 계산 불안정", WARN
    margin = (x_cp - x_cg) / max(length, 1e-6)   # 길이로 정규화한 정적여유 근사
    if margin > 0.01:
        return "CP가 CG보다 뒤 (세로 안정)", OK
    if margin < -0.01:
        return "CP가 CG보다 앞 (세로 불안정)", DANGER
    return "CP와 CG 거의 일치 (중립)", WARN


def _amplitude(arr: np.ndarray, lo: float, hi: float) -> float:
    """배열에서 구간 [lo, hi] (비율) 의 진폭(최대-최소)."""
    n = len(arr)
    a, b = int(n * lo), max(int(n * hi), int(n * lo) + 1)
    seg = arr[a:b]
    return float(seg.max() - seg.min()) if len(seg) else 0.0


def _converging(arr: np.ndarray) -> bool:
    """후반 진폭이 전반보다 작으면 수렴으로 본다."""
    early = _amplitude(arr, 0.0, 0.3)
    late = _amplitude(arr, 0.7, 1.0)
    return late <= early * 0.9 + 1e-9


def pitch_trend(res: SimResult) -> tuple[str, str]:
    final = float(np.mean(res.pitch[int(len(res.pitch) * 0.8):]))
    conv = _converging(res.pitch)
    diverging = _amplitude(res.pitch, 0.7, 1.0) > _amplitude(res.pitch, 0.0, 0.3) * 1.3
    if diverging:
        return "발산(불안정)", DANGER
    if conv and abs(final) < max(2.0, abs(res.pitch0) * 0.4):
        return "안정 수렴", OK
    if final > 2.0:
        return "기수 들림", WARN
    if final < -2.0:
        return "기수 숙임", WARN
    return "안정 수렴", OK


def _axis_trend(arr: np.ndarray, pos_label: str, neg_label: str,
                init_val: float) -> tuple[str, str]:
    final = float(np.mean(arr[int(len(arr) * 0.8):]))
    diverging = _amplitude(arr, 0.7, 1.0) > _amplitude(arr, 0.0, 0.3) * 1.3
    if diverging or abs(final) > 60.0:
        return "발산(불안정)", DANGER
    if abs(final) < 2.0:
        return "안정", OK
    if final > 0:
        return pos_label, WARN
    return neg_label, WARN


def roll_trend(res: SimResult) -> tuple[str, str]:
    return _axis_trend(res.roll, "우측 기울어짐", "좌측 기울어짐", res.roll0)


def yaw_trend(res: SimResult) -> tuple[str, str]:
    return _axis_trend(res.yaw, "우측 편향", "좌측 편향", res.yaw0)


# ---------------------------------------------------------------------------
# 전체 안정성 + 제작 적합성 점수
# ---------------------------------------------------------------------------
def overall_assessment(ac: Aircraft, env: Environment, res: SimResult,
                       k_theta_override: float | None = None,
                       k_psi_override: float | None = None) -> dict:
    """모든 판정을 모아 전체 안정성 라벨과 0~100 점수를 만든다."""
    aoa_label, aoa_lvl = aoa_status(float(np.max(np.abs(res.aoa))))
    cp_label, cp_lvl = cp_cg_relation(float(res.cp[0]), ac.cg, ac.length)
    p_label, p_lvl = pitch_trend(res)
    r_label, r_lvl = roll_trend(res)
    y_label, y_lvl = yaw_trend(res)

    k_theta = 0.0 if k_theta_override is None else float(k_theta_override)
    k_psi = 0.0 if k_psi_override is None else float(k_psi_override)

    # ---- 점수 구성 (각 항목 가중합) ----
    score = 0.0

    # (1) 세로 정적안정 35점
    if k_theta > 0:
        score += 35 * min(k_theta / (abs(k_theta) + _ref_pitch_k(ac, env)), 1.0)
    # (2) pitch 수렴 15점
    score += {OK: 15, WARN: 8, DANGER: 0}[p_lvl]
    # (3) AoA 안전 20점
    score += {OK: 20, WARN: 10, DANGER: 0}[aoa_lvl]
    # (4) 방향(yaw) 안정 15점
    if k_psi > 0:
        score += 10
    score += {OK: 5, WARN: 3, DANGER: 0}[y_lvl]
    # (5) 좌우 대칭 15점 (비대칭 클수록 감점)
    score += 15 * max(0.0, 1.0 - abs(ac.asymmetry) / 0.3)

    score = float(np.clip(score, 0, 100))

    # ---- 전체 안정성 라벨 ----
    levels = [aoa_lvl, cp_lvl, p_lvl, r_lvl, y_lvl]
    if aoa_lvl == DANGER:
        overall = ("실속 위험", DANGER)
    elif levels.count(DANGER) >= 1:
        overall = ("불안정", DANGER)
    elif levels.count(WARN) >= 2:
        overall = ("주의", WARN)
    else:
        overall = ("안정", OK)

    return {
        "aoa": (aoa_label, aoa_lvl),
        "cp_cg": (cp_label, cp_lvl),
        "pitch": (p_label, p_lvl),
        "roll": (r_label, r_lvl),
        "yaw": (y_label, y_lvl),
        "overall": overall,
        "score": round(score),
        "k_theta": k_theta,
        "k_psi": k_psi,
    }


def _ref_pitch_k(ac: Aircraft, env: Environment) -> float:
    """점수 정규화용 기준 강성 (q·S·c 규모)."""
    q = 0.5 * env.rho * env.V * env.V
    return max(q * ac.wing.area * ac.length * 0.1, 1e-6)


# ---------------------------------------------------------------------------
# 자동 해석 문장 (과학탐구 보고서용)
# ---------------------------------------------------------------------------
def interpretation_sentences(ac: Aircraft, env: Environment,
                             res: SimResult, assess: dict) -> list[str]:
    s: list[str] = []

    aoa_max = float(np.max(np.abs(res.aoa)))
    x_cp0 = float(res.cp[0])
    margin = x_cp0 - ac.cg

    # CP–CG 관계 + pitch 경향
    if margin < -0.005:
        s.append(
            f"양력중심(CP={x_cp0:.2f} m)이 무게중심(CG={ac.cg:.2f} m)보다 앞쪽에 있어, "
            "받음각이 커질 때 기수가 더 들리는 방향으로 모멘트가 커집니다. "
            "이는 세로 방향으로 불안정해지기 쉬운 배치입니다.")
    elif margin > 0.005:
        s.append(
            f"양력중심(CP={x_cp0:.2f} m)이 무게중심(CG={ac.cg:.2f} m)보다 뒤쪽에 위치하여, "
            "자세가 흐트러져도 원래 자세로 되돌리려는 복원 모멘트가 작용합니다. "
            "세로 방향으로 안정적인 설계입니다.")
    else:
        s.append(
            "양력중심과 무게중심이 거의 일치하여 세로 방향으로 중립에 가깝습니다. "
            "작은 외란에도 자세가 잘 변할 수 있으므로 주의가 필요합니다.")

    # pitch 수렴/발산
    p_label = assess["pitch"][0]
    if p_label == "안정 수렴":
        s.append(
            f"초기 pitch {res.pitch0:.1f}° 에서 출발한 자세가 시간이 지나며 "
            f"약 {np.mean(res.pitch[-10:]):.1f}° 부근으로 수렴하므로, "
            "복원 안정성이 있는 설계로 볼 수 있습니다.")
    elif p_label in ("기수 들림", "기수 숙임"):
        s.append(
            f"시간이 지나도 pitch 가 약 {np.mean(res.pitch[-10:]):.1f}° 의 "
            f"치우친 자세로 유지되어 ‘{p_label}’ 경향이 나타납니다. "
            "무게중심·꼬리날개 조건을 조정하면 개선할 수 있습니다.")
    else:
        s.append(
            "pitch 진동의 진폭이 시간이 지날수록 커져 세로 방향으로 발산(불안정)하는 "
            "경향이 보입니다. 꼬리날개 면적을 키우거나 무게중심을 앞으로 옮겨 보세요.")

    # AoA / 실속
    if aoa_max >= 18.0:
        s.append(
            f"최대 받음각이 약 {aoa_max:.1f}° 로 실속 기준(18°)을 넘어, "
            "양력이 급격히 감소하고 자세가 불안정해질 위험이 큽니다.")
    elif aoa_max >= 12.0:
        s.append(
            f"최대 받음각이 약 {aoa_max:.1f}° 로 실속 주의 영역(12° 이상)에 들어, "
            "양력 증가가 둔화되기 시작합니다.")
    else:
        s.append(
            f"받음각이 최대 약 {aoa_max:.1f}° 로 선형 양력 구간 안에 있어 "
            "양력계수 근사가 비교적 잘 들어맞습니다.")

    # 좌우 비대칭 (roll/yaw)
    if abs(ac.asymmetry) >= 0.05:
        s.append(
            f"좌우 비대칭이 {ac.asymmetry:+.2f} 로 설정되어 한쪽 양력이 크기 때문에 "
            f"roll(현재 {np.mean(res.roll[-10:]):.1f}°) 또는 "
            f"yaw(현재 {np.mean(res.yaw[-10:]):.1f}°) 방향으로 치우치는 경향이 있습니다. "
            "모형 제작 시 좌우 날개를 최대한 대칭으로 만드는 것이 중요합니다.")
    else:
        s.append(
            "좌우가 거의 대칭이어서 roll·yaw 방향의 초기 외란이 "
            "수직꼬리날개와 감쇠에 의해 점차 줄어듭니다.")

    # 방향 안정
    if assess["k_psi"] > 0:
        s.append(
            "수직꼬리날개가 풍향계(weathercock) 역할을 하여 yaw 방향으로 "
            "기수를 바람 방향에 맞추려는 복원력을 만듭니다. "
            "수직꼬리날개 면적·개수를 늘리면 방향 안정이 강해집니다.")

    return s
