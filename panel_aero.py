# -*- coding: utf-8 -*-
"""
panel_aero.py
=============
STL 표면을 **패널(삼각형)** 단위로 보고, 각 패널이 기류를 향하는지를
법선·기류방향 내적(= 기류 ray 가 그 면을 때리는지)으로 판정해
표면 경사(surface-inclination / impact) 압력 모델로 공력을 계산한다.

 - 각 패널 압력계수:
     m = -(n̂ · d̂)               (>0 이면 바람을 맞는 windward 면)
     Cp = CW · m        (windward, 양압)
     Cp = CL_SUC · m    (leeward,  m<0 → 음압=흡입)
 - 패널 힘(동압 q 당): f = -Cp · A · n̂
 - 전체 힘 F=Σf, 무게중심 둘레 모멘트 M=Σ(r×f),  r = 패널중심 − CG

받음각 α, 옆미끄럼각 β 격자에서 F,M(둘 다 q 당)을 미리 계산해 표로 저장하고,
시뮬레이션에서는 자세로부터 (α,β)를 구해 표를 보간 → q 를 곱해 실제 힘·모멘트를 얻는다.

좌표(정렬 후 기체 프레임): x=기수(앞), y=위, z=가로(측).
pitch=±z축 회전(기수 들림 +), roll=±x축, yaw=±y축.

⚠️ 비점성 표면 impact 근사라 실제 익형의 순환양력을 정밀 재현하진 않지만,
형상·자세에 따라 양력/모멘트가 실제로 분포·변화하는 교육용 모델이다.
"""

from __future__ import annotations
import math
import numpy as np

CW = 3.2        # windward(양압) 기울기
CL_SUC = 1.1    # leeward(흡입) 기울기


def _ensure_outward(tris: np.ndarray) -> np.ndarray:
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    sv = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()
    if sv < 0:
        tris = tris[:, [0, 2, 1], :].copy()
    return tris


def _tri_props(tris):
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    cen = (v0 + v1 + v2) / 3.0
    nrm = np.cross(v1 - v0, v2 - v0)
    twoA = np.linalg.norm(nrm, axis=1)
    area = 0.5 * twoA
    nhat = np.divide(nrm, twoA[:, None], out=np.zeros_like(nrm), where=twoA[:, None] > 1e-15)
    return cen, nhat, area


def _dir(al, be):
    """(α,β) → 기류 진행방향 단위벡터 (기체프레임). α=β=0 → (-1,0,0)."""
    d = np.array([-1.0, math.tan(al), math.tan(be)])
    return d / np.linalg.norm(d)


def ab_from_wind(w):
    """기체프레임 상대풍 단위벡터 → (α, β) [rad]."""
    dx, dy, dz = float(w[0]), float(w[1]), float(w[2])
    return math.atan2(dy, -dx), math.atan2(dz, -dx)


def build_aero_model(tris: np.ndarray, cm: np.ndarray,
                     n_alpha: int = 27, n_beta: int = 11,
                     a_max: float = 45.0, b_max: float = 35.0,
                     max_tris: int = 60000) -> dict:
    """패널 공력 표 생성. 모멘트는 **원점 기준**(Fg, Mg 모두 동압 q 당)으로 저장하여
    실행 시 임의의 무게중심에 대해 M_cg = M_origin − cg×F 로 옮길 수 있게 한다.

    cm : 정렬프레임 부피중심(무게중심의 y,z 기본값 및 기준점 표시용).
    """
    tris = _ensure_outward(np.asarray(tris, dtype=float))
    if len(tris) > max_tris:                       # 과대 메시는 샘플링
        idx = np.random.default_rng(0).choice(len(tris), max_tris, replace=False)
        tris = tris[idx]
    cen, nhat, area = _tri_props(tris)              # r = 원점 기준 = cen

    alphas = np.radians(np.linspace(-a_max, a_max, n_alpha))
    betas = np.radians(np.linspace(-b_max, b_max, n_beta))
    Fg = np.zeros((n_alpha, n_beta, 3))
    Mg = np.zeros((n_alpha, n_beta, 3))

    for i, al in enumerate(alphas):
        for j, be in enumerate(betas):
            d = _dir(al, be)
            m = -(nhat @ d)                         # >0 windward
            Cp = np.where(m > 0.0, CW * m, CL_SUC * m)
            f = (-(Cp * area))[:, None] * nhat      # q 당 패널 힘
            Fg[i, j] = f.sum(0)
            Mg[i, j] = np.cross(cen, f).sum(0)      # 원점 기준 모멘트

    return {"alphas": alphas, "betas": betas, "Fg": Fg, "Mg": Mg,
            "ref_area": float(area.sum()), "n_tri": int(len(tris)),
            "nose_x": float(tris[:, :, 0].max()),
            "cm": [float(c) for c in np.asarray(cm)]}


def _bilinear(grid, alphas, betas, al, be):
    al = min(max(al, alphas[0]), alphas[-1])
    be = min(max(be, betas[0]), betas[-1])
    ia = min(max(int(np.searchsorted(alphas, al)) - 1, 0), len(alphas) - 2)
    ib = min(max(int(np.searchsorted(betas, be)) - 1, 0), len(betas) - 2)
    ta = (al - alphas[ia]) / (alphas[ia + 1] - alphas[ia])
    tb = (be - betas[ib]) / (betas[ib + 1] - betas[ib])
    return (grid[ia, ib] * (1 - ta) * (1 - tb) + grid[ia + 1, ib] * ta * (1 - tb)
            + grid[ia, ib + 1] * (1 - ta) * tb + grid[ia + 1, ib + 1] * ta * tb)


def aero(model: dict, w_body: np.ndarray, q: float, cg_point):
    """상대풍 w_body, 동압 q, 무게중심점 cg_point 에서
    (F[3], M_cg[3], α, β) 반환(기체프레임, 실제 단위)."""
    al, be = ab_from_wind(w_body)
    F = _bilinear(model["Fg"], model["alphas"], model["betas"], al, be) * q
    Mo = _bilinear(model["Mg"], model["alphas"], model["betas"], al, be) * q
    M = Mo - np.cross(np.asarray(cg_point, dtype=float), F)   # 무게중심으로 이동
    return F, M, al, be


# ---------------------------------------------------------------------------
# 자세 → 기체프레임 상대풍
# ---------------------------------------------------------------------------
def _Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_W_WORLD = np.array([-1.0, 0.0, 0.0])   # 바람: 정면(+x)에서 불어옴(공기는 -x 로 진행)


def relative_wind_body(theta, phi, psi):
    """pitch θ, roll φ, yaw ψ(rad) 에서 기체프레임 상대풍 단위벡터."""
    theta, phi, psi = float(theta), float(phi), float(psi)   # numpy 스칼라 방지
    R_bw = _Ry(-psi) @ _Rz(theta) @ _Rx(phi)     # 기체→세계 (애니메이션과 동일 규약)
    w = R_bw.T @ _W_WORLD
    n = float(np.linalg.norm(w))
    return w / n if n > 0 else _W_WORLD.copy()


def body_moments(M):
    """패널 모멘트 벡터 → (roll, pitch, yaw) 동역학 모멘트.

    자세각 규약상:
      pitch θ(기수↑) = +z축 회전 → pitch 모멘트 = +M_z
      roll  φ        = +x축 회전 → roll  모멘트 = +M_x
      yaw   ψ(기수 +z향) = +y축 회전은 ψ 를 줄이므로 → yaw 모멘트 = -M_y
    """
    return M[0], M[2], -M[1]


def cg_point_from_nose(model, cg_from_nose):
    """무게중심(기수 기준 스칼라) → 정렬프레임 무게중심점 [x,y,z]."""
    cm = model["cm"]
    return np.array([model["nose_x"] - cg_from_nose, cm[1], cm[2]])


def pitch_stiffness(model, env_q, cg_point, theta0=0.0, eps=math.radians(3.0)):
    """k_θ = -dM_pitch/dθ (>0 이면 세로 안정)."""
    m1 = body_moments(aero(model, relative_wind_body(theta0 + eps, 0, 0), env_q, cg_point)[1])[1]
    m0 = body_moments(aero(model, relative_wind_body(theta0 - eps, 0, 0), env_q, cg_point)[1])[1]
    return -(m1 - m0) / (2 * eps)


def yaw_stiffness(model, env_q, cg_point, eps=math.radians(3.0)):
    """k_ψ = -dM_yaw/dψ (>0 이면 방향 안정)."""
    m1 = body_moments(aero(model, relative_wind_body(0, 0, eps), env_q, cg_point)[1])[2]
    m0 = body_moments(aero(model, relative_wind_body(0, 0, -eps), env_q, cg_point)[1])[2]
    return -(m1 - m0) / (2 * eps)


# ---------------------------------------------------------------------------
# 자체 검증
# ---------------------------------------------------------------------------
def _plate_xz(cx, span_z, chord_x, y=0.0, n=6):
    """x-z 평면 수평판(주날개). 위/아래 양면 삼각형."""
    xs = np.linspace(cx - chord_x/2, cx + chord_x/2, n)
    zs = np.linspace(-span_z/2, span_z/2, n)
    tris = []
    for i in range(n-1):
        for j in range(n-1):
            p00=[xs[i],y,zs[j]]; p10=[xs[i+1],y,zs[j]]; p11=[xs[i+1],y,zs[j+1]]; p01=[xs[i],y,zs[j+1]]
            tris += [[p00,p10,p11],[p00,p11,p01]]      # 윗면
            tris += [[p00,p11,p10],[p00,p01,p11]]      # 아랫면(반대 와인딩)
    return tris


def _plate_xy(cx, height_y, chord_x, z=0.0, n=5):
    """x-y 평면 수직판(수직꼬리)."""
    xs = np.linspace(cx - chord_x/2, cx + chord_x/2, n)
    ys = np.linspace(0, height_y, n)
    tris = []
    for i in range(n-1):
        for j in range(n-1):
            p00=[xs[i],ys[j],z]; p10=[xs[i+1],ys[j],z]; p11=[xs[i+1],ys[j+1],z]; p01=[xs[i],ys[j+1],z]
            tris += [[p00,p10,p11],[p00,p11,p01], [p00,p11,p10],[p00,p01,p11]]
    return tris


if __name__ == "__main__":
    # 합성 글라이더: 주날개(앞), 수평꼬리(뒤), 수직꼬리(뒤). CG 를 날개보다 앞에.
    tris = []
    tris += _plate_xz(cx=0.10, span_z=1.0, chord_x=0.20)     # 주날개
    tris += _plate_xz(cx=-0.55, span_z=0.4, chord_x=0.10)    # 수평꼬리
    tris += _plate_xy(cx=-0.55, height_y=0.25, chord_x=0.12) # 수직꼬리
    tris = np.array(tris, float)
    cg = np.array([0.06, 0.0, 0.0])     # 날개 앞전 근처(약간 앞)
    model = build_aero_model(tris, cg)
    q = 0.5 * 1.225 * 15**2

    print("=== 양력 vs α (Fy 가 증가해야) ===")
    for a in [-10,-5,0,5,10,15]:
        w = relative_wind_body(math.radians(a),0,0)
        F,M,al,be = aero(model, w, q, cg)
        print(f"  α={a:4d}  Fy(양력)={F[1]:8.2f} N   M_pitch(Mz)={body_moments(M)[1]:8.3f}")
    kth = pitch_stiffness(model, q, cg)
    kps = yaw_stiffness(model, q, cg)
    print(f"k_theta = {kth:.3f} (>0 안정),  k_psi = {kps:.3f} (>0 방향안정)")
    # CG 를 뒤로(0.06→-0.30) 옮기면 불안정해져야
    cg_aft = np.array([-0.30, 0.0, 0.0])
    print(f"CG 뒤로: k_theta = {pitch_stiffness(model, q, cg_aft):.3f} (<0 불안정 기대)")
