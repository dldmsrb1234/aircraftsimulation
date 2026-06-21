# -*- coding: utf-8 -*-
"""
stl_analysis.py
===============
업로드한 STL 메시에서 **항공역학적 근사 특성**을 추출한다.

표준 기체 좌표(분석용)로 정렬한 뒤 계산:
  * x = 기수(앞) 방향(+),  y = 위(+),  z = 우측 날개(+)

추출 항목
  - 기체 길이 / 날개폭 / 높이      (바운딩박스)
  - 주날개 기준면적 S_wing          (윗면 실루엣 = planform 면적, 앞 70% 영역)
  - 수평/수직 꼬리 면적·거리(거친 추정)
  - 무게중심 CG                     (균일밀도 부피중심)
  - 양력중심 CP 근사                (planform 면적 중심 ≈ 공력중심 근사)
  - 관성모멘트 Ix, Iy, Iz          (균일밀도 폴리헤드론 적분, 사용자 질량으로 스케일)

⚠️ 익형/점성/실속 등은 다루지 않는 **형상 기반 근사**이며 교육용이다.
무게중심은 '균일 밀도' 가정이라 실제 무게 분포와 다를 수 있다.
"""

from __future__ import annotations
import struct
import numpy as np


# ---------------------------------------------------------------------------
# STL 파싱
# ---------------------------------------------------------------------------
def parse_stl(data: bytes) -> np.ndarray:
    """STL 바이트 → 삼각형 배열 (M, 3, 3) [float64]. 이진/ASCII 자동 판별."""
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + n * 50:          # 이진 STL
            dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
            arr = np.frombuffer(data, dtype=dt, count=n, offset=84)
            return arr["v"].astype(np.float64)
    # ASCII STL
    verts = []
    for line in data.decode("ascii", "ignore").splitlines():
        s = line.strip()
        if s.startswith("vertex"):
            p = s.split()
            verts.append([float(p[1]), float(p[2]), float(p[3])])
    v = np.asarray(verts, dtype=np.float64)
    m = (len(v) // 3) * 3
    return v[:m].reshape(-1, 3, 3)


# ---------------------------------------------------------------------------
# 축 매핑 / 회전
# ---------------------------------------------------------------------------
_AXIS = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0),
         "-Y": (0, -1, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}

AXIS_NAMES = list(_AXIS.keys())


def axis_vec(name: str) -> np.ndarray:
    return np.asarray(_AXIS[name], dtype=float)


def align_matrix(fwd: str, up: str) -> np.ndarray:
    """모델 좌표 → 표준 좌표(x=앞,y=위,z=우) 회전행렬 R (v_std = R·v)."""
    f = axis_vec(fwd)
    u = axis_vec(up)
    # up 에서 f 성분 제거(직교화) 후 정규화
    u = u - np.dot(u, f) * f
    if np.linalg.norm(u) < 1e-9:
        raise ValueError("기수(앞) 축과 위(상단) 축은 서로 다른 방향이어야 합니다.")
    u = u / np.linalg.norm(u)
    r = np.cross(f, u)
    return np.vstack([f, u, r])           # 행이 f,u,r → R·f=e1 등


# ---------------------------------------------------------------------------
# 폴리헤드론 질량특성 (Eberly, density=1 → mass=volume)
# ---------------------------------------------------------------------------
_MULT = np.array([1/6, 1/24, 1/24, 1/24, 1/60, 1/60, 1/60, 1/120, 1/120, 1/120])


def _sub(w0, w1, w2):
    t0 = w0 + w1
    f1 = t0 + w2
    t1 = w0 * w0
    t2 = t1 + w1 * t0
    f2 = t2 + w2 * f1
    f3 = w0 * t1 + w1 * t2 + w2 * f2
    g0 = f2 + w0 * (f1 + w0)
    g1 = f2 + w1 * (f1 + w1)
    g2 = f2 + w2 * (f1 + w2)
    return f1, f2, f3, g0, g1, g2


def _mass_properties(v0, v1, v2):
    """density=1 가정. (volume, cm[3], inertia_diag[xx,yy,zz] about CM) 반환."""
    x0, y0, z0 = v0[:, 0], v0[:, 1], v0[:, 2]
    x1, y1, z1 = v1[:, 0], v1[:, 1], v1[:, 2]
    x2, y2, z2 = v2[:, 0], v2[:, 1], v2[:, 2]
    a1, b1, c1 = x1 - x0, y1 - y0, z1 - z0
    a2, b2, c2 = x2 - x0, y2 - y0, z2 - z0
    d0 = b1 * c2 - b2 * c1
    d1 = a2 * c1 - a1 * c2
    d2 = a1 * b2 - a2 * b1

    f1x, f2x, f3x, g0x, g1x, g2x = _sub(x0, x1, x2)
    f1y, f2y, f3y, g0y, g1y, g2y = _sub(y0, y1, y2)
    f1z, f2z, f3z, g0z, g1z, g2z = _sub(z0, z1, z2)

    intg = np.array([
        np.sum(d0 * f1x),
        np.sum(d0 * f2x), np.sum(d1 * f2y), np.sum(d2 * f2z),
        np.sum(d0 * f3x), np.sum(d1 * f3y), np.sum(d2 * f3z),
        np.sum(d0 * (y0 * g0x + y1 * g1x + y2 * g2x)),
        np.sum(d1 * (z0 * g0y + z1 * g1y + z2 * g2y)),
        np.sum(d2 * (x0 * g0z + x1 * g1z + x2 * g2z)),
    ]) * _MULT

    vol = intg[0]
    if abs(vol) < 1e-12:
        return 0.0, np.zeros(3), np.zeros(3)
    cm = np.array([intg[1], intg[2], intg[3]]) / vol
    xx = intg[5] + intg[6] - vol * (cm[1]**2 + cm[2]**2)
    yy = intg[4] + intg[6] - vol * (cm[2]**2 + cm[0]**2)
    zz = intg[4] + intg[5] - vol * (cm[0]**2 + cm[1]**2)
    return vol, cm, np.array([xx, yy, zz])


# ---------------------------------------------------------------------------
# 메인 분석
# ---------------------------------------------------------------------------
def analyze(tris: np.ndarray, fwd: str, up: str, mass: float) -> dict:
    """삼각형 배열을 분석해 항공역학적 근사 특성 dict 반환."""
    if tris is None or len(tris) == 0:
        raise ValueError("빈 STL 입니다.")

    R = align_matrix(fwd, up)
    V = (tris.reshape(-1, 3) @ R.T).reshape(-1, 3, 3)
    v0, v1, v2 = V[:, 0], V[:, 1], V[:, 2]

    # 외향 와인딩 보정(부호 있는 부피<0 이면 v1,v2 교환)
    sv = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()
    if sv < 0:
        v1, v2 = v2.copy(), v1.copy()

    allp = V.reshape(-1, 3)
    mn, mx = allp.min(0), allp.max(0)
    length = float(mx[0] - mn[0])
    height = float(mx[1] - mn[1])
    span = float(mx[2] - mn[2])
    nose = float(mx[0])                       # 기수 = +x 최대

    # 면 법선/면적
    nrm = np.cross(v1 - v0, v2 - v0)
    twoA = np.linalg.norm(nrm, axis=1)
    area = 0.5 * twoA
    unit = np.divide(nrm, twoA[:, None], out=np.zeros_like(nrm), where=twoA[:, None] > 1e-12)
    cen = (v0 + v1 + v2) / 3.0
    dist = nose - cen[:, 0]                    # 기수로부터 거리(뒤로 +)
    ymid = 0.5 * (mn[1] + mx[1])

    up_area = np.maximum(0.0, unit[:, 1]) * area      # 윗면 실루엣(planform)
    side_area = np.maximum(0.0, unit[:, 2]) * area    # 측면 실루엣

    # 주날개(앞 70%) vs 수평꼬리(뒤 30%)
    aft = dist > 0.70 * length
    S_wing = float(up_area[~aft].sum())
    S_htail = float(up_area[aft].sum())
    S_plan = float(up_area.sum())
    if S_wing <= 1e-9:                          # 분리 실패 시 전체를 주날개로
        S_wing, S_htail = S_plan, 0.0

    # CP 근사 = 주날개(앞 영역) planform 면적 중심 (양력 위치와 면적을 일관되게)
    wf = up_area * (~aft)
    wsum = float(wf.sum())
    if wsum > 1e-9:
        cpx = float((wf * cen[:, 0]).sum() / wsum)
    else:
        tot = float(up_area.sum())
        cpx = float((up_area * cen[:, 0]).sum() / tot) if tot > 1e-9 else 0.5 * (mn[0] + mx[0])
    x_cp = nose - cpx

    # 주날개 위치(앞 영역 면적중심)
    wf = up_area[~aft]
    wing_pos = float((wf * cen[~aft, 0]).sum() / wf.sum()) if wf.sum() > 1e-9 else cpx
    wing_pos = nose - wing_pos

    # 수평꼬리 거리(뒤 영역 면적중심)
    ht = up_area[aft]
    if ht.sum() > 1e-9:
        ht_dist = float((ht * dist[aft]).sum() / ht.sum())
    else:
        ht_dist = 0.85 * length

    # 수직꼬리: 뒤 30% & 중심 위쪽의 측면 실루엣
    vmask = aft & (cen[:, 1] > ymid)
    S_vtail = float(side_area[vmask].sum())
    if S_vtail > 1e-9:
        vt_dist = float((side_area[vmask] * dist[vmask]).sum() / side_area[vmask].sum())
    else:
        S_vtail, vt_dist = 0.05 * S_wing, 0.9 * length

    # 질량특성(부피중심=CG, 관성)
    vol, cm, Idiag = _mass_properties(v0, v1, v2)
    vol = abs(vol)
    x_cg = nose - float(cm[0]) if vol > 1e-12 else nose - 0.5 * length
    dens = (mass / vol) if vol > 1e-12 else 0.0
    # 회전축 매핑: roll=전후축(x), pitch=가로축(z, 측), yaw=수직축(y, 위)
    Ix = float(Idiag[0] * dens)               # roll  (x, 전후)
    Iy = float(Idiag[2] * dens)               # pitch (z, 가로)
    Iz = float(Idiag[1] * dens)               # yaw   (y, 수직)

    return {
        "length": length, "span": span, "height": height,
        "S_wing": S_wing, "S_htail": S_htail, "S_plan": S_plan,
        "S_vtail": S_vtail,
        "wing_pos": wing_pos, "cp_base": x_cp, "cg": x_cg,
        "htail_arm": max(ht_dist - x_cg, 0.05 * length),
        "vtail_arm": max(vt_dist - x_cg, 0.05 * length),
        "Ix": max(Ix, 1e-9), "Iy": max(Iy, 1e-9), "Iz": max(Iz, 1e-9),
        "volume": vol, "frontal_area": float((np.maximum(0, unit[:, 0]) * area).sum()),
        "n_tri": int(len(tris)),
        "cm": [float(cm[0]), float(cm[1]), float(cm[2])],   # 정렬프레임 무게중심점
    }


def align_mesh(tris: np.ndarray, fwd: str, up: str) -> np.ndarray:
    """삼각형 메시를 표준 기체프레임(x=앞,y=위,z=측)으로 회전한 배열 반환."""
    R = align_matrix(fwd, up)
    return (tris.reshape(-1, 3) @ R.T).reshape(-1, 3, 3)


# ---------------------------------------------------------------------------
# 자체 검증 (박스: 해석해와 비교)
# ---------------------------------------------------------------------------
def _box_stl(Lx, Ly, Lz):
    """원점중심 박스의 외향 와인딩 삼각형 (12,3,3)."""
    hx, hy, hz = Lx/2, Ly/2, Lz/2
    p = np.array([[-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz],
                  [-hx,-hy, hz],[hx,-hy, hz],[hx,hy, hz],[-hx,hy, hz]], float)
    faces = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
             (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    return np.array([[p[a],p[b],p[c]] for a,b,c in faces])


if __name__ == "__main__":
    Lx, Ly, Lz, M = 4.0, 1.0, 2.0, 10.0
    tris = _box_stl(Lx, Ly, Lz)
    r = analyze(tris, "+X", "+Y", M)
    print("box L/H/span:", round(r["length"],3), round(r["height"],3), round(r["span"],3),
          "(기대 4,1,2)")
    print("volume:", round(r["volume"],4), "(기대", Lx*Ly*Lz, ")")
    print("Ix:", round(r["Ix"],4), "기대", round(M/12*(Ly**2+Lz**2),4))
    print("Iy:", round(r["Iy"],4), "기대", round(M/12*(Lx**2+Ly**2),4))
    print("Iz:", round(r["Iz"],4), "기대", round(M/12*(Lx**2+Lz**2),4))
    print("planform S(top):", round(r["S_plan"],4), "(기대", Lx*Lz, ")")
    print("cg from nose:", round(r["cg"],4), "(기대", Lx/2, ")")
