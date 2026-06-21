# -*- coding: utf-8 -*-
"""
visualization.py
================
Plotly 로 측면도 / 정면도 / 상면도 / 3D 자세 / 시계열 그래프를 그린다.

각 2D 뷰는 "초기 자세(반투명 회색)" 와 "현재 자세(진한 색)" 를 겹쳐 그리고,
양력 벡터와 회전 모멘트 방향 화살표를 함께 표시한다.
"""

from __future__ import annotations
import json
import math
import numpy as np
import plotly.graph_objects as go

from aircraft import Aircraft, Environment
from simulation import SimResult

# 색상
COL_INIT = "rgba(120,120,120,0.45)"
COL_NOW = "#1f77b4"
COL_WING_LIFT = "#2ca02c"
COL_TAIL_LIFT = "#ff7f0e"
COL_MOMENT = "#d62728"
COL_CG = "#000000"
COL_CP = "#9467bd"
LEVEL_COLOR = {"ok": "#2e7d32", "warn": "#ef6c00", "danger": "#c62828"}


# ===========================================================================
# 공통 유틸
# ===========================================================================
def _rot2d(xs, zs, ang):
    """2D 점들을 각 ang(rad) 만큼 회전. x'=x·c−z·s, z'=x·s+z·c (pitch-up 부호)."""
    c, s = math.cos(ang), math.sin(ang)
    xs = np.asarray(xs, dtype=float)
    zs = np.asarray(zs, dtype=float)
    return xs * c - zs * s, xs * s + zs * c


def _add_polyline(fig, xs, zs, color, width=2, fill=None, name=None,
                  showlegend=False, opacity=1.0, dash=None):
    fig.add_trace(go.Scatter(
        x=xs, y=zs, mode="lines", line=dict(color=color, width=width, dash=dash),
        fill=fill, fillcolor=color if fill else None,
        name=name, showlegend=showlegend, opacity=opacity,
        hoverinfo="skip"))


def _add_arrow(fig, x0, y0, dx, dy, color, name=None, width=4, showlegend=False):
    """(x0,y0) 에서 (dx,dy) 방향 화살표(축+머리)."""
    x1, y1 = x0 + dx, y0 + dy
    fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                             line=dict(color=color, width=width),
                             name=name, showlegend=showlegend, hoverinfo="skip"))
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return
    ux, uy = dx / L, dy / L
    h = max(L * 0.22, 1e-9)
    ang = math.radians(28)
    for sgn in (+1, -1):
        ca, sa = math.cos(sgn * ang), math.sin(sgn * ang)
        hx = -(ux * ca - uy * sa) * h
        hy = -(ux * sa + uy * ca) * h
        fig.add_trace(go.Scatter(x=[x1, x1 + hx], y=[y1, y1 + hy], mode="lines",
                                 line=dict(color=color, width=width),
                                 showlegend=False, hoverinfo="skip"))


def _moment_marker(fig, x, y, sign, color, label):
    """회전 모멘트 방향 표시(↺/↻ 기호와 라벨)."""
    sym = "↻" if sign >= 0 else "↺"
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="text", text=[f"{sym} {label}"],
        textfont=dict(color=color, size=18), showlegend=False, hoverinfo="skip"))


def _equal_axes(fig, title, xlab, ylab, height=380):
    fig.update_yaxes(scaleanchor="x", scaleratio=1,
                     title_text=ylab, zeroline=False)
    fig.update_xaxes(title_text=xlab, zeroline=False)
    fig.update_layout(title=title, height=height,
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.02, x=0),
                      plot_bgcolor="rgba(245,247,250,1)")
    return fig


# ===========================================================================
# 기체 형상 (몸체 좌표, 원점=CG)
# ===========================================================================
def _side_parts(ac: Aircraft):
    """측면도용 폴리라인들 (xs, zs). 원점은 CG, x=뒤+, z=위+."""
    L, h = ac.length, ac.height
    cg = ac.cg
    def X(p): return p - cg
    hh = max(h * 0.5, L * 0.04)         # 동체 반높이

    # 동체 외곽(폐곡선)
    fus_x = [0, 0.12 * L, 0.85 * L, L, L, 0.85 * L, 0.12 * L, 0]
    fus_z = [0, hh, hh * 0.9, hh * 0.4, -hh * 0.4, -hh * 0.7, -hh, 0]
    fus_x = [X(v) for v in fus_x]

    cw = 0.16 * L                       # 주날개 시위
    wx = [X(ac.wing.position - cw / 2), X(ac.wing.position + cw / 2)]
    wz = [-hh * 0.2, -hh * 0.2]

    xt = ac.cg + ac.htail.arm           # 수평꼬리 위치(절대)
    ct = 0.08 * L
    tx = [X(xt - ct / 2), X(xt + ct / 2)]
    tz = [ac.htail.height + hh * 0.1, ac.htail.height + hh * 0.1]

    # 수직꼬리(측면에서는 삼각형)
    vt_x = [X(xt - ct), X(xt + ct * 0.5), X(xt + ct * 0.5)]
    vt_z = [hh * 0.5, hh * 0.5, hh * 0.5 + h * 0.6]

    return {
        "fuselage": (fus_x, fus_z),
        "wing": (wx, wz),
        "htail": (tx, tz),
        "vtail": (vt_x, vt_z),
        "cg": (0.0, 0.0),
        "cp": (X(ac.wing.cp_base), -hh * 0.2),
        "tail_pt": (X(xt), 0.0),
    }


# ===========================================================================
# 1) 측면도 (pitch)
# ===========================================================================
def side_view_figure(ac: Aircraft, env: Environment, res: SimResult, idx: int):
    fig = go.Figure()
    parts = _side_parts(ac)
    th0 = math.radians(res.pitch0)
    thN = math.radians(res.pitch[idx])

    def draw(theta, color, opacity, name):
        fx, fz = _rot2d(*parts["fuselage"], theta)
        _add_polyline(fig, fx, fz, color, 2, fill="toself", opacity=opacity)
        for key in ("wing", "htail", "vtail"):
            rx, rz = _rot2d(*parts[key], theta)
            _add_polyline(fig, rx, rz, color, 4, opacity=opacity)
        # 범례용 더미
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(color=color, width=3),
                                 name=name, showlegend=True))

    draw(th0, COL_INIT, 0.45, "초기 자세")
    draw(thN, COL_NOW, 1.0, "현재 자세")

    # CG / CP (현재 자세)
    cgx, cgz = _rot2d([parts["cg"][0]], [parts["cg"][1]], thN)
    cpx, cpz = _rot2d([parts["cp"][0]], [parts["cp"][1]], thN)
    fig.add_trace(go.Scatter(x=cgx, y=cgz, mode="markers+text",
                             marker=dict(color=COL_CG, size=11, symbol="circle"),
                             text=["CG"], textposition="bottom center",
                             name="CG", showlegend=True, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=cpx, y=cpz, mode="markers+text",
                             marker=dict(color=COL_CP, size=11, symbol="x"),
                             text=["CP"], textposition="top center",
                             name="CP", showlegend=True, hoverinfo="skip"))

    # 양력 벡터(세계 수직 방향) — 크기 정규화
    ref = max(np.max(np.abs(res.L_wing)), 1e-9)
    scale = 0.45 * ac.length
    lw = res.L_wing[idx] / ref * scale
    _add_arrow(fig, cpx[0], cpz[0], 0, lw, COL_WING_LIFT, "주날개 양력", showlegend=True)

    tpx, tpz = _rot2d([parts["tail_pt"][0]], [parts["tail_pt"][1]], thN)
    reft = max(np.max(np.abs(res.L_tail)), 1e-9)
    lt = res.L_tail[idx] / reft * scale * 0.7
    _add_arrow(fig, tpx[0], tpz[0], 0, lt, COL_TAIL_LIFT, "꼬리날개 양력", showlegend=True)

    # pitch 모멘트 방향
    _moment_marker(fig, 0, ac.height * 1.2 + 0.1 * ac.length,
                   res.M_pitch[idx], COL_MOMENT,
                   f"M_pitch ({'기수↑' if res.M_pitch[idx] >= 0 else '기수↓'})")

    return _equal_axes(fig, "측면도 — Pitch", "기체 길이 방향 (m)", "높이 (m)")


# ===========================================================================
# 2) 정면도 (roll)
# ===========================================================================
def front_view_figure(ac: Aircraft, env: Environment, res: SimResult, idx: int):
    """정면도: y(우+)-z(위+) 평면. roll φ 적용."""
    fig = go.Figure()
    b, h = ac.span, ac.height

    # 부품(몸체 좌표): 동체 단면(타원 근사 사각), 좌우 날개, 수직미익
    body_y = [-0.06 * b, 0.06 * b, 0.06 * b, -0.06 * b]
    body_z = [-0.5 * h, -0.5 * h, 0.5 * h, 0.5 * h]
    wing_y = [-0.5 * b, 0.5 * b]
    wing_z = [0.0, 0.0]
    fin_y = [0.0, 0.0]
    fin_z = [0.5 * h, 0.5 * h + 0.5 * h]

    def rot_roll(ys, zs, phi):
        # roll>0 = 우측 날개 아래로: y'=y·c+z·s, z'=−y·s+z·c
        c, s = math.cos(phi), math.sin(phi)
        ys = np.asarray(ys, float); zs = np.asarray(zs, float)
        return ys * c + zs * s, -ys * s + zs * c

    def draw(phi, color, opacity, name):
        by, bz = rot_roll(body_y, body_z, phi)
        _add_polyline(fig, by, bz, color, 2, fill="toself", opacity=opacity)
        wy, wz = rot_roll(wing_y, wing_z, phi)
        _add_polyline(fig, wy, wz, color, 6, opacity=opacity)
        fy, fz = rot_roll(fin_y, fin_z, phi)
        _add_polyline(fig, fy, fz, color, 4, opacity=opacity)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(color=color, width=3),
                                 name=name, showlegend=True))

    phi0 = math.radians(res.roll0)
    phiN = math.radians(res.roll[idx])
    draw(phi0, COL_INIT, 0.45, "초기 자세")
    draw(phiN, COL_NOW, 1.0, "현재 자세")

    # 좌우 비대칭 힘(양 날개 끝의 양력 차이) 표시
    if abs(ac.asymmetry) > 1e-3:
        wy, wz = rot_roll(wing_y, wing_z, phiN)
        base = 0.35 * h
        # 우측 날개에 (1+asym), 좌측에 (1-asym) 비례 화살표
        _add_arrow(fig, wy[1], wz[1], 0, base * (1 + ac.asymmetry),
                   COL_WING_LIFT, "우측 양력", showlegend=True)
        _add_arrow(fig, wy[0], wz[0], 0, base * (1 - ac.asymmetry),
                   COL_TAIL_LIFT, "좌측 양력", showlegend=True)

    _moment_marker(fig, 0, h * 1.4,
                   res.M_roll[idx], COL_MOMENT,
                   f"M_roll ({'우측↓' if res.M_roll[idx] >= 0 else '좌측↓'})")

    fig.update_layout(annotations=[dict(
        x=0.5 * b, y=-0.9 * h, text="우(R)", showarrow=False),
        dict(x=-0.5 * b, y=-0.9 * h, text="좌(L)", showarrow=False)])
    return _equal_axes(fig, "정면도 — Roll", "좌 ← 가로 (m) → 우", "높이 (m)")


# ===========================================================================
# 3) 상면도 (yaw)
# ===========================================================================
def top_view_figure(ac: Aircraft, env: Environment, res: SimResult, idx: int):
    """상면도: x(앞쪽 위로)-y(우+). yaw ψ 적용. 화면상 위쪽이 기수."""
    fig = go.Figure()
    L, b, cg = ac.length, ac.span, ac.cg
    def X(p): return p - cg

    # 동체(위에서 본 외곽)
    fus_x = [X(0), X(0.15 * L), X(0.85 * L), X(L), X(0.85 * L), X(0.15 * L)]
    fus_y = [0, 0.05 * b, 0.05 * b, 0, -0.05 * b, -0.05 * b]
    # 주날개(좌우)
    wing_x = [X(ac.wing.position)] * 2
    wing_y = [-0.5 * b, 0.5 * b]
    # 수평꼬리
    xt = cg + ac.htail.arm
    htw = 0.35 * b
    ht_x = [X(xt)] * 2
    ht_y = [-0.5 * htw, 0.5 * htw]
    # 수직꼬리(상면에서는 동체선상의 점/짧은 선; 개수만큼 좌우 배치)
    vt_pts = []
    if ac.vtail.count >= 2:
        off = 0.15 * b
        vt_pts = [(X(xt), -off), (X(xt), off)]
    else:
        vt_pts = [(X(xt), 0.0)]

    def rot_yaw(xs, ys, psi):
        # 화면: 위(+)=앞쪽 => x를 위로. yaw>0(기수 우향): 기수가 +y로.
        c, s = math.cos(psi), math.sin(psi)
        xs = np.asarray(xs, float); ys = np.asarray(ys, float)
        xr = xs * c - ys * s
        yr = xs * s + ys * c
        return yr, xr   # 화면 x=가로(yr), 화면 y=세로(xr, 앞쪽 위)

    def draw(psi, color, opacity, name):
        sx, sy = rot_yaw(fus_x, fus_y, psi)
        _add_polyline(fig, sx, sy, color, 2, fill="toself", opacity=opacity)
        for xs, ys, w in ((wing_x, wing_y, 6), (ht_x, ht_y, 4)):
            rx, ry = rot_yaw(xs, ys, psi)
            _add_polyline(fig, rx, ry, color, w, opacity=opacity)
        for (vx, vy) in vt_pts:
            rx, ry = rot_yaw([vx], [vy], psi)
            fig.add_trace(go.Scatter(x=rx, y=ry, mode="markers",
                                     marker=dict(color=color, size=8, symbol="square"),
                                     opacity=opacity, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(color=color, width=3),
                                 name=name, showlegend=True))

    psi0 = math.radians(res.yaw0)
    psiN = math.radians(res.yaw[idx])
    draw(psi0, COL_INIT, 0.45, "초기 자세")
    draw(psiN, COL_NOW, 1.0, "현재 자세")

    # 기수 방향 화살표
    nx, ny = rot_yaw([X(L)], [0], psiN)
    _add_arrow(fig, 0, 0, nx[0], ny[0], COL_NOW, "기수 방향", width=2)

    _moment_marker(fig, 0, 0.7 * L,
                   res.M_yaw[idx], COL_MOMENT,
                   f"M_yaw ({'우향' if res.M_yaw[idx] >= 0 else '좌향'})")
    return _equal_axes(fig, "상면도 — Yaw (위쪽이 기수)", "좌 ← 가로 (m) → 우", "앞 ↑ (m)")


# ===========================================================================
# 4) 3D 자세
# ===========================================================================
def _rot_matrix(pitch, roll, yaw):
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rp = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    Rr = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
    Ry = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Ry @ Rp @ Rr


def _aircraft_3d_parts(ac: Aircraft):
    """3D 부품을 (verts Nx3, tris list) 또는 라인으로 반환."""
    L, b, h, cg = ac.length, ac.span, ac.height, ac.cg
    def X(p): return p - cg
    parts = {}

    # 동체 라인
    parts["fuselage"] = np.array([[X(0), 0, 0], [X(L), 0, 0]])

    # 주날개 사각판
    xw = X(ac.wing.position); cw = 0.14 * L
    parts["wing"] = (np.array([
        [xw - cw / 2, -b / 2, 0], [xw + cw / 2, -b / 2, 0],
        [xw + cw / 2, b / 2, 0], [xw - cw / 2, b / 2, 0]]),
        [(0, 1, 2), (0, 2, 3)])

    # 수평꼬리
    xt = X(ac.cg + ac.htail.arm); ct = 0.06 * L; bt = 0.35 * b
    parts["htail"] = (np.array([
        [xt - ct / 2, -bt / 2, ac.htail.height], [xt + ct / 2, -bt / 2, ac.htail.height],
        [xt + ct / 2, bt / 2, ac.htail.height], [xt - ct / 2, bt / 2, ac.htail.height]]),
        [(0, 1, 2), (0, 2, 3)])

    # 수직꼬리(개수)
    fin_h = 0.5 * h + 0.4 * L * 0.1
    fins = []
    offs = [0.0] if ac.vtail.count < 2 else [-0.15 * b, 0.15 * b]
    for o in offs:
        v = np.array([
            [xt - ct, o, 0], [xt + ct * 0.6, o, 0],
            [xt + ct * 0.6, o, fin_h], [xt - ct, o, fin_h * 0.6]])
        fins.append((v, [(0, 1, 2), (0, 2, 3)]))
    parts["vtails"] = fins
    return parts


def attitude_3d_figure(ac: Aircraft, res: SimResult, idx: int):
    fig = go.Figure()
    parts = _aircraft_3d_parts(ac)

    def transform(P, R):
        return (R @ P.T).T

    def draw(R, color, opacity, name):
        # 동체
        f = transform(parts["fuselage"], R)
        fig.add_trace(go.Scatter3d(x=f[:, 0], y=f[:, 1], z=f[:, 2], mode="lines",
                                   line=dict(color=color, width=8),
                                   name=name, showlegend=True))
        for key in ("wing", "htail"):
            V, T = parts[key]
            Vr = transform(V, R)
            i, j, k = zip(*T)
            fig.add_trace(go.Mesh3d(x=Vr[:, 0], y=Vr[:, 1], z=Vr[:, 2],
                                    i=i, j=j, k=k, color=color, opacity=opacity * 0.8,
                                    showlegend=False, hoverinfo="skip"))
        for (V, T) in parts["vtails"]:
            Vr = transform(V, R)
            i, j, k = zip(*T)
            fig.add_trace(go.Mesh3d(x=Vr[:, 0], y=Vr[:, 1], z=Vr[:, 2],
                                    i=i, j=j, k=k, color=color, opacity=opacity * 0.8,
                                    showlegend=False, hoverinfo="skip"))

    R0 = _rot_matrix(math.radians(res.pitch0), math.radians(res.roll0),
                     math.radians(res.yaw0))
    RN = _rot_matrix(math.radians(res.pitch[idx]), math.radians(res.roll[idx]),
                     math.radians(res.yaw[idx]))
    draw(R0, "rgba(150,150,150,0.9)", 0.25, "초기 자세")
    draw(RN, COL_NOW, 0.9, "현재 자세")

    rng = max(ac.length, ac.span) * 0.65
    fig.update_layout(
        title="3D 자세 (반투명=초기, 진함=현재)",
        height=480, margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis=dict(title="앞(+) / 뒤", range=[-rng, rng]),
            yaxis=dict(title="좌 / 우(+)", range=[-rng, rng]),
            zaxis=dict(title="아래 / 위(+)", range=[-rng, rng]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9))),
        legend=dict(orientation="h", y=1.0))
    return fig


# ===========================================================================
# 5) STL 적용 전 3D 진단
# ===========================================================================
def stl_diagnostic_figure(tris: np.ndarray, props: dict, cg_from_nose: float,
                          stability: dict, max_tris: int = 6000):
    """STL 적용 전 CG/CP/축 안정성을 3D로 보여주는 미리보기."""
    fig = go.Figure()
    V = np.asarray(tris, dtype=float)
    if len(V) > max_tris:
        idx = np.random.default_rng(0).choice(len(V), max_tris, replace=False)
        V = V[idx]

    P = V.reshape(-1, 3)
    tri_idx = np.arange(len(V) * 3).reshape(-1, 3)
    fig.add_trace(go.Mesh3d(
        x=P[:, 0], y=P[:, 2], z=P[:, 1],
        i=tri_idx[:, 0], j=tri_idx[:, 1], k=tri_idx[:, 2],
        color="#7aa6d8", opacity=0.58, name="STL 모델",
        flatshading=True, hoverinfo="skip"))

    allp = np.asarray(tris, dtype=float).reshape(-1, 3)
    mn, mx = allp.min(0), allp.max(0)
    c = 0.5 * (mn + mx)
    nose_x = float(mx[0])
    cg = np.array([nose_x - float(cg_from_nose), props["cm"][1], props["cm"][2]], dtype=float)
    cp = np.array([nose_x - float(props["cp_base"]), props["cm"][1], props["cm"][2]], dtype=float)
    auto_cg = np.array([nose_x - float(props["cg"]), props["cm"][1], props["cm"][2]], dtype=float)

    def plot_point(p, label, color, symbol="circle", size=7):
        fig.add_trace(go.Scatter3d(
            x=[p[0]], y=[p[2]], z=[p[1]], mode="markers+text",
            marker=dict(size=size, color=color, symbol=symbol),
            text=[label], textposition="top center", name=label))

    plot_point(cg, "현재 CG", COL_CG, "circle", 8)
    if abs(float(cg_from_nose) - float(props["cg"])) > max(float(props["length"]) * 0.01, 1e-6):
        plot_point(auto_cg, "자동 CG", "#666666", "diamond", 6)
    plot_point(cp, "CP", COL_CP, "x", 8)

    fig.add_trace(go.Scatter3d(
        x=[cg[0], cp[0]], y=[cg[2], cp[2]], z=[cg[1], cp[1]],
        mode="lines", line=dict(color="#444", width=4),
        name="CG-CP 거리", hoverinfo="skip"))

    span = max(float(mx[0] - mn[0]), float(mx[1] - mn[1]), float(mx[2] - mn[2]), 1e-6)
    axis_len = span * 0.28
    axis_defs = [
        ("Roll 축", np.array([1.0, 0.0, 0.0]), stability["roll"][1]),
        ("Pitch 축", np.array([0.0, 0.0, 1.0]), stability["pitch"][1]),
        ("Yaw 축", np.array([0.0, 1.0, 0.0]), stability["yaw"][1]),
    ]
    for label, direction, level in axis_defs:
        p0 = cg - direction * axis_len
        p1 = cg + direction * axis_len
        fig.add_trace(go.Scatter3d(
            x=[p0[0], p1[0]], y=[p0[2], p1[2]], z=[p0[1], p1[1]],
            mode="lines+text", text=["", label], textposition="top center",
            line=dict(color=LEVEL_COLOR.get(level, "#555"), width=7),
            name=label, hoverinfo="skip"))

    # 기수/바람/바운딩 중심선
    fig.add_trace(go.Scatter3d(
        x=[mn[0], mx[0]], y=[c[2], c[2]], z=[c[1], c[1]],
        mode="lines+text", text=["꼬리", "기수"], textposition="top center",
        line=dict(color="#1f7ae0", width=5), name="기체 전후축", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(
        x=[mx[0] + span * 0.35, mx[0] + span * 0.05],
        y=[c[2], c[2]], z=[c[1] + span * 0.18, c[1] + span * 0.18],
        mode="lines+text", text=["바람", ""], textposition="top center",
        line=dict(color="#57b0ff", width=6), name="상대풍", hoverinfo="skip"))

    rng = span * 0.68
    center = np.array([c[0], c[2], c[1]])
    fig.update_layout(
        title="STL 적용 전 3D 진단 (CG/CP/축 안정성)",
        height=520, margin=dict(l=0, r=0, t=42, b=0),
        scene=dict(
            xaxis=dict(title="앞(+x) / 뒤", range=[center[0] - rng, center[0] + rng]),
            yaxis=dict(title="좌 / 우(+z)", range=[center[1] - rng, center[1] + rng]),
            zaxis=dict(title="아래 / 위(+y)", range=[center[2] - rng, center[2] + rng]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.35, y=1.45, z=0.95))),
        legend=dict(orientation="h", y=1.0))
    return fig


# ===========================================================================
# 6) 시계열 그래프
# ===========================================================================
def time_series_figure(t, series: list[tuple], title, ylabel, t_now=None, height=300):
    """series: [(y배열, 이름, 색)] 여러 개를 한 그래프에."""
    fig = go.Figure()
    for y, name, color in series:
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=name,
                                 line=dict(color=color, width=2)))
    if t_now is not None:
        fig.add_vline(x=t_now, line=dict(color="rgba(0,0,0,0.4)", dash="dash"))
    fig.update_layout(title=title, height=height,
                      margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="시간 t (s)", yaxis_title=ylabel,
                      legend=dict(orientation="h", y=1.15, x=0),
                      plot_bgcolor="rgba(245,247,250,1)")
    return fig
