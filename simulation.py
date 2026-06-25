# -*- coding: utf-8 -*-
"""
Ray-only STL flight simulation.

The old parameterized wing/tail/CP physics path is intentionally gone.  A
simulation run now requires an STL-derived ``panel_aero`` lookup model, and all
forces and moments come from ray/panel aerodynamic samples on that model.

``run_simulation`` still returns a finite trace because the Streamlit graphs
need arrays.  The browser animation uses the same ray tables but has no time
limit and keeps integrating until the user pauses or stops it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from aircraft import Aircraft, Environment, InitialState, SimConfig
import panel_aero


@dataclass
class SimResult:
    """Finite analysis trace.  Angles and rates are stored in degrees."""

    t: np.ndarray
    pitch: np.ndarray
    roll: np.ndarray
    yaw: np.ndarray
    aoa: np.ndarray
    cp: np.ndarray
    L_wing: np.ndarray
    L_tail: np.ndarray
    M_pitch: np.ndarray
    M_roll: np.ndarray
    M_yaw: np.ndarray
    ray_area: np.ndarray
    ray_wake_distance: np.ndarray
    pitch_rate: np.ndarray
    roll_rate: np.ndarray
    yaw_rate: np.ndarray

    pitch0: float = 0.0
    roll0: float = 0.0
    yaw0: float = 0.0

    def index_at(self, t_query: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t_query), 0, len(self.t) - 1))


def _dynamic_pressure(env: Environment) -> float:
    return 0.5 * float(env.rho) * float(env.V) * float(env.V)


def _ray_damping(k_raw: float, inertia: float, damping_mult: float) -> float:
    """Use ray-derived static stiffness scale for numerical attitude damping."""

    zeta = 0.4
    return 2.0 * zeta * math.sqrt(max(abs(k_raw), 1e-12) * inertia) * damping_mult


def run_simulation(
    ac: Aircraft,
    env: Environment,
    init: InitialState,
    sim: SimConfig,
    aero_model=None,
) -> SimResult:
    if aero_model is None:
        raise ValueError("ray-only simulation requires an STL aero_model")
    if sim.dt <= 0:
        raise ValueError("simulation dt must be positive")

    n = max(int(round(max(sim.t_end, sim.dt) / sim.dt)) + 1, 2)

    theta = math.radians(float(init.pitch0_deg))
    phi = math.radians(float(init.roll0_deg))
    psi = math.radians(float(init.yaw0_deg))
    q_rate = math.radians(float(init.q0_deg))
    p_rate = math.radians(float(init.p0_deg))
    r_rate = math.radians(float(init.r0_deg))

    q_dyn = _dynamic_pressure(env)
    i_pitch = max(float(ac.Iy), 1e-12)
    i_roll = max(float(ac.Ix), 1e-12)
    i_yaw = max(float(ac.Iz), 1e-12)
    cg_point = panel_aero.cg_point_from_nose(aero_model, float(ac.cg))

    k_pitch_raw = float(panel_aero.pitch_stiffness(aero_model, q_dyn, cg_point))
    k_yaw_raw = float(panel_aero.yaw_stiffness(aero_model, q_dyn, cg_point))
    cd_pitch = _ray_damping(k_pitch_raw, i_pitch, float(sim.damping_mult))
    cd_yaw = _ray_damping(k_yaw_raw, i_yaw, float(sim.damping_mult))
    cd_roll = i_roll * 3.0 * float(sim.damping_mult)

    def moments(th: float, ph: float, ps: float):
        w_body = panel_aero.relative_wind_body(th, ph, ps)
        force, moment_origin, alpha, _beta = panel_aero.aero(
            aero_model, w_body, q_dyn, cg_point
        )
        m_roll, m_pitch, m_yaw = panel_aero.body_moments(moment_origin)
        m_roll = float(m_roll)
        m_pitch = float(m_pitch)
        m_yaw = float(m_yaw)

        fy = float(force[1])
        f_norm = max(float(np.linalg.norm(force)), 1e-9)
        if abs(fy) > max(1e-6, f_norm * 1e-4):
            cp = float(np.clip(float(ac.cg) - m_pitch / fy, 0.0, max(float(ac.length), 1e-3)))
        else:
            cp = float("nan")

        aux = (
            math.degrees(float(alpha)),
            cp,
            fy,
            float(-force[0]),
            panel_aero.ray_hit_area(aero_model, w_body),
            panel_aero.ray_wake_distance(aero_model, w_body),
        )
        return m_pitch, m_roll, m_yaw, aux

    w_pitch = math.sqrt(abs(k_pitch_raw) / i_pitch) if i_pitch > 0 else 0.0
    w_yaw = math.sqrt(abs(k_yaw_raw) / i_yaw) if i_yaw > 0 else 0.0
    w_max = max(w_pitch, w_yaw, 1e-9)
    n_sub = int(min(240, max(1, math.ceil(float(sim.dt) * w_max / 1.5))))
    h = float(sim.dt) / n_sub

    rec = {
        key: np.zeros(n)
        for key in (
            "t",
            "pitch",
            "roll",
            "yaw",
            "aoa",
            "cp",
            "L_wing",
            "L_tail",
            "M_pitch",
            "M_roll",
            "M_yaw",
            "ray_area",
            "ray_wake_distance",
            "pitch_rate",
            "roll_rate",
            "yaw_rate",
        )
    }

    def wrap_angle(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    for i in range(n):
        m_pitch, m_roll, m_yaw, aux = moments(theta, phi, psi)

        rec["t"][i] = i * float(sim.dt)
        rec["pitch"][i] = math.degrees(theta)
        rec["roll"][i] = math.degrees(phi)
        rec["yaw"][i] = math.degrees(psi)
        rec["aoa"][i] = aux[0]
        rec["cp"][i] = aux[1]
        rec["L_wing"][i] = aux[2]
        rec["L_tail"][i] = aux[3]
        rec["M_pitch"][i] = m_pitch
        rec["M_roll"][i] = m_roll
        rec["M_yaw"][i] = m_yaw
        rec["ray_area"][i] = aux[4]
        rec["ray_wake_distance"][i] = aux[5]
        rec["pitch_rate"][i] = math.degrees(q_rate)
        rec["roll_rate"][i] = math.degrees(p_rate)
        rec["yaw_rate"][i] = math.degrees(r_rate)

        for _ in range(n_sub):
            m_pitch, m_roll, m_yaw, _aux = moments(theta, phi, psi)
            q_rate = (q_rate + (m_pitch / i_pitch) * h) / (1.0 + (cd_pitch / i_pitch) * h)
            p_rate = (p_rate + (m_roll / i_roll) * h) / (1.0 + (cd_roll / i_roll) * h)
            r_rate = (r_rate + (m_yaw / i_yaw) * h) / (1.0 + (cd_yaw / i_yaw) * h)

            theta = wrap_angle(theta + q_rate * h)
            phi = wrap_angle(phi + p_rate * h)
            psi = wrap_angle(psi + r_rate * h)

    return SimResult(
        t=rec["t"],
        pitch=rec["pitch"],
        roll=rec["roll"],
        yaw=rec["yaw"],
        aoa=rec["aoa"],
        cp=rec["cp"],
        L_wing=rec["L_wing"],
        L_tail=rec["L_tail"],
        M_pitch=rec["M_pitch"],
        M_roll=rec["M_roll"],
        M_yaw=rec["M_yaw"],
        ray_area=rec["ray_area"],
        ray_wake_distance=rec["ray_wake_distance"],
        pitch_rate=rec["pitch_rate"],
        roll_rate=rec["roll_rate"],
        yaw_rate=rec["yaw_rate"],
        pitch0=init.pitch0_deg,
        roll0=init.roll0_deg,
        yaw0=init.yaw0_deg,
    )
