#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Missile Simulator V8
- Structure de données améliorée de V7
- Précision aérodynamique complète de V6
- Mode "quick" et "precise" (RK4)
- Cache atmosphérique
- Recherche parallélisée
- Interface CLI interactive
- Affichage matplotlib interactif (pas de PNG)
"""

from __future__ import annotations
import math
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import sys


# --------------------------
# CONSTANTES
# --------------------------
class Const:
    G = 9.80665
    R = 287.05
    GAMMA = 1.405


# --------------------------
# CONFIGURATION DU VEHICULE
# --------------------------
@dataclass
class VehicleConfig:
    name: str
    initial_mass: float  # kg
    mass_after_booster: float  # kg
    mass_after_sustainer: float  # kg
    booster_time: float  # s
    sustainer_time: float  # s
    booster_thrust: float  # N
    sustainer_thrust: float  # N
    diameter: float  # m
    nose_length: float  # m
    total_length: float  # m
    nozzle_diameter: float  # m
    initial_velocity: float  # m/s
    initial_altitude: float  # m
    target_altitude: float  # m
    battery_time: float  # s
    major_axis: float = 0.08  # a
    minor_axis: float = 0.08  # b
    wing_surface: float = 0.02  # m²
    wing_length: float = 0.5  # m
    wing_position: float = 0.7  # position relative (0-1)

    def __post_init__(self):
        self.cross_section_area = math.pi * (self.diameter / 2) ** 2


def default_vehicle_database() -> Dict[str, VehicleConfig]:
    return {
        "V_MICA": VehicleConfig(
            name="V_MICA",
            initial_mass=112.0,
            mass_after_booster=88.3,
            mass_after_sustainer=70.1,
            booster_time=2.75,
            sustainer_time=4.0,
            booster_thrust=20250.0,
            sustainer_thrust=10720.0,
            diameter=0.16,
            nose_length=0.40,
            total_length=3.1,
            nozzle_diameter=0.12,
            initial_velocity=300.0,
            initial_altitude=5000.0,
            target_altitude=2000.0,
            battery_time=70.0,
            major_axis=0.08,
            minor_axis=0.08,
            wing_surface=0.02,
            wing_length=0.5,
            wing_position=0.7
        ),
        "V_AIM120": VehicleConfig(
            name="V_AIM120",
            initial_mass=157.0,
            mass_after_booster=125.0,
            mass_after_sustainer=95.0,
            booster_time=3.2,
            sustainer_time=5.5,
            booster_thrust=25000.0,
            sustainer_thrust=12000.0,
            diameter=0.178,
            nose_length=0.45,
            total_length=3.66,
            nozzle_diameter=0.13,
            initial_velocity=320.0,
            initial_altitude=6000.0,
            target_altitude=2000.0,
            battery_time=80.0,
            wing_surface=0.025,
            wing_length=0.6
        ),
    }


# --------------------------
# ATMOSPHERE PRECISE (avec cache)
# --------------------------
@lru_cache(maxsize=8192)
def atmosphere_ISA_precise(altitude_m: float) -> Tuple[float, float, float]:
    """
    Modèle atmosphérique précis de V6 avec cache.
    Returns (T[K], p[Pa], rho[kg/m³])
    """
    z = float(max(0.0, min(altitude_m, 79000.0)))

    if z <= 11000:
        T = 288.15 - 0.0065 * z
        P = 101325 * (1 - 0.0065 * z / 288.15) ** 5.25588
    elif z <= 20000:
        T = 216.65
        P = 22632 * math.exp(-0.000157 * (z - 11000))
    elif z <= 32000:
        T = 216.65 + 0.001 * (z - 20000)
        P = 5474.87 * (1 + 0.001 * (z - 20000) / 216.65) ** -34.1632
    elif z <= 47000:
        T = 228.65 + 0.0028 * (z - 32000)
        P = 868.014 * (1 + 0.0028 * (z - 32000) / 228.65) ** -12.2011
    elif z <= 52000:
        T = 270.65
        P = 110.906 * math.exp(-0.000157 * (z - 47000))
    elif z <= 61000:
        T = 270.65 - 0.0028 * (z - 52000)
        P = 66.9389 * (T / 270.65) ** -12.2011
    elif z <= 79000:
        T = 252.65 - 0.002 * (z - 61000)
        P = 3.95642 * (T / 214.65) ** -12.2011
    else:
        T = 214.65
        P = 3.95642

    rho = P / (Const.R * T)
    return T, P, rho


# --------------------------
# AERODYNAMICS PRECIS (de V6)
# --------------------------
def calculate_drag_coefficients_precise(cfg: VehicleConfig, M: float, alpha_rad: float,
                                        q: float, A_e: float, S_Ref: float) -> Tuple[
    float, float, float, float, float, float]:
    """
    Calcul précis des coefficients de traînée (modèle V6).
    """
    # Conversion en pieds pour compatibilité avec formules originales
    l_n = cfg.nose_length * 3.28084
    d = cfg.diameter * 3.28084
    l = cfg.total_length * 3.28084

    # Wave drag
    if M > 1:
        CD0_Body_Wave = (1.59 + 1.83 / M ** 2) * (math.atan(0.5 / (l_n / d))) ** 1.69
    else:
        CD0_Body_Wave = 0

    # Base drag
    if M > 1:
        CD0_Base_Coast = 0.25 / M
        CD0_Base_Powered = (1 - A_e / S_Ref) * (0.25 / M)
    else:
        CD0_Base_Coast = 0.12 + 0.13 * M ** 2
        CD0_Base_Powered = (1 - A_e / S_Ref) * (0.12 + 0.13 * M ** 2)

    # Friction drag
    CD0_Body_Friction = 0.053 * (l / d) * (M / (q * l)) ** 0.2

    # Wing drag avec décrochage
    alpha_abs = abs(alpha_rad)
    CD_Wing_base = 0.1 * (cfg.wing_surface / cfg.cross_section_area) * (math.sin(alpha_abs) ** 2)
    if M > 1:
        CD_Wing_base += 0.05 * (cfg.wing_surface / cfg.cross_section_area)

    # Décrochage
    alpha_crit_rad = math.radians(20.0)
    alpha_stall_end_rad = math.radians(36.0)
    if alpha_abs > alpha_crit_rad:
        drag_increase_factor = 1.0 + 0.1 * min(
            (alpha_abs - alpha_crit_rad) / (alpha_stall_end_rad - alpha_crit_rad), 1.0)
        CD_Wing = CD_Wing_base * drag_increase_factor
    else:
        CD_Wing = CD_Wing_base

    # Total drag
    if A_e > 0:
        Ca = CD0_Body_Wave + CD0_Base_Powered + CD0_Body_Friction + CD_Wing
    else:
        Ca = CD0_Body_Wave + CD0_Base_Coast + CD0_Body_Friction + CD_Wing

    return CD0_Body_Wave, CD0_Base_Coast, CD0_Base_Powered, CD0_Body_Friction, CD_Wing, Ca


def calculate_normal_force_coefficient(cfg: VehicleConfig, alpha_rad: float, phi: float = 0) -> float:
    """
    Calcul du coefficient de force normale (modèle V6).
    """
    # Contribution du corps
    if alpha_rad < 0:
        CN_body = -abs((cfg.major_axis / cfg.minor_axis) * math.cos(phi) +
                       (cfg.minor_axis / cfg.major_axis) * math.sin(phi)) * (
                          abs(math.sin(2 * alpha_rad) * math.cos(alpha_rad / 2)) +
                          2 * (cfg.total_length / cfg.diameter) * math.sin(alpha_rad) ** 2)
    else:
        CN_body = abs((cfg.major_axis / cfg.minor_axis) * math.cos(phi) +
                      (cfg.minor_axis / cfg.major_axis) * math.sin(phi)) * (
                          abs(math.sin(2 * alpha_rad) * math.cos(alpha_rad / 2)) +
                          2 * (cfg.total_length / cfg.diameter) * math.sin(alpha_rad) ** 2)

    # Contribution des ailerons/ailes
    alpha_abs = abs(alpha_rad)
    CN_wing_base = 2 * math.pi * alpha_abs * (cfg.wing_surface / cfg.cross_section_area)

    # Décrochage
    alpha_crit_rad = math.radians(20.0)
    if alpha_abs > alpha_crit_rad:
        num = alpha_abs - math.radians(28.0)
        den = math.radians(36.0) - math.radians(28.0)
        lift_reduction = 1.0 / (1.0 + math.exp(5.0 * num / den))
        CN_wing = CN_wing_base * lift_reduction
    else:
        CN_wing = CN_wing_base

    CN_wing = max(-1.0, min(1.0, CN_wing))
    CN_total = CN_body + CN_wing * (cfg.wing_position * (cfg.total_length - cfg.nose_length) / cfg.total_length)

    return CN_total


# --------------------------
# PARAMETRES DE VOL
# --------------------------
@dataclass
class FlightParams:
    launch_angle_deg: float
    ascent_alpha_deg: float
    descent_alpha_deg: float


@dataclass
class SimState:
    t: float
    x: float
    z: float
    vx: float
    vz: float
    mass: float
    thrust: float
    drag: float
    Cd: float
    mach: float
    alpha_deg: float
    T: float
    P: float


@dataclass
class SimResult:
    range_m: float
    flight_time: float
    reached_target_alt: bool
    states: List[SimState]


# --------------------------
# SIMULATEUR
# --------------------------
class Simulator:
    def __init__(self, config: VehicleConfig, verbose: bool = False):
        self.cfg = config
        self.verbose = verbose

    def mass_at(self, t: float) -> float:
        if t <= self.cfg.booster_time:
            return (self.cfg.initial_mass -
                    (self.cfg.initial_mass - self.cfg.mass_after_booster) * (t / self.cfg.booster_time))
        elif t <= (self.cfg.booster_time + self.cfg.sustainer_time):
            dt = t - self.cfg.booster_time
            return (self.cfg.mass_after_booster -
                    (self.cfg.mass_after_booster - self.cfg.mass_after_sustainer) *
                    (dt / self.cfg.sustainer_time))
        else:
            return self.cfg.mass_after_sustainer

    def thrust_at(self, t: float) -> float:
        if t <= self.cfg.booster_time:
            return self.cfg.booster_thrust
        elif t <= (self.cfg.booster_time + self.cfg.sustainer_time):
            return self.cfg.sustainer_thrust
        else:
            return 0.0

    def simulate(self, flight: FlightParams, mode: str = "precise",
                 dt: Optional[float] = None, max_time: Optional[float] = None,
                 record_states: bool = True) -> SimResult:
        """
        Simulation avec précision complète du modèle V6.
        - mode: "quick" (dt=0.1, Euler) ou "precise" (dt=0.05, RK4)
        """
        if mode not in ("quick", "precise"):
            raise ValueError("mode must be 'quick' or 'precise'")

        if dt is None:
            dt = 0.1 if mode == "quick" else 0.05
        if max_time is None:
            max_time = self.cfg.battery_time

        # Initial conditions
        angle_rad = math.radians(flight.launch_angle_deg)
        vx = self.cfg.initial_velocity * math.cos(angle_rad)
        vz = self.cfg.initial_velocity * math.sin(angle_rad)
        x = 0.0
        z = self.cfg.initial_altitude
        t = 0.0

        states: List[SimState] = []
        reached_target_alt = False
        en_descente = False
        launched = False

        # Nozzle exit area
        Nozzle_ft = self.cfg.nozzle_diameter * 3.28084
        A_e = (math.pi * (Nozzle_ft / 2) ** 2) * 144
        d_ft = self.cfg.diameter * 3.28084
        S_Ref = math.pi * (d_ft / 2) ** 2 * 144

        while t < max_time and z > 0:
            mass = self.mass_at(t)
            thrust = self.thrust_at(t)

            v = math.hypot(vx, vz) + 1e-8

            if v > 100:
                launched = True

            if launched and z <= self.cfg.target_altitude:
                reached_target_alt = True
                break

            if launched and v <= 100:
                break

            # Atmosphère précise
            T, P, rho = atmosphere_ISA_precise(z)
            a_sound = math.sqrt(Const.GAMMA * Const.R * T)
            mach = v / a_sound
            q = (0.5 * rho * v ** 2) / 47.88

            # Détection phase descendante
            if not en_descente and len(states) > 10 and vz < 0:
                en_descente = True

            # Angle d'attaque selon phase
            alpha_deg = flight.descent_alpha_deg if en_descente else flight.ascent_alpha_deg
            alpha_rad = math.radians(alpha_deg)

            # Aérodynamique précise
            _, _, _, _, _, Ca = calculate_drag_coefficients_precise(
                self.cfg, mach, alpha_rad, q, A_e, S_Ref)

            Fa = 0.5 * Ca * rho * self.cfg.cross_section_area * v ** 2

            # Force normale
            Cn = calculate_normal_force_coefficient(self.cfg, alpha_rad)
            Lift_drag_ratio = (Cn * math.cos(alpha_rad) - Ca * math.sin(alpha_rad)) / (
                    Cn * math.sin(alpha_rad) + Ca * math.cos(alpha_rad) + 1e-8)
            Fn = Lift_drag_ratio * Fa

            if mode == "quick":
                # Euler simple
                F_horizontal = thrust * math.cos(angle_rad)
                F_vertical = thrust * math.sin(angle_rad)
                Fa_horizontal = Fa * math.cos(alpha_rad)
                Fa_vertical = Fa * math.sin(alpha_rad)
                Fn_horizontal = Fn * math.sin(alpha_rad)
                Fn_vertical = Fn * math.cos(alpha_rad)

                ax = (F_horizontal - Fa_horizontal + Fn_horizontal) / mass
                az = (F_vertical - Fa_vertical + Fn_vertical) / mass - Const.G

                vx += ax * dt
                vz += az * dt
                x += vx * dt
                z += vz * dt
                t += dt
            else:
                # RK4 précis
                def derivatives(state):
                    sx, sz, svx, svz, st = state
                    mass_local = self.mass_at(st)
                    thrust_local = self.thrust_at(st)

                    Tloc, Ploc, rholoc = atmosphere_ISA_precise(sz)
                    vloc = math.hypot(svx, svz) + 1e-8
                    a_sound_local = math.sqrt(Const.GAMMA * Const.R * Tloc)
                    mach_local = vloc / a_sound_local
                    q_local = (0.5 * rholoc * vloc ** 2) / 47.88

                    # Déterminer l'alpha local
                    en_desc_local = svz < 0 and st > 1.0
                    alpha_deg_local = flight.descent_alpha_deg if en_desc_local else flight.ascent_alpha_deg
                    alpha_rad_local = math.radians(alpha_deg_local)

                    _, _, _, _, _, Ca_local = calculate_drag_coefficients_precise(
                        self.cfg, mach_local, alpha_rad_local, q_local, A_e, S_Ref)

                    Fa_local = 0.5 * Ca_local * rholoc * self.cfg.cross_section_area * vloc ** 2
                    Cn_local = calculate_normal_force_coefficient(self.cfg, alpha_rad_local)
                    LDR = (Cn_local * math.cos(alpha_rad_local) - Ca_local * math.sin(alpha_rad_local)) / (
                            Cn_local * math.sin(alpha_rad_local) + Ca_local * math.cos(alpha_rad_local) + 1e-8)
                    Fn_local = LDR * Fa_local

                    ux = svx / vloc
                    uz = svz / vloc

                    thrust_ax = thrust_local * ux
                    thrust_az = thrust_local * uz
                    Fa_ax = Fa_local * ux
                    Fa_az = Fa_local * uz
                    Fn_ax = Fn_local * (-uz if alpha_rad_local > 0 else uz) * abs(math.sin(alpha_rad_local))
                    Fn_az = Fn_local * (ux if alpha_rad_local > 0 else -ux) * abs(math.cos(alpha_rad_local))

                    ax_local = (thrust_ax - Fa_ax + Fn_ax) / max(1e-6, mass_local)
                    az_local = (thrust_az - Fa_az + Fn_az) / max(1e-6, mass_local) - Const.G

                    return np.array([svx, svz, ax_local, az_local, 1.0])

                state0 = np.array([x, z, vx, vz, t], dtype=float)
                k1 = derivatives(state0)
                k2 = derivatives(state0 + 0.5 * dt * k1)
                k3 = derivatives(state0 + 0.5 * dt * k2)
                k4 = derivatives(state0 + dt * k3)
                state_new = state0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                x, z, vx, vz, t = state_new.tolist()

            if record_states:
                states.append(SimState(
                    t=t, x=x, z=z, vx=vx, vz=vz, mass=mass,
                    thrust=thrust, drag=Fa, Cd=Ca, mach=mach,
                    alpha_deg=alpha_deg, T=T, P=P
                ))

        flight_time = t
        range_m = x
        return SimResult(range_m=range_m, flight_time=flight_time,
                         reached_target_alt=reached_target_alt, states=states)


# --------------------------
# RECHERCHE PARALLELE
# --------------------------
def _worker_eval(args):
    vehicle_dict, launch_angle, ascent_alpha, descent_alpha, mode, dt = args
    cfg = VehicleConfig(**vehicle_dict)
    sim = Simulator(cfg, verbose=False)
    params = FlightParams(launch_angle_deg=launch_angle,
                          ascent_alpha_deg=ascent_alpha,
                          descent_alpha_deg=descent_alpha)
    res = sim.simulate(params, mode=mode, dt=dt, record_states=False)
    return ((launch_angle, ascent_alpha, descent_alpha),
            res.range_m, res.flight_time, res.reached_target_alt)


def parallel_sweep(cfg: VehicleConfig,
                   angle_range: Tuple[float, float],
                   ascent_range: Tuple[float, float],
                   descent_range: Tuple[float, float],
                   step: float = 1.0,
                   mode: str = "quick",
                   dt: Optional[float] = None,
                   max_workers: Optional[int] = None,
                   verbose: bool = True):
    angles = np.arange(angle_range[0], angle_range[1] + 1e-9, step)
    ascents = np.arange(ascent_range[0], ascent_range[1] + 1e-9, step)
    descents = np.arange(descent_range[0], descent_range[1] + 1e-9, step)

    combos = []
    vdict = cfg.__dict__.copy()
    for a in angles:
        for b in ascents:
            for c in descents:
                combos.append((vdict, float(a), float(b), float(c), mode, dt))

    total = len(combos)
    if total == 0:
        return None

    workers = max_workers or max(1, mp.cpu_count() - 1)
    if verbose:
        print(f"Recherche parallèle ({total} combinaisons) sur {workers} workers...")

    best = None
    best_range = -1.0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker_eval, c): c for c in combos}
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            try:
                params, rng, ft, reached = fut.result()
                if reached and rng > best_range:
                    best_range = rng
                    best = (params, rng, ft, reached)
            except Exception as e:
                if verbose:
                    print(f"Erreur worker: {e}", file=sys.stderr)
            if verbose and completed % max(1, total // 10) == 0:
                print(f"Progression: {completed}/{total}")

    return best


# --------------------------
# PLOTTING (figures matplotlib)
# --------------------------
def plot_results(states: List[SimState], cfg: VehicleConfig, params: FlightParams):
    """Génère les figures matplotlib (comme V6)."""
    if not states:
        print("Pas de données à afficher.")
        return

    times = [s.t for s in states]
    xs = [s.x for s in states]
    zs = [s.z for s in states]
    vxs = [s.vx for s in states]
    vzs = [s.vz for s in states]
    masses = [s.mass for s in states]
    thrusts = [s.thrust for s in states]
    drags = [s.drag for s in states]
    Cds = [s.Cd for s in states]
    machs = [s.mach for s in states]
    alphas = [s.alpha_deg for s in states]
    Ts = [s.T for s in states]
    Ps = [s.P for s in states]

    # Figure 1: Atmosphère
    fig1 = plt.figure(figsize=(16, 12))

    plt.subplot(4, 1, 1)
    plt.plot(times, Ts, 'r-', linewidth=2)
    plt.xlabel('Temps (s)')
    plt.ylabel('Température (K)')
    plt.title('Température en fonction du temps')
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(times, Ps, 'b-', linewidth=2)
    plt.xlabel('Temps (s)')
    plt.ylabel('Pression (Pa)')
    plt.title('Pression en fonction du temps')
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(zs, Ps, 'b-', linewidth=2)
    plt.xlabel('Altitude (m)')
    plt.ylabel('Pression (Pa)')
    plt.title('Pression en fonction de l\'altitude')
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(zs, Ts, 'r-', linewidth=2)
    plt.xlabel('Altitude (m)')
    plt.ylabel('Température (K)')
    plt.title('Température en fonction de l\'altitude')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Figure 2: Trajectoire et performances
    fig2 = plt.figure(figsize=(16, 12))

    plt.subplot(4, 2, 1)
    plt.plot(xs, zs, 'b-', linewidth=2)
    plt.axhline(cfg.target_altitude, color='r', linestyle='--', label=f'Cible {cfg.target_altitude}m')
    plt.xlabel('Distance horizontale (m)')
    plt.ylabel('Altitude (m)')
    plt.title(
        f'Trajectoire - {cfg.name}\n(tir: {params.launch_angle_deg:.1f}°, α_asc: {params.ascent_alpha_deg:.1f}°, α_desc: {params.descent_alpha_deg:.1f}°)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 2, 2)
    plt.plot(times, masses, 'g-', linewidth=2)
    plt.xlabel('Temps (s)')
    plt.ylabel('Masse (kg)')
    plt.title('Évolution de la masse')
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 2, 3)
    plt.plot(times, vxs, 'b-', label='Vx', linewidth=2)
    plt.plot(times, vzs, 'r-', label='Vz', linewidth=2)
    plt.xlabel('Temps (s)')
    plt.ylabel('Vitesse (m/s)')
    plt.title('Évolution des vitesses')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 2, 4)
    plt.plot(times, thrusts, 'r-', label='Poussée', linewidth=2)
    plt.plot(times, drags, 'b-', label='Traînée', linewidth=2)
    plt.xlabel('Temps (s)')
    plt.ylabel('Force (N)')
    plt.title('Forces de propulsion et traînée')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 2, 5)
    plt.plot(times, zs, 'b-', linewidth=2)
    plt.xlabel('Temps (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Évolution de l\'altitude')
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 2, 6)
    plt.plot(times, xs, 'g-', linewidth=2)
    plt.xlabel('Temps (s)')
    plt.ylabel('Portée (m)')
    plt.title('Évolution de la portée')
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 2, 7)
    plt.plot(times, Cds, 'purple', linewidth=2)
    plt.xlabel('Temps (s)')
    plt.ylabel('Coefficient Cd')
    plt.title('Évolution du coefficient de traînée')
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 2, 8)
    plt.plot(machs, Cds, 'purple', linewidth=2)
    plt.xlabel('Nombre de Mach')
    plt.ylabel('Coefficient Cd')
    plt.title('Cd en fonction du Mach')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Figure 3: Angle d'attaque
    fig3 = plt.figure(figsize=(12, 6))
    plt.plot(times, alphas, 'orange', linewidth=2)
    plt.xlabel('Temps (s)')
    plt.ylabel('Angle d\'attaque α (degrés)')
    plt.title(
        f'Évolution de l\'angle d\'attaque\n(α_asc: {params.ascent_alpha_deg:.1f}°, α_desc: {params.descent_alpha_deg:.1f}°)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


# --------------------------
# INTERFACE CLI
# --------------------------
def interactive_menu():
    db = default_vehicle_database()
    current = list(db.keys())[0]
    cfg = db[current]
    sim = Simulator(cfg)

    print("=" * 60)
    print("=== Simulateur de Missile V8 ===")
    print("=== Précision aérodynamique complète ===")
    print("=" * 60)

    while True:
        print(f"\nVéhicule actuel: {current}")
        print("1) Sélectionner un véhicule")
        print("2) Simulation rapide (mode quick)")
        print("3) Simulation précise (mode precise, RK4)")
        print("4) Recherche parallélisée des paramètres optimaux")
        print("5) Quitter")

        c = input("\nChoix: ").strip()

        if c == "1":
            print("\nVéhicules disponibles:")
            for i, k in enumerate(db.keys()):
                print(f" {i + 1}) {k}")
            try:
                idx = int(input("Sélectionnez l'index: ").strip()) - 1
                current = list(db.keys())[idx]
                cfg = db[current]
                sim = Simulator(cfg)
                print(f"✓ Véhicule sélectionné: {current}")
            except Exception as e:
                print(f"Erreur de sélection: {e}")

        elif c == "2":
            try:
                la = float(input("Angle de tir (deg) [-45..45]: ").strip())
                aa = float(input("Angle d'attaque ascendant (deg) [-20..20]: ").strip())
                da = float(input("Angle d'attaque descendant (deg) [-36..36]: ").strip())
                params = FlightParams(la, aa, da)

                print("\nSimulation rapide en cours...")
                start = time.time()
                res = sim.simulate(params, mode="quick", dt=0.1, record_states=True)
                elapsed = time.time() - start

                print(f"\n{'=' * 60}")
                print(f"Résultats (calculés en {elapsed:.2f}s):")
                print(f"  Portée: {res.range_m:.1f} m")
                print(f"  Temps de vol: {res.flight_time:.1f} s")
                print(f"  Cible atteinte: {'OUI' if res.reached_target_alt else 'NON'}")
                print(f"{'=' * 60}")

                plot = input("\nAfficher les graphiques? (o/n): ").strip().lower()
                if plot == 'o':
                    plot_results(res.states, cfg, params)

            except Exception as e:
                print(f"Erreur: {e}")

        elif c == "3":
            try:
                la = float(input("Angle de tir (deg) [-45..45]: ").strip())
                aa = float(input("Angle d'attaque ascendant (deg) [-20..20]: ").strip())
                da = float(input("Angle d'attaque descendant (deg) [-36..36]: ").strip())
                params = FlightParams(la, aa, da)

                print("\nSimulation précise en cours (RK4)...")
                start = time.time()
                res = sim.simulate(params, mode="precise", dt=0.05, record_states=True)
                elapsed = time.time() - start

                print(f"\n{'=' * 60}")
                print(f"Résultats (calculés en {elapsed:.2f}s):")
                print(f"  Portée: {res.range_m:.1f} m")
                print(f"  Temps de vol: {res.flight_time:.1f} s")
                print(f"  Cible atteinte: {'OUI' if res.reached_target_alt else 'NON'}")
                print(f"{'=' * 60}")

                plot = input("\nAfficher les graphiques? (o/n): ").strip().lower()
                if plot == 'o':
                    plot_results(res.states, cfg, params)

            except Exception as e:
                print(f"Erreur: {e}")

        elif c == "4":
            try:
                print("\nParamètres de recherche:")
                a0 = float(input("  Angle de tir min (deg): "))
                a1 = float(input("  Angle de tir max (deg): "))
                step = float(input("  Pas de recherche (deg) [recommandé: 1.0]: "))

                asc0 = float(input("  Angle d'attaque ascendant min (deg): "))
                asc1 = float(input("  Angle d'attaque ascendant max (deg): "))

                desc0 = float(input("  Angle d'attaque descendant min (deg): "))
                desc1 = float(input("  Angle d'attaque descendant max (deg): "))

                mode = input("  Mode d'évaluation (quick/precise) [quick]: ").strip() or "quick"

                dt = 0.1 if mode == "quick" else 0.05

                print(f"\n{'=' * 60}")
                print("Lancement de la recherche parallélisée...")
                print(f"{'=' * 60}")

                start = time.time()
                best = parallel_sweep(
                    cfg,
                    angle_range=(a0, a1),
                    ascent_range=(asc0, asc1),
                    descent_range=(desc0, desc1),
                    step=step,
                    mode=mode,
                    dt=dt,
                    verbose=True
                )
                elapsed = time.time() - start

                print(f"\n{'=' * 60}")
                print(f"Recherche terminée en {elapsed:.1f}s")

                if best:
                    params, rng, ft, reached = best
                    la, aa, da = params
                    print(f"\nMeilleure configuration trouvée:")
                    print(f"  Angle de tir: {la:.2f}°")
                    print(f"  Angle d'attaque ascendant: {aa:.2f}°")
                    print(f"  Angle d'attaque descendant: {da:.2f}°")
                    print(f"  Portée: {rng:.1f} m")
                    print(f"  Temps de vol: {ft:.1f} s")
                    print(f"{'=' * 60}")

                    visualize = input("\nVisualiser cette trajectoire? (o/n): ").strip().lower()
                    if visualize == 'o':
                        best_params = FlightParams(la, aa, da)
                        res = sim.simulate(best_params, mode="precise", dt=0.05, record_states=True)
                        plot_results(res.states, cfg, best_params)
                else:
                    print("Aucune configuration valide n'a atteint la cible.")
                    print(f"{'=' * 60}")

            except Exception as e:
                print(f"Erreur: {e}")
                import traceback
                traceback.print_exc()

        elif c == "5" or c.lower() in ("q", "quit", "exit"):
            print("\nAu revoir!")
            break
        else:
            print("Choix invalide.")


def cli_main():
    parser = argparse.ArgumentParser(
        description="Simulateur de Missile V8 - Précision aérodynamique complète"
    )
    parser.add_argument("--vehicle", type=str, default="V_MICA",
                        help="Clé du véhicule dans la base de données")
    parser.add_argument("--mode", type=str, choices=["quick", "precise"], default="precise",
                        help="Mode de simulation")
    parser.add_argument("--launch", type=float, default=0.0,
                        help="Angle de tir (deg)")
    parser.add_argument("--ascent_alpha", type=float, default=2.0,
                        help="Angle d'attaque ascendant (deg)")
    parser.add_argument("--descent_alpha", type=float, default=-5.0,
                        help="Angle d'attaque descendant (deg)")
    parser.add_argument("--dt", type=float, default=None,
                        help="Pas de temps (override)")
    parser.add_argument("--plot", action="store_true",
                        help="Afficher les graphiques")
    parser.add_argument("--interactive", action="store_true",
                        help="Menu interactif")

    args = parser.parse_args()

    db = default_vehicle_database()

    if args.interactive:
        interactive_menu()
        return

    if args.vehicle not in db:
        print(f"Véhicule non trouvé. Disponibles: {', '.join(db.keys())}")
        return

    cfg = db[args.vehicle]
    sim = Simulator(cfg)
    params = FlightParams(args.launch, args.ascent_alpha, args.descent_alpha)

    print(f"Simulation {args.mode} pour {cfg.name}")
    start = time.time()
    res = sim.simulate(params, mode=args.mode, dt=args.dt, record_states=True)
    elapsed = time.time() - start

    print(f"\nRésultats (calculés en {elapsed:.2f}s):")
    print(f"  Portée: {res.range_m:.1f} m")
    print(f"  Temps de vol: {res.flight_time:.1f} s")
    print(f"  Cible atteinte: {res.reached_target_alt}")

    if args.plot:
        plot_results(res.states, cfg, params)


if __name__ == "__main__":
    try:
        cli_main()
    except KeyboardInterrupt:
        print("\n\nInterrompu par l'utilisateur")
    except Exception as e:
        print(f"Erreur fatale: {e}")
        import traceback

        traceback.print_exc()