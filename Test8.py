#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Missile / projectile simulator - V7
- Mode "quick"  : simulation approximative, très rapide (coarse dt, modèle de traînée simplifié)
- Mode "precise": simulation détaillée (petit dt, RK4, aérodynamique plus complet)
- Interface CLI + menu interactif
- Parameter sweep parallèle (ProcessPool) pour recherche coarse
- Conçu pour être copié-collé dans un fichier .py et exécuté

Dependencies: numpy, matplotlib
Run: python missile_sim_v7.py --help
"""

from __future__ import annotations
import math
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # pour utilisation dans PyCharm sans display
import matplotlib.pyplot as plt
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import sys
import os

# --------------------------
# CONSTANTES
# --------------------------
class Const:
    G = 9.80665
    R = 287.05
    GAMMA = 1.405
    STALL_ANGLE_DEG = 20.0

# --------------------------
# CONFIGURATION MINIMALE DU VEHICULE (MISSILE)
# --------------------------
@dataclass
class VehicleConfig:
    name: str
    initial_mass: float                 # kg
    mass_after_booster: float           # kg
    mass_after_sustainer: float         # kg
    booster_time: float                 # s
    sustainer_time: float               # s
    booster_thrust: float               # N
    sustainer_thrust: float             # N
    diameter: float                     # m (used for cross-section)
    total_length: float                 # m
    initial_velocity: float             # m/s (magnitude)
    initial_altitude: float             # m
    target_altitude: float              # m
    wing_surface: float = 0.02          # m2 (small control surfaces)
    wing_position: float = 0.7          # coefficient 0-1

    def __post_init__(self):
        self.cross_section_area = math.pi * (self.diameter/2)**2

# Exemples "neutres" (tu peux modifier)
def default_vehicle_database() -> Dict[str, VehicleConfig]:
    return {
        "V_MICA": VehicleConfig(
            "V_MICA", 112.0, 88.3, 70.1, 2.75, 4.0, 20250.0, 10720.0,
            diameter=0.16, total_length=3.1, initial_velocity=300.0,
            initial_altitude=5000.0, target_altitude=2000.0
        ),
        "V_AIM120": VehicleConfig(
            "V_AIM120", 157.0, 125.0, 95.0, 3.2, 5.5, 25000.0, 12000.0,
            diameter=0.178, total_length=3.66, initial_velocity=320.0,
            initial_altitude=6000.0, target_altitude=2000.0
        ),
        "V_TEST": VehicleConfig(
            "V_TEST", 50.0, 40.0, 35.0, 2.0, 3.0, 8000.0, 4000.0,
            diameter=0.12, total_length=2.0, initial_velocity=250.0,
            initial_altitude=2000.0, target_altitude=1000.0
        ),
    }

# --------------------------
# ATMOSPHERE (cached)
# --------------------------
@lru_cache(maxsize=4096)
def atmosphere_ISA(altitude_m: float) -> Tuple[float, float, float]:
    """
    Simplified ISA (troposphere + stratos approx). Returns (T[K], p[Pa], rho[kg/m3])
    altitude_m clipped to [0, 79000]
    """
    z = float(max(0.0, min(altitude_m, 79000.0)))
    # simple layer model (only first two layers implemented with continuity)
    if z <= 11000:
        T = 288.15 - 0.0065 * z
        p = 101325.0 * (T / 288.15) ** (-Const.G / (0.0065 * Const.R))
    elif z <= 20000:
        T = 216.65
        p11 = 22632.06
        p = p11 * math.exp(-Const.G * (z - 11000) / (Const.R * T))
    else:
        # fallback approximate exponential decay
        T = 216.65
        p = 5474.866 * math.exp(-0.000157 * (z - 20000))
    rho = p / (Const.R * T)
    return T, p, rho

# --------------------------
# AERODYNAMICS (light & advanced)
# --------------------------
def drag_coefficient_simplified() -> float:
    # fixed coarse drag coefficient used in "quick" mode
    return 0.35

def aerodynamic_coefficients_precise(config: VehicleConfig, alpha_deg: float,
                                     mach: float, altitude: float) -> Tuple[float, float]:
    """
    Return (C_d, C_l) - reasonably fast analytic approximations.
    Not a CFD model; intended for educational/academic uses.
    """
    alpha = math.radians(alpha_deg)
    # CL ~ a*alpha until stall
    a = 2 * math.pi  # per radian (thin airfoil approx)
    CL = a * alpha * (config.wing_surface / max(1e-6, config.cross_section_area))
    # reduction after stall (smooth)
    if abs(alpha_deg) > Const.STALL_ANGLE_DEG:
        CL *= 1.0 / (1.0 + (abs(alpha_deg) - Const.STALL_ANGLE_DEG)**1.5 / 30.0)

    # Cd: base + induced + wave (simple)
    Cd0 = 0.02 + 0.04 * (config.wing_surface / config.cross_section_area)
    k = 1.0 / (math.pi * (config.wing_surface / (config.diameter + 1e-6)) * 0.8 + 1e-6)
    Cd_induced = k * CL**2
    Cd_wave = 0.0
    if mach > 0.95:
        Cd_wave = 0.02 * (mach - 0.95) ** 2
    Cd = Cd0 + Cd_induced + Cd_wave
    return max(0.001, Cd), CL

# --------------------------
# SIMULATOR
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

@dataclass
class SimResult:
    range_m: float
    flight_time: float
    reached_target_alt: bool
    states: List[SimState]

class Simulator:
    def __init__(self, config: VehicleConfig, verbose: bool = False):
        self.cfg = config
        self.verbose = verbose

    def mass_at(self, t: float) -> float:
        # piecewise linear mass burn (booster -> sustainer -> residual)
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

    def simulate(self, flight: FlightParams, mode: str = "quick",
                 dt: Optional[float] = None, max_time: Optional[float] = None,
                 record_states: bool = True) -> SimResult:
        """
        Run simulation.
        - mode: "quick" (fast, approximate) or "precise" (RK4, smaller dt)
        - dt: time step override (None => default chosen per mode)
        - returns SimResult
        """
        if mode not in ("quick", "precise"):
            raise ValueError("mode must be 'quick' or 'precise'")

        # defaults
        if dt is None:
            dt = 0.5 if mode == "quick" else 0.05
        if max_time is None:
            max_time = max(60.0, self.cfg.booster_time + self.cfg.sustainer_time + 2 * 60.0)

        # initial conditions
        angle_rad = math.radians(flight.launch_angle_deg)
        vx = self.cfg.initial_velocity * math.cos(angle_rad)
        vz = self.cfg.initial_velocity * math.sin(angle_rad)
        x = 0.0
        z = self.cfg.initial_altitude
        t = 0.0

        states: List[SimState] = []
        reached_target_alt = False

        # small helper: aerodynamic forces
        while t < max_time and z > 0 and t < 3600.0:
            mass = self.mass_at(t)
            thrust = self.thrust_at(t)

            T, p, rho = atmosphere_ISA(z)
            v = math.hypot(vx, vz) + 1e-8
            a_sound = math.sqrt(Const.GAMMA * Const.R * T)
            mach = v / a_sound

            # choose alpha (angle of attack) depending on flight phase:
            # quick heuristic: if ascending (vz>0) use ascent_alpha, else descent_alpha
            alpha_deg = flight.ascent_alpha_deg if (vz > 0 or t < 0.5) else flight.descent_alpha_deg

            if mode == "quick":
                # approximate drag: constant Cd, simple acceleration (Euler)
                Cd = drag_coefficient_simplified()
                drag = 0.5 * Cd * rho * self.cfg.cross_section_area * v * v
                # thrust projected along velocity vector (assume launch angle for initial phase)
                thrust_ax = thrust * math.cos(angle_rad)
                thrust_az = thrust * math.sin(angle_rad)
                ax = (thrust_ax - drag * (vx / v)) / max(1e-6, mass)
                az = (thrust_az - drag * (vz / v)) / max(1e-6, mass) - Const.G
                # integrate (simple explicit Euler)
                vx += ax * dt
                vz += az * dt
                x += vx * dt
                z += vz * dt
                t += dt
            else:
                # precise: RK4 integration on velocities and positions with aerodynamic model
                def derivatives(state):
                    sx, sz, svx, svz, st = state
                    mass_local = self.mass_at(st)
                    thrust_local = self.thrust_at(st)
                    Tloc, ploc, rholoc = atmosphere_ISA(sz)
                    vloc = math.hypot(svx, svz) + 1e-8
                    a_sound_local = math.sqrt(Const.GAMMA * Const.R * Tloc)
                    mach_local = vloc / a_sound_local
                    Cd, Cl = aerodynamic_coefficients_precise(self.cfg, alpha_deg, mach_local, sz)
                    drag_local = 0.5 * Cd * rholoc * self.cfg.cross_section_area * vloc**2
                    # thrust decomposed along current velocity direction to give acceleration (approx)
                    ux = svx / vloc
                    uz = svz / vloc
                    # approximate thrust vector: align with instantaneous velocity direction (simple guidance)
                    thrust_ax_local = thrust_local * ux
                    thrust_az_local = thrust_local * uz
                    ax_local = (thrust_ax_local - drag_local * ux) / max(1e-6, mass_local)
                    az_local = (thrust_az_local - drag_local * uz) / max(1e-6, mass_local) - Const.G
                    return np.array([svx, svz, ax_local, az_local, 1.0])  # dt derivative for time

                # state vector [x, z, vx, vz, t]
                state0 = np.array([x, z, vx, vz, t], dtype=float)
                # RK4
                k1 = derivatives(state0)
                k2 = derivatives(state0 + 0.5*dt*k1)
                k3 = derivatives(state0 + 0.5*dt*k2)
                k4 = derivatives(state0 + dt*k3)
                state_new = state0 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
                x, z, vx, vz, t = state_new.tolist()

            if record_states:
                states.append(SimState(t=t, x=x, z=z, vx=vx, vz=vz, mass=mass))

            # stopping criteria: reached or passed below target altitude (when descending)
            if z <= self.cfg.target_altitude and (vz <= 0):
                reached_target_alt = True
                break

            # safety clamps
            if z < -1000:
                break

        # final range is horizontal position at end
        flight_time = t
        range_m = x
        return SimResult(range_m=range_m, flight_time=flight_time, reached_target_alt=reached_target_alt, states=states)

# --------------------------
# PARAMETER SWEEP (parallélisé)
# --------------------------
# Worker must be top-level for ProcessPool pickling
def _worker_eval(args):
    """
    args: (vehicle_dict, launch_angle, ascent_alpha, descent_alpha, mode, dt)
    Returns: tuple(params_tuple, result.range_m, result.flight_time, result.reached_target_alt)
    """
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
                   angle_range: Tuple[float,float], ascent_range: Tuple[float,float],
                   descent_range: Tuple[float,float], step: float = 1.0,
                   mode: str = "quick", dt: Optional[float] = None,
                   max_workers: Optional[int] = None,
                   verbose: bool = True):
    """
    Coarse parallel sweep over parameter grid.
    Returns best (params, result) among evaluated points (only those that reached target altitude).
    """
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
        print(f"Running parallel sweep ({total} combos) on {workers} workers...")

    best = None
    best_range = -1.0
    results = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker_eval, c): c for c in combos}
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            try:
                params, rng, ft, reached = fut.result()
                # store or compare only if reached target alt (or could store all)
                if reached and rng > best_range:
                    best_range = rng
                    best = (params, rng, ft, reached)
            except Exception as e:
                # ignore single failures, continue
                if verbose:
                    print(f"Worker error: {e}", file=sys.stderr)
            if verbose and completed % max(1, total//10) == 0:
                print(f"Progress: {completed}/{total}")

    return best

# --------------------------
# PLOTTING HELPERS
# --------------------------
def save_trajectory_plot(states: List[SimState], cfg: VehicleConfig, params: FlightParams, filename: Optional[str] = None):
    if not states:
        print("No states to plot.")
        return None
    times = [s.t for s in states]
    zs = [s.z for s in states]
    xs = [s.x for s in states]
    vxs = [s.vx for s in states]
    # basic figure
    plt.figure(figsize=(10,6))
    plt.plot(xs, zs, '-', linewidth=2)
    plt.axhline(cfg.target_altitude, color='r', linestyle='--', label=f"target alt {cfg.target_altitude} m")
    plt.xlabel("Horizontal distance (m)")
    plt.ylabel("Altitude (m)")
    plt.title(f"Trajectory - {cfg.name}  launch {params.launch_angle_deg:.1f}°")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if filename is None:
        filename = f"trajectory_v7_{int(time.time())}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    return filename

# --------------------------
# CLI / INTERACTIVE MENU
# --------------------------
def interactive_menu():
    db = default_vehicle_database()
    current = list(db.keys())[0]
    cfg = db[current]
    sim = Simulator(cfg)
    print("=== Missile Simulator V7 ===")
    while True:
        print(f"\nCurrent vehicle: {current}")
        print("1) Select vehicle")
        print("2) Quick sim (fast)")
        print("3) Precise sim (slow, accurate)")
        print("4) Parallel parameter sweep (coarse)")
        print("5) Exit")
        c = input("Choice: ").strip()
        if c == "1":
            print("Available vehicles:")
            for i, k in enumerate(db.keys()):
                print(f" {i+1}) {k}")
            try:
                idx = int(input("Select index: ").strip()) - 1
                current = list(db.keys())[idx]
                cfg = db[current]
                sim = Simulator(cfg)
                print(f"Selected {current}")
            except Exception:
                print("Invalid selection")
        elif c == "2":
            try:
                la = float(input("Launch angle (deg) [-45..45]: ").strip())
                aa = float(input("Ascent alpha (deg) [-20..20]: ").strip())
                da = float(input("Descent alpha (deg) [-36..36]: ").strip())
                params = FlightParams(la, aa, da)
                print("Running quick sim...")
                res = sim.simulate(params, mode="quick", dt=0.5, record_states=True)
                print(f"Range: {res.range_m:.1f} m | Time: {res.flight_time:.1f} s | reached_target: {res.reached_target_alt}")
                fname = save_trajectory_plot(res.states, cfg, params)
                if fname:
                    print("Plot saved to", fname)
            except Exception as e:
                print("Error:", e)
        elif c == "3":
            try:
                la = float(input("Launch angle (deg) [-45..45]: ").strip())
                aa = float(input("Ascent alpha (deg) [-20..20]: ").strip())
                da = float(input("Descent alpha (deg) [-36..36]: ").strip())
                params = FlightParams(la, aa, da)
                print("Running precise sim (this can be slow)...")
                res = sim.simulate(params, mode="precise", dt=0.05, record_states=True, max_time=300.0)
                print(f"Range: {res.range_m:.1f} m | Time: {res.flight_time:.1f} s | reached_target: {res.reached_target_alt}")
                fname = save_trajectory_plot(res.states, cfg, params)
                if fname:
                    print("Plot saved to", fname)
            except Exception as e:
                print("Error:", e)
        elif c == "4":
            try:
                a0 = float(input("Angle start: "))
                a1 = float(input("Angle end: "))
                step = float(input("Step (deg): "))
                asc0 = float(input("Ascent alpha start: "))
                asc1 = float(input("Ascent alpha end: "))
                desc0 = float(input("Descent alpha start: "))
                desc1 = float(input("Descent alpha end: "))
                mode = input("Mode for evaluation (quick/precise) [quick]: ").strip() or "quick"
                dt = None
                if mode == "precise":
                    dt = 0.05
                else:
                    dt = 0.5
                print("Launching parallel sweep (coarse). This will spawn worker processes.")
                best = parallel_sweep(cfg,
                                      angle_range=(a0, a1),
                                      ascent_range=(asc0, asc1),
                                      descent_range=(desc0, desc1),
                                      step=step, mode=mode, dt=dt, verbose=True)
                if best:
                    params, rng, ft, reached = best
                    la, aa, da = params
                    print(f"Best found: launch={la:.2f}°, ascent_alpha={aa:.2f}°, descent_alpha={da:.2f}° -> range={rng:.1f} m (t={ft:.1f}s)")
                else:
                    print("No valid result reached target altitude.")
            except Exception as e:
                print("Error:", e)
        elif c == "5" or c.lower() in ("q","quit","exit"):
            print("bye")
            break
        else:
            print("unknown choice")

def cli_main():
    parser = argparse.ArgumentParser(description="Missile Simulator V7 - quick/precise modes")
    parser.add_argument("--vehicle", type=str, default="V_MICA", help="vehicle key from internal database")
    parser.add_argument("--mode", type=str, choices=["quick","precise"], default="quick", help="simulation mode")
    parser.add_argument("--launch", type=float, default=0.0, help="launch angle deg")
    parser.add_argument("--ascent_alpha", type=float, default=2.0, help="ascent alpha deg")
    parser.add_argument("--descent_alpha", type=float, default=-5.0, help="descent alpha deg")
    parser.add_argument("--dt", type=float, default=None, help="time step override")
    parser.add_argument("--plot", action="store_true", help="save a trajectory plot")
    parser.add_argument("--interactive", action="store_true", help="open interactive menu")
    args = parser.parse_args()

    db = default_vehicle_database()
    if args.interactive:
        interactive_menu()
        return

    if args.vehicle not in db:
        print("Vehicle not found. Available:", ", ".join(db.keys()))
        return

    cfg = db[args.vehicle]
    sim = Simulator(cfg)
    params = FlightParams(args.launch, args.ascent_alpha, args.descent_alpha)
    print(f"Running simulation {args.mode} for {cfg.name}")
    res = sim.simulate(params, mode=args.mode, dt=args.dt, record_states=True)
    print(f"Range: {res.range_m:.1f} m")
    print(f"Flight time: {res.flight_time:.1f} s")
    print(f"Reached target altitude: {res.reached_target_alt}")
    if args.plot:
        fname = save_trajectory_plot(res.states, cfg, params)
        if fname:
            print("Plot saved:", fname)

if __name__ == "__main__":
    try:
        cli_main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print("Fatal error:", e)
        import traceback
        traceback.print_exc()
