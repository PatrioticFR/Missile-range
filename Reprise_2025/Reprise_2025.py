# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Missile / projectile simulator - V8
- Combines the robust architecture of V7 with the high-fidelity physics of V6.
- Structure: Modular (VehicleConfig, Simulator).
- Physics: Detailed multi-layer atmosphere and complex aerodynamic model from V6.
- Simulation: Dual-mode with 'quick' (Euler) and 'precise' (RK4) options.
- Features: Parallel parameter sweep, vehicle database, interactive CLI, detailed Matplotlib plots.
"""

from __future__ import annotations
import math
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
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
    STALL_ANGLE_DEG = 20.0


# --------------------------
# CONFIGURATION DU VEHICULE (MISSILE)
# --------------------------
@dataclass
class VehicleConfig:
    name: str
    initial_mass: float
    mass_after_booster: float
    mass_after_sustainer: float
    booster_time: float
    sustainer_time: float
    booster_thrust: float
    sustainer_thrust: float
    diameter: float
    total_length: float
    nose_length: float
    nozzle_diameter: float
    initial_velocity: float
    initial_altitude: float
    target_altitude: float
    wing_surface: float = 0.02
    wing_position: float = 0.7

    def __post_init__(self):
        self.cross_section_area = math.pi * (self.diameter / 2) ** 2


def default_vehicle_database() -> Dict[str, VehicleConfig]:
    return {
        "V_MICA_V8": VehicleConfig(
            "V_MICA_V8", 112.0, 88.3, 70.1, 2.75, 4.0, 20250.0, 10720.0,
            diameter=0.16, total_length=3.1, nose_length=0.4, nozzle_diameter=0.12,
            initial_velocity=300.0, initial_altitude=5000.0, target_altitude=2000.0
        ),
        "V_AIM120_V8": VehicleConfig(
            "V_AIM120_V8", 157.0, 125.0, 95.0, 3.2, 5.5, 25000.0, 12000.0,
            diameter=0.178, total_length=3.66, nose_length=0.5, nozzle_diameter=0.14,
            initial_velocity=320.0, initial_altitude=6000.0, target_altitude=2000.0
        ),
    }


# --------------------------
# ATMOSPHERE (Modèle détaillé de la V6 avec cache)
# --------------------------
@lru_cache(maxsize=4096)
def atmosphere_ISA(altitude_m: float) -> Tuple[float, float, float]:
    """
    Modèle atmosphérique détaillé basé sur la V6.
    Retourne (T[K], p[Pa], rho[kg/m3]).
    """
    z = float(max(0.0, min(altitude_m, 79000.0)))
    if z <= 11000:
        T = 288.15 - 0.0065 * z
        p = 101325 * (1 - 0.0065 * z / 288.15) ** 5.25588
    elif z <= 20000:
        T = 216.65
        p = 22632 * math.exp(-0.000157 * (z - 11000))
    elif z <= 32000:
        T = 216.65 + 0.001 * (z - 20000)
        p = 5474.87 * (1 + 0.001 * (z - 20000) / 216.65) ** -34.1632
    elif z <= 47000:
        T = 228.65 + 0.0028 * (z - 32000)
        p = 868.014 * (1 + 0.0028 * (z - 32000) / 228.65) ** -12.2011
    elif z <= 52000:
        T = 270.65
        p = 110.906 * math.exp(-0.000157 * (z - 47000))
    elif z <= 61000:
        T = 270.65 - 0.0028 * (z - 52000)
        p = 66.9389 * (T / 270.65) ** -12.2011
    else:  # z <= 79000
        T = 252.65 - 0.002 * (z - 61000)
        p = 3.95642 * (T / 214.65) ** -17.0816  # Correction de l'exposant pour la continuité
    rho = p / (Const.R * T)
    return T, p, rho


# --------------------------
# AERODYNAMIQUE (Modèle détaillé de la V6)
# --------------------------
def aerodynamic_forces_precise(config: VehicleConfig, alpha_rad: float, mach: float, rho: float, v: float,
                               is_powered: bool) -> Tuple[float, float, float, float]:
    """
    Calcule les forces aérodynamiques en utilisant le modèle détaillé de la V6.
    Retourne (Force de traînée [N], Force normale [N], Coeff. axial Ca, Coeff. normal Cn).
    """
    # Conversion des unités pour les formules de la V6 (qui utilisaient des unités impériales)
    l_n_ft = config.nose_length * 3.28084
    d_ft = config.diameter * 3.28084
    l_ft = config.total_length * 3.28084
    nozzle_d_ft = config.nozzle_diameter * 3.28084

    A_e_sqin = (math.pi * (nozzle_d_ft / 2) ** 2) * 144
    S_Ref_sqin = math.pi * (d_ft / 2) ** 2 * 144
    q_psf = (0.5 * rho * v ** 2) / 47.88  # Pression dynamique en lb/ft^2

    # --- Calcul du Coefficient Axial (Ca) ---
    if mach > 1:
        CD0_Body_Wave = (1.59 + 1.83 / mach ** 2) * (math.atan(0.5 / (l_n_ft / d_ft))) ** 1.69
    else:
        CD0_Body_Wave = 0

    if is_powered:
        CD0_Base = (1 - A_e_sqin / S_Ref_sqin) * (0.25 / mach if mach > 1 else 0.12 + 0.13 * mach ** 2)
    else:
        CD0_Base = 0.25 / mach if mach > 1 else 0.12 + 0.13 * mach ** 2

    CD0_Body_Friction = 0.053 * (l_ft / d_ft) * (mach / (q_psf * l_ft + 1e-9)) ** 0.2

    alpha_abs = abs(alpha_rad)
    CD_Wing_base = 0.1 * (config.wing_surface / config.cross_section_area) * (math.sin(alpha_abs) ** 2)
    if mach > 1:
        CD_Wing_base += 0.05 * (config.wing_surface / config.cross_section_area)

    alpha_crit_rad = math.radians(Const.STALL_ANGLE_DEG)
    if alpha_abs > alpha_crit_rad:
        stall_end_rad = math.radians(36.0)
        factor = 1.0 + 0.1 * min((alpha_abs - alpha_crit_rad) / (stall_end_rad - alpha_crit_rad), 1.0)
        CD_Wing = CD_Wing_base * factor
    else:
        CD_Wing = CD_Wing_base

    Ca = CD0_Body_Wave + CD0_Base + CD0_Body_Friction + CD_Wing

    # --- Calcul du Coefficient Normal (Cn) ---
    # Pour le corps, on simplifie à un terme proportionnel à sin(2a)
    CN_body = (2 * config.total_length / config.diameter) * math.sin(alpha_rad) ** 2

    CN_wing_base = 2 * math.pi * alpha_rad * (config.wing_surface / config.cross_section_area)

    if alpha_abs > alpha_crit_rad:
        num = alpha_abs - math.radians(28.0)
        den = math.radians(36.0) - math.radians(28.0)
        lift_reduction = 1.0 / (1.0 + math.exp(5.0 * num / (den + 1e-9)))
        CN_wing = CN_wing_base * lift_reduction
    else:
        CN_wing = CN_wing_base

    Cn = CN_body + CN_wing

    # --- Calcul des Forces ---
    dynamic_pressure_pa = 0.5 * rho * v ** 2
    drag_force = Ca * dynamic_pressure_pa * config.cross_section_area
    normal_force = Cn * dynamic_pressure_pa * config.cross_section_area

    return drag_force, normal_force, Ca, Cn


# --------------------------
# SIMULATEUR
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
    normal_force: float
    Ca: float
    Cn: float
    mach: float


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
        if t <= self.cfg.booster_time:
            return self.cfg.initial_mass - (self.cfg.initial_mass - self.cfg.mass_after_booster) * (
                        t / self.cfg.booster_time)
        elif t <= (self.cfg.booster_time + self.cfg.sustainer_time):
            dt_sustainer = t - self.cfg.booster_time
            return self.cfg.mass_after_booster - (self.cfg.mass_after_booster - self.cfg.mass_after_sustainer) * (
                        dt_sustainer / self.cfg.sustainer_time)
        else:
            return self.cfg.mass_after_sustainer

    def thrust_at(self, t: float) -> float:
        if t <= self.cfg.booster_time:
            return self.cfg.booster_thrust
        elif t <= (self.cfg.booster_time + self.cfg.sustainer_time):
            return self.cfg.sustainer_thrust
        else:
            return 0.0

    def simulate(self, flight: FlightParams, mode: str = "quick", dt: Optional[float] = None,
                 max_time: Optional[float] = None, record_states: bool = True) -> SimResult:
        dt = dt if dt is not None else (0.5 if mode == "quick" else 0.05)
        max_time = max_time if max_time is not None else 300.0

        angle_rad = math.radians(flight.launch_angle_deg)
        state = np.array([0.0, self.cfg.initial_altitude, self.cfg.initial_velocity * math.cos(angle_rad),
                          self.cfg.initial_velocity * math.sin(angle_rad)], dtype=float)
        t = 0.0

        states_history: List[SimState] = []

        def derivatives(current_state, current_t):
            x, z, vx, vz = current_state

            mass = self.mass_at(current_t)
            thrust_magnitude = self.thrust_at(current_t)

            T, p, rho = atmosphere_ISA(z)
            v = math.hypot(vx, vz) + 1e-8

            if mode == "quick":
                # Simplification pour le mode rapide
                drag_magnitude = 0.5 * 0.3 * rho * self.cfg.cross_section_area * v ** 2
                normal_force_magnitude = 0.0
            else:  # mode == "precise"
                a_sound = math.sqrt(Const.GAMMA * Const.R * T)
                mach = v / a_sound

                alpha_deg = flight.ascent_alpha_deg if (vz > 0 or current_t < 1.0) else flight.descent_alpha_deg
                alpha_rad = math.radians(alpha_deg)

                drag_magnitude, normal_force_magnitude, _, _ = aerodynamic_forces_precise(self.cfg, alpha_rad, mach,
                                                                                          rho, v, thrust_magnitude > 0)

            # Décomposition des forces
            # Vecteur vitesse unitaire
            ux, uz = vx / v, vz / v

            # Poussée alignée avec la vitesse (guidage simple)
            thrust_x, thrust_z = thrust_magnitude * ux, thrust_magnitude * uz

            # Traînée opposée à la vitesse
            drag_x, drag_z = -drag_magnitude * ux, -drag_magnitude * uz

            # Force normale perpendiculaire à la vitesse
            normal_x, normal_z = -normal_force_magnitude * uz, normal_force_magnitude * ux

            # Somme des forces
            total_force_x = thrust_x + drag_x + normal_x
            total_force_z = thrust_z + drag_z + normal_z - mass * Const.G

            # Accélérations
            ax, az = total_force_x / mass, total_force_z / mass

            return np.array([vx, vz, ax, az])

        while t < max_time and state[1] > -100:  # state[1] is altitude z
            # RK4 pour l'intégration
            k1 = derivatives(state, t)
            k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = derivatives(state + dt * k3, t + dt)
            state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt

            if record_states:
                # Stocker l'état détaillé pour les graphiques
                x, z, vx, vz = state
                mass = self.mass_at(t)
                thrust = self.thrust_at(t)
                T, _, rho = atmosphere_ISA(z)
                v = math.hypot(vx, vz)
                a_sound = math.sqrt(Const.GAMMA * Const.R * T)
                mach = v / a_sound
                alpha_deg = flight.ascent_alpha_deg if vz > 0 else flight.descent_alpha_deg
                drag, normal_force, Ca, Cn = aerodynamic_forces_precise(self.cfg, math.radians(alpha_deg), mach, rho, v,
                                                                        thrust > 0)

                states_history.append(SimState(t, x, z, vx, vz, mass, thrust, drag, normal_force, Ca, Cn, mach))

            if state[1] <= self.cfg.target_altitude and state[3] <= 0:
                break

        reached = state[1] <= self.cfg.target_altitude and state[3] <= 0
        return SimResult(range_m=state[0], flight_time=t, reached_target_alt=reached, states=states_history)


# --------------------------
# RECHERCHE PARALLELE
# --------------------------
def _worker_eval(args):
    vehicle_dict, launch_angle, ascent_alpha, descent_alpha, mode, dt = args
    cfg = VehicleConfig(**vehicle_dict)
    sim = Simulator(cfg, verbose=False)
    params = FlightParams(launch_angle, ascent_alpha, descent_alpha)
    res = sim.simulate(params, mode=mode, dt=dt, record_states=False)
    return ((launch_angle, ascent_alpha, descent_alpha), res.range_m, res.flight_time, res.reached_target_alt)


def parallel_sweep(cfg: VehicleConfig, angle_range: Tuple[float, float], ascent_range: Tuple[float, float],
                   descent_range: Tuple[float, float], step: float, mode: str, dt: float,
                   max_workers: Optional[int] = None, verbose: bool = True):
    angles = np.arange(angle_range[0], angle_range[1] + 1e-9, step)
    ascents = np.arange(ascent_range[0], ascent_range[1] + 1e-9, step)
    descents = np.arange(descent_range[0], descent_range[1] + 1e-9, step)

    combos = [(cfg.__dict__, a, b, c, mode, dt) for a in angles for b in ascents for c in descents]
    if not combos: return None

    workers = max_workers or max(1, mp.cpu_count() - 1)
    if verbose: print(f"Lancement de la recherche parallèle ({len(combos)} combinaisons) sur {workers} processus...")

    best_result = None
    best_range = -1.0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker_eval, c): c for c in combos}
        for i, fut in enumerate(as_completed(futures)):
            if verbose: print(f"Progrès: {i + 1}/{len(combos)}", end='\r')
            try:
                params, rng, _, reached = fut.result()
                if reached and rng > best_range:
                    best_range = rng
                    best_result = (params, rng)
            except Exception as e:
                if verbose: print(f"\nErreur dans un processus: {e}")
    if verbose: print("\nRecherche terminée.")
    return best_result


# --------------------------
# GRAPHIQUES (Style V6)
# --------------------------
def generate_detailed_plots(result: SimResult, cfg: VehicleConfig, params: FlightParams):
    if not result.states:
        print("Aucune donnée à afficher.")
        return

    s = result.states
    t = [st.t for st in s]
    x = [st.x / 1000 for st in s]  # en km
    z = [st.z / 1000 for st in s]  # en km
    mass = [st.mass for st in s]
    vx = [st.vx for st in s]
    vz = [st.vz for st in s]
    thrust = [st.thrust for st in s]
    drag = [st.drag for st in s]
    Ca = [st.Ca for st in s]
    mach = [st.mach for st in s]

    # --- Figure 1: Trajectoire et performance ---
    fig1, axs1 = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle(f'Résultats de Simulation pour {cfg.name}', fontsize=16)

    # Trajectoire
    axs1[0, 0].plot(x, z, label='Trajectoire')
    axs1[0, 0].axhline(cfg.target_altitude / 1000, color='r', linestyle='--',
                       label=f'Altitude Cible ({cfg.target_altitude} m)')
    axs1[0, 0].set_xlabel('Distance Horizontale (km)')
    axs1[0, 0].set_ylabel('Altitude (km)')
    axs1[0, 0].set_title(f'Trajectoire (Angle: {params.launch_angle_deg}°)')
    axs1[0, 0].grid(True)
    axs1[0, 0].legend()

    # Masse
    axs1[0, 1].plot(t, mass)
    axs1[0, 1].set_xlabel('Temps (s)')
    axs1[0, 1].set_ylabel('Masse (kg)')
    axs1[0, 1].set_title('Évolution de la Masse')
    axs1[0, 1].grid(True)

    # Vitesses
    axs1[1, 0].plot(t, vx, label='Vitesse Horizontale')
    axs1[1, 0].plot(t, vz, label='Vitesse Verticale')
    axs1[1, 0].set_xlabel('Temps (s)')
    axs1[1, 0].set_ylabel('Vitesse (m/s)')
    axs1[1, 0].set_title('Évolution des Vitesses')
    axs1[1, 0].grid(True)
    axs1[1, 0].legend()

    # Forces
    axs1[1, 1].plot(t, thrust, label='Poussée')
    axs1[1, 1].plot(t, drag, label='Traînée')
    axs1[1, 1].set_xlabel('Temps (s)')
    axs1[1, 1].set_ylabel('Force (N)')
    axs1[1, 1].set_title('Forces Principales')
    axs1[1, 1].grid(True)
    axs1[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Figure 2: Données aérodynamiques ---
    fig2, axs2 = plt.subplots(1, 2, figsize=(15, 6))
    fig2.suptitle('Analyse Aérodynamique', fontsize=16)

    # Coeff de traînée vs Temps
    axs2[0].plot(t, Ca)
    axs2[0].set_xlabel('Temps (s)')
    axs2[0].set_ylabel('Coefficient de traînée (Ca)')
    axs2[0].set_title('Ca en fonction du temps')
    axs2[0].grid(True)

    # Coeff de traînée vs Mach
    axs2[1].plot(mach, Ca, 'o', markersize=2)
    axs2[1].set_xlabel('Nombre de Mach')
    axs2[1].set_ylabel('Coefficient de traînée (Ca)')
    axs2[1].set_title('Ca en fonction du nombre de Mach')
    axs2[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --------------------------
# INTERFACE UTILISATEUR
# --------------------------
def interactive_menu():
    db = default_vehicle_database()
    current_key = list(db.keys())[0]

    print("=== Simulateur de Missile V8 ===")
    while True:
        cfg = db[current_key]
        sim = Simulator(cfg)
        print(f"\nVéhicule actuel : {cfg.name}")
        print("1) Choisir un autre véhicule")
        print("2) Lancer une simulation précise")
        print("3) Lancer une recherche de paramètres (parallèle)")
        print("4) Quitter")
        choice = input("Votre choix : ").strip()

        if choice == '1':
            for i, key in enumerate(db.keys()): print(f" {i + 1}) {key}")
            try:
                idx = int(input("Index : ")) - 1
                if 0 <= idx < len(db):
                    current_key = list(db.keys())[idx]
                else:
                    print("Index invalide.")
            except ValueError:
                print("Entrée invalide.")

        elif choice == '2':
            try:
                la = float(input("Angle de tir (deg) [-45, 45]: "))
                aa = float(input("Angle d'attaque montée (deg) [-20, 20]: "))
                da = float(input("Angle d'attaque descente (deg) [-36, 36]: "))
                params = FlightParams(la, aa, da)
                print("Lancement de la simulation précise...")
                res = sim.simulate(params, mode="precise", record_states=True)
                print(
                    f"Résultat -> Portée: {res.range_m / 1000:.2f} km | Temps de vol: {res.flight_time:.1f} s | Cible atteinte: {res.reached_target_alt}")
                generate_detailed_plots(res, cfg, params)
            except ValueError:
                print("Entrée invalide.")

        elif choice == '3':
            try:
                print("Définir les plages de recherche :")
                a_start, a_end = map(float, input("Angle de tir [start end]: ").split())
                asc_start, asc_end = map(float, input("AoA montée [start end]: ").split())
                desc_start, desc_end = map(float, input("AoA descente [start end]: ").split())
                step = float(input("Pas de recherche (deg): "))

                best = parallel_sweep(cfg, (a_start, a_end), (asc_start, asc_end), (desc_start, desc_end), step=step,
                                      mode="quick", dt=0.5)

                if best:
                    (la, aa, da), rng = best
                    print("\n--- Meilleur résultat trouvé (recherche rapide) ---")
                    print(f"  Angle de tir: {la:.2f}°")
                    print(f"  AoA montée: {aa:.2f}°")
                    print(f"  AoA descente: {da:.2f}°")
                    print(f"  Portée estimée: {rng / 1000:.2f} km")
                    if input("Lancer une simulation précise avec ces paramètres ? (o/n): ").lower() == 'o':
                        params = FlightParams(la, aa, da)
                        res = sim.simulate(params, mode="precise", record_states=True)
                        print(
                            f"Résultat -> Portée: {res.range_m / 1000:.2f} km | Temps de vol: {res.flight_time:.1f} s | Cible atteinte: {res.reached_target_alt}")
                        generate_detailed_plots(res, cfg, params)
                else:
                    print("Aucune combinaison valide n'a été trouvée.")
            except ValueError:
                print("Entrée invalide.")

        elif choice == '4':
            print("Au revoir !")
            break
        else:
            print("Choix invalide.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Missile Simulator V8")
    parser.add_argument("--interactive", action="store_true", help="Lancer le menu interactif")
    args = parser.parse_args()

    if args.interactive:
        interactive_menu()
    else:
        print("Lancer avec '--interactive' pour utiliser le simulateur.")
        # Exemple de lancement non-interactif
        cfg = default_vehicle_database()["V_MICA_V8"]
        sim = Simulator(cfg)
        params = FlightParams(launch_angle_deg=15, ascent_alpha_deg=5, descent_alpha_deg=-5)
        print(f"Lancement d'un exemple pour {cfg.name}...")
        res = sim.simulate(params, mode="precise")
        print(
            f"Résultat -> Portée: {res.range_m / 1000:.2f} km | Temps de vol: {res.flight_time:.1f} s | Cible atteinte: {res.reached_target_alt}")

