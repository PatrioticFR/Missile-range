"""
Simulateur de trajectoire de missile optimisé
Auteur: Version améliorée
Plus rapide que les codes précedents
Atention etude paramétrique + rapide mais - précise
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
plt.ioff()  # Désactive le mode interactif
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from functools import lru_cache
import logging


# ============================================================================
# CONSTANTES PHYSIQUES ET AÉRODYNAMIQUES
# ============================================================================

class PhysicalConstants:
    """Constantes physiques utilisées dans la simulation"""
    GRAVITY = 9.81  # m/s²
    GAS_CONSTANT = 287.05  # J/(kg·K)
    GAMMA = 1.405  # Rapport des chaleurs spécifiques

    # Conversion d'unités
    M_TO_FT = 3.28084
    FT2_TO_M2 = 1 / 10.764

    # Paramètres aérodynamiques
    STALL_ANGLE_DEG = 20.0
    MAX_ANGLE_DEG = 36.0
    FRICTION_COEFFICIENT = 0.053


class AtmosphereLayer(NamedTuple):
    """Définition d'une couche atmosphérique"""
    altitude_max: float
    temp_gradient: float
    temp_base: float
    pressure_base: float
    pressure_exp: float


# ============================================================================
# STRUCTURES DE DONNÉES
# ============================================================================

@dataclass
class MissileConfiguration:
    """Configuration complète d'un missile"""
    # Masses (kg)
    initial_mass: float
    mass_after_booster: float
    mass_after_sustainer: float

    # Temps (s)
    booster_time: float
    sustainer_time: float
    battery_life: float

    # Forces (N)
    booster_thrust: float
    sustainer_thrust: float

    # Géométrie (m)
    nose_length: float
    diameter: float
    total_length: float
    nozzle_diameter: float

    # Géométrie elliptique
    major_axis: float
    minor_axis: float

    # Paramètres de vol
    initial_velocity: float
    initial_altitude: float
    target_altitude: float
    target_horizontal_velocity: float

    # Surfaces de contrôle (m² et m)
    wing_surface: float = 0.02
    wing_length: float = 0.5
    wing_position: float = 0.7  # Position relative (0-1)

    def __post_init__(self):
        """Validation des paramètres d'entrée"""
        self._validate_parameters()
        self._precompute_values()

    def _validate_parameters(self):
        """Valide les paramètres d'entrée"""
        if self.initial_mass <= 0:
            raise ValueError("La masse initiale doit être positive")
        if self.booster_time < 0 or self.sustainer_time < 0:
            raise ValueError("Les temps de combustion ne peuvent être négatifs")
        if self.diameter <= 0 or self.total_length <= 0:
            raise ValueError("Les dimensions géométriques doivent être positives")
        if not 0 <= self.wing_position <= 1:
            raise ValueError("La position des ailes doit être entre 0 et 1")

    def _precompute_values(self):
        """Pré-calcule les valeurs utilisées fréquemment"""
        # Conversions en pieds (calculées une seule fois)
        self.nose_length_ft = self.nose_length * PhysicalConstants.M_TO_FT
        self.diameter_ft = self.diameter * PhysicalConstants.M_TO_FT
        self.total_length_ft = self.total_length * PhysicalConstants.M_TO_FT
        self.nozzle_diameter_ft = self.nozzle_diameter * PhysicalConstants.M_TO_FT

        # Surfaces de référence
        self.nozzle_area_ft2 = math.pi * (self.nozzle_diameter_ft / 2) ** 2 * 144
        self.reference_area_ft2 = math.pi * (self.diameter_ft / 2) ** 2 * 144
        self.cross_section_area = math.pi * (self.diameter / 2) ** 2


@dataclass
class FlightParameters:
    """Paramètres de vol pour une simulation"""
    launch_angle: float
    ascent_alpha: float
    descent_alpha: float


@dataclass
class SimulationState:
    """État instantané de la simulation"""
    time: float
    altitude: float
    horizontal_position: float
    horizontal_velocity: float
    vertical_velocity: float
    mass: float
    angle_of_attack: float
    mach_number: float
    drag_coefficient: float
    thrust: float
    drag_force: float


class SimulationResult(NamedTuple):
    """Résultat d'une simulation complète"""
    range_m: float
    score: float
    target_reached: bool
    flight_time: float
    states: List[SimulationState]


# ============================================================================
# MODÈLES PHYSIQUES
# ============================================================================

class AtmosphereModel:
    """Modèle atmosphérique optimisé avec cache"""

    # Définition des couches atmosphériques
    LAYERS = [
        AtmosphereLayer(11000, -0.0065, 288.15, 101325, 5.25588),
        AtmosphereLayer(20000, 0.0, 216.65, 22632, 0.000157),
        AtmosphereLayer(32000, 0.001, 216.65, 5474.87, -34.1632),
        AtmosphereLayer(47000, 0.0028, 228.65, 868.014, -12.2011),
        AtmosphereLayer(52000, 0.0, 270.65, 110.906, 0.000157),
        AtmosphereLayer(61000, -0.0028, 270.65, 66.9389, -12.2011),
        AtmosphereLayer(79000, -0.002, 252.65, 3.95642, -12.2011)
    ]

    @staticmethod
    @lru_cache(maxsize=10000)
    def get_atmospheric_properties(altitude: float) -> Tuple[float, float, float]:
        """
        Calcule les propriétés atmosphériques avec mise en cache

        Amélioration: Cache LRU pour éviter les recalculs répétitifs
        """
        altitude = max(0, min(altitude, 79000))  # Limite les valeurs

        if altitude <= 11000:
            layer = AtmosphereModel.LAYERS[0]
            temp = layer.temp_base + layer.temp_gradient * altitude
            pressure = layer.pressure_base * (
                        1 + layer.temp_gradient * altitude / layer.temp_base) ** layer.pressure_exp
        elif altitude <= 20000:
            layer = AtmosphereModel.LAYERS[1]
            temp = layer.temp_base
            pressure = layer.pressure_base * math.exp(-layer.pressure_exp * (altitude - 11000))
        elif altitude <= 32000:
            layer = AtmosphereModel.LAYERS[2]
            temp = layer.temp_base + layer.temp_gradient * (altitude - 20000)
            pressure = layer.pressure_base * (
                        1 + layer.temp_gradient * (altitude - 20000) / layer.temp_base) ** layer.pressure_exp
        elif altitude <= 47000:
            layer = AtmosphereModel.LAYERS[3]
            temp = layer.temp_base + layer.temp_gradient * (altitude - 32000)
            pressure = layer.pressure_base * (
                        1 + layer.temp_gradient * (altitude - 32000) / layer.temp_base) ** layer.pressure_exp
        elif altitude <= 52000:
            layer = AtmosphereModel.LAYERS[4]
            temp = layer.temp_base
            pressure = layer.pressure_base * math.exp(-layer.pressure_exp * (altitude - 47000))
        elif altitude <= 61000:
            layer = AtmosphereModel.LAYERS[5]
            temp = 270.65 - 0.0028 * (altitude - 52000)
            pressure = layer.pressure_base * (temp / 270.65) ** layer.pressure_exp
        else:
            layer = AtmosphereModel.LAYERS[6]
            temp = 252.65 - 0.002 * (altitude - 61000)
            pressure = layer.pressure_base * (temp / 214.65) ** layer.pressure_exp

        density = pressure / (PhysicalConstants.GAS_CONSTANT * temp)
        return temp, pressure, density


class AerodynamicsModel:
    """Modèle aérodynamique optimisé"""

    @staticmethod
    def calculate_drag_coefficients(config: MissileConfiguration, mach: float,
                                    dynamic_pressure: float, alpha_deg: float) -> Tuple[float, float]:
        """
        Calcule les coefficients de traînée

        Amélioration: Interface simplifiée, calculs optimisés
        """
        # Traînée d'onde (supersonique)
        if mach > 1:
            wave_drag = (1.59 + 1.83 / mach ** 2) * (
                math.atan(0.5 / (config.nose_length_ft / config.diameter_ft))) ** 1.69
            base_drag = 0.25 / mach
        else:
            wave_drag = 0
            base_drag = 0.12 + 0.13 * mach ** 2

        # Traînée de frottement
        friction_drag = (PhysicalConstants.FRICTION_COEFFICIENT *
                         (config.total_length_ft / config.diameter_ft) *
                         (mach / (dynamic_pressure * config.total_length_ft)) ** 0.2)

        # Traînée des surfaces de contrôle avec modèle de décrochage amélioré
        wing_drag = AerodynamicsModel._calculate_wing_drag(config, alpha_deg, mach)

        total_drag = wave_drag + base_drag + friction_drag + wing_drag

        return total_drag, wing_drag

    @staticmethod
    def _calculate_wing_drag(config: MissileConfiguration, alpha_deg: float, mach: float) -> float:
        """
        Calcule la traînée des surfaces de contrôle avec modèle de décrochage

        Amélioration: Modèle de décrochage plus réaliste et progressif
        """
        alpha_abs = abs(alpha_deg)
        base_drag = 0.1 * (config.wing_surface / config.cross_section_area) * (math.sin(math.radians(alpha_abs)) ** 2)

        if mach > 1:
            base_drag += 0.05 * (config.wing_surface / config.cross_section_area)

        # Modèle de décrochage progressif amélioré
        if alpha_abs > PhysicalConstants.STALL_ANGLE_DEG:
            # Fonction sigmoïde pour une transition plus réaliste
            stall_factor = 1 + 2 * (1 / (1 + math.exp(-0.3 * (alpha_abs - 25))))
            return base_drag * stall_factor

        return base_drag

    @staticmethod
    def calculate_normal_force_coefficient(config: MissileConfiguration, alpha_deg: float) -> float:
        """
        Calcule le coefficient de force normale

        Amélioration: Modèle plus précis avec gestion du décrochage
        """
        alpha_rad = math.radians(alpha_deg)
        alpha_abs = abs(alpha_deg)

        # Contribution du corps
        phi = 0  # Simplifié pour cette version
        body_factor = abs((config.major_axis / config.minor_axis) * math.cos(phi) +
                          (config.minor_axis / config.major_axis) * math.sin(phi))

        cn_body = (body_factor *
                   (abs(math.sin(2 * alpha_rad) * math.cos(alpha_rad / 2)) +
                    2 * (config.total_length / config.diameter) * math.sin(alpha_rad) ** 2))

        if alpha_deg < 0:
            cn_body = -cn_body

        # Contribution des surfaces de contrôle avec modèle de décrochage
        cn_wing = 2 * math.pi * abs(alpha_rad) * (config.wing_surface / config.cross_section_area)

        # Réduction progressive de la portance après décrochage
        if alpha_abs > PhysicalConstants.STALL_ANGLE_DEG:
            stall_reduction = 1.0 / (1.0 + math.exp(5.0 * (alpha_abs - 28.0) / 8.0))
            cn_wing *= stall_reduction

        if alpha_deg < 0:
            cn_wing = -cn_wing

        # Limitation physique
        cn_wing = max(-1.5, min(1.5, cn_wing))

        # Coefficient total avec effet de position des ailerons
        cn_total = cn_body + cn_wing * config.wing_position

        return cn_total


# ============================================================================
# SIMULATEUR PRINCIPAL
# ============================================================================

class MissileSimulator:
    """
    Simulateur de trajectoire de missile optimisé

    Amélioration: Architecture modulaire avec séparation des responsabilités
    """

    def __init__(self, config: MissileConfiguration, time_step: float = 0.1):
        self.config = config
        self.time_step = time_step
        self.atmosphere = AtmosphereModel()
        self.aerodynamics = AerodynamicsModel()
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Configure le système de logging"""
        logger = logging.getLogger('MissileSimulator')
        logger.setLevel(logging.INFO)
        return logger

    def simulate_trajectory(self, flight_params: FlightParameters,
                            generate_plots: bool = False) -> SimulationResult:
        """
        Simule la trajectoire complète du missile

        Amélioration: Interface claire, séparation simulation/visualisation
        """
        try:
            states = self._run_simulation(flight_params)
            result = self._analyze_results(states, flight_params)

            if generate_plots:
                self._generate_comprehensive_plots(states, flight_params)

            return result

        except Exception as e:
            self.logger.error(f"Erreur lors de la simulation: {e}")
            raise

    def _run_simulation(self, flight_params: FlightParameters) -> List[SimulationState]:
        """
        Exécute la simulation numérique

        Amélioration: Logique de simulation séparée et optimisée
        """
        states = []

        # Conditions initiales
        time = 0.0
        altitude = self.config.initial_altitude
        horizontal_position = 0.0
        horizontal_velocity = self.config.initial_velocity * math.cos(math.radians(flight_params.launch_angle))
        vertical_velocity = self.config.initial_velocity * math.sin(math.radians(flight_params.launch_angle))

        alpha = math.radians(flight_params.ascent_alpha)
        descent_phase = False
        launched = False
        max_altitude_reached = altitude

        while time < self.config.battery_life:
            # Vérification des conditions d'arrêt
            if self._should_stop_simulation(horizontal_velocity, altitude, launched):
                break

            # Mise à jour des paramètres de vol
            if horizontal_velocity > 100:
                launched = True

            if launched and altitude < max_altitude_reached and not descent_phase:
                descent_phase = True
                alpha = math.radians(flight_params.descent_alpha)

            max_altitude_reached = max(max_altitude_reached, altitude)

            # Calcul de la masse actuelle
            mass = self._calculate_current_mass(time)

            # Calcul de la poussée
            thrust = self._calculate_thrust(time)

            # Propriétés atmosphériques
            temp, pressure, density = self.atmosphere.get_atmospheric_properties(altitude)

            # Calculs aérodynamiques
            velocity_magnitude = math.sqrt(horizontal_velocity ** 2 + vertical_velocity ** 2)
            mach_number = velocity_magnitude / math.sqrt(
                PhysicalConstants.GAMMA * PhysicalConstants.GAS_CONSTANT * temp)
            dynamic_pressure = (0.5 * density * velocity_magnitude ** 2) / 47.88

            # Coefficients aérodynamiques
            drag_coeff, wing_drag_coeff = self.aerodynamics.calculate_drag_coefficients(
                self.config, mach_number, dynamic_pressure, math.degrees(alpha))

            normal_force_coeff = self.aerodynamics.calculate_normal_force_coefficient(
                self.config, math.degrees(alpha))

            # Forces
            drag_force = 0.5 * drag_coeff * density * self.config.cross_section_area * velocity_magnitude ** 2

            # Composantes de la poussée
            thrust_horizontal = thrust * math.cos(math.radians(flight_params.launch_angle))
            thrust_vertical = thrust * math.sin(math.radians(flight_params.launch_angle))

            # Composantes de la traînée
            drag_horizontal = drag_force * math.cos(alpha)
            drag_vertical = drag_force * math.sin(alpha)

            # Force normale (portance)
            lift_drag_ratio = ((normal_force_coeff * math.cos(alpha) - drag_coeff * math.sin(alpha)) /
                               (normal_force_coeff * math.sin(alpha) + drag_coeff * math.cos(alpha)))

            normal_force = lift_drag_ratio * drag_force
            normal_horizontal = normal_force * math.sin(alpha)
            normal_vertical = normal_force * math.cos(alpha)

            # Accélérations
            accel_horizontal = (thrust_horizontal - drag_horizontal + normal_horizontal) / mass
            accel_vertical = (thrust_vertical - drag_vertical + normal_vertical) / mass - PhysicalConstants.GRAVITY

            # Intégration numérique (Euler amélioré)
            horizontal_velocity += accel_horizontal * self.time_step
            vertical_velocity += accel_vertical * self.time_step

            altitude += vertical_velocity * self.time_step
            horizontal_position += horizontal_velocity * self.time_step

            # Enregistrement de l'état
            state = SimulationState(
                time=time,
                altitude=altitude,
                horizontal_position=horizontal_position,
                horizontal_velocity=horizontal_velocity,
                vertical_velocity=vertical_velocity,
                mass=mass,
                angle_of_attack=math.degrees(alpha),
                mach_number=mach_number,
                drag_coefficient=drag_coeff,
                thrust=thrust,
                drag_force=drag_force
            )
            states.append(state)

            time += self.time_step

        return states

    def _should_stop_simulation(self, horizontal_velocity: float, altitude: float, launched: bool) -> bool:
        """Détermine si la simulation doit s'arrêter"""
        if launched and altitude <= self.config.target_altitude:
            return True
        if launched and horizontal_velocity <= 100:
            return True
        return False

    def _calculate_current_mass(self, time: float) -> float:
        """Calcule la masse actuelle en fonction du temps"""
        if time <= self.config.booster_time:
            return (self.config.initial_mass -
                    (self.config.initial_mass - self.config.mass_after_booster) *
                    time / self.config.booster_time)
        elif time <= self.config.booster_time + self.config.sustainer_time:
            return (self.config.mass_after_booster -
                    (self.config.mass_after_booster - self.config.mass_after_sustainer) *
                    (time - self.config.booster_time) / self.config.sustainer_time)
        else:
            return self.config.mass_after_sustainer

    def _calculate_thrust(self, time: float) -> float:
        """Calcule la poussée actuelle"""
        if time <= self.config.booster_time:
            return self.config.booster_thrust
        elif time <= self.config.booster_time + self.config.sustainer_time:
            return self.config.sustainer_thrust
        else:
            return 0.0

    def _analyze_results(self, states: List[SimulationState], flight_params: FlightParameters) -> SimulationResult:
        """
        Analyse les résultats de simulation

        Amélioration: Métriques de performance améliorées
        """
        if not states:
            return SimulationResult(0, 0, False, 0, [])

        final_state = states[-1]
        flight_time = final_state.time
        target_reached = final_state.altitude <= self.config.target_altitude

        # Calcul du score avec pénalités
        battery_margin = abs(flight_time - self.config.battery_life)
        score = final_state.horizontal_position - battery_margin * 100 if target_reached else 0

        return SimulationResult(
            range_m=final_state.horizontal_position,
            score=score,
            target_reached=target_reached,
            flight_time=flight_time,
            states=states
        )

    def _generate_comprehensive_plots(self, states: List[SimulationState], flight_params: FlightParameters):
        """
        Génère des graphiques complets de la simulation

        Amélioration: Visualisations optimisées et informatives
        """
        if not states:
            return

        times = [s.time for s in states]
        altitudes = [s.altitude for s in states]
        positions = [s.horizontal_position for s in states]
        h_velocities = [s.horizontal_velocity for s in states]
        v_velocities = [s.vertical_velocity for s in states]
        masses = [s.mass for s in states]
        machs = [s.mach_number for s in states]
        drags = [s.drag_coefficient for s in states]
        thrusts = [s.thrust for s in states]
        alphas = [s.angle_of_attack for s in states]

        # Configuration des graphiques
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

        # Graphique principal : Trajectoire
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Analyse Complète - Angle: {flight_params.launch_angle:.1f}°, '
                     f'α_asc: {flight_params.ascent_alpha:.1f}°, α_desc: {flight_params.descent_alpha:.1f}°',
                     fontsize=16, fontweight='bold')

        # Trajectoire
        axes[0, 0].plot(positions, altitudes, 'b-', linewidth=2)
        axes[0, 0].axhline(y=self.config.target_altitude, color='r', linestyle='--',
                           label=f'Cible: {self.config.target_altitude}m')
        axes[0, 0].set_xlabel('Distance horizontale (m)')
        axes[0, 0].set_ylabel('Altitude (m)')
        axes[0, 0].set_title('Trajectoire du missile')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Vitesses
        axes[0, 1].plot(times, h_velocities, 'g-', label='Horizontale', linewidth=2)
        axes[0, 1].plot(times, v_velocities, 'r-', label='Verticale', linewidth=2)
        axes[0, 1].set_xlabel('Temps (s)')
        axes[0, 1].set_ylabel('Vitesse (m/s)')
        axes[0, 1].set_title('Évolution des vitesses')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # Altitude vs temps
        axes[0, 2].plot(times, altitudes, 'b-', linewidth=2)
        axes[0, 2].axhline(y=self.config.target_altitude, color='r', linestyle='--')
        axes[0, 2].set_xlabel('Temps (s)')
        axes[0, 2].set_ylabel('Altitude (m)')
        axes[0, 2].set_title('Profil d\'altitude')
        axes[0, 2].grid(True, alpha=0.3)

        # Masse
        axes[1, 0].plot(times, masses, 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Temps (s)')
        axes[1, 0].set_ylabel('Masse (kg)')
        axes[1, 0].set_title('Évolution de la masse')
        axes[1, 0].grid(True, alpha=0.3)

        # Nombre de Mach
        axes[1, 1].plot(times, machs, 'orange', linewidth=2)
        axes[1, 1].axhline(y=1, color='r', linestyle='--', label='Mach 1')
        axes[1, 1].set_xlabel('Temps (s)')
        axes[1, 1].set_ylabel('Nombre de Mach')
        axes[1, 1].set_title('Évolution du nombre de Mach')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        # Coefficient de traînée vs Mach
        axes[1, 2].plot(machs, drags, 'brown', linewidth=2)
        axes[1, 2].set_xlabel('Nombre de Mach')
        axes[1, 2].set_ylabel('Coefficient de traînée')
        axes[1, 2].set_title('Cx = f(Mach)')
        axes[1, 2].grid(True, alpha=0.3)

        # Poussée
        axes[2, 0].plot(times, thrusts, 'red', linewidth=2)
        axes[2, 0].set_xlabel('Temps (s)')
        axes[2, 0].set_ylabel('Poussée (N)')
        axes[2, 0].set_title('Évolution de la poussée')
        axes[2, 0].grid(True, alpha=0.3)

        # Angle d'attaque
        axes[2, 1].plot(times, alphas, 'cyan', linewidth=2)
        axes[2, 1].axhline(y=PhysicalConstants.STALL_ANGLE_DEG, color='r', linestyle='--',
                           label=f'Décrochage: {PhysicalConstants.STALL_ANGLE_DEG}°')
        axes[2, 1].axhline(y=-PhysicalConstants.STALL_ANGLE_DEG, color='r', linestyle='--')
        axes[2, 1].set_xlabel('Temps (s)')
        axes[2, 1].set_ylabel('Angle d\'attaque (°)')
        axes[2, 1].set_title('Évolution de l\'angle d\'attaque')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend()

        # Portée vs temps
        axes[2, 2].plot(times, positions, 'green', linewidth=2)
        axes[2, 2].set_xlabel('Temps (s)')
        axes[2, 2].set_ylabel('Distance horizontale (m)')
        axes[2, 2].set_title('Évolution de la portée')
        axes[2, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'trajectory_analysis_{int(time.time())}.png', dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"📊 Graphiques sauvegardés: trajectory_analysis_{int(time.time())}.png")


import multiprocessing as mp
from itertools import product
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
# SUITE DE L'OPTIMISEUR DE TRAJECTOIRE
# ============================================================================

class TrajectoryOptimizer:
    """
    Optimiseur de trajectoire avec algorithmes avancés

    Amélioration: Optimisation parallèle avec stratégie multi-niveaux
    """

    def __init__(self, simulator: MissileSimulator, max_workers: int = None):
        self.simulator = simulator
        self.max_workers = max_workers or min(8, (mp.cpu_count() or 1))
        self.logger = logging.getLogger('TrajectoryOptimizer')

    def optimize(self,
                 angle_range: Tuple[float, float] = (-45, 45),
                 ascent_alpha_range: Tuple[float, float] = (-20, 20),
                 descent_alpha_range: Tuple[float, float] = (-36, 36),
                 coarse_step: float = 1.0,
                 fine_step: float = 0.1,
                 verbose: bool = True) -> Tuple[FlightParameters, SimulationResult]:
        """
        Optimise les paramètres de vol

        Amélioration: Algorithme d'optimisation à deux passes avec parallélisation
        """
        start_time = time.time()

        if verbose:
            print("🚀 Démarrage de l'optimisation de trajectoire")
            print("=" * 60)

        # Phase 1: Exploration grossière
        if verbose:
            print("📊 Phase 1: Exploration grossière...")

        coarse_results = self._coarse_search(angle_range, ascent_alpha_range,
                                             descent_alpha_range, coarse_step)

        if not coarse_results:
            if verbose:
                print("❌ Aucune solution valide trouvée en phase grossière")
            return FlightParameters(0, 0, 0), SimulationResult(0, 0, False, 0, [])

        # Sélection des meilleures candidates
        top_candidates = sorted(coarse_results, key=lambda x: x[1].range_m, reverse=True)[:5]

        if verbose:
            print(f"✅ {len(coarse_results)} solutions valides trouvées")
            print(f"📈 Meilleure portée grossière: {top_candidates[0][1].range_m:.1f}m")
            print("\n🔍 Phase 2: Affinage local...")

        # Phase 2: Affinage local
        fine_results = self._fine_search(top_candidates, angle_range, ascent_alpha_range,
                                         descent_alpha_range, fine_step)

        if fine_results:
            best_params, best_result = max(fine_results, key=lambda x: x[1].range_m)
        else:
            if verbose:
                print("⚠️  Pas d'amélioration en phase fine, conservation du meilleur grossier")
            best_params, best_result = top_candidates[0]

        # Résultats finaux
        elapsed = time.time() - start_time
        if verbose:
            print(f"\n🎯 Optimisation terminée en {elapsed:.2f}s")
            print("=" * 60)
            print(f"🏆 RÉSULTATS OPTIMAUX:")
            print(f"   • Angle de tir: {best_params.launch_angle:.2f}°")
            print(f"   • Alpha ascendant: {best_params.ascent_alpha:.2f}°")
            print(f"   • Alpha descendant: {best_params.descent_alpha:.2f}°")
            print(f"   • Portée maximale: {best_result.range_m:.1f}m")
            print(f"   • Temps de vol: {best_result.flight_time:.1f}s")
            print(f"   • Cible atteinte: {'✅' if best_result.target_reached else '❌'}")

        return best_params, best_result

    def _coarse_search(self, angle_range: Tuple[float, float],
                       ascent_range: Tuple[float, float],
                       descent_range: Tuple[float, float],
                       step: float) -> List[Tuple[FlightParameters, SimulationResult]]:
        """
        Amélioration: Recherche grossière parallélisée avec gestion d'erreurs robuste
        """
        # Génération des combinaisons
        angles = np.arange(angle_range[0], angle_range[1] + step, step)
        ascent_alphas = np.arange(ascent_range[0], ascent_range[1] + step, step)
        descent_alphas = np.arange(descent_range[0], descent_range[1] + step, step)

        combinations = list(product(angles, ascent_alphas, descent_alphas))

        # Parallélisation avec gestion d'erreurs
        valid_results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Soumission des tâches
            future_to_params = {
                executor.submit(self._evaluate_combination, combo): combo
                for combo in combinations
            }

            # Collecte des résultats
            for future in as_completed(future_to_params):
                try:
                    result = future.result(timeout=30)  # Timeout de sécurité
                    if result is not None:
                        valid_results.append(result)
                except Exception as e:
                    params = future_to_params[future]
                    self.logger.warning(f"Erreur pour {params}: {e}")

        return valid_results

    def _fine_search(self, candidates: List[Tuple[FlightParameters, SimulationResult]],
                     angle_range: Tuple[float, float],
                     ascent_range: Tuple[float, float],
                     descent_range: Tuple[float, float],
                     step: float) -> List[Tuple[FlightParameters, SimulationResult]]:
        """
        Amélioration: Recherche fine avec voisinage adaptatif
        """
        all_fine_combinations = []

        # Génération des voisinages pour chaque candidat
        for params, _ in candidates:
            # Définition du voisinage local (±2 unités grossières)
            local_angles = np.arange(
                max(angle_range[0], params.launch_angle - 2),
                min(angle_range[1], params.launch_angle + 2) + step,
                step
            )
            local_ascent = np.arange(
                max(ascent_range[0], params.ascent_alpha - 2),
                min(ascent_range[1], params.ascent_alpha + 2) + step,
                step
            )
            local_descent = np.arange(
                max(descent_range[0], params.descent_alpha - 2),
                min(descent_range[1], params.descent_alpha + 2) + step,
                step
            )

            all_fine_combinations.extend(list(product(local_angles, local_ascent, local_descent)))

        # Suppression des doublons
        unique_combinations = list(set(all_fine_combinations))

        # Évaluation parallèle
        valid_results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_params = {
                executor.submit(self._evaluate_combination, combo): combo
                for combo in unique_combinations
            }

            for future in as_completed(future_to_params):
                try:
                    result = future.result(timeout=30)
                    if result is not None:
                        valid_results.append(result)
                except Exception as e:
                    params = future_to_params[future]
                    self.logger.warning(f"Erreur fine pour {params}: {e}")

        return valid_results

    def _evaluate_combination(self, combination: Tuple[float, float, float]) -> Optional[
        Tuple[FlightParameters, SimulationResult]]:
        """
        Amélioration: Évaluation robuste avec gestion d'erreurs et validation
        """
        try:
            angle, ascent_alpha, descent_alpha = combination

            # Validation des paramètres
            if not self._is_valid_combination(angle, ascent_alpha, descent_alpha):
                return None

            params = FlightParameters(angle, ascent_alpha, descent_alpha)
            result = self.simulator.simulate_trajectory(params, generate_plots=False)

            # Validation des résultats
            if (result.target_reached and
                    result.range_m > 0 and
                    result.flight_time <= self.simulator.config.battery_life and
                    result.flight_time > 0):
                return (params, result)

            return None

        except Exception as e:
            self.logger.debug(f"Erreur évaluation {combination}: {e}")
            return None

    def _is_valid_combination(self, angle: float, ascent_alpha: float, descent_alpha: float) -> bool:
        """
        Amélioration: Validation physique des paramètres
        """
        # Contraintes physiques
        if abs(angle) > 89:  # Éviter les tirs verticaux
            return False
        if abs(ascent_alpha) > 45 or abs(descent_alpha) > 45:  # Angles d'attaque raisonnables
            return False
        if ascent_alpha < descent_alpha - 10:  # Cohérence ascendant/descendant
            return False

        return True


# ============================================================================
# GESTIONNAIRE DE CAMPAGNES D'OPTIMISATION
# ============================================================================

class OptimizationCampaign:
    """
    Gestionnaire de campagnes optimisé pour analyses paramétriques
    Amélioration: Gestion mémoire et performances optimisées
    """

    def __init__(self, base_config: MissileConfiguration):
        self.base_config = base_config
        self.results_history = []
        self.logger = logging.getLogger('OptimizationCampaign')

    def run_parametric_study(self, parameter_variations: dict,
                             verbose: bool = True,
                             max_workers: int = 2,  # Réduit le parallélisme
                             coarse_step: float = 2.0,  # Pas plus large
                             fine_step: float = 0.5) -> dict:  # Pas plus large
        """
        Étude paramétrique optimisée avec gestion mémoire

        Améliorations:
        - Parallélisme réduit pour économiser la RAM
        - Pas d'optimisation plus larges (moins de combinaisons)
        - Nettoyage mémoire entre les paramètres
        - Simulation sans stockage des états détaillés
        """
        results = {}

        if verbose:
            print("🔬 Lancement de l'étude paramétrique optimisée")
            print("=" * 50)
            print(f"⚙️  Paramètres d'optimisation:")
            print(f"   • Workers max: {max_workers}")
            print(f"   • Pas grossier: {coarse_step}°")
            print(f"   • Pas fin: {fine_step}°")

        total_combinations = sum(len(values) for values in parameter_variations.values())
        current_combination = 0

        for param_name, values in parameter_variations.items():
            if verbose:
                print(f"\n📊 Analyse du paramètre: {param_name}")
                print(f"   Valeurs à tester: {values}")

            param_results = []

            for i, value in enumerate(values):
                current_combination += 1

                if verbose:
                    progress = (current_combination / total_combinations) * 100
                    print(f"   [{current_combination}/{total_combinations}] "
                          f"({progress:.1f}%) {param_name}={value}... ", end="", flush=True)

                try:
                    # Création d'une configuration modifiée
                    modified_config = self._modify_config(param_name, value)

                    # Optimisation LÉGÈRE avec moins de workers et pas plus larges
                    simulator = MissileSimulator(modified_config)
                    optimizer = TrajectoryOptimizer(simulator, max_workers=max_workers)

                    # Optimisation avec paramètres réduits
                    best_params, best_result = optimizer.optimize(
                        angle_range=(-45, 45),
                        ascent_alpha_range=(-20, 20),
                        descent_alpha_range=(-36, 36),
                        coarse_step=coarse_step,
                        fine_step=fine_step,
                        verbose=False
                    )

                    # Stockage uniquement des métriques essentielles
                    param_results.append({
                        'value': value,
                        'range': best_result.range_m,
                        'flight_time': best_result.flight_time,
                        'target_reached': best_result.target_reached,
                        'optimal_angle': best_params.launch_angle,
                        'optimal_ascent_alpha': best_params.ascent_alpha,
                        'optimal_descent_alpha': best_params.descent_alpha,
                        'score': best_result.score
                    })

                    if verbose:
                        status = "✅" if best_result.target_reached else "❌"
                        print(f"{best_result.range_m:.0f}m {status}")

                except Exception as e:
                    self.logger.error(f"Erreur pour {param_name}={value}: {e}")
                    param_results.append({
                        'value': value,
                        'range': 0,
                        'flight_time': 0,
                        'target_reached': False,
                        'optimal_angle': 0,
                        'optimal_ascent_alpha': 0,
                        'optimal_descent_alpha': 0,
                        'score': 0
                    })

                    if verbose:
                        print("❌ Erreur")

                # Nettoyage mémoire explicite après chaque simulation
                self._cleanup_memory()

            results[param_name] = param_results

            # Affichage du résumé pour ce paramètre
            if verbose:
                successful_results = [r for r in param_results if r['target_reached']]
                if successful_results:
                    best_range = max(successful_results, key=lambda x: x['range'])
                    print(f"   🏆 Meilleur résultat: {best_range['range']:.0f}m "
                          f"(valeur={best_range['value']})")
                else:
                    print("   ⚠️  Aucun résultat valide pour ce paramètre")

                # Nettoyage mémoire entre paramètres
                self._cleanup_memory()

        if verbose:
            print(f"\n✅ Étude paramétrique terminée!")
            self._print_summary(results)

        return results

    def _cleanup_memory(self):
        """Nettoyage explicite de la mémoire"""
        import gc
        gc.collect()  # Force le garbage collector

    def _print_summary(self, results: dict):
        """Affiche un résumé des résultats"""
        print("\n📈 RÉSUMÉ DE L'ÉTUDE PARAMÉTRIQUE:")
        print("=" * 50)

        for param_name, param_results in results.items():
            successful = [r for r in param_results if r['target_reached']]
            total = len(param_results)
            success_rate = len(successful) / total * 100 if total > 0 else 0

            print(f"\n🔍 {param_name}:")
            print(f"   • Taux de succès: {success_rate:.1f}% ({len(successful)}/{total})")

            if successful:
                ranges = [r['range'] for r in successful]
                best_result = max(successful, key=lambda x: x['range'])
                print(f"   • Portée min/max: {min(ranges):.0f}m - {max(ranges):.0f}m")
                print(f"   • Meilleure config: {param_name}={best_result['value']} → {best_result['range']:.0f}m")

    def _modify_config(self, param_name: str, value: float) -> MissileConfiguration:
        """
        Modification dynamique de configuration (inchangée)
        """
        # Création d'une copie de la configuration de base
        config_dict = {
            'initial_mass': self.base_config.initial_mass,
            'mass_after_booster': self.base_config.mass_after_booster,
            'mass_after_sustainer': self.base_config.mass_after_sustainer,
            'booster_time': self.base_config.booster_time,
            'sustainer_time': self.base_config.sustainer_time,
            'battery_life': self.base_config.battery_life,
            'booster_thrust': self.base_config.booster_thrust,
            'sustainer_thrust': self.base_config.sustainer_thrust,
            'nose_length': self.base_config.nose_length,
            'diameter': self.base_config.diameter,
            'total_length': self.base_config.total_length,
            'nozzle_diameter': self.base_config.nozzle_diameter,
            'major_axis': self.base_config.major_axis,
            'minor_axis': self.base_config.minor_axis,
            'initial_velocity': self.base_config.initial_velocity,
            'initial_altitude': self.base_config.initial_altitude,
            'target_altitude': self.base_config.target_altitude,
            'target_horizontal_velocity': self.base_config.target_horizontal_velocity,
            'wing_surface': self.base_config.wing_surface,
            'wing_length': self.base_config.wing_length,
            'wing_position': self.base_config.wing_position
        }

        # Modification du paramètre spécifié
        if param_name in config_dict:
            config_dict[param_name] = value
        else:
            raise ValueError(f"Paramètre inconnu: {param_name}")

        return MissileConfiguration(**config_dict)

    def generate_parametric_plots(self, results: dict):
        """
        Visualisation optimisée des études paramétriques
        """
        n_params = len(results)
        if n_params == 0:
            print("⚠️  Aucun résultat à visualiser")
            return

        try:
            # Configuration de base pour éviter les problèmes mémoire
            plt.style.use('default')  # Style plus simple

            fig, axes = plt.subplots(2, n_params, figsize=(5 * n_params, 8))  # Plus petit
            if n_params == 1:
                axes = axes.reshape(-1, 1)

            for i, (param_name, param_results) in enumerate(results.items()):
                values = [r['value'] for r in param_results]
                ranges = [r['range'] for r in param_results]
                success = [r['target_reached'] for r in param_results]

                # Graphique de portée (uniquement les succès)
                successful_indices = [j for j, s in enumerate(success) if s]
                if successful_indices:
                    success_values = [values[j] for j in successful_indices]
                    success_ranges = [ranges[j] for j in successful_indices]

                    axes[0, i].plot(success_values, success_ranges, 'bo-', linewidth=2, markersize=4)
                    axes[0, i].set_xlabel(param_name)
                    axes[0, i].set_ylabel('Portée (m)')
                    axes[0, i].set_title(f'Portée vs {param_name}')
                    axes[0, i].grid(True, alpha=0.3)
                else:
                    axes[0, i].text(0.5, 0.5, 'Aucun\nsuccès', ha='center', va='center',
                                    transform=axes[0, i].transAxes, fontsize=12)
                    axes[0, i].set_xlabel(param_name)
                    axes[0, i].set_ylabel('Portée (m)')
                    axes[0, i].set_title(f'Portée vs {param_name}')

                # Graphique de taux de succès
                success_rate = [1 if s else 0 for s in success]
                colors = ['green' if s else 'red' for s in success]
                axes[1, i].bar(values, success_rate, alpha=0.7, color=colors)
                axes[1, i].set_xlabel(param_name)
                axes[1, i].set_ylabel('Cible atteinte')
                axes[1, i].set_title(f'Succès vs {param_name}')
                axes[1, i].set_ylim(0, 1.1)
                axes[1, i].grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f'parametric_study_{int(time.time())}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')  # DPI réduit
            plt.close('all')  # Fermeture explicite

            print(f"📊 Étude paramétrique sauvegardée: {filename}")

        except Exception as e:
            print(f"❌ Erreur lors de la génération des graphiques: {e}")
            plt.close('all')  # Nettoyage en cas d'erreur


# ============================================================================
# EXEMPLE D'UTILISATION AVANCÉE
# ============================================================================

def main_advanced_example():
    """
    Amélioration: Exemple complet avec toutes les fonctionnalités
    """

    # Configuration du missile MICA optimisée
    mica_config = MissileConfiguration(
        # Masses (kg)
        initial_mass=112.0,
        mass_after_booster=88.3,
        mass_after_sustainer=70.1,

        # Temps (s)
        booster_time=2.75,
        sustainer_time=4.0,
        battery_life=70.0,

        # Forces (N)
        booster_thrust=20250.0,
        sustainer_thrust=10720.0,

        # Géométrie (m)
        nose_length=0.40,
        diameter=0.16,
        total_length=3.1,
        nozzle_diameter=0.12,

        # Paramètres elliptiques
        major_axis=0.08,
        minor_axis=0.08,

        # Conditions de vol
        initial_velocity=300.0,
        initial_altitude=5000.0,
        target_altitude=2000.0,
        target_horizontal_velocity=300.0,

        # Surfaces de contrôle
        wing_surface=0.02,
        wing_length=0.5,
        wing_position=0.7
    )

    print("🎯 SIMULATEUR DE MISSILE OPTIMISÉ")
    print("=" * 50)
    print(f"Configuration: MICA")
    print(f"Masse initiale: {mica_config.initial_mass} kg")
    print(f"Altitude de lancement: {mica_config.initial_altitude} m")
    print(f"Altitude cible: {mica_config.target_altitude} m")

    # 1. Optimisation standard
    print("\n🚀 1. OPTIMISATION STANDARD")
    simulator = MissileSimulator(mica_config)
    optimizer = TrajectoryOptimizer(simulator)

    best_params, best_result = optimizer.optimize(
        angle_range=(-30, 30),
        ascent_alpha_range=(-15, 15),
        descent_alpha_range=(-30, 30),
        coarse_step=2.0,
        fine_step=0.2
    )

    # Génération des graphiques détaillés
    print("\n📊 Génération des graphiques détaillés...")
    detailed_result = simulator.simulate_trajectory(best_params, generate_plots=True)

    # 2. Étude paramétrique
    print("\n🔬 2. ÉTUDE PARAMÉTRIQUE")
    campaign = OptimizationCampaign(mica_config)

    # Variations à étudier
    variations = {
        'initial_velocity': [200, 250, 300, 350, 400],
        'initial_altitude': [2000, 4000, 6000, 8000, 10000],
        'target_altitude': [2000, 4000, 6000, 8000, 10000],
        'target_horizontal_velocity': [200, 250, 300, 350, 400],
    }

    parametric_results = campaign.run_parametric_study(variations)
    campaign.generate_parametric_plots(parametric_results)

    # 3. Analyse de sensibilité
    print("\n📈 3. ANALYSE DE SENSIBILITÉ")
    sensitivity_analysis(mica_config, best_params)

    print("\n✅ Analyse complète terminée!")


def sensitivity_analysis(config: MissileConfiguration, nominal_params: FlightParameters):
    """
    Amélioration: Analyse de sensibilité aux paramètres de vol
    """
    simulator = MissileSimulator(config)
    nominal_result = simulator.simulate_trajectory(nominal_params, generate_plots=False)

    print(f"Résultat nominal: {nominal_result.range_m:.1f}m")

    # Test de sensibilité
    sensitivities = {}

    # Sensibilité à l'angle de tir
    for delta in [-1, -0.5, 0.5, 1]:
        test_params = FlightParameters(
            nominal_params.launch_angle + delta,
            nominal_params.ascent_alpha,
            nominal_params.descent_alpha
        )
        test_result = simulator.simulate_trajectory(test_params, generate_plots=False)
        if test_result.target_reached:
            sensitivity = (test_result.range_m - nominal_result.range_m) / delta
            sensitivities[f'angle_tir_{delta:+.1f}'] = sensitivity

    # Sensibilité aux alphas
    for delta in [-2, -1, 1, 2]:
        test_params = FlightParameters(
            nominal_params.launch_angle,
            nominal_params.ascent_alpha + delta,
            nominal_params.descent_alpha
        )
        test_result = simulator.simulate_trajectory(test_params, generate_plots=False)
        if test_result.target_reached:
            sensitivity = (test_result.range_m - nominal_result.range_m) / delta
            sensitivities[f'alpha_asc_{delta:+.1f}'] = sensitivity

    # Affichage des sensibilités
    print("\nSensibilités (m/degré):")
    for param, sens in sensitivities.items():
        print(f"  {param}: {sens:.2f}")


# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    try:
        main_advanced_example()
    except KeyboardInterrupt:
        print("\n⚠️  Interruption utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur critique: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Nettoyage complet des ressources matplotlib
        try:
            plt.close('all')
            plt.clf()
            plt.cla()
        except:
            pass