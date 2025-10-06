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
import random
from typing import List, Tuple
import multiprocessing as mp
from itertools import product

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

# ============================================================================
# CONFIGURATIONS DE MISSILES PRÉDÉFINIS
# ============================================================================

class MissileDatabase:
    """Base de données des missiles avec leurs caractéristiques"""

    @staticmethod
    def get_mica_config() -> MissileConfiguration:
        """Configuration du missile MICA"""
        return MissileConfiguration(
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

    @staticmethod
    def get_aim120b_config() -> MissileConfiguration:
        """Configuration du missile AIM-120B AMRAAM"""
        return MissileConfiguration(
            # Masses (kg)
            initial_mass=157.0,
            mass_after_booster=125.0,
            mass_after_sustainer=95.0,
            # Temps (s)
            booster_time=3.2,
            sustainer_time=5.5,
            battery_life=90.0,
            # Forces (N)
            booster_thrust=25000.0,
            sustainer_thrust=12000.0,
            # Géométrie (m)
            nose_length=0.45,
            diameter=0.178,
            total_length=3.66,
            nozzle_diameter=0.13,
            # Paramètres elliptiques
            major_axis=0.089,
            minor_axis=0.089,
            # Conditions de vol
            initial_velocity=320.0,
            initial_altitude=6000.0,
            target_altitude=2000.0,
            target_horizontal_velocity=320.0,
            # Surfaces de contrôle
            wing_surface=0.025,
            wing_length=0.6,
            wing_position=0.75
        )

    @staticmethod
    def get_r77_config() -> MissileConfiguration:
        """Configuration du missile R-77 (AA-12 Adder)"""
        return MissileConfiguration(
            # Masses (kg)
            initial_mass=175.0,
            mass_after_booster=140.0,
            mass_after_sustainer=110.0,
            # Temps (s)
            booster_time=3.5,
            sustainer_time=6.0,
            battery_life=85.0,
            # Forces (N)
            booster_thrust=28000.0,
            sustainer_thrust=13500.0,
            # Géométrie (m)
            nose_length=0.50,
            diameter=0.20,
            total_length=3.60,
            nozzle_diameter=0.14,
            # Paramètres elliptiques
            major_axis=0.10,
            minor_axis=0.10,
            # Conditions de vol
            initial_velocity=310.0,
            initial_altitude=5500.0,
            target_altitude=2000.0,
            target_horizontal_velocity=310.0,
            # Surfaces de contrôle (grilles lattice)
            wing_surface=0.030,
            wing_length=0.4,
            wing_position=0.8
        )

    @staticmethod
    def get_available_missiles() -> dict:
        """Retourne la liste des missiles disponibles"""
        return {
            'MICA': MissileDatabase.get_mica_config,
            'AIM-120B': MissileDatabase.get_aim120b_config,
            'R-77': MissileDatabase.get_r77_config
        }

    @staticmethod
    def get_missile_config(missile_name: str) -> MissileConfiguration:
        """Récupère la configuration d'un missile par son nom"""
        missiles = MissileDatabase.get_available_missiles()
        if missile_name not in missiles:
            available = ', '.join(missiles.keys())
            raise ValueError(f"Missile '{missile_name}' non disponible. Missiles disponibles: {available}")
        return missiles[missile_name]()

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

class AdvancedAerodynamicsModel(AerodynamicsModel):
    """Modèle aérodynamique avancé avec effets compressibles"""

    @staticmethod
    def calculate_compressibility_effects(mach: float) -> float:
        """Calcule les effets de compressibilité"""
        if mach < 0.8:
            return 1.0
        elif mach < 1.2:
            # Zone transsonique
            return 1.0 + 0.5 * ((mach - 0.8) / 0.4) ** 2
        else:
            # Zone supersonique
            return 1.2 + 0.3 * (mach - 1.2) ** 0.5

    @staticmethod
    def calculate_reynolds_correction(altitude: float, velocity: float, length: float) -> float:
        """Correction du nombre de Reynolds"""
        temp, _, density = AtmosphereModel.get_atmospheric_properties(altitude)
        viscosity = 1.458e-6 * (temp ** 1.5) / (temp + 110.4)  # Sutherland
        reynolds = density * velocity * length / viscosity

        if reynolds < 1e6:
            return 1.2  # Correction pour bas Reynolds
        return 1.0

    @classmethod
    def calculate_advanced_drag(cls, config: MissileConfiguration, mach: float,
                              altitude: float, velocity: float, alpha_deg: float) -> float:
        """Calcul de traînée avec effets avancés"""
        base_drag, _ = cls.calculate_drag_coefficients(config, mach, 0.5, alpha_deg)

        # Corrections
        comp_factor = cls.calculate_compressibility_effects(mach)
        reynolds_factor = cls.calculate_reynolds_correction(altitude, velocity, config.total_length)

        return base_drag * comp_factor * reynolds_factor

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

            # Intégration Runge-Kutta 4
            h_vel, v_vel, alt, h_pos = self._runge_kutta_step(
                horizontal_velocity, vertical_velocity, altitude, horizontal_position,
                accel_horizontal, accel_vertical, self.time_step
            )
            horizontal_velocity, vertical_velocity = h_vel, v_vel
            altitude, horizontal_position = alt, h_pos

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

    def _runge_kutta_step(self, vx, vy, alt, x, ax, ay, dt):
        """Intégration Runge-Kutta d'ordre 4"""
        # k1
        k1_vx = ax * dt
        k1_vy = ay * dt
        k1_x = vx * dt
        k1_alt = vy * dt

        # k2
        k2_vx = ax * dt  # Approximation constante pour les accélérations
        k2_vy = ay * dt
        k2_x = (vx + k1_vx/2) * dt
        k2_alt = (vy + k1_vy/2) * dt

        # k3
        k3_vx = ax * dt
        k3_vy = ay * dt
        k3_x = (vx + k2_vx/2) * dt
        k3_alt = (vy + k2_vy/2) * dt

        # k4
        k4_vx = ax * dt
        k4_vy = ay * dt
        k4_x = (vx + k3_vx) * dt
        k4_alt = (vy + k3_vy) * dt

        # Intégration finale
        new_vx = vx + (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx) / 6
        new_vy = vy + (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy) / 6
        new_x = x + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        new_alt = alt + (k1_alt + 2*k2_alt + 2*k3_alt + k4_alt) / 6

        return new_vx, new_vy, new_alt, new_x

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

class GeneticOptimizer:
    """Optimiseur utilisant des algorithmes génétiques"""

    def __init__(self, simulator: MissileSimulator, population_size: int = 50, generations: int = 100):
        self.simulator = simulator
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8

    def optimize(self, bounds: dict) -> Tuple[FlightParameters, SimulationResult]:
        """Optimisation par algorithme génétique"""
        print("🧬 Optimisation par algorithme génétique")

        # Initialisation de la population
        population = self._initialize_population(bounds)
        best_individual = None
        best_fitness = 0

        for generation in range(self.generations):
            # Évaluation de la population
            fitness_scores = []
            for individual in population:
                params = FlightParameters(*individual)
                result = self.simulator.simulate_trajectory(params, generate_plots=False)
                fitness = result.range_m if result.target_reached else 0
                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual

            if generation % 20 == 0:
                print(f"Génération {generation}: Meilleur fitness = {best_fitness:.1f}m")

            # Sélection et reproduction
            population = self._evolve_population(population, fitness_scores, bounds)

        best_params = FlightParameters(*best_individual)
        best_result = self.simulator.simulate_trajectory(best_params, generate_plots=False)

        print(f"🏆 Algorithme génétique terminé: {best_result.range_m:.1f}m")
        return best_params, best_result

    def _initialize_population(self, bounds: dict) -> List[List[float]]:
        """Initialise la population aléatoirement"""
        population = []
        for _ in range(self.population_size):
            individual = []
            for param, (min_val, max_val) in bounds.items():
                individual.append(random.uniform(min_val, max_val))
            population.append(individual)
        return population

    def _evolve_population(self, population: List[List[float]],
                          fitness_scores: List[float], bounds: dict) -> List[List[float]]:
        """Évolution de la population (sélection, croisement, mutation)"""
        new_population = []

        # Élitisme : garder les 10% meilleurs
        elite_count = max(1, self.population_size // 10)
        elite_indices = sorted(range(len(fitness_scores)),
                             key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
        for i in elite_indices:
            new_population.append(population[i][:])

        # Génération du reste par croisement et mutation
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)

            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = self._mutate(child1, bounds)
            child2 = self._mutate(child2, bounds)

            new_population.extend([child1, child2])

        return new_population[:self.population_size]

    def _tournament_selection(self, population: List[List[float]],
                            fitness_scores: List[float], tournament_size: int = 3) -> List[float]:
        """Sélection par tournoi"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_index][:]

    def _crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Croisement uniforme"""
        child1, child2 = parent1[:], parent2[:]
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def _mutate(self, individual: List[float], bounds: dict) -> List[float]:
        """Mutation gaussienne"""
        mutated = individual[:]
        param_names = list(bounds.keys())

        for i, param in enumerate(param_names):
            if random.random() < self.mutation_rate:
                min_val, max_val = bounds[param]
                sigma = (max_val - min_val) * 0.1  # 10% de l'intervalle
                mutation = random.gauss(0, sigma)
                mutated[i] = max(min_val, min(max_val, mutated[i] + mutation))

        return mutated

class RealTimePredictor:
    """Prédicteur de trajectoire en temps réel"""

    def __init__(self, simulator: MissileSimulator):
        self.simulator = simulator
        self.current_state = None
        self.prediction_horizon = 20.0  # secondes

    def predict_trajectory(self, current_params: dict) -> List[SimulationState]:
        """Prédit la trajectoire à partir de l'état actuel"""
        # Création d'un état initial basé sur les paramètres actuels
        time = current_params.get('time', 0)
        altitude = current_params.get('altitude', self.simulator.config.initial_altitude)
        h_pos = current_params.get('horizontal_position', 0)
        h_vel = current_params.get('horizontal_velocity', self.simulator.config.initial_velocity)
        v_vel = current_params.get('vertical_velocity', 0)
        alpha = current_params.get('angle_of_attack', 0)

        # Simulation rapide avec pas de temps réduit
        predicted_states = []
        end_time = time + self.prediction_horizon
        dt = 0.1

        while time < end_time and altitude > 0:
            # Calculs simplifiés pour la vitesse
            mass = self.simulator._calculate_current_mass(time)
            thrust = self.simulator._calculate_thrust(time)

            # Propriétés atmosphériques
            temp, _, density = AtmosphereModel.get_atmospheric_properties(altitude)
            velocity_mag = math.sqrt(h_vel**2 + v_vel**2)

            # Forces simplifiées
            drag_force = 0.5 * 0.3 * density * self.simulator.config.cross_section_area * velocity_mag**2

            # Accélérations
            accel_h = (thrust - drag_force) / mass
            accel_v = -PhysicalConstants.GRAVITY

            # Intégration simple
            h_vel += accel_h * dt
            v_vel += accel_v * dt
            altitude += v_vel * dt
            h_pos += h_vel * dt

            # État prédit
            mach = velocity_mag / math.sqrt(PhysicalConstants.GAMMA * PhysicalConstants.GAS_CONSTANT * temp)

            state = SimulationState(
                time=time, altitude=altitude, horizontal_position=h_pos,
                horizontal_velocity=h_vel, vertical_velocity=v_vel,
                mass=mass, angle_of_attack=alpha, mach_number=mach,
                drag_coefficient=0.3, thrust=thrust, drag_force=drag_force
            )
            predicted_states.append(state)

            time += dt

        return predicted_states

    def update_prediction(self, telemetry_data: dict) -> dict:
        """Met à jour la prédiction avec de nouvelles données de télémétrie"""
        predicted_states = self.predict_trajectory(telemetry_data)

        if predicted_states:
            final_state = predicted_states[-1]
            return {
                'predicted_range': final_state.horizontal_position,
                'predicted_flight_time': final_state.time,
                'impact_altitude': final_state.altitude,
                'trajectory_points': [(s.horizontal_position, s.altitude) for s in predicted_states]
            }

        return {'predicted_range': 0, 'predicted_flight_time': 0, 'impact_altitude': 0, 'trajectory_points': []}

class InteractiveSimulator:
    """Interface utilisateur interactive pour la simulation"""

    def __init__(self):
        self.current_missile = "MICA"
        self.config = MissileDatabase.get_missile_config(self.current_missile)
        self.simulator = MissileSimulator(self.config)
        self.predictor = RealTimePredictor(self.simulator)

    def run_interactive_session(self):
        """Lance une session interactive"""
        print("🎮 SIMULATEUR INTERACTIF DE MISSILE")
        print("=" * 50)

        while True:
            self._show_menu()
            choice = input("\nVotre choix: ").strip()

            if choice == '1':
                self._select_missile()
            elif choice == '2':
                self._quick_simulation()
            elif choice == '3':
                self._parameter_tuning()
            elif choice == '4':
                self._real_time_prediction()
            elif choice == '5':
                self._genetic_optimization()
            elif choice == '6':
                self._monte_carlo_analysis()
            elif choice == '0':
                print("👋 Au revoir!")
                break
            else:
                print("❌ Choix invalide!")

    def _show_menu(self):
        """Affiche le menu principal"""
        print(f"\n📊 Missile actuel: {self.current_missile}")
        print("1. 🚀 Changer de missile")
        print("2. ⚡ Simulation rapide")
        print("3. 🔧 Réglage des paramètres")
        print("4. 📡 Prédiction temps réel")
        print("5. 🧬 Optimisation génétique")
        print("6. 🎲 Analyse Monte Carlo")
        print("0. 🚪 Quitter")

    def _select_missile(self):
        """Sélection du missile"""
        missiles = list(MissileDatabase.get_available_missiles().keys())
        print("\nMissiles disponibles:")
        for i, missile in enumerate(missiles, 1):
            print(f"{i}. {missile}")

        try:
            choice = int(input("Choisissez un missile (numéro): ")) - 1
            if 0 <= choice < len(missiles):
                self.current_missile = missiles[choice]
                self.config = MissileDatabase.get_missile_config(self.current_missile)
                self.simulator = MissileSimulator(self.config)
                self.predictor = RealTimePredictor(self.simulator)
                print(f"✅ Missile sélectionné: {self.current_missile}")
            else:
                print("❌ Choix invalide!")
        except ValueError:
            print("❌ Veuillez entrer un numéro valide!")

    def _quick_simulation(self):
        """Simulation rapide avec paramètres par défaut"""
        print("\n⚡ Simulation rapide en cours...")
        optimizer = TrajectoryOptimizer(self.simulator, max_workers=2)
        best_params, best_result = optimizer.optimize(verbose=False)

        print(f"🎯 Résultats:")
        print(f"   • Portée: {best_result.range_m:.0f}m")
        print(f"   • Temps de vol: {best_result.flight_time:.1f}s")
        print(f"   • Angle optimal: {best_params.launch_angle:.1f}°")
        print(f"   • Cible atteinte: {'✅' if best_result.target_reached else '❌'}")

    def _parameter_tuning(self):
        """Interface de réglage des paramètres"""
        print("\n🔧 RÉGLAGE DES PARAMÈTRES")
        try:
            angle = float(input(f"Angle de tir ({-45}-{45}°): "))
            ascent_alpha = float(input(f"Alpha ascendant ({-20}-{20}°): "))
            descent_alpha = float(input(f"Alpha descendant ({-36}-{36}°): "))

            params = FlightParameters(angle, ascent_alpha, descent_alpha)
            result = self.simulator.simulate_trajectory(params, generate_plots=True)

            print(f"\n📊 Résultats:")
            print(f"   • Portée: {result.range_m:.0f}m")
            print(f"   • Temps de vol: {result.flight_time:.1f}s")
            print(f"   • Cible atteinte: {'✅' if result.target_reached else '❌'}")

        except ValueError:
            print("❌ Veuillez entrer des valeurs numériques valides!")

    def _real_time_prediction(self):
        """Simulation de prédiction temps réel"""
        print("\n📡 PRÉDICTION TEMPS RÉEL")
        print("Simulation de télémétrie en cours...")

        # Simulation de données de télémétrie
        telemetry = {
            'time': 5.0,
            'altitude': 3000,
            'horizontal_position': 1200,
            'horizontal_velocity': 280,
            'vertical_velocity': -50,
            'angle_of_attack': 5
        }

        prediction = self.predictor.update_prediction(telemetry)

        print(f"📈 Prédictions:")
        print(f"   • Portée prédite: {prediction['predicted_range']:.0f}m")
        print(f"   • Temps de vol restant: {prediction['predicted_flight_time']-5:.1f}s")
        print(f"   • Altitude d'impact: {prediction['impact_altitude']:.0f}m")

    def _genetic_optimization(self):
        """Optimisation par algorithme génétique"""
        print("\n🧬 OPTIMISATION GÉNÉTIQUE")

        bounds = {
            'launch_angle': (-30, 30),
            'ascent_alpha': (-15, 15),
            'descent_alpha': (-30, 30)
        }

        genetic_optimizer = GeneticOptimizer(self.simulator, population_size=30, generations=50)
        best_params, best_result = genetic_optimizer.optimize(bounds)

        print(f"\n🏆 Meilleur résultat génétique:")
        print(f"   • Portée: {best_result.range_m:.0f}m")
        print(f"   • Angle: {best_params.launch_angle:.2f}°")
        print(f"   • Alpha asc: {best_params.ascent_alpha:.2f}°")
        print(f"   • Alpha desc: {best_params.descent_alpha:.2f}°")

class MonteCarloAnalyzer:
    """Analyseur Monte Carlo pour l'évaluation de robustesse"""

    def __init__(self, simulator: MissileSimulator):
        self.simulator = simulator

    def analyze_robustness(self, nominal_params: FlightParameters,
                          n_simulations: int = 1000, uncertainty_percent: float = 5.0) -> dict:
        """Analyse de robustesse par Monte Carlo"""
        print(f"🎲 Analyse Monte Carlo ({n_simulations} simulations)")
        print(f"Incertitude: ±{uncertainty_percent}%")

        results = []
        successful_simulations = 0

        for i in range(n_simulations):
            if i % 100 == 0:
                print(f"Progression: {i}/{n_simulations}")

            # Génération de paramètres perturbés
            perturbed_params = self._perturb_parameters(nominal_params, uncertainty_percent)

            try:
                result = self.simulator.simulate_trajectory(perturbed_params, generate_plots=False)
                results.append({
                    'range': result.range_m,
                    'flight_time': result.flight_time,
                    'target_reached': result.target_reached,
                    'params': perturbed_params
                })

                if result.target_reached:
                    successful_simulations += 1

            except Exception:
                # Simulation échouée, considérée comme non réussie
                results.append({
                    'range': 0,
                    'flight_time': 0,
                    'target_reached': False,
                    'params': perturbed_params
                })

        # Analyse statistique
        successful_results = [r for r in results if r['target_reached']]

        if successful_results:
            ranges = [r['range'] for r in successful_results]
            flight_times = [r['flight_time'] for r in successful_results]

            analysis = {
                'success_rate': successful_simulations / n_simulations,
                'mean_range': np.mean(ranges),
                'std_range': np.std(ranges),
                'min_range': np.min(ranges),
                'max_range': np.max(ranges),
                'mean_flight_time': np.mean(flight_times),
                'std_flight_time': np.std(flight_times),
                'percentile_95_range': np.percentile(ranges, 95),
                'percentile_5_range': np.percentile(ranges, 5),
                'all_results': results
            }
        else:
            analysis = {
                'success_rate': 0,
                'mean_range': 0,
                'std_range': 0,
                'min_range': 0,
                'max_range': 0,
                'mean_flight_time': 0,
                'std_flight_time': 0,
                'percentile_95_range': 0,
                'percentile_5_range': 0,
                'all_results': results
            }

        self._print_monte_carlo_results(analysis)
        return analysis

    def _perturb_parameters(self, params: FlightParameters, uncertainty_percent: float) -> FlightParameters:
        """Génère des paramètres perturbés selon une distribution normale"""
        sigma = uncertainty_percent / 100.0

        angle_noise = np.random.normal(0, abs(params.launch_angle) * sigma)
        ascent_noise = np.random.normal(0, abs(params.ascent_alpha) * sigma if params.ascent_alpha != 0 else 1.0 * sigma)
        descent_noise = np.random.normal(0, abs(params.descent_alpha) * sigma if params.descent_alpha != 0 else 1.0 * sigma)

        return FlightParameters(
            params.launch_angle + angle_noise,
            params.ascent_alpha + ascent_noise,
            params.descent_alpha + descent_noise
        )

    def _print_monte_carlo_results(self, analysis: dict):
        """Affiche les résultats de l'analyse Monte Carlo"""
        print(f"\n📊 RÉSULTATS MONTE CARLO:")
        print(f"   • Taux de succès: {analysis['success_rate']*100:.1f}%")

        if analysis['success_rate'] > 0:
            print(f"   • Portée moyenne: {analysis['mean_range']:.0f} ± {analysis['std_range']:.0f}m")
            print(f"   • Portée min/max: {analysis['min_range']:.0f}m - {analysis['max_range']:.0f}m")
            print(f"   • Intervalle 90%: {analysis['percentile_5_range']:.0f}m - {analysis['percentile_95_range']:.0f}m")
            print(f"   • Temps de vol moyen: {analysis['mean_flight_time']:.1f} ± {analysis['std_flight_time']:.1f}s")

    def generate_monte_carlo_plots(self, analysis: dict):
        """Génère les graphiques de l'analyse Monte Carlo"""
        successful_results = [r for r in analysis['all_results'] if r['target_reached']]

        if not successful_results:
            print("❌ Pas de résultats valides à visualiser")
            return

        ranges = [r['range'] for r in successful_results]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Histogramme des portées
        axes[0,0].hist(ranges, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0,0].axvline(analysis['mean_range'], color='red', linestyle='--', label=f'Moyenne: {analysis["mean_range"]:.0f}m')
        axes[0,0].set_xlabel('Portée (m)')
        axes[0,0].set_ylabel('Fréquence')
        axes[0,0].set_title('Distribution des portées')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()

        # Box plot
        axes[0,1].boxplot(ranges)
        axes[0,1].set_ylabel('Portée (m)')
        axes[0,1].set_title('Box plot des portées')
        axes[0,1].grid(True, alpha=0.3)

        # Évolution des statistiques
        cumulative_mean = np.cumsum(ranges) / np.arange(1, len(ranges) + 1)
        axes[1,0].plot(cumulative_mean)
        axes[1,0].axhline(analysis['mean_range'], color='red', linestyle='--')
        axes[1,0].set_xlabel('Nombre de simulations')
        axes[1,0].set_ylabel('Portée moyenne (m)')
        axes[1,0].set_title('Convergence de la moyenne')
        axes[1,0].grid(True, alpha=0.3)

        # Taux de succès cumulatif
        successes = [1 if r['target_reached'] else 0 for r in analysis['all_results']]
        cumulative_success_rate = np.cumsum(successes) / np.arange(1, len(successes) + 1)
        axes[1,1].plot(cumulative_success_rate)
        axes[1,1].axhline(analysis['success_rate'], color='red', linestyle='--')
        axes[1,1].set_xlabel('Nombre de simulations')
        axes[1,1].set_ylabel('Taux de succès')
        axes[1,1].set_title('Convergence du taux de succès')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'monte_carlo_analysis_{int(time.time())}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 Analyse Monte Carlo sauvegardée: {filename}")

def compare_missiles():
    """Compare les performances de tous les missiles disponibles"""
    missiles = MissileDatabase.get_available_missiles()

    print("🔍 COMPARAISON DES MISSILES")
    print("=" * 60)

    for missile_name in missiles.keys():
        config = MissileDatabase.get_missile_config(missile_name)
        simulator = MissileSimulator(config)
        optimizer = TrajectoryOptimizer(simulator)

        print(f"\n📊 {missile_name}:")
        best_params, best_result = optimizer.optimize(verbose=False)
        print(f"   • Portée maximale: {best_result.range_m:.0f}m")
        print(f"   • Temps de vol: {best_result.flight_time:.1f}s")
        print(f"   • Angle optimal: {best_params.launch_angle:.1f}°")


def main_advanced_example():
    """Exemple principal avec toutes les nouvelles fonctionnalités"""

    # Lancement de l'interface interactive
    interactive_sim = InteractiveSimulator()

    print("🎯 SIMULATEUR DE MISSILE AVANCÉ")
    print("=" * 50)
    print("Choisissez votre mode:")
    print("1. 🎮 Mode interactif")
    print("2. 🤖 Mode automatique (analyse complète)")

    choice = input("Votre choix (1 ou 2): ").strip()

    if choice == '1':
        interactive_sim.run_interactive_session()
    else:
        # Mode automatique avec toutes les analyses
        SELECTED_MISSILE = "MICA"
        config = MissileDatabase.get_missile_config(SELECTED_MISSILE)

        print(f"\n🚀 Analyse complète du missile {SELECTED_MISSILE}")

        # 1. Optimisation classique
        print("\n1️⃣ Optimisation classique...")
        simulator = MissileSimulator(config)
        optimizer = TrajectoryOptimizer(simulator)
        best_params, best_result = optimizer.optimize(verbose=False)

        print(f"   • Portée: {best_result.range_m:.0f}m")
        print(f"   • Temps de vol: {best_result.flight_time:.1f}s")
        print(f"   • Cible atteinte: {'✅' if best_result.target_reached else '❌'}")

        # 2. Optimisation génétique
        print("\n2️⃣ Optimisation génétique...")
        bounds = {'launch_angle': (-30, 30), 'ascent_alpha': (-15, 15), 'descent_alpha': (-30, 30)}
        genetic_optimizer = GeneticOptimizer(simulator, population_size=30, generations=50)
        genetic_params, genetic_result = genetic_optimizer.optimize(bounds)

        print(f"   • Portée: {genetic_result.range_m:.0f}m")
        print(f"   • Temps de vol: {genetic_result.flight_time:.1f}s")
        print(f"   • Cible atteinte: {'✅' if genetic_result.target_reached else '❌'}")

        # 3. Analyse Monte Carlo
        print("\n3️⃣ Analyse Monte Carlo...")
        monte_carlo = MonteCarloAnalyzer(simulator)
        monte_carlo_analysis = monte_carlo.analyze_robustness(best_params, n_simulations=300)
        monte_carlo.generate_monte_carlo_plots(monte_carlo_analysis)

        # 4. Analyse de sensibilité
        print("\n4️⃣ Analyse de sensibilité...")
        sensitivity_analysis(config, best_params)

        print("\n✅ Analyse automatique terminée.")


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
