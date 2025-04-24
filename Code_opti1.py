import math
import numpy as np
import matplotlib.pyplot as plt

import time

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional



class Missile:
    def __init__(self, masse_debut, masse_apres_booster, masse_apres_sustainer, temps_booster, temps_sustainer,
                 F_booster, F_sustainer, vitesse_initiale, angle_tir, batterie, altitude, l_n, d, l, Nozzle, a, b,
                 angle_of_attack, Vt, h_cible, S_wing=0.02, l_wing=0.5, wing_pos=0.7):
        self.masse_debut = masse_debut
        self.masse_apres_booster = masse_apres_booster
        self.masse_apres_sustainer = masse_apres_sustainer
        self.temps_booster = temps_booster
        self.temps_sustainer = temps_sustainer
        self.F_booster = F_booster
        self.F_sustainer = F_sustainer
        self.vitesse_initiale = vitesse_initiale
        self.angle_tir = angle_tir
        self.batterie = batterie
        self.altitude = altitude
        self.l_n = l_n  # Longueur du nez
        self.d = d  # Diamètre du missile
        self.l = l  # Longueur du missile
        self.Nozzle = Nozzle  # Diamètre de la tuyère
        self.a = a  # Major axis
        self.b = b  # Minor axis
        self.angle_of_attack = angle_of_attack
        self.Vt = Vt  # Vitesse horizontale de l'ennemi
        self.h_cible = h_cible  # Altitude de l'ennemi
        self.altitudes = [self.altitude]
        # Paramètres pour les ailes/ailerons
        self.S_wing = S_wing  # Surface des ailes/ailerons (m²)
        self.l_wing = l_wing  # Longueur des ailes/ailerons (m)
        self.wing_pos = wing_pos  # Position relative des ailes sur la longueur totale (0 à 1)

    def calculate_drag_coefficients(self, M, l_n, d, l, A_e, S_Ref, q, alpha):
        if M > 1:
            CD0_Body_Wave = (1.59 + 1.83 / M ** 2) * (math.atan(0.5 / (l_n / d))) ** 1.69
        else:
            CD0_Body_Wave = 0

        if M > 1:
            CD0_Base_Coast = 0.25 / M
            CD0_Base_Powered = (1 - A_e / S_Ref) * (0.25 / M)
        else:
            CD0_Base_Coast = 0.12 + 0.13 * M ** 2
            CD0_Base_Powered = (1 - A_e / S_Ref) * (0.12 + 0.13 * M ** 2)

        CD0_Body_Friction = 0.053 * (l / d) * (M / (q * l)) ** 0.2

        # Contribution des ailerons/ailes à la traînée
        alpha_deg = abs(alpha)  # Utiliser la valeur absolue pour le décrochage
        CD_Wing_base = 0.1 * (self.S_wing / S_Ref) * (math.sin(math.radians(alpha_deg)) ** 2)
        if M > 1:
            CD_Wing_base += 0.05 * (self.S_wing / S_Ref)

        # Modélisation du décrochage : augmentation de la traînée après l'angle critique
        alpha_crit = 20.0  # Angle critique de décrochage (en degrés)
        if alpha_deg > alpha_crit:
            # Facteur d'augmentation de la traînée (linéaire jusqu'à 36°, puis plateau)
            drag_increase_factor = 1.0 + 0.1 * min((alpha_deg - alpha_crit) / (36.0 - alpha_crit), 1.0)
            CD_Wing = CD_Wing_base * drag_increase_factor
        else:
            CD_Wing = CD_Wing_base

        # Traînée totale
        if A_e > 0:
            Ca = CD0_Body_Wave + CD0_Base_Powered + CD0_Body_Friction + CD_Wing
        else:
            Ca = CD0_Body_Wave + CD0_Base_Coast + CD0_Body_Friction + CD_Wing

        return CD0_Body_Wave, CD0_Base_Coast, CD0_Base_Powered, CD0_Body_Friction, CD_Wing, Ca

    def calculate_normal_force_coefficient(self, a, b, phi, alpha, l, d):
        # Contribution du corps
        if alpha < 0:
            CN_body = -abs((a / b) * math.cos(phi) + (b / a) * math.sin(phi)) * (
                    abs(math.sin(2 * math.radians(alpha)) * math.cos(math.radians(alpha) / 2)) + 2 * (l / d) * math.sin(math.radians(alpha)) ** 2)
        else:
            CN_body = abs((a / b) * math.cos(phi) + (b / a) * math.sin(phi)) * (
                    abs(math.sin(2 * math.radians(alpha)) * math.cos(math.radians(alpha) / 2)) + 2 * (l / d) * math.sin(math.radians(alpha)) ** 2)

        # Contribution des ailerons/ailes
        alpha_rad = math.radians(abs(alpha))  # Utiliser la valeur absolue pour le décrochage
        CN_wing_base = 2 * math.pi * alpha_rad * (self.S_wing / (math.pi * (d / 2) ** 2))

        # Modélisation du décrochage : perte de portance après l'angle critique
        alpha_crit = 20.0  # Angle critique de décrochage (en degrés)
        if abs(alpha) > alpha_crit:
            # Réduction sigmoïde de la portance (0 à 1, avec 1 avant décrochage et 0 après 36°)
            lift_reduction = 1.0 / (1.0 + math.exp(5.0 * (abs(alpha) - 28.0) / (36.0 - 28.0)))
            CN_wing = CN_wing_base * lift_reduction
        else:
            CN_wing = CN_wing_base

        CN_wing = max(-1.0, min(1.0, CN_wing))  # Limiter entre -1 et 1

        # Coefficient de force normale total
        CN_total = CN_body + CN_wing * (self.wing_pos * (l - self.l_n) / l)

        return CN_total

    def calculer_portee(self, generer_graphiques=True, alpha_ascendant=None, alpha_descendant=None):
        self.altitudes = [self.altitude]

        vitesse_horizontale = self.vitesse_initiale * math.cos(math.radians(self.angle_tir))
        vitesse_verticale = self.vitesse_initiale * math.sin(math.radians(self.angle_tir))
        altitude = self.altitude
        portee = 0
        temps_total = max(self.temps_booster + self.temps_sustainer, self.batterie)
        delta_t = 0.1

        altitudes = [self.altitude]
        portees = [0]
        masses = [self.masse_debut]
        vitesses_horizontales = [self.vitesse_initiale * math.cos(math.radians(self.angle_tir))]
        vitesses_verticales = [self.vitesse_initiale * math.sin(math.radians(self.angle_tir))]
        forces = [self.F_booster if self.temps_booster > 0 else self.F_sustainer]
        resistances = [0]
        Cxs = [0]
        Machs = [0]
        alphas = [0]
        F_horizontales = [0]
        F_verticales = [0]
        Fa_horizontales = [0]
        Fa_verticales = [0]
        Fn_horizontales = [0]
        Fn_verticales = [0]
        Accel_horizontales = [0]
        Accel_verticales = [0]
        Ts = [0]
        Ps = [0]

        alpha = math.radians(alpha_ascendant) if alpha_ascendant is not None else 0  # Alpha ascendant par défaut
        en_descente = False
        launched = False

        for t in np.arange(delta_t, temps_total + delta_t, delta_t):
            if vitesse_horizontale > 100:
                launched = True

            if t > self.batterie:
                break

            if launched and altitude <= self.h_cible:
                break

            if launched and vitesse_horizontale <= 100:
                break

            if t <= self.temps_booster:
                masse = self.masse_debut - (self.masse_debut - self.masse_apres_booster) * t / self.temps_booster
            elif t <= self.temps_booster + self.temps_sustainer:
                masse = self.masse_apres_booster - (self.masse_apres_booster - self.masse_apres_sustainer) * (
                        t - self.temps_booster) / self.temps_sustainer
            else:
                masse = self.masse_apres_sustainer

            if t <= self.temps_booster:
                F = self.F_booster
            elif t <= self.temps_booster + self.temps_sustainer:
                F = self.F_sustainer
            else:
                F = 0

            l_n = self.l_n * 3.28084
            d = self.d * 3.28084
            l = self.l * 3.28084
            Nozzle = self.Nozzle * 3.28084
            A_e = (math.pi * (Nozzle / 2) ** 2) * 144
            S_Ref = math.pi * (d / 2) ** 2 * 144

            R = 287.05
            gamma = 1.405

            if altitude <= 11000:
                T = 288.15 - 0.0065 * altitude
                P = 101325 * (1 - 0.0065 * altitude / 288.15) ** 5.25588
            elif altitude <= 20000:
                T = 216.65
                P = 22632 * math.exp(-0.000157 * (altitude - 11000))
            elif altitude <= 32000:
                T = 216.65 + 0.001 * (altitude - 20000)
                P = 5474.87 * (1 + 0.001 * (altitude - 20000) / 216.65) ** -34.1632
            elif altitude <= 47000:
                T = 228.65 + 0.0028 * (altitude - 32000)
                P = 868.014 * (1 + 0.0028 * (altitude - 32000) / 228.65) ** -12.2011
            elif altitude <= 52000:
                T = 270.65
                P = 110.906 * math.exp(-0.000157 * (altitude - 47000))
            elif altitude <= 61000:
                T = 270.65 - 0.0028 * (altitude - 52000)
                P = 66.9389 * (T / 270.65) ** -12.2011
            elif altitude <= 79000:
                T = 252.65 - 0.002 * (altitude - 61000)
                P = 3.95642 * (T / 214.65) ** -12.2011

            rho = P / (R * T)
            V = (vitesse_horizontale ** 2 + vitesse_verticale ** 2) ** 0.5
            q = (0.5 * rho * V ** 2) / 47.88
            S = math.pi * (self.d / 2) ** 2
            son = math.sqrt(gamma * R * T)
            M = V / son
            CD0_Body_Wave, CD0_Base_Coast, CD0_Base_Powered, CD0_Body_Friction, CD_Wing, Ca = self.calculate_drag_coefficients(
                M, l_n, d, l, A_e, S_Ref, q, math.degrees(alpha))

            Fa = 0.5 * Ca * rho * S * V ** 2

            F_horizontal = F * math.cos(math.radians(self.angle_tir))
            F_vertical = F * math.sin(math.radians(self.angle_tir))

            Fa_horizontal = Fa * math.cos(alpha)
            Fa_vertical = Fa * math.sin(alpha)

            temps_restant = self.batterie - t

            if not en_descente and altitude < max(altitudes):
                en_descente = True
                alpha = math.radians(alpha_descendant) if alpha_descendant is not None else alpha

            phi = 0
            Cn = self.calculate_normal_force_coefficient(self.a, self.b, phi, math.degrees(alpha), self.l, self.d)

            Lift_drag_ratio = (Cn * math.cos(alpha) - Ca * math.sin(alpha)) / (
                    Cn * math.sin(alpha) + Ca * math.cos(alpha))

            Fn = Lift_drag_ratio * Fa

            Fn_horizontal = Fn * math.sin(alpha)
            Fn_vertical = Fn * math.cos(alpha)

            Accel_horizontal = (F_horizontal - Fa_horizontal) / masse
            Accel_vertical = (F_vertical - Fa_vertical) / masse

            vitesse_horizontale += (Accel_horizontal + Fn_horizontal / masse) * delta_t
            vitesse_verticale += (Accel_vertical - 9.81 + Fn_vertical / masse) * delta_t

            altitude += vitesse_verticale * delta_t
            self.altitudes.append(altitude)

            portee += vitesse_horizontale * delta_t

            altitudes.append(altitude)
            portees.append(portee)
            masses.append(masse)
            vitesses_horizontales.append(vitesse_horizontale)
            vitesses_verticales.append(vitesse_verticale)
            forces.append(F)
            resistances.append(Fa)
            Cxs.append(Ca)
            Machs.append(M)
            F_horizontales.append(F_horizontal)
            F_verticales.append(F_vertical)
            Fa_horizontales.append(Fa_horizontal)
            Fa_verticales.append(Fa_vertical)
            Fn_horizontales.append(Fn_horizontal)
            Fn_verticales.append(Fn_vertical)
            Accel_horizontales.append(Accel_horizontal)
            Accel_verticales.append(Accel_vertical)
            alphas.append(math.degrees(alpha))
            Ps.append(P)
            Ts.append(T)

        if generer_graphiques:
            fig = plt.figure(figsize=(20, 20))

            plt.subplot(4, 1, 1)
            plt.plot(np.arange(0, len(Ts) * delta_t, delta_t), Ts)
            plt.xlabel('Temps (s)')
            plt.ylabel('Température (K)')
            plt.title('Température en fonction du temps')
            plt.grid(True)

            plt.subplot(4, 1, 2)
            plt.plot(np.arange(0, len(Ps) * delta_t, delta_t), Ps)
            plt.xlabel('Temps (s)')
            plt.ylabel('Pression (Pa)')
            plt.title('Pression en fonction du temps')
            plt.grid(True)

            plt.subplot(4, 1, 3)
            plt.plot(altitudes, Ps)
            plt.xlabel('Altitude (m)')
            plt.ylabel('Pression (Pa)')
            plt.title('Pression en fonction de l\'altitude')
            plt.grid(True)

            plt.subplot(4, 1, 4)
            plt.plot(altitudes, Ts)
            plt.xlabel('Altitude (m)')
            plt.ylabel('Température (K)')
            plt.title('Température en fonction de l\'altitude')
            plt.grid(True)

            plt.tight_layout()
            plt.show()

            fig = plt.figure(figsize=(20, 20))

            plt.subplot(4, 2, 1)
            plt.plot(portees, altitudes)
            plt.xlabel('Distance horizontale (m)')
            plt.ylabel('Altitude (m)')
            plt.title(
                f'Trajectoire du missile (angle de tir: {self.angle_tir}°, alpha_ascendant: {alpha_ascendant}°, alpha_descendant: {alpha_descendant}°)')
            plt.grid(True)

            plt.subplot(4, 2, 2)
            plt.plot(np.arange(0, len(masses) * delta_t, delta_t), masses)
            plt.xlabel('Temps (s)')
            plt.ylabel('Masse (kg)')
            plt.title('Évolution de la masse du missile')
            plt.grid(True)

            plt.subplot(4, 2, 3)
            plt.plot(np.arange(0, len(vitesses_horizontales) * delta_t, delta_t), vitesses_horizontales,
                     label='Vitesse horizontale')
            plt.plot(np.arange(0, len(vitesses_verticales) * delta_t, delta_t), vitesses_verticales,
                     label='Vitesse verticale')
            plt.xlabel('Temps (s)')
            plt.ylabel('Vitesse (m/s)')
            plt.title('Évolution de la vitesse du missile')
            plt.legend()
            plt.grid(True)

            plt.subplot(4, 2, 4)
            plt.plot(np.arange(0, len(forces) * delta_t, delta_t), forces, label='Force de poussée')
            plt.plot(np.arange(0, len(resistances) * delta_t, delta_t), resistances, label='Résistance de l\'air')
            plt.xlabel('Temps (s)')
            plt.ylabel('Force (N)')
            plt.title('Évolution de la force de poussée et de la résistance de l\'air')
            plt.legend()
            plt.grid(True)

            plt.subplot(4, 2, 5)
            plt.plot(np.arange(0, len(altitudes) * delta_t, delta_t), altitudes)
            plt.xlabel('Temps (s)')
            plt.ylabel('Altitude (m)')
            plt.title('Évolution de l\'altitude du missile')
            plt.grid(True)

            plt.subplot(4, 2, 6)
            plt.plot(np.arange(0, len(portees) * delta_t, delta_t), portees)
            plt.xlabel('Temps (s)')
            plt.ylabel('Portée (m)')
            plt.title('Évolution de la portée du missile')
            plt.grid(True)

            plt.subplot(4, 2, 7)
            plt.plot(np.arange(0, len(Cxs) * delta_t, delta_t), Cxs)
            plt.xlabel('Temps (s)')
            plt.ylabel('Coefficient de traînée Ca')
            plt.title('Évolution du coefficient de traînée Cx du missile')
            plt.grid(True)

            plt.subplot(4, 2, 8)
            plt.plot(Machs, Cxs)
            plt.xlabel('Nombre de Mach')
            plt.ylabel('Coefficient de traînée Cx')
            plt.title('Évolution du coefficient de traînée Cx en fonction du nombre de Mach')
            plt.grid(True)

            plt.tight_layout()
            plt.show()

            fig = plt.figure(figsize=(20, 20))

            plt.subplot(4, 1, 1)
            plt.plot(np.arange(0, len(F_horizontales) * delta_t, delta_t), F_horizontales, label='F_horizontal')
            plt.plot(np.arange(0, len(F_verticales) * delta_t, delta_t), F_verticales, label='F_vertical')
            plt.xlabel('Temps (s)')
            plt.ylabel('Force (N)')
            plt.title('Forces horizontales et verticales')
            plt.legend()
            plt.grid(True)

            plt.subplot(4, 1, 2)
            plt.plot(np.arange(0, len(Fa_horizontales) * delta_t, delta_t), Fa_horizontales, label='Fa_horizontal')
            plt.plot(np.arange(0, len(Fa_verticales) * delta_t, delta_t), Fa_verticales, label='Fa_vertical')
            plt.xlabel('Temps (s)')
            plt.ylabel('Force de traînée (N)')
            plt.title('Forces de traînée horizontales et verticales')
            plt.legend()
            plt.grid(True)

            plt.subplot(4, 1, 3)
            plt.plot(np.arange(0, len(Fn_horizontales) * delta_t, delta_t), Fn_horizontales, label='Fn_horizontal')
            plt.plot(np.arange(0, len(Fn_verticales) * delta_t, delta_t), Fn_verticales, label='Fn_vertical')
            plt.xlabel('Temps (s)')
            plt.ylabel('Force normale (N)')
            plt.title('Forces normales horizontales et verticales')
            plt.legend()
            plt.grid(True)

            plt.subplot(4, 1, 4)
            plt.plot(np.arange(0, len(Accel_horizontales) * delta_t, delta_t), Accel_horizontales,
                     label='Accel_horizontal')
            plt.plot(np.arange(0, len(Accel_verticales) * delta_t, delta_t), Accel_verticales, label='Accel_vertical')
            plt.xlabel('Temps (s)')
            plt.ylabel('Accélération (m/s^2)')
            plt.title('Accélérations horizontales et verticales')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(np.arange(0, len(alphas) * delta_t, delta_t), alphas)
            plt.xlabel('Temps (s)')
            plt.ylabel('Angle d\'attaque alpha (degrés)')
            plt.title(f'Évolution de l\'angle d\'attaque alpha du missile (ascendant: {alpha_ascendant}°, descendant: {alpha_descendant}°)')
            plt.grid(True)
            plt.show()

        cible_atteinte = False
        if self.altitudes[-1] <= self.h_cible:
            cible_atteinte = True

        duree_vol = len(self.altitudes) * 0.1
        marge_batterie = abs(duree_vol - self.batterie)
        score = portee - marge_batterie * 100 if cible_atteinte else 0

        return portee, score, cible_atteinte

    def evaluer_combinaison(self, angle_tir: float, alpha_asc: float, alpha_desc: float) -> Optional[Tuple[float, float, float, float, float]]:
        self.angle_tir = angle_tir
        self.altitudes = [self.altitude]
        portee, score, cible_atteinte = self.calculer_portee(generer_graphiques=False, alpha_ascendant=alpha_asc, alpha_descendant=alpha_desc)

        if cible_atteinte:
            duree_vol = len(self.altitudes) * 0.1
            if duree_vol <= self.batterie:
                return (angle_tir, alpha_asc, alpha_desc, portee, score)
        return None

    def generer_combinaisons(self, angle_tir_range, alpha_asc_range, alpha_desc_range) -> List[Tuple[float, float, float]]:
        return [(a, b, c) for a in angle_tir_range for b in alpha_asc_range for c in alpha_desc_range]

    def trouver_meilleure_combinaison(self,
                                      angle_tir_min: float = -45, angle_tir_max: float = 45,
                                      alpha_asc_min: float = -20, alpha_asc_max: float = 20,
                                      alpha_desc_min: float = -36, alpha_desc_max: float = 36,
                                      pas_tir_grossier: float = 1, pas_alpha_grossier: float = 1,
                                      pas_tir_fin: float = 0.1, pas_alpha_fin: float = 0.1,
                                      verbose: bool = True) -> Tuple[float, float, float, float]:

        # Démarrer le timer
        start_time = time.time()

        def executer_evaluation(combinaisons) -> List[Tuple[float, float, float, float, float]]:
            with ThreadPoolExecutor() as executor:
                resultats = list(executor.map(lambda comb: self.evaluer_combinaison(*comb), combinaisons))
            return [r for r in resultats if r is not None]

        if verbose:
            print("Passe 1 : Recherche grossière...")

        angles = np.arange(angle_tir_min, angle_tir_max + pas_tir_grossier, pas_tir_grossier)
        alphas_asc = np.arange(alpha_asc_min, alpha_asc_max + pas_alpha_grossier, pas_alpha_grossier)
        alphas_desc = np.arange(alpha_desc_min, alpha_desc_max + pas_alpha_grossier, pas_alpha_grossier)

        combinaisons = self.generer_combinaisons(angles, alphas_asc, alphas_desc)
        valid_1 = executer_evaluation(combinaisons)

        if not valid_1:
            if verbose:
                print("Aucune combinaison valide (passe 1).")
            return 0, 0, 0, 0

        valid_1.sort(key=lambda x: x[3], reverse=True)
        top_combinaisons = valid_1[:min(5, len(valid_1))]

        if verbose:
            print("Passe 2 : Affinage autour des meilleures combinaisons...")

        toutes_combinaisons_fines = []
        for angle_tir, alpha_asc, alpha_desc, _, _ in top_combinaisons:
            angles_fins = np.arange(max(angle_tir_min, angle_tir - 2), min(angle_tir_max, angle_tir + 2) + pas_tir_fin, pas_tir_fin)
            alphas_asc_fins = np.arange(max(alpha_asc_min, alpha_asc - 2), min(alpha_asc_max, alpha_asc + 2) + pas_alpha_fin, pas_alpha_fin)
            alphas_desc_fins = np.arange(max(alpha_desc_min, alpha_desc - 2), min(alpha_desc_max, alpha_desc + 2) + pas_alpha_fin, pas_alpha_fin)
            toutes_combinaisons_fines += self.generer_combinaisons(angles_fins, alphas_asc_fins, alphas_desc_fins)

        valid_2 = executer_evaluation(toutes_combinaisons_fines)

        if valid_2:
            meilleure = max(valid_2, key=lambda x: x[3])
        else:
            if verbose:
                print("Aucune amélioration en passe fine, on garde la meilleure de la passe 1.")
            meilleure = max(top_combinaisons, key=lambda x: x[3])

        if verbose:
            print(f"✅ Meilleure combinaison : angle_tir={meilleure[0]:.2f}°, alpha_asc={meilleure[1]:.2f}°, alpha_desc={meilleure[2]:.2f}°, portée={meilleure[3]:.2f}m")

        # Fin du timer
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nDurée de la simulation : {elapsed_time:.2f} secondes")

        return meilleure[0], meilleure[1], meilleure[2], meilleure[3]


# Spécifications du missile MICA
missile = Missile(112, 88.3, 70.1,
                  2.75, 4, 20250, 10720,
                  300, None, 70, 5000,
                  0.40, 0.16, 3.1, 0.12, 0.08, 0.08, 36,
                  300, 2000,
                  S_wing=0.02, l_wing=0.5, wing_pos=0.7)

# Test et optimisation
print("Test avec vitesse initiale = 300 m/s")
angle_tir_optimal, alpha_asc_optimal, alpha_desc_optimal, portee_max = missile.trouver_meilleure_combinaison()
print(f"L'angle de tir optimal est de {angle_tir_optimal} degrés")
print(f"L'angle d'attaque ascendant optimal est de {alpha_asc_optimal} degrés")
print(f"L'angle d'attaque descendant optimal est de {alpha_desc_optimal} degrés")
print(f"La portée maximale est de {portee_max:.2f} mètres")

# Générer les graphiques pour les valeurs optimales
missile.angle_tir = angle_tir_optimal
missile.calculer_portee(
    alpha_ascendant=alpha_asc_optimal,
    alpha_descendant=alpha_desc_optimal,
    generer_graphiques=True
)
