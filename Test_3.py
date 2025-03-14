import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar


class Missile:
    def __init__(self, masse_debut, masse_apres_booster, masse_apres_sustainer, temps_booster, temps_sustainer,
                 F_booster, F_sustainer, vitesse_initiale, angle_tir, batterie, altitude, l_n, d, l, Nozzle, a, b,
                 angle_of_attack, Vt, h_cible):
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
        self.alpha_optimal = 0  # Ajouter cet attribut pour stocker l'angle d'attaque optimal
        self.Vt = Vt  # Vitesse horizontale de l'ennemi
        self.h_cible = h_cible  # Altitude de l'ennemi
        self.altitudes = [self.altitude]

    def calculate_drag_coefficients(self, M, l_n, d, l, A_e, S_Ref, q):
        # Calculer le coefficient de traînée d'onde du corps
        if M > 1:
            CD0_Body_Wave = (1.59 + 1.83 / M ** 2) * (math.atan(0.5 / (l_n / d))) ** 1.69
        else:
            CD0_Body_Wave = 0  # La formule n'est valable que pour M > 1

        # Calculer le coefficient de traînée de base
        if M > 1:
            CD0_Base_Coast = 0.25 / M
            CD0_Base_Powered = (1 - A_e / S_Ref) * (0.25 / M)
        else:
            CD0_Base_Coast = 0.12 + 0.13 * M ** 2
            CD0_Base_Powered = (1 - A_e / S_Ref) * (0.12 + 0.13 * M ** 2)

        # Calculer le coefficient de traînée de frottement du corps
        CD0_Body_Friction = 0.053 * (l / d) * (M / (q * l)) ** 0.2

        return CD0_Body_Wave, CD0_Base_Coast, CD0_Base_Powered, CD0_Body_Friction

    def calculate_normal_force_coefficient(self, a, b, phi, alpha, l, d):
        # Calculer le coefficient de force normale
        if alpha < 0:
            CN = -abs((a / b) * math.cos(phi) + (b / a) * math.sin(phi)) * (
                    abs(math.sin(2 * alpha) * math.cos(alpha / 2)) + 2 * (l / d) * math.sin(alpha) ** 2)
        else:
            CN = abs((a / b) * math.cos(phi) + (b / a) * math.sin(phi)) * (
                    abs(math.sin(2 * alpha) * math.cos(alpha / 2)) + 2 * (l / d) * math.sin(alpha) ** 2)
        return CN

    def calculer_portee(self, generer_graphiques=True):
        # Réinitialiser les listes pour stocker les données de trajectoire
        self.altitudes = [self.altitude]

        # Initialiser les variables
        vitesse_horizontale = self.vitesse_initiale * math.cos(math.radians(self.angle_tir))
        vitesse_verticale = self.vitesse_initiale * math.sin(math.radians(self.angle_tir))
        altitude = self.altitude
        portee = 0
        temps_total = max(self.temps_booster + self.temps_sustainer, self.batterie)
        delta_t = 0.1  # s

        # Initialiser les listes pour stocker les données de trajectoire
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

        # Initialiser l'altitude maximale et l'angle d'attaque
        altitude_max = 0
        alpha = 0  # Angle d'attaque initial (en radians)
        en_descente = False

        # Initialiser la variable launched
        launched = False

        # Boucle sur chaque seconde de vol
        for t in np.arange(delta_t, temps_total + delta_t, delta_t):
            # Vérifier si le missile a été lancé
            if vitesse_horizontale > 100:
                launched = True

            # Conditions d'arrêt de la simulation
            if t > self.batterie:
                break  # La batterie est épuisée

            if launched and altitude <= self.h_cible:
                # Le missile a atteint la cible, on s'arrête
                break

            if launched and vitesse_horizontale <= 100:
                break  # Le missile a perdu trop de vitesse

            # Calculer la masse actuelle du missile
            if t <= self.temps_booster:
                masse = self.masse_debut - (self.masse_debut - self.masse_apres_booster) * t / self.temps_booster
            elif t <= self.temps_booster + self.temps_sustainer:
                masse = self.masse_apres_booster - (self.masse_apres_booster - self.masse_apres_sustainer) * (
                        t - self.temps_booster) / self.temps_sustainer
            else:
                masse = self.masse_apres_sustainer

            # Calculer la force de poussée actuelle
            if t <= self.temps_booster:
                F = self.F_booster
            elif t <= self.temps_booster + self.temps_sustainer:
                F = self.F_sustainer
            else:
                F = 0

            # Calculer le coefficient de traînée
            l_n = self.l_n * 3.28084  # Longueur du nez (en pied)
            d = self.d * 3.28084  # diamètre du missile (en pied)
            l = self.l * 3.28084  # Longueur du missile (en pied)
            Nozzle = self.Nozzle * 3.28084  # diamètre du nozzle (en pied)
            A_e = (math.pi * (Nozzle / 2) ** 2) * 144  # Aire du nozzle en in^2
            S_Ref = math.pi * (d / 2) ** 2 * 144  # Surface de ref en in^2

            # Constantes
            R = 287.05  # Constante spécifique de l'air sec en J/(kg*K)
            gamma = 1.405

            # Palier 0 (0 à 11000 m)
            if altitude <= 11000:
                T = 288.15 - 0.0065 * altitude  # Température en K
                P = 101325 * (1 - 0.0065 * altitude / 288.15) ** 5.25588  # Pression en Pa

            # Palier 1 (11000 à 20000 m)
            elif altitude <= 20000:
                T = 216.65  # Température en K
                P = 22632 * math.exp(-0.000157 * (altitude - 11000))  # Pression en Pa

            # Palier 2 (20000 à 32000 m)
            elif altitude <= 32000:
                T = 216.65 + 0.001 * (altitude - 20000)  # Température en K
                P = 5474.87 * (1 + 0.001 * (altitude - 20000) / 216.65) ** -34.1632  # Pression en Pa

            # Palier 3 (32000 à 47000 m)
            elif altitude <= 47000:
                T = 228.65 + 0.0028 * (altitude - 32000)  # Température en K
                P = 868.014 * (1 + 0.0028 * (altitude - 32000) / 228.65) ** -12.2011  # Pression en Pa

            # Palier 4 (47000 à 52000 m)
            elif altitude <= 52000:
                T = 270.65  # Température en K
                P = 110.906 * math.exp(-0.000157 * (altitude - 47000))  # Pression en Pa

            # Palier 3 (52000 à 61000 m)
            elif altitude <= 61000:
                T = 270.65 - 0.0028 * (altitude - 52000)  # Température en K
                P = 66.9389 * (T / 270.65) ** -12.2011  # Pression en Pa

            # Palier 3 (61000 à 79000 m)
            elif altitude <= 79000:
                T = 252.65 - 0.002 * (altitude - 61000)  # Température en K
                P = 3.95642 * (T / 214.65) ** -12.2011  # Pression en Pa

            rho = P / (R * T)  # Masse volumique en kg/m^3
            V = (vitesse_horizontale ** 2 + vitesse_verticale ** 2) ** 0.5  # Vitesse totale
            q = (0.5 * rho * V ** 2) / 47.88  # Pression dynamique (en psf)
            S = math.pi * (self.d / 2) ** 2  # m^2, surface de référence
            son = math.sqrt(gamma * R * T)
            M = V / son
            CD0_Body_Wave, CD0_Base_Coast, CD0_Base_Powered, CD0_Body_Friction = self.calculate_drag_coefficients(M,
                                                                                                                  l_n,
                                                                                                                  d, l,
                                                                                                                  A_e,
                                                                                                                  S_Ref,
                                                                                                                  q)
            if t <= self.temps_booster + self.temps_sustainer:
                Ca = CD0_Body_Wave + CD0_Base_Powered + CD0_Body_Friction
            else:
                Ca = CD0_Body_Wave + CD0_Base_Coast + CD0_Body_Friction

            # Calculer la résistance de l'air en utilisant l'équation Fx = 0.5 * rho * S * Ca * V^2
            Fa = 0.5 * Ca * rho * S * V ** 2  # N, Force de traînée

            F_horizontal = F * math.cos(math.radians(self.angle_tir))
            F_vertical = F * math.sin(math.radians(self.angle_tir))

            Fa_horizontal = Fa * math.cos(alpha)
            Fa_vertical = Fa * math.sin(alpha)

            temps_restant = self.batterie - t

            # Détection de l'apogée - vérifier si nous commençons à descendre
            if not en_descente and altitude < max(altitudes):
                en_descente = True
                # Convertir l'angle optimal en radians
                alpha = math.radians(self.alpha_optimal)

            phi = 0
            Cn = self.calculate_normal_force_coefficient(self.a, self.b, phi, alpha, self.l, self.d)

            Lift_drag_ratio = (Cn * math.cos(alpha) - Ca * math.sin(alpha)) / (
                    Cn * math.sin(alpha) + Ca * math.cos(alpha))

            Fn = Lift_drag_ratio * Fa

            # Calculer les composantes horizontale et verticale de la force normale
            Fn_horizontal = Fn * math.sin(alpha)
            Fn_vertical = Fn * math.cos(alpha)

            # Calculer l'accélération due à la résistance de l'air et à la force de poussée
            Accel_horizontal = (F_horizontal - Fa_horizontal) / masse  # m/s^2
            Accel_vertical = (F_vertical - Fa_vertical) / masse  # m/s^2

            # Mettre à jour la vitesse et l'altitude
            vitesse_horizontale += (Accel_horizontal + Fn_horizontal / masse) * delta_t
            vitesse_verticale += (
                                         Accel_vertical - 9.81 + Fn_vertical / masse) * delta_t  # m/s^2, acceleration due to gravity

            altitude += vitesse_verticale * delta_t
            self.altitudes.append(altitude)

            # Mettre à jour la portée
            portee += vitesse_horizontale * delta_t

            # Ajouter les données actuelles aux listes de trajectoire
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
            # Stocker l'angle en degrés pour l'affichage
            alphas.append(math.degrees(alpha))
            Ps.append(P)
            Ts.append(T)

        if generer_graphiques:
            # Créer une figure
            fig = plt.figure(figsize=(20, 20))

            # Créer un sous-graphique pour la température en fonction du temps
            plt.subplot(4, 1, 1)
            plt.plot(np.arange(0, len(Ts) * delta_t, delta_t), Ts)
            plt.xlabel('Temps (s)')
            plt.ylabel('Température (K)')
            plt.title('Température en fonction du temps')
            plt.grid(True)

            # Créer un sous-graphique pour la pression en fonction du temps
            plt.subplot(4, 1, 2)
            plt.plot(np.arange(0, len(Ps) * delta_t, delta_t), Ps)
            plt.xlabel('Temps (s)')
            plt.ylabel('Pression (Pa)')
            plt.title('Pression en fonction du temps')
            plt.grid(True)

            # Créer un sous-graphique pour la pression en fonction de l'altitude
            plt.subplot(4, 1, 3)
            plt.plot(altitudes, Ps)
            plt.xlabel('Altitude (m)')
            plt.ylabel('Pression (Pa)')
            plt.title('Pression en fonction de l\'altitude')
            plt.grid(True)

            # Créer un sous-graphique pour la température en fonction de l'altitude
            plt.subplot(4, 1, 4)
            plt.plot(altitudes, Ts)
            plt.xlabel('Altitude (m)')
            plt.ylabel('Température (K)')
            plt.title('Température en fonction de l\'altitude')
            plt.grid(True)

            # Afficher tous les sous-graphiques
            plt.tight_layout()
            plt.show()

            # Créer une figure
            fig = plt.figure(figsize=(20, 20))

            # Créer un graphique de la trajectoire du missile
            plt.subplot(4, 2, 1)
            plt.plot(portees, altitudes)
            plt.xlabel('Distance horizontale (m)')
            plt.ylabel('Altitude (m)')
            plt.title(
                f'Trajectoire du missile (angle de tir: {self.angle_tir}°, angle d\'attaque: {self.alpha_optimal}°)')
            plt.grid(True)

            # Créer un graphique de l'évolution de la masse du missile
            plt.subplot(4, 2, 2)
            plt.plot(np.arange(0, len(masses) * delta_t, delta_t), masses)
            plt.xlabel('Temps (s)')
            plt.ylabel('Masse (kg)')
            plt.title('Évolution de la masse du missile')
            plt.grid(True)

            # Créer un graphique de l'évolution de la vitesse horizontale et verticale
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

            # Créer un graphique de l'évolution de la force de poussée et de la résistance de l'air
            plt.subplot(4, 2, 4)
            plt.plot(np.arange(0, len(forces) * delta_t, delta_t), forces, label='Force de poussée')
            plt.plot(np.arange(0, len(resistances) * delta_t, delta_t), resistances, label='Résistance de l\'air')
            plt.xlabel('Temps (s)')
            plt.ylabel('Force (N)')
            plt.title('Évolution de la force de poussée et de la résistance de l\'air')
            plt.legend()
            plt.grid(True)

            # Créer un graphique de l'évolution de l'altitude en fonction du temps
            plt.subplot(4, 2, 5)
            plt.plot(np.arange(0, len(altitudes) * delta_t, delta_t), altitudes)
            plt.xlabel('Temps (s)')
            plt.ylabel('Altitude (m)')
            plt.title('Évolution de l\'altitude du missile')
            plt.grid(True)

            # Créer un graphique de l'évolution de la portée en fonction du temps
            plt.subplot(4, 2, 6)
            plt.plot(np.arange(0, len(portees) * delta_t, delta_t), portees)
            plt.xlabel('Temps (s)')
            plt.ylabel('Portée (m)')
            plt.title('Évolution de la portée du missile')
            plt.grid(True)

            # Créer un graphique de l'évolution du coefficient de traînée Cx
            plt.subplot(4, 2, 7)
            plt.plot(np.arange(0, len(Cxs) * delta_t, delta_t), Cxs)
            plt.xlabel('Temps (s)')
            plt.ylabel('Coefficient de traînée Ca')
            plt.title('Évolution du coefficient de traînée Cx du missile')
            plt.grid(True)

            # Créer un graphique de l'évolution du coefficient de traînée Cx en fonction du nombre de Mach
            plt.subplot(4, 2, 8)
            plt.plot(Machs, Cxs)
            plt.xlabel('Nombre de Mach')
            plt.ylabel('Coefficient de traînée Cx')
            plt.title('Évolution du coefficient de traînée Cx en fonction du nombre de Mach')
            plt.grid(True)

            # Afficher tous les graphiques
            plt.tight_layout()
            plt.show()

            # Créer une figure
            fig = plt.figure(figsize=(20, 20))

            # Créer un sous-graphique pour les forces horizontales et verticales
            plt.subplot(4, 1, 1)
            plt.plot(np.arange(0, len(F_horizontales) * delta_t, delta_t), F_horizontales, label='F_horizontal')
            plt.plot(np.arange(0, len(F_verticales) * delta_t, delta_t), F_verticales, label='F_vertical')
            plt.xlabel('Temps (s)')
            plt.ylabel('Force (N)')
            plt.title('Forces horizontales et verticales')
            plt.legend()
            plt.grid(True)

            # Créer un sous-graphique pour les forces de traînée horizontales et verticales
            plt.subplot(4, 1, 2)
            plt.plot(np.arange(0, len(Fa_horizontales) * delta_t, delta_t), Fa_horizontales, label='Fa_horizontal')
            plt.plot(np.arange(0, len(Fa_verticales) * delta_t, delta_t), Fa_verticales, label='Fa_vertical')
            plt.xlabel('Temps (s)')
            plt.ylabel('Force de traînée (N)')
            plt.title('Forces de traînée horizontales et verticales')
            plt.legend()
            plt.grid(True)

            # Créer un sous-graphique pour les forces normales horizontales et verticales
            plt.subplot(4, 1, 3)
            plt.plot(np.arange(0, len(Fn_horizontales) * delta_t, delta_t), Fn_horizontales, label='Fn_horizontal')
            plt.plot(np.arange(0, len(Fn_verticales) * delta_t, delta_t), Fn_verticales, label='Fn_vertical')
            plt.xlabel('Temps (s)')
            plt.ylabel('Force normale (N)')
            plt.title('Forces normales horizontales et verticales')
            plt.legend()
            plt.grid(True)

            # Créer un sous-graphique pour les accélérations horizontales et verticales
            plt.subplot(4, 1, 4)
            plt.plot(np.arange(0, len(Accel_horizontales) * delta_t, delta_t), Accel_horizontales,
                     label='Accel_horizontal')
            plt.plot(np.arange(0, len(Accel_verticales) * delta_t, delta_t), Accel_verticales, label='Accel_vertical')
            plt.xlabel('Temps (s)')
            plt.ylabel('Accélération (m/s^2)')
            plt.title('Accélérations horizontales et verticales')
            plt.legend()
            plt.grid(True)

            # Afficher tous les sous-graphiques
            plt.tight_layout()
            plt.show()

            # Créer un graphique de l'évolution de l'angle d'attaque alpha
            plt.figure(figsize=(10, 5))
            plt.plot(np.arange(0, len(alphas) * delta_t, delta_t), alphas)
            plt.xlabel('Temps (s)')
            plt.ylabel('Angle d\'attaque alpha (degrés)')
            plt.title(f'Évolution de l\'angle d\'attaque alpha du missile (optimisé à {self.alpha_optimal} degrés)')
            plt.grid(True)
            plt.show()

        # Vérifier si la cible a été atteinte
        cible_atteinte = False
        if self.altitudes[-1] <= self.h_cible:
            cible_atteinte = True

        # Calculer la durée de vol
        duree_vol = len(self.altitudes) * 0.1  # Durée en secondes (delta_t = 0.1)

        # Calculer la différence entre la durée de vol et la batterie
        marge_batterie = abs(duree_vol - self.batterie)

        # On veut minimiser cette marge pour atteindre la cible au moment où la batterie s'épuise
        score = portee - marge_batterie * 100 if cible_atteinte else 0

        return portee, score, cible_atteinte

    def trouver_alpha_optimal(self):
        # Vérifier que l'angle est bien dans les limites
        alpha_min = -min(self.angle_of_attack, 36)  # S'assurer que c'est dans les limites
        alpha_max = min(self.angle_of_attack, 36)  # S'assurer que c'est dans les limites

        portee_max = 0
        alpha_optimal = 0
        score_max = 0

        # Sauvegarde de l'angle d'attaque original
        angle_original = self.angle_of_attack

        # Première passe avec un pas plus large pour trouver la zone d'intérêt
        print("Première passe d'optimisation pour l'angle d'attaque...")
        for alpha_deg in np.arange(alpha_min, alpha_max, 2):  # Pas de 2° pour accélérer
            # Réinitialiser la liste des altitudes
            self.altitudes = [self.altitude]

            # Stocker l'angle d'attaque à tester
            self.alpha_optimal = alpha_deg

            portee, score, cible_atteinte = self.calculer_portee(generer_graphiques=False)

            if cible_atteinte and score > score_max:
                score_max = score
                portee_max = portee
                alpha_optimal = alpha_deg
                print(
                    f"Trouvé: alpha={alpha_deg}°, portée={portee:.2f}m, score={score:.2f}")

        # Seconde passe plus précise autour de l'angle optimal trouvé
        print(f"Seconde passe d'optimisation autour de {alpha_optimal}°...")
        for alpha_deg in np.arange(max(alpha_min, alpha_optimal - 2), min(alpha_max, alpha_optimal + 2),
                                   0.2):  # Pas réduit
            # Réinitialiser la liste des altitudes
            self.altitudes = [self.altitude]

            # Stocker l'angle d'attaque à tester
            self.alpha_optimal = alpha_deg

            portee, score, cible_atteinte = self.calculer_portee(generer_graphiques=False)

            if cible_atteinte and score > score_max:
                score_max = score
                portee_max = portee
                alpha_optimal = alpha_deg
                print(
                    f"Affiné: alpha={alpha_deg}°, portée={portee:.2f}m, score={score:.2f}")

        # Si aucune solution n'a été trouvée (portee_max est toujours 0)
        if portee_max == 0:
            print(
                "Aucune solution n'a été trouvée pour atteindre la cible. Recherche de la trajectoire la plus proche.")
            meilleure_altitude = float('inf')

            for alpha_deg in np.arange(alpha_min, alpha_max, 1):
                self.altitudes = [self.altitude]
                self.alpha_optimal = alpha_deg
                self.calculer_portee(generer_graphiques=False)

                # Calculer la distance minimale à la cible
                min_distance = float('inf')
                for i, alt in enumerate(self.altitudes):
                    distance = abs(alt - self.h_cible)
                    if distance < min_distance:
                        min_distance = distance

                if min_distance < meilleure_altitude:
                    meilleure_altitude = min_distance
                    alpha_optimal = alpha_deg

            print(f"Meilleure approximation: alpha={alpha_optimal}°, distance à la cible={meilleure_altitude:.2f}m")

            # La portée est 0 car la cible n'est pas atteinte
            portee_max = 0

        # Restaurer l'angle d'attaque original pour le reste du programme
        self.angle_of_attack = angle_original

        # Définir l'angle optimal trouvé
        self.alpha_optimal = alpha_optimal

        # Réinitialiser les altitudes pour le prochain appel
        self.altitudes = [self.altitude]

        # Calculer la portée réelle avec l'angle optimal
        portee_reelle, _, _ = self.calculer_portee(generer_graphiques=False)

        return alpha_optimal, portee_reelle

    def trouver_angle_tir_optimal(self, angle_min=-45, angle_max=45, pas_initial=5, pas_fin=0.5):
        """
        Trouve l'angle de tir optimal pour maximiser la portée.

        Parameters:
        -----------
        angle_min : float
            Angle minimum à tester (en degrés).
        angle_max : float
            Angle maximum à tester (en degrés).
        pas_initial : float
            Pas pour la première passe d'optimisation (en degrés).
        pas_fin : float
            Pas pour la seconde passe d'optimisation (en degrés).

        Returns:
        --------
        tuple
            (angle_optimal, portee_max, alpha_optimal) - L'angle de tir optimal,
            la portée maximale correspondante, et l'angle d'attaque optimal.
        """
        angle_optimal = self.angle_tir
        portee_max = 0
        alpha_optimal = self.alpha_optimal

        # Sauvegarde de l'angle de tir original
        angle_tir_original = self.angle_tir

        # Première passe avec un pas plus large
        print("Première passe d'optimisation pour l'angle de tir...")
        for angle_deg in np.arange(angle_min, angle_max + pas_initial, pas_initial):
            # Définir l'angle de tir à tester
            self.angle_tir = angle_deg

            # Trouver l'angle d'attaque optimal pour cet angle de tir
            alpha_opt, portee = self.trouver_alpha_optimal()

            if portee > portee_max:
                portee_max = portee
                angle_optimal = angle_deg
                alpha_optimal = alpha_opt
                print(f"Trouvé: angle_tir={angle_deg}°, alpha={alpha_opt}°, portée={portee:.2f}m")

        # Seconde passe plus précise autour de l'angle optimal trouvé
        print(f"Seconde passe d'optimisation autour de angle_tir={angle_optimal}°...")
        for angle_deg in np.arange(max(angle_min, angle_optimal - pas_initial),
                                   min(angle_max, angle_optimal + pas_initial),
                                   pas_fin):
            # Définir l'angle de tir à tester
            self.angle_tir = angle_deg

            # Trouver l'angle d'attaque optimal pour cet angle de tir
            alpha_opt, portee = self.trouver_alpha_optimal()

            if portee > portee_max:
                portee_max = portee
                angle_optimal = angle_deg
                alpha_optimal = alpha_opt
                print(f"Affiné: angle_tir={angle_deg}°, alpha={alpha_opt}°, portée={portee:.2f}m")

        # Restaurer l'angle de tir original
        self.angle_tir = angle_tir_original

        return angle_optimal, portee_max, alpha_optimal


# Spécifications du missile MICA
missile = Missile(112, 88.3, 70.1,
                 2.75, 4, 20250, 10720,
                 1, 15, 70, 5000,
                 0.40, 0.16, 3.1, 0.12, 0.08, 0.08, 36,
                 300, 2000)

# Test avec vitesse initiale = 1 m/s (missile standard)
print("Test avec vitesse initiale = 1 m/s")
angle_tir_optimal, portee_max, alpha_optimal = missile.trouver_angle_tir_optimal()
print(f"L'angle de tir optimal est de {angle_tir_optimal} degrés")
print(f"L'angle d'attaque optimal est de {alpha_optimal} degrés")
print(f"La portée maximale est de {portee_max:.2f} mètres")

# Générer les graphiques pour l'angle de tir et d'attaque optimaux
missile.angle_tir = angle_tir_optimal
missile.alpha_optimal = alpha_optimal
missile.calculer_portee(generer_graphiques=True)

# Test avec vitesse initiale = 300 m/s (missile tiré depuis un avion)
print("\nTest avec vitesse initiale = 300 m/s")
missile_avion = Missile(112, 88.3, 70.1,
                       2.75, 4, 20250, 10720,
                       300, 15, 70, 5000,
                       0.40, 0.16, 3.1, 0.12, 0.08, 0.08, 36,
                       300, 2000)
angle_tir_optimal, portee_max, alpha_optimal = missile_avion.trouver_angle_tir_optimal()
print(f"L'angle de tir optimal est de {angle_tir_optimal} degrés")
print(f"L'angle d'attaque optimal est de {alpha_optimal} degrés")
print(f"La portée maximale est de {portee_max:.2f} mètres")

# Générer les graphiques pour l'angle de tir et d'attaque optimaux
missile_avion.angle_tir = angle_tir_optimal
missile_avion.alpha_optimal = alpha_optimal
missile_avion.calculer_portee(generer_graphiques=True)