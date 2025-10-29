# Simulateur de Trajectoire de Missile

## 📚 Description

Ce dépôt contient une série de scripts Python développés pour **simuler et analyser la trajectoire balistique de missiles tactiques**. L'objectif principal est de calculer la **portée maximale** atteignable en fonction de divers paramètres physiques, aérodynamiques et de mission, tout en optimisant certains paramètres de vol comme l'**angle de tir** et l'**angle d'attaque (AoA)**.

Les simulations prennent en compte :

* La **masse variable** du missile due à la consommation de propergol (booster et sustainer).
* La **poussée** variable des moteurs.
* Les **forces aérodynamiques** (traînée et portance) qui dépendent de :
    * L'**altitude** (via un modèle atmosphérique multi-couches ISA).
    * La **vitesse** (nombre de Mach).
    * L'**angle d'attaque**.
    * La **géométrie** du missile (diamètre, longueur, forme du nez, surfaces portantes).
* Les **contraintes opérationnelles** comme la durée de vie de la batterie et l'altitude de la cible.

## ✨ Fonctionnalités Principales

* **Simulation de Trajectoire 2D :** Intégration numérique des équations du mouvement pour obtenir la position (x, z) et la vitesse (vx, vz) au cours du temps. Différentes méthodes d'intégration (Euler, Runge-Kutta 4) sont explorées dans les diverses versions.
* **Modélisation Aérodynamique :** Calcul des coefficients de traînée (onde, base, friction) et de force normale (portance) basés sur des formules semi-empiriques et la géométrie du missile. Inclut des modèles de décrochage pour les angles d'attaque élevés.
* **Optimisation des Paramètres de Vol :**
    * Recherche de l'angle d'attaque optimal (en phase de descente ou phases distinctes ascendante/descendante) pour maximiser la portée.
    * Optimisation combinée de l'angle de tir et des angles d'attaque via une recherche sur grille (parfois affinée en deux passes).
    * Utilisation du **parallélisme** (multithreading ou multiprocessing) pour accélérer la recherche exhaustive des paramètres optimaux.
* **Visualisation :** Génération de graphiques détaillés avec `matplotlib` illustrant la trajectoire, l'évolution des vitesses, de la masse, des forces, du Mach, de l'angle d'attaque, des coefficients aérodynamiques, etc.
* **Base de Données de Missiles :** Certaines versions incluent des configurations prédéfinies pour différents types de missiles (MICA, AIM-120B, R-77) facilitant les tests comparatifs.
* **Analyses Avancées :** Exploration d'études paramétriques, d'analyses de sensibilité et de méthodes d'optimisation alternatives (génétique, Monte Carlo) dans les versions les plus récentes (`Code_opti5.py`, `Code_opti6.py`).
* **Interface Utilisateur :** Présence d'une interface en ligne de commande (CLI) interactive dans certaines versions pour faciliter le lancement des simulations et des optimisations.

## 📜 Évolution du Code (Résumé)

Le dépôt contient plusieurs versions du simulateur, reflétant une évolution progressive:

1.  **Versions initiales (`Reprise_code`, `Test_1` à `Test_3`) :** Mise en place de la simulation de base, avec des implémentations et corrections successives concernant notamment le calcul et l'application de l'angle d'attaque.
2.  **Introduction de l'optimisation (`Test_4`, `Test_5`) :** Ajout de la recherche de l'angle de tir optimal et de l'angle d'attaque (post-apogée). Prise en compte plus détaillée de l'aérodynamique des ailes.
3.  **Optimisation multi-phases (`Test_6`, `Code_opti1`) :** Tentative d'optimisation de l'angle d'attaque sur les phases ascendante et descendante, améliorant la précision au prix d'un temps de calcul significativement plus long. Introduction du multithreading.
4.  **Modèles Aérodynamiques Avancés (`Test_7`, `Test_7v2`) :** Ajout d'un modèle de décrochage pour les grands angles d'attaque et correction des conversions d'unités angulaires.
5.  **Performances et Structure (`Code_opti2` à `Code_opti6`, `Test8`) :** Focalisation sur l'amélioration drastique des performances via le multiprocessing, refactorisation du code pour une meilleure modularité, ajout de bases de données de missiles, d'analyses paramétriques/Monte Carlo, et d'interfaces utilisateur.
6.  **Intégration (`Reprise_2025`, `Reprise_Sonnet4.5`) :** Tentatives de fusion des aspects de précision aérodynamique (`Test_7v2`) avec les structures de code plus récentes et interactives (`Test8`), ainsi que l'exploration de différentes méthodes de calcul des forces normales.

*(Note : Ce dépôt sert d'espace d'expérimentation et d'amélioration continue. Les différentes versions peuvent avoir des niveaux de fonctionnalité et de précision variables.)*

## 🚀 Utilisation (Exemple avec une version avancée comme `Code_opti6.py`)

*(Assurez-vous d'avoir Python et les bibliothèques `numpy` et `matplotlib` installées)*

1.  Exécutez le script. Il lancera par défaut une analyse complète pour le missile MICA :
    ```bash
    python Code_opti6.py
    ```
2.  Le script effectuera :
    * Une optimisation standard pour trouver les meilleurs angles.
    * Une étude paramétrique (si activée et ressources suffisantes).
    * Une analyse de sensibilité autour des paramètres optimaux.
    * Une comparaison rapide des performances avec d'autres missiles de la base de données.
3.  Les **graphiques** résultants (`trajectory_analysis_*.png`, `parametric_study_*.png`, `monte_carlo_analysis_*.png`) seront sauvegardés dans le même répertoire que le script.

*(Pour les versions avec interface interactive comme `Test8.py` ou `Reprise_2025/Reprise_Sonet4.5.py`, lancez le script et suivez les instructions du menu.)*

---
