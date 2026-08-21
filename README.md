# Modélisation de l'Action du Vent sur une Forêt (L-Systems)

## 1. Introduction au Projet
Ce projet personnel a pour objectif de modéliser mathématiquement et physiquement le comportement d'un couvert végétal (une forêt) soumis à des rafales de vent.

Plutôt que de modéliser des arbres de manière arbitraire, nous utilisons les **L-Systems** (Systèmes de Lindenmayer) pour générer procéduralement la structure fractale des arbres. Ensuite, nous appliquons un modèle physique dynamique (basé sur le théorème du moment cinétique et l'intégration d'Euler) sur cette structure ramifiée.

---

## 2. Architecture du Projet

Le projet sera construit de manière incrémentale, chaque étape validant un principe physique ou mathématique :

1. **Génération par L-Systems** : Création de la structure topologique des arbres.
2. **Moteur Physique (PFD & Euler)** : Application des forces (raideur, amortissement visqueux, vent, couplage inter-branches) sur un réseau de tiges.
3. **Simulation Complète** : Couplage de la physique et de la géométrie, avec un vent oscillant temporellement.
4. **Visualisation (Matplotlib / Animation)** : Affichage dynamique de la simulation.

---

## 3. Étape Actuelle : Les L-Systems
*(En cours de développement)*

Les systèmes de Lindenmayer reposent sur un alphabet, un axiome de départ, et des règles de réécriture. À chaque itération, l'axiome grandit pour former une chaîne complexe qui sera ensuite traduite en instructions géométriques (Turtle Graphics).
