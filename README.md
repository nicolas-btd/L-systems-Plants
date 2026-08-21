# Modélisation de l'Action du Vent sur une Forêt (L-Systems)

## 1. Introduction au Projet
Ce projet personnel a pour objectif de modéliser mathématiquement et physiquement le comportement d'un couvert végétal (une forêt) soumis à des rafales de vent.

Plutôt que de modéliser des arbres de manière arbitraire, nous utilisons les **L-Systems** (Systèmes de Lindenmayer) pour générer procéduralement la structure fractale des arbres. Ensuite, nous appliquons un modèle physique dynamique (basé sur le théorème du moment cinétique et l'intégration d'Euler) sur cette structure ramifiée.

---

## 2. Architecture du Projet

Le projet sera construit de manière très structurée et incrémentale, en validant chaque palier de complexité avant de passer au suivant :

1. **Arbre Isolé (2D) avec Vent** :
   - Génération d'un arbre simple en 2D via L-Systems.
   - Application du moteur physique (Euler) pour simuler la torsion des branches sous l'effet du vent.
   - Visualisation et affinage des paramètres (raideur, masse, amortissement).
2. **Arbre Isolé (3D) Réaliste** :
   - Passage de la génération et de la physique en 3 dimensions.
   - Amélioration du réalisme visuel et mécanique (prise au vent selon l'orientation spatiale 3D).
3. **Modélisation d'une Forêt** :
   - Instanciation de multiples arbres.
   - Prise en compte de la dynamique des fluides simplifiée (atténuation du vent par le feuillage des premiers arbres) et des collisions/couplages.

---

## 3. Étape Actuelle : 2D - Moteur L-Systems
*(En cours de développement)*

La première brique est le moteur de génération procédurale. Il repose sur un axiome de départ et des règles de réécriture. La chaîne complexe générée est ensuite traduite en un graphe de noeuds géométriques (segments) via une interprétation type "Turtle Graphics".
