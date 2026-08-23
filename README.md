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

## 3. Étape Actuelle : Validation du Moteur 3D
*(En cours de développement)*

L'étape 2 (Arbre 3D) ainsi que la génération d'une forêt (ombrage aérodynamique) sont désormais **terminées**. Un moteur 3D complet a été implémenté avec succès :
- **L-System Spatial** : Ajout d'une véritable tortue 3D gérant le roulis, le tangage et le lacet via des matrices de rotation.
- **Physique 3D Rigoureuse** : Les moments de force, l'inertie et le calcul de la torsion utilisent l'algèbre linéaire 3D (Vecteurs d'Euler, formule de Rodrigues).
- **Moteur Graphique** : Rendu optimisé via PyVista et VTK.

---

## 4. Applications à la Sylviculture

Maintenant que le modèle mathématique et physique est robuste, le projet se concentre sur des expériences pratiques liées à la sylviculture et à la gestion forestière.

**Objectif Actuel : Topologie de plantation et résistance au vent**
L'objectif est de placer une forêt de dimension fixe sous une forte tempête et de mesurer mathématiquement le stress mécanique subi par les arbres en fonction de leur schéma de plantation. 
Nous comparerons deux configurations classiques :
- **La plantation alignée (Grille)** : Les arbres sont disposés en rangées et colonnes parfaites.
- **La plantation en quinconce** : Les rangées sont décalées pour tenter de bloquer les couloirs de vent.

Cette expérience permettra d'évaluer la robustesse de notre modèle aérodynamique face aux observations réelles des forestiers (qui recommandent généralement le quinconce ou des lisières progressives) et d'affiner notre simulation de la dynamique des fluides.
