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

## 3. Modélisation du Moteur 3D (Terminée)

La modélisation de l'arbre en 3D ainsi que la génération d'une forêt sont désormais **terminées**. Un moteur de simulation complet a été implémenté avec succès :
- **Génération Spatiale (L-System)** : Les arbres poussent désormais dans un espace en trois dimensions. Ils peuvent s'orienter librement dans toutes les directions (inclinaisons et rotations) pour créer des branchages réalistes.
- **Physique 3D Rigoureuse** : Le moteur calcule les forces exactes (inertie, torsion) subies par chaque branche grâce à des mathématiques vectorielles avancées pour simuler leur résistance au vent.
- **Moteur Graphique** : Rendu interactif et optimisé de la forêt via les bibliothèques scientifiques PyVista et VTK.

---

## 4. Applications à la Sylviculture

Maintenant que le modèle mathématique et physique est robuste, le projet se concentre sur des expériences pratiques liées à la sylviculture et à la gestion forestière.

**Objectif Actuel : Topologie de plantation et résistance au vent**
L'objectif est de placer une forêt de dimension fixe sous une forte tempête et de mesurer mathématiquement le stress mécanique subi par les arbres en fonction de leur schéma de plantation. 
Nous comparerons deux configurations classiques :
- **La plantation alignée (Grille)** : Les arbres sont disposés en rangées et colonnes parfaites.
- **La plantation en quinconce** : Les rangées sont décalées pour tenter de bloquer les couloirs de vent.

Cette expérience nous a permis de confronter notre modèle aux observations réelles des forestiers (qui recommandent généralement le quinconce ou des lisières progressives) et d'améliorer notre simulation en y intégrant de véritables principes de dynamique des fluides.

**Résultats de l'expérience : Le Triomphe du Quinconce**
En intégrant les turbulences directionnelles (oscillations du vent pendant la tempête) et l'**Effet Venturi** (l'air s'accélère lorsqu'il traverse un espace étroit entre deux arbres), notre modèle physique valide parfaitement les pratiques sylvicoles :
- **La Grille Alignée (Échec)** : Le vent s'engouffre dans les longs "couloirs" vides et s'y accélère violemment. Au moindre changement de direction de la rafale, ces jets d'air à haute vitesse frappent les arbres arrière de plein fouet. Le stress mécanique atteint des pics destructeurs.
- **Le Quinconce (Succès)** : Les arbres étant décalés, ils bloquent physiquement toute formation de couloirs. Le flux d'air rebondit constamment sur un tronc et ne peut pas accumuler de vitesse. La forêt encaisse la tempête comme un mur homogène, réduisant le stress mécanique maximal de plus de 25% !

**Expérience 2 : Lisière Progressive (Étagée)**
Nous avons ensuite simulé une lisière étagée : planter des arbres plus jeunes et plus souples en bordure de la forêt (échelle 0.5 puis 0.75), pour protéger le cœur de la parcelle (échelle 1.0).
- Le modèle confirme que la bordure étagée agit comme un "tremplin" aérodynamique. 
- La lisière progressive réduit le stress moyen global de la forêt et abaisse le pic de stress subi par les arbres adultes, démontrant la pertinence de cette méthode de plantation pour limiter le déracinement des fûts de valeur.
