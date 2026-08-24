import math
import time
import numpy as np
import pyvista as pv

from lsystem_3d import LSystem3D, parse_to_graph_3d
from physics_3d import PhysicsEngine3D

# ==========================================
# 1. PARAMÉTRAGE 3D
# ==========================================
AXIOM = "FFFFX"  
RULES = {
    
    "X": [
        "F[+X][-X]FX", # Axe 1
        "F[&X][^X]FX", # Axe 2
        "F[+X][^X]FX", # Diagonale 1
        "F[-X][&X]FX", # Diagonale 2
        "F[+X][&X]FX", # Diagonale 3
        "F[-X][^X]FX"  # Diagonale 4
    ], 
    "F": "F"
}
ITERATIONS = 6 
ANGLE_INCREMENT = math.radians(24.0)
SEGMENT_LENGTH = 1.0
NOISE = math.radians(12.0)
TROPISM_VECTOR = [-1.0, 0.0, 0.0] 
TROPISM_FACTOR = 0.06 

FRAME_DT = 1.0 / 60.0  # 60 FPS strict

PHYSICS_DT = FRAME_DT 
STEPS_PER_FRAME = 1
WIND_PARAMS = {
    "wind_speed": 22.0, 
    "wind_dir": [1.0, 1.0, 0.0], 
    "wind_frequency": 0.15 
}

# ==========================================
# 2. GÉNÉRATION DE L'ARBRE 3D
# ==========================================
print("Génération du modèle mathématique de la forêt...")
system = LSystem3D(AXIOM, RULES)
sentence = system.generate(ITERATIONS)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--layout", type=str, choices=["random", "grid", "quinconce"], default="random", help="Schéma de plantation de la forêt.")
args = parser.parse_args()

NUM_TREES = 7
FOREST_SIZE = 30.0 # Terrain de 30x30 mètres

forest_roots = []
tree_positions = []

import random

# Génération des positions selon le schéma choisi
if args.layout == "random":
    for i in range(NUM_TREES):
        x = random.uniform(-FOREST_SIZE/2, FOREST_SIZE/2)
        y = random.uniform(-FOREST_SIZE/2, FOREST_SIZE/2)
        tree_positions.append(np.array([x, y, 0.0], dtype=np.float32))
else:
    # Schémas structurés : 4x4 arbres (16 arbres)
    rows, cols = 4, 4
    spacing_x = 6.0 # Le vent souffle selon X, c'est la distance entre les lignes de front
    spacing_y = 6.0 # Distance latérale
    start_x = - (cols - 1) * spacing_x / 2.0
    start_y = - (rows - 1) * spacing_y / 2.0
    
    for r in range(rows):
        for c in range(cols):
            x = start_x + c * spacing_x
            y = start_y + r * spacing_y
            if args.layout == "quinconce" and c % 2 == 1:
                # En quinconce, on décale une colonne sur deux (car le vent souffle selon X, donc on décale latéralement en Y)
                y += spacing_y / 2.0
            tree_positions.append(np.array([x, y, 0.0], dtype=np.float32))
    NUM_TREES = len(tree_positions)

for i in range(NUM_TREES):
    # Phase 3 - Étape 1 : Pour la rigueur scientifique, on utilise la même échelle (1.0)
    tree_scale = 1.0 if args.layout != "random" else random.uniform(0.7, 1.4)
    
    # Variations structurelles via le bruit
    r = parse_to_graph_3d(sentence, ANGLE_INCREMENT, SEGMENT_LENGTH * tree_scale, noise=NOISE, tropism_vector=TROPISM_VECTOR, tropism_factor=TROPISM_FACTOR * tree_scale)
    forest_roots.extend(r)

def init_physics_properties(segment, depth=0):
    thickness_pow = 0
    segment.inertia = 0.0
    for child in segment.children:
        init_physics_properties(child, depth + 1)
        thickness_pow += child.thickness ** 2.5 # Loi de Murray (n=2.5) pour un tronc plus élancé
        segment.inertia += child.inertia
        
    if not segment.children:
        segment.thickness = 0.05 # Une brindille fait 5cm d'épaisseur, pas 70cm !
    else:
        segment.thickness = thickness_pow ** (1.0 / 2.5)

    segment.mass = (segment.thickness ** 2) * 0.2
    segment.inertia += segment.mass
    segment.stiffness = (segment.thickness ** 4) * 1200.0  # Rigidité accrue
    segment.damping = segment.stiffness * 0.5
    segment.is_kinematic = segment.thickness > 0.09  # Tronc et branches fermes

for root in forest_roots:
    init_physics_properties(root)

engine = PhysicsEngine3D(dt=PHYSICS_DT)

# ======================================
# 3. MOTEUR GRAPHIQUE  (PYVISTA / VTK)
# ======================================
print("Initialisation du moteur graphique VTK...")

all_segments = []
def collect_segments(seg):
    all_segments.append(seg)
    for c in seg.children:
        collect_segments(c)
        
for r in forest_roots:
    collect_segments(r)

num_segments = len(all_segments)

points = np.zeros((num_segments * 2, 3), dtype=np.float32)
lines = np.zeros((num_segments, 3), dtype=np.int32)

thicknesses = np.zeros(num_segments * 2, dtype=np.float32)

# Remplissage des tableaux géométriques et création d'un feuillage volumétrique
import random
leaf_offsets_list = []
leaf_parent_indices_list = []
leaf_orient_list = []
for i, seg in enumerate(all_segments):
    lines[i, 0] = 2
    lines[i, 1] = i * 2
    lines[i, 2] = i * 2 + 1
    
    # Tapering continu
    if seg.parent is not None:
        thicknesses[i * 2] = seg.parent.thickness
    else:
        thicknesses[i * 2] = seg.thickness * 1.3 # Racine de l'arbre un peu plus évasée (réalisme)
        
    # Le sommet du segment prend l'épaisseur de la branche actuelle
    thicknesses[i * 2 + 1] = seg.thickness
    
    if getattr(seg, 'has_leaf', False):
        # Ajout des feuilles
        for _ in range(3):
            # Décalage aléatoire autour de la branche
            u = np.random.normal(0, 1, 3)
            u = u / np.linalg.norm(u) * np.random.uniform(0.1, 0.7)
            leaf_offsets_list.append(u)
            leaf_parent_indices_list.append(i)
            # Orientation aléatoire de la feuille
            n = np.random.normal(0, 1, 3)
            leaf_orient_list.append(n / np.linalg.norm(n))

leaf_offsets = np.array(leaf_offsets_list, dtype=np.float32)
leaf_parent_indices = np.array(leaf_parent_indices_list, dtype=np.int32)
num_leaves = len(leaf_offsets)
leaf_points = np.zeros((num_leaves, 3), dtype=np.float32)

# ==========================================
# 3. INITIALISATION DE LA PHYSIQUE
# ==========================================
rot_y_up = np.array([
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0]
], dtype=float)

def calc_absolute_positions(segment, start_pos):
    H_abs = segment.absolute_R[:, 0]
    end_pos = start_pos + segment.length * H_abs
    segment.start_pos = start_pos
    segment.end_pos = end_pos
    for child in segment.children:
        calc_absolute_positions(child, end_pos)

# Positions initiales
for root in forest_roots:
    engine.update_kinematics(root, parent_R_abs=rot_y_up)
for i, root in enumerate(forest_roots):
    calc_absolute_positions(root, tree_positions[i])
for i, seg in enumerate(all_segments):
    points[i*2] = seg.start_pos
    points[i*2+1] = seg.end_pos
if num_leaves > 0:
    leaf_points[:] = points[leaf_parent_indices * 2 + 1] + leaf_offsets

# Création des objets géométriques VTK
mesh_branches = pv.PolyData(points, lines=lines)
# Ajustement de l'épaisseur du tronc
mesh_branches.point_data['thickness'] = thicknesses * 0.45

mesh_leaves = pv.PolyData(leaf_points)
mesh_leaves['orient'] = np.array(leaf_orient_list, dtype=np.float32)
mesh_leaves.active_vectors_name = 'orient'

# Forme géométrique d'une vraie petite feuille (sphère aplatie optimisée)
base_leaf = pv.Sphere(theta_resolution=4, phi_resolution=4, radius=0.65)
base_leaf.points[:, 2] *= 0.15 # Forme de feuille plate et ovale

# Génération initiale des feuilles
leaf_glyphs = mesh_leaves.glyph(geom=base_leaf, orient='orient', factor=1.0)

plotter = pv.Plotter(title="Simulation 3D Hyper-Réaliste - PyVista")
plotter.set_background('#87ceeb') # Bleu ciel

# Ajout du terrain
ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=FOREST_SIZE * 1.2, j_size=FOREST_SIZE * 1.2)
plotter.add_mesh(ground, color='#3c5a14', lighting=True)

# On génère des tubes 3D pour le tronc
mesh_tubes = mesh_branches.tube(scalars='thickness', absolute=True, n_sides=8)

# On dessine les branches
branches_actor = plotter.add_mesh(mesh_tubes, color='#4A3320', 
                                  show_scalar_bar=False, smooth_shading=True)

# On dessine le feuillage avec un vert riche et profond
leaf_actor = plotter.add_mesh(leaf_glyphs, color='#2c7a26', opacity=0.9, lighting=True)

def update_points():
    for i, root in enumerate(forest_roots):
        calc_absolute_positions(root, tree_positions[i])
    for i, seg in enumerate(all_segments):
        points[i*2] = seg.start_pos
        points[i*2+1] = seg.end_pos

    leaf_points[:] = points[leaf_parent_indices * 2 + 1] + leaf_offsets
        
    mesh_branches.points = points
    mesh_leaves.points = leaf_points
    new_glyphs = mesh_leaves.glyph(geom=base_leaf, orient='orient', factor=1.0)
    leaf_actor.mapper.dataset = new_glyphs
    
    new_tubes = mesh_branches.tube(scalars='thickness', absolute=True, n_sides=8)
    branches_actor.mapper.dataset = new_tubes

# Première frame
for root in forest_roots:
    engine.update_kinematics(root, parent_R_abs=rot_y_up)
update_points()

plotter.camera_position = 'yz'
plotter.camera.elevation = 15

plotter.show(interactive_update=True, auto_close=False)

step = 0
while not getattr(plotter, '_closed', False):
    animation_callback(step)
    plotter.update()
    step += 1
    time.sleep(0.01)
