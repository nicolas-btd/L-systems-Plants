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
    # Un chêne a une croissance asymétrique. On tire au hasard parmi 4 motifs.
    "X": [
        "F[+X][\\\\-X]FX",
        "F[-X][/X]FX",
        "F[&X][^X]FX",
        "F[+X]FX" # Branche qui ne se sépare qu'en deux
    ], 
    "F": "F"
}
ITERATIONS = 6 # Densité idéale pour allier un aspect visuel luxuriant et des performances fluides
ANGLE_INCREMENT = math.radians(24.0)
SEGMENT_LENGTH = 1.0
NOISE = math.radians(12.0)
TROPISM_VECTOR = [-1.0, 0.0, 0.0] # La gravité tire l'axe local X vers l'arrière
TROPISM_FACTOR = 0.06 # Force de la gravité (donne un joli port retombant)

FRAME_DT = 1.0 / 60.0  # 60 FPS strict
# La physique est désormais si stable (grâce à l'harmonisation) qu'elle peut tourner à 60Hz natif
PHYSICS_DT = FRAME_DT 
STEPS_PER_FRAME = 1
WIND_PARAMS = {
    "wind_speed": 22.0,
    "wind_dir": [1.0, 0.0, 0.0],
    "wind_frequency": 0.35
}

# ==========================================
# 2. GÉNÉRATION DE L'ARBRE 3D
# ==========================================
print("Génération du modèle mathématique...")
system = LSystem3D(AXIOM, RULES)
sentence = system.generate(ITERATIONS)
roots = parse_to_graph_3d(sentence, ANGLE_INCREMENT, SEGMENT_LENGTH, noise=NOISE, tropism_vector=TROPISM_VECTOR, tropism_factor=TROPISM_FACTOR)

def init_physics_properties(segment, depth=0):
    thickness_pow = 0
    segment.inertia = 0.0
    for child in segment.children:
        init_physics_properties(child, depth + 1)
        thickness_pow += child.thickness ** 2.5 # Loi de Murray (n=2.5) pour un tronc plus élancé
        segment.inertia += child.inertia
        
    if not segment.children:
        segment.thickness = 0.7
    else:
        segment.thickness = thickness_pow ** (1.0 / 2.5)

    # Physique réaliste d'une poutre : la rigidité dépend du rayon à la puissance 4
    # Cela garantit un tronc extrêmement rigide qui ne tourbillonne pas,
    # tout en laissant les petites branches souples.
    segment.mass = (segment.thickness ** 2) * 0.2
    segment.inertia += segment.mass
    segment.stiffness = (segment.thickness ** 4) * 500.0
    segment.damping = segment.stiffness * 0.6

for root in roots:
    init_physics_properties(root)

engine = PhysicsEngine3D(dt=PHYSICS_DT)

# ==========================================
# 3. MOTEUR GRAPHIQUE HAUTE PERFORMANCE (PYVISTA / VTK)
# ==========================================
print("Initialisation du moteur graphique VTK...")

all_segments = []
def collect_segments(seg):
    all_segments.append(seg)
    for c in seg.children:
        collect_segments(c)
        
for r in roots:
    collect_segments(r)

num_segments = len(all_segments)
# Les tableaux Numpy sont directement passés à la carte graphique (zéro surcoût)
points = np.zeros((num_segments * 2, 3), dtype=np.float32)
lines = np.zeros((num_segments, 3), dtype=np.int32)
thicknesses = np.zeros(num_segments, dtype=np.float32)

# Remplissage des tableaux géométriques et création d'un feuillage volumétrique
import random
leaf_offsets_list = []
leaf_parent_indices_list = []
leaf_orient_list = []
for i, seg in enumerate(all_segments):
    lines[i, 0] = 2
    lines[i, 1] = i * 2
    lines[i, 2] = i * 2 + 1
    thicknesses[i] = seg.thickness
    
    if getattr(seg, 'has_leaf', False):
        # 1 grand cluster/feuille par extrémité pour garantir 60 FPS constants sans surcharger Python
        for _ in range(1):
            # Décalage aléatoire
            u = np.random.normal(0, 1, 3)
            u = u / np.linalg.norm(u) * np.random.uniform(0.2, 1.8)
            leaf_offsets_list.append(u)
            leaf_parent_indices_list.append(i)
            # Orientation aléatoire de la feuille
            n = np.random.normal(0, 1, 3)
            leaf_orient_list.append(n / np.linalg.norm(n))

leaf_offsets = np.array(leaf_offsets_list, dtype=np.float32)
leaf_parent_indices = np.array(leaf_parent_indices_list, dtype=np.int32)
num_leaves = len(leaf_offsets)
leaf_points = np.zeros((num_leaves, 3), dtype=np.float32)

# Création des objets géométriques VTK
mesh_branches = pv.PolyData(points, lines=lines)
mesh_branches.cell_data['thickness'] = thicknesses

mesh_leaves = pv.PolyData(leaf_points)
mesh_leaves['orient'] = np.array(leaf_orient_list, dtype=np.float32)
mesh_leaves.active_vectors_name = 'orient'

# Forme géométrique d'une feuille (un grand losange aplati pour compenser la réduction du nombre)
base_leaf = pv.Sphere(theta_resolution=4, phi_resolution=4, radius=1.5)
base_leaf.points[:, 2] *= 0.1 # On aplatit la sphère pour en faire une feuille plate

# Génération initiale des feuilles
leaf_glyphs = mesh_leaves.glyph(geom=base_leaf, orient='orient', factor=1.0)

plotter = pv.Plotter(title="Simulation 3D Hyper-Réaliste - PyVista")
plotter.set_background('#87ceeb') # Bleu ciel

# On dessine les branches (shaders natifs)
plotter.add_mesh(mesh_branches, scalars='thickness', cmap='copper', 
                 render_lines_as_tubes=True, line_width=5, show_scalar_bar=False)

# On dessine le feuillage
leaf_actor = plotter.add_mesh(leaf_glyphs, color='#35b02a', opacity=0.9, lighting=True)

# Matrice pour mettre l'arbre debout
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

def update_points():
    for root in roots:
        calc_absolute_positions(root, np.array([0.0, 0.0, 0.0]))
    for i, seg in enumerate(all_segments):
        points[i*2] = seg.start_pos
        points[i*2+1] = seg.end_pos
        
    # Mise à jour hyper rapide (vectorisée) des positions des feuilles
    leaf_points[:] = points[leaf_parent_indices * 2 + 1] + leaf_offsets
        
    # On met à jour directement la mémoire vidéo
    mesh_branches.points = points
    
    # Recalcul hyper-rapide des feuilles (12000 objets = < 1ms)
    mesh_leaves.points = leaf_points
    new_glyphs = mesh_leaves.glyph(geom=base_leaf, orient='orient', factor=1.0)
    leaf_actor.mapper.dataset = new_glyphs

# Première frame
for root in roots:
    engine.update_kinematics(root, parent_R_abs=rot_y_up)
update_points()

plotter.camera_position = 'yz'
plotter.camera.elevation = 15

# ==========================================
# 4. BOUCLE D'ANIMATION EN TEMPS RÉEL
# ==========================================
print("Démarrage de la simulation 3D fluide à 60 FPS (Fermez la fenêtre pour arrêter)...")
current_time = 0.0

def animation_callback(step):
    global current_time
    # Résolution de la physique
    for _ in range(STEPS_PER_FRAME):
        for root in roots:
            engine.update_kinematics(root, parent_R_abs=rot_y_up)
        for root in roots:
            engine.update_segment(root, current_time, WIND_PARAMS)
        current_time += PHYSICS_DT
        
    # Mise à jour graphique
    update_points()
    # Le rendu est géré automatiquement par VTK

# Sur MacOS, l'interface graphique DOIT tourner sur le thread principal.
# L'utilisation d'un timer VTK est la seule solution stable.
duration_ms = max(1, int(FRAME_DT * 1000))
plotter.add_timer_event(max_steps=1000000, duration=duration_ms, callback=animation_callback)

plotter.show()
