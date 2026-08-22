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
    "X": "F[+X][\\\\-X]FX", 
    "F": "F"
}
ITERATIONS = 6
ANGLE_INCREMENT = math.radians(24.0)
SEGMENT_LENGTH = 1.0
NOISE = math.radians(12.0)

FRAME_DT = 1.0 / 60.0  # 60 FPS strict
PHYSICS_DT = 0.002
STEPS_PER_FRAME = max(1, int(FRAME_DT / PHYSICS_DT))
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
roots = parse_to_graph_3d(sentence, ANGLE_INCREMENT, SEGMENT_LENGTH, noise=NOISE)

def init_physics_properties(segment, depth=0):
    thickness_sq = 0
    segment.inertia = 0.0
    for child in segment.children:
        init_physics_properties(child, depth + 1)
        thickness_sq += child.thickness ** 2
        segment.inertia += child.inertia
        
    if not segment.children:
        segment.thickness = 1.0
    else:
        segment.thickness = math.sqrt(thickness_sq)

    segment.mass = (segment.thickness ** 2) * 0.2
    segment.inertia += segment.mass
    segment.stiffness = (segment.thickness ** 4) * 6.0
    segment.damping = segment.stiffness * 0.2

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

leaf_indices = []
for i, seg in enumerate(all_segments):
    lines[i, 0] = 2
    lines[i, 1] = i * 2
    lines[i, 2] = i * 2 + 1
    thicknesses[i] = seg.thickness
    if getattr(seg, 'has_leaf', False):
        leaf_indices.append(i * 2 + 1)

# Création des objets géométriques VTK
mesh_branches = pv.PolyData(points, lines=lines)
mesh_branches.cell_data['thickness'] = thicknesses

mesh_leaves = pv.PolyData(points[leaf_indices])

plotter = pv.Plotter(title="Simulation 3D - PyVista")
plotter.set_background('black')

# Rendu avec les shaders natifs (très rapide)
# On colore les branches selon leur épaisseur (copper)
plotter.add_mesh(mesh_branches, scalars='thickness', cmap='copper', 
                 render_lines_as_tubes=True, line_width=4, show_scalar_bar=False)

# On dessine les feuilles comme de vraies sphères lumineuses
plotter.add_mesh(mesh_leaves, color='#2ca02c', point_size=8, 
                 render_points_as_spheres=True)

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
    # On met à jour directement la mémoire vidéo
    mesh_branches.points = points
    mesh_leaves.points = points[leaf_indices]

# Première frame
for root in roots:
    engine.update_kinematics(root, parent_R_abs=rot_y_up)
update_points()

plotter.show(interactive_update=True)
plotter.camera_position = 'yz'
plotter.camera.elevation = 15

# ==========================================
# 4. BOUCLE D'ANIMATION EN TEMPS RÉEL
# ==========================================
print("Démarrage de la simulation 3D fluide à 60 FPS (Fermez la fenêtre pour arrêter)...")
current_time = 0.0

try:
    while not plotter.window_size == [0,0]:
        t0 = time.time()
        
        # Résolution de la physique
        for _ in range(STEPS_PER_FRAME):
            for root in roots:
                engine.update_kinematics(root, parent_R_abs=rot_y_up)
            for root in roots:
                engine.update_segment(root, current_time, WIND_PARAMS)
            current_time += PHYSICS_DT
            
        # Mise à jour graphique
        update_points()
        plotter.update()
        
        # Limitation à 60 FPS pour que la vitesse perçue soit parfaite
        t1 = time.time()
        sleep_time = FRAME_DT - (t1 - t0)
        if sleep_time > 0:
            time.sleep(sleep_time)
            
except KeyboardInterrupt:
    pass
finally:
    plotter.close()
