import math
import numpy as np
import time
from vpython import *

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

FRAME_DT = 1.0 / 60.0  # 60 FPS pour un affichage parfaitement fluide
PHYSICS_DT = 0.002     # Pas de temps physique resserré pour la stabilité
STEPS_PER_FRAME = max(1, int(FRAME_DT / PHYSICS_DT))
WIND_PARAMS = {
    "wind_speed": 22.0,
    "wind_dir": [1.0, 0.0, 0.0], # Le vent souffle selon l'axe X (VPython: Y est vertical)
    "wind_frequency": 0.35
}

# ==========================================
# 2. GÉNÉRATION DE L'ARBRE 3D
# ==========================================
print("Génération de l'arbre et de la géométrie 3D...")
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
# 3. VISUALISATION VPYTHON
# ==========================================
# Initialisation de la scène
scene = canvas(title="Simulation 3D Fluide - Action du vent sur l'arbre", 
               width=1000, height=800, background=color.black)
scene.camera.pos = vector(0, 15, 35)
scene.camera.axis = vector(0, -10, -35)

# Matrice pour orienter l'arbre vers le haut dans VPython (Axe Y = axe du tronc)
# Rotation de +pi/2 autour de Z pour que le vecteur [1,0,0] devienne [0,1,0]
rot_x_to_y = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
], dtype=float)

def calc_absolute_positions(segment, start_pos):
    H_abs = segment.absolute_R[:, 0]
    end_pos = start_pos + segment.length * H_abs
    segment.start_pos = start_pos
    segment.end_pos = end_pos
    for child in segment.children:
        calc_absolute_positions(child, end_pos)

# Première initialisation
for root in roots:
    engine.update_kinematics(root, parent_R_abs=rot_x_to_y)
    calc_absolute_positions(root, np.array([0.0, 0.0, 0.0]))

# Création des objets graphiques
def build_vpython_geometry(segment):
    sx, sy, sz = segment.start_pos
    ex, ey, ez = segment.end_pos
    
    # Épaisseur du cylindre ajustée pour le rendu volumétrique
    radius = segment.thickness * 0.15 
    col = vector(84/255, 59/255, 42/255) if segment.thickness > 2.0 else vector(140/255, 107/255, 81/255)
    
    segment.v_cyl = cylinder(pos=vector(sx, sy, sz), 
                             axis=vector(ex-sx, ey-sy, ez-sz), 
                             radius=radius, color=col)
    
    if getattr(segment, 'has_leaf', False):
        segment.v_leaf = sphere(pos=vector(ex, ey, ez), radius=0.6, color=color.green, opacity=0.9)
        
    for child in segment.children:
        build_vpython_geometry(child)

for root in roots:
    build_vpython_geometry(root)

# Sol décoratif
box(pos=vector(0, -0.5, 0), size=vector(50, 1, 50), color=vector(0.2, 0.3, 0.2))

# ==========================================
# 4. BOUCLE D'ANIMATION FLUIDE
# ==========================================
def update_vpython_geometry(segment):
    sx, sy, sz = segment.start_pos
    ex, ey, ez = segment.end_pos
    
    segment.v_cyl.pos = vector(sx, sy, sz)
    segment.v_cyl.axis = vector(ex-sx, ey-sy, ez-sz)
    
    if getattr(segment, 'has_leaf', False):
        segment.v_leaf.pos = vector(ex, ey, ez)
        
    for child in segment.children:
        update_vpython_geometry(child)

print("Démarrage de l'animation... Naviguez dans la scène avec la souris (Clic Droit ou Molette) !")
current_time = 0.0

while True:
    # VPython limite l'exécution à 60 images par seconde
    rate(60) 
    
    # Moteur Physique
    for _ in range(STEPS_PER_FRAME):
        for root in roots:
            engine.update_kinematics(root, parent_R_abs=rot_x_to_y)
        for root in roots:
            engine.update_segment(root, current_time, WIND_PARAMS)
        current_time += PHYSICS_DT
        
    # Mise à jour Géométrique (Calcul Numpy)
    for root in roots:
        calc_absolute_positions(root, np.array([0.0, 0.0, 0.0]))
        
    # Mise à jour du Rendu 3D (VPython)
    for root in roots:
        update_vpython_geometry(root)
