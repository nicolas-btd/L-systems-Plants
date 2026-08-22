import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from lsystem_3d import LSystem3D, parse_to_graph_3d
from physics_3d import PhysicsEngine3D

# ==========================================
# 1. PARAMÉTRAGE 3D
# ==========================================
# L-System Monopodial avec rotation spatiale
AXIOM = "FFFFX"  # Base du tronc
RULES = {
    # \\\\ correspond à 4 fois l'angle (soit 90 degrés si angle=22.5)
    # L'arbre va donc générer ses branches en spirale autour du tronc !
    "X": "F[+X][\\\\-X]FX", 
    "F": "F"
}
ITERATIONS = 6
ANGLE_INCREMENT = math.radians(22.5)
SEGMENT_LENGTH = 1.0
NOISE = math.radians(10.0)

# Paramètres Physiques
FRAME_DT = 0.05
PHYSICS_DT = 0.005
STEPS_PER_FRAME = int(FRAME_DT / PHYSICS_DT)
WIND_PARAMS = {
    "wind_speed": 18.0,
    "wind_dir": [0.0, 1.0, 0.0], # Le vent souffle selon l'axe Y
    "wind_frequency": 0.3
}

# ==========================================
# 2. GÉNÉRATION DE L'ARBRE 3D
# ==========================================
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
# 3. VISUALISATION 3D
# ==========================================
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

def calc_absolute_positions(segment, start_pos):
    # L'axe principal local est X (colonne 0 de la matrice de rotation absolue)
    # Note : Dans matplotlib 3D, l'axe Z est la hauteur. 
    # Or, au repos (matrice identité), X pointe vers [1,0,0], ce qui est horizontal.
    # Pour que l'arbre pousse vers le haut (axe Z absolu), nous devons appliquer
    # une rotation initiale de l'arbre tout entier de 90° autour de Y.
    # Cela se fait en passant une matrice initiale modifiée au moteur physique !
    
    H_abs = segment.absolute_R[:, 0]
    end_pos = start_pos + segment.length * H_abs
    segment.start_pos = start_pos
    segment.end_pos = end_pos
    for child in segment.children:
        calc_absolute_positions(child, end_pos)

# Matrice pour orienter l'arbre vers le haut (Axe Z = axe du tronc)
# Rotation de -pi/2 autour de Y pour que le vecteur [1,0,0] devienne [0,0,1]
rot_y_up = np.array([
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0]
], dtype=float)

# Initialisation pour avoir les longueurs et limites
for root in roots:
    engine.update_kinematics(root, parent_R_abs=rot_y_up)
    calc_absolute_positions(root, np.array([0.0, 0.0, 0.0]))

max_val = 0.0
def find_max(segment):
    global max_val
    max_val = max(max_val, np.linalg.norm(segment.end_pos))
    for child in segment.children:
        find_max(child)

for root in roots:
    find_max(root)

lim = max_val * 0.6
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([0, max_val * 1.05])
ax.axis('off')

lines = []

# Collecteurs pour l'optimisation
branch_segments = []
branch_colors = []
branch_lws = []
leaf_xs, leaf_ys, leaf_zs = [], [], []

def build_geometry_recursive(segment):
    sx, sy, sz = segment.start_pos
    ex, ey, ez = segment.end_pos
    
    branch_segments.append([(sx, sy, sz), (ex, ey, ez)])
    branch_colors.append('#543b2a' if segment.thickness > 2.0 else '#8c6b51')
    branch_lws.append(max(1.0, segment.thickness * 1.2))
    
    if getattr(segment, 'has_leaf', False):
        leaf_xs.append(ex)
        leaf_ys.append(ey)
        leaf_zs.append(ez)
        
    for child in segment.children:
        build_geometry_recursive(child)

# Première passe pour initialiser les collections
for root in roots:
    build_geometry_recursive(root)
    
# Création d'une collection unique pour toutes les branches (beaucoup plus rapide !)
lc = Line3DCollection(branch_segments, colors=branch_colors, linewidths=branch_lws, capstyle='round')
ax.add_collection3d(lc)

# Un seul objet pour toutes les feuilles
leaves_plot, = ax.plot(leaf_xs, leaf_ys, leaf_zs, marker='o', markersize=3, color='#45a83a', alpha=0.9, linestyle='None', zorder=3)

def update(frame):
    current_time = frame * FRAME_DT
    
    for _ in range(STEPS_PER_FRAME):
        # 1. Cinématique (en appliquant la rotation pour mettre l'arbre debout)
        for root in roots:
            engine.update_kinematics(root, parent_R_abs=rot_y_up)
            
        # 2. Dynamique
        for root in roots:
            engine.update_segment(root, current_time, WIND_PARAMS)
            
        current_time += PHYSICS_DT
        
    # 3. Calcul des positions 3D absolues pour l'affichage
    for root in roots:
        calc_absolute_positions(root, np.array([0.0, 0.0, 0.0]))
        
    # 4. Mise à jour de la géométrie optimisée
    branch_segments.clear()
    leaf_xs.clear()
    leaf_ys.clear()
    leaf_zs.clear()
    
    for root in roots:
        build_geometry_recursive(root)
        
    lc.set_segments(branch_segments)
    leaves_plot.set_data_3d(leaf_xs, leaf_ys, leaf_zs)
        
    # Rotation douce de la caméra pour admirer la 3D
    ax.view_init(elev=20, azim=frame * 0.5)
    
    return [lc, leaves_plot]

ani = FuncAnimation(fig, update, frames=300, interval=FRAME_DT*1000, blit=False)

if __name__ == '__main__':
    print("Génération de l'arbre 3D et démarrage de la simulation (cela peut prendre quelques secondes)...")
    plt.show()
