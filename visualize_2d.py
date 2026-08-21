"""
Script de visualisation 2D d'un arbre soumis au vent.
Utilise matplotlib pour l'affichage dynamique.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from lsystem import LSystem, parse_to_graph
from physics import PhysicsEngine

# ==========================================
# 1. PARAMÉTRAGE
# ==========================================
# Paramètres du L-System (Arbre Monopodial d'apparence naturelle)
AXIOM = "FFFFX"  # Base du tronc dégarnie
RULES = {
    "X": "F[+X][-X]FX", # Le tronc principal (F) continue tout droit, les branches secondaires s'écartent
    "F": "F"            # F ne se multiplie pas (croissance linéaire)
}
ITERATIONS = 6
ANGLE_INCREMENT = math.radians(24.0)
SEGMENT_LENGTH = 1.0
NOISE = math.radians(12.0) # Bruit pour casser la symétrie parfaite

# Paramètres Physiques
FRAME_DT = 0.05
PHYSICS_DT = 0.005  # Sous-échantillonnage physique pour stabiliser Euler
STEPS_PER_FRAME = int(FRAME_DT / PHYSICS_DT)
WIND_PARAMS = {
    "wind_speed": 15.0,
    "wind_frequency": 0.5,
    "gust_duration": 4.0
}

# ==========================================
# 2. GÉNÉRATION DE L'ARBRE
# ==========================================
system = LSystem(AXIOM, RULES)
sentence = system.generate(ITERATIONS)
roots = parse_to_graph(sentence, ANGLE_INCREMENT, SEGMENT_LENGTH, noise=NOISE)

# Fonction récursive pour initialiser les paramètres physiques
def init_physics_properties(segment, depth=0):
    thickness_sq = 0
    segment.inertia = 0.0
    for child in segment.children:
        init_physics_properties(child, depth + 1)
        thickness_sq += child.thickness ** 2
        segment.inertia += child.inertia
        
    if not segment.children:
        # Base thickness for terminal branches
        segment.thickness = 1.0
    else:
        # Règle de Léonard de Vinci pour la section
        segment.thickness = math.sqrt(thickness_sq)

    # Propriétés physiques basées sur l'épaisseur calculée
    # La masse est proportionnelle à la section (épaisseur au carré)
    segment.mass = (segment.thickness ** 2) * 0.2
    # L'inertie totale perçue par ce noeud inclut sa propre masse et celle de tout l'arbre au-dessus de lui !
    segment.inertia += segment.mass
    # La rigidité d'une branche (moment d'inertie) est proportionnelle au rayon à la puissance 4
    segment.stiffness = (segment.thickness ** 4) * 5.0
    # Amortissement de Rayleigh (proportionnel à la raideur) pour dissiper l'énergie oscillatoire
    segment.damping = segment.stiffness * 0.2

for root in roots:
    init_physics_properties(root)

engine = PhysicsEngine(dt=PHYSICS_DT)

# ==========================================
# 3. VISUALISATION AVEC MATPLOTLIB
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
# Limites d'affichage adaptées dynamiquement à l'arbre généré
max_x, max_y = 0.0, 0.0
def find_bounds(segment, x, y):
    global max_x, max_y
    abs_a = segment.get_absolute_angle()
    nx = x + segment.length * math.sin(abs_a)
    ny = y + segment.length * math.cos(abs_a)
    max_x = max(max_x, abs(nx))
    max_y = max(max_y, ny)
    for c in segment.children:
        find_bounds(c, nx, ny)

for root in roots:
    find_bounds(root, 0, 0)

padding = max(max_y * 0.1, 5)
ax.set_xlim(-max_x - padding, max_x + padding)
ax.set_ylim(0, max_y + padding)
ax.axis('off')  # On cache les axes pour faire joli
ax.set_title("Simulation 2D d'un arbre sous le vent")

lines = []

def draw_segment_recursive(segment, start_x, start_y, lines_list, line_index):
    """
    Parcourt l'arbre récursivement pour tracer ou mettre à jour les lignes matplotlib.
    """
    abs_angle = segment.get_absolute_angle()
    # Convention : 0 rad = orienté vers le HAUT (Y positif)
    end_x = start_x + segment.length * math.sin(abs_angle)
    end_y = start_y + segment.length * math.cos(abs_angle)
    
    # On trace la branche
    if line_index >= len(lines_list):
        lw = max(1.0, segment.thickness * 1.5)
        # Bois marron
        color = '#3b2f2f' if segment.thickness > 2.0 else '#6e5c47'
        line, = ax.plot([start_x, end_x], [start_y, end_y], color=color, lw=lw, solid_capstyle='round')
        lines_list.append(line)
    else:
        lines_list[line_index].set_data([start_x, end_x], [start_y, end_y])
        
    line_index += 1
    
    # On trace la feuille si le segment en porte une
    if segment.has_leaf:
        if line_index >= len(lines_list):
            leaf, = ax.plot([end_x], [end_y], marker='o', markersize=6, color='#2ca02c', alpha=0.8, zorder=3)
            lines_list.append(leaf)
        else:
            lines_list[line_index].set_data([end_x], [end_y])
        line_index += 1
    
    for child in segment.children:
        line_index = draw_segment_recursive(child, end_x, end_y, lines_list, line_index)
        
    return line_index

def update(frame):
    # 1. Avancer le moteur physique en plusieurs petits pas (sub-stepping)
    current_time = frame * FRAME_DT
    for _ in range(STEPS_PER_FRAME):
        engine.step(roots, current_time, WIND_PARAMS)
        current_time += PHYSICS_DT
    
    # 2. Mettre à jour l'affichage géométrique
    line_index = 0
    for root in roots:
        line_index = draw_segment_recursive(root, 0, 0, lines, line_index)
        
    return lines

# Création de l'animation
ani = FuncAnimation(fig, update, frames=200, interval=FRAME_DT*1000, blit=True)

if __name__ == '__main__':
    plt.show()
