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
# Paramètres du L-System
AXIOM = "F"
# Un système fractal classique
RULES = {"F": "F[+F]F[-F]F"}
ITERATIONS = 4
ANGLE_INCREMENT = math.radians(20)
SEGMENT_LENGTH = 1.0

# Paramètres Physiques
DT = 0.05
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
roots = parse_to_graph(sentence, ANGLE_INCREMENT, SEGMENT_LENGTH)

# Fonction récursive pour initialiser les paramètres physiques en fonction de la profondeur
def init_physics_properties(segment, depth=0):
    # Les branches basses sont plus lourdes et plus rigides
    # On diminue la masse et la raideur avec la profondeur
    segment.mass = max(0.1, 2.0 * (0.7 ** depth))
    segment.stiffness = max(1.0, 30.0 * (0.6 ** depth))
    segment.damping = max(0.5, 5.0 * (0.8 ** depth))
    
    for child in segment.children:
        init_physics_properties(child, depth + 1)

for root in roots:
    init_physics_properties(root)

engine = PhysicsEngine(dt=DT)

# ==========================================
# 3. VISUALISATION AVEC MATPLOTLIB
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
# Limites d'affichage adaptées à l'arbre
max_height = ITERATIONS * SEGMENT_LENGTH * 3
ax.set_xlim(-max_height, max_height)
ax.set_ylim(0, max_height)
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
    
    # On ajoute la ligne si elle n'existe pas encore
    if line_index >= len(lines_list):
        # L'épaisseur de la ligne dépend de la masse (approximatif)
        lw = max(1, segment.mass * 3)
        line, = ax.plot([start_x, end_x], [start_y, end_y], color='green', lw=lw, solid_capstyle='round')
        lines_list.append(line)
    else:
        # On met à jour la ligne existante
        lines_list[line_index].set_data([start_x, end_x], [start_y, end_y])
        
    line_index += 1
    
    for child in segment.children:
        line_index = draw_segment_recursive(child, end_x, end_y, lines_list, line_index)
        
    return line_index

def update(frame):
    # 1. Avancer le moteur physique
    current_time = frame * DT
    engine.step(roots, current_time, WIND_PARAMS)
    
    # 2. Mettre à jour l'affichage géométrique
    line_index = 0
    for root in roots:
        line_index = draw_segment_recursive(root, 0, 0, lines, line_index)
        
    return lines

# Création de l'animation
ani = FuncAnimation(fig, update, frames=200, interval=DT*1000, blit=True)

if __name__ == '__main__':
    plt.show()
