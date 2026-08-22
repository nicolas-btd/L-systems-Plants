import numpy as np
import random

class Segment3D:
    """
    Représente un segment physique de l'arbre en 3D.
    """
    def __init__(self, segment_id, parent=None, length=1.0, R_base=None):
        self.id = segment_id
        self.parent = parent
        self.children = []
        
        self.length = length
        
        # R_base est la matrice de rotation 3x3 de ce segment par rapport à son parent
        # (Ou par rapport au monde si c'est la racine).
        # La direction principale du segment est l'axe X local (vecteur [1, 0, 0])
        self.R_base = R_base if R_base is not None else np.eye(3)
            
        # Dynamique (Vecteur d'Euler pour la flexion relative au repos, en repère local)
        self.theta = np.zeros(3) # [theta_x (torsion), theta_y (flexion), theta_z (flexion)]
        self.omega = np.zeros(3) # Vitesse angulaire locale
        
        self.has_leaf = False
        self.thickness = 1.0
        
        # Attributs physiques qui seront calculés
        self.mass = 0.0
        self.inertia = 0.0
        self.stiffness = 0.0
        self.damping = 0.0
        
        if self.parent is not None:
            self.parent.add_child(self)
            
    def add_child(self, child_segment):
        self.children.append(child_segment)


class LSystem3D:
    def __init__(self, axiom, rules):
        self.axiom = axiom
        self.rules = rules
        
    def generate(self, iterations):
        current = self.axiom
        for _ in range(iterations):
            next_string = "".join([self.rules.get(c, c) for c in current])
            current = next_string
        return current


def parse_to_graph_3d(sentence, angle_increment, segment_length=1.0, noise=0.0):
    """
    Interprète une chaîne générée par un L-System en 3D (Turtle Graphics spatiale).
    H (Heading) = X axis, L (Left) = Y axis, U (Up) = Z axis.
    """
    root_segments = []
    state_stack = []
    
    # Matrice de rotation courante. 
    # Les colonnes représentent les vecteurs H, L, U dans le repère absolu.
    current_R = np.eye(3)
    
    # Matrices de rotation de base
    def R_U(a): # Lacet (Yaw) : tourne autour de U (Z local) -> symboles + / -
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
    def R_L(a): # Tangage (Pitch) : tourne autour de L (Y local) -> symboles & / ^
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
    def R_H(a): # Roulis (Roll) : tourne autour de H (X local) -> symboles / / \
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        
    current_segment = None
    next_id = 0
    
    for char in sentence:
        if char in ('F', 'A', 'B'):
            # Léger bruit directionnel pour la nature organique
            if noise > 0:
                current_R = current_R @ R_U(random.uniform(-noise/3, noise/3))
                current_R = current_R @ R_L(random.uniform(-noise/3, noise/3))
                
            # Calcul de la rotation relative par rapport au parent
            if current_segment is None:
                R_rel = current_R.copy()
            else:
                # current_R = parent_R @ R_rel  =>  R_rel = parent_R.T @ current_R
                parent_R = current_segment.absolute_R
                R_rel = parent_R.T @ current_R
                
            new_seg = Segment3D(next_id, current_segment, segment_length, R_rel)
            new_seg.absolute_R = current_R.copy() # Stockage temporaire pour le parsing
            next_id += 1
            
            if current_segment is None:
                root_segments.append(new_seg)
                
            current_segment = new_seg
            
        elif char == '+':
            a = angle_increment + random.uniform(-noise, noise)
            current_R = current_R @ R_U(a)
        elif char == '-':
            a = angle_increment + random.uniform(-noise, noise)
            current_R = current_R @ R_U(-a)
        elif char == '&':
            a = angle_increment + random.uniform(-noise, noise)
            current_R = current_R @ R_L(a)
        elif char == '^':
            a = angle_increment + random.uniform(-noise, noise)
            current_R = current_R @ R_L(-a)
        elif char == '\\':
            a = angle_increment + random.uniform(-noise, noise)
            current_R = current_R @ R_H(a)
        elif char == '/':
            a = angle_increment + random.uniform(-noise, noise)
            current_R = current_R @ R_H(-a)
        elif char == 'X':
            if current_segment is not None:
                current_segment.has_leaf = True
        elif char == '[':
            state_stack.append((current_segment, current_R.copy()))
        elif char == ']':
            if state_stack:
                current_segment, current_R = state_stack.pop()
                
    return root_segments
