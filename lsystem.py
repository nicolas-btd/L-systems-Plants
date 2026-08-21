"""
Moteur de génération procédurale par L-Systems.

Ce module permet de générer des structures arborescentes (chaînes de caractères)
à partir d'un axiome et de règles de réécriture, puis de traduire ces chaînes
en une structure de graphe (noeuds / segments) via une interprétation type "Turtle".
"""

class Segment:
    """
    Représente un segment physique (une branche ou un morceau de tronc) de l'arbre.
    C'est un noeud dans le graphe de l'arbre, qui possède potentiellement un parent
    et plusieurs enfants.
    """
    def __init__(self, segment_id, parent=None, length=1.0, base_angle=0.0):
        self.id = segment_id
        self.parent = parent
        self.children = []
        
        self.length = length
        
        # L'angle de base est l'angle de repos relatif au parent
        self.base_angle = base_angle
        # L'angle actuel de la branche (utilisé plus tard par le moteur physique)
        self.theta = base_angle
        self.omega = 0.0 # Vitesse angulaire
        
        if self.parent is not None:
            self.parent.add_child(self)
            
    def add_child(self, child_segment):
        self.children.append(child_segment)
        
    def get_absolute_angle(self):
        """Calcule l'angle absolu du segment en remontant jusqu'à la racine."""
        abs_angle = self.theta
        curr = self.parent
        while curr is not None:
            abs_angle += curr.theta
            curr = curr.parent
        return abs_angle


class LSystem:
    """
    Générateur de chaîne basée sur les L-Systems.
    """
    def __init__(self, axiom, rules):
        self.axiom = axiom
        self.rules = rules
        
    def generate(self, iterations):
        """Applique les règles de réécriture un nombre d'itérations donné."""
        current = self.axiom
        for _ in range(iterations):
            next_string = ""
            for char in current:
                # Si le caractère a une règle, on le remplace, sinon on le garde
                next_string += self.rules.get(char, char)
            current = next_string
        return current


def parse_to_graph(sentence, angle_increment, segment_length=1.0):
    """
    Interprète une chaîne générée par un L-System (Turtle Graphics)
    et la convertit en un graphe de `Segment`.
    
    Alphabet standard supporté :
    - F, A, B : Avancer en traçant un segment
    - + : Tourner à droite d'un certain angle
    - - : Tourner à gauche d'un certain angle
    - [ : Sauvegarder la position et l'angle actuel
    - ] : Restaurer la position et l'angle sauvegardé
    
    Retourne la liste des segments "racines".
    """
    root_segments = []
    state_stack = []
    
    current_angle = 0.0
    current_segment = None
    
    next_id = 0
    
    for char in sentence:
        if char in ('F', 'A', 'B'):
            # On calcule l'angle absolu du parent pour déduire l'angle relatif
            parent_abs_angle = current_segment.get_absolute_angle() if current_segment else 0.0
            relative_angle = current_angle - parent_abs_angle
            
            new_segment = Segment(next_id, current_segment, segment_length, relative_angle)
            next_id += 1
            
            if current_segment is None:
                root_segments.append(new_segment)
                
            current_segment = new_segment
            
        elif char == '+':
            current_angle += angle_increment
        elif char == '-':
            current_angle -= angle_increment
        elif char == '[':
            # On empile l'état courant
            state_stack.append((current_segment, current_angle))
        elif char == ']':
            # On dépile pour revenir à l'embranchement
            if state_stack:
                current_segment, current_angle = state_stack.pop()
                
    return root_segments

if __name__ == '__main__':
    # Test simple du module
    # Arbre fractal simple
    regles = {'F': 'F[+F]F[-F]F'}
    axiome = 'F'
    
    systeme = LSystem(axiome, regles)
    phrase = systeme.generate(iterations=2)
    print("Chaîne générée (2 itérations) :", phrase)
    
    import math
    racines = parse_to_graph(phrase, angle_increment=math.radians(25))
    print(f"L'arbre généré a {len(racines)} segment(s) racine(s).")
