import math
import numpy as np
from lsystem_3d import LSystem3D, parse_to_graph_3d
from physics_3d import PhysicsEngine3D
from wind_model import compute_wind_effects

def init_physics_properties(segment):
    thickness_pow = 0
    segment.inertia = 0.0
    for child in segment.children:
        init_physics_properties(child)
        thickness_pow += child.thickness ** 2.5
        segment.inertia += child.inertia
        
    if not segment.children:
        segment.thickness = 0.05
    else:
        segment.thickness = thickness_pow ** (1.0 / 2.5)

    segment.mass = (segment.thickness ** 2) * 0.2
    segment.inertia += segment.mass
    segment.stiffness = (segment.thickness ** 4) * 500.0 
    segment.damping = segment.stiffness * 0.45
    segment.is_kinematic = segment.thickness > 0.12

def generate_layout(layout_type):
    tree_positions = []
    rows, cols = 4, 4
    spacing_x = 6.0 
    spacing_y = 6.0 
    start_x = - (cols - 1) * spacing_x / 2.0
    start_y = - (rows - 1) * spacing_y / 2.0
    
    for r in range(rows):
        for c in range(cols):
            x = start_x + c * spacing_x
            y = start_y + r * spacing_y
            if layout_type == "quinconce" and c % 2 == 1:
                y += spacing_y / 2.0
            tree_positions.append(np.array([x, y, 0.0], dtype=np.float32))
    return tree_positions

def evaluate_topology(layout_type):
    print(f"\n--- Évaluation du schéma : {layout_type.upper()} ---")
    
    # 1. Création de l'arbre (tous les arbres seront identiques)
    # On prend une règle simple par défaut
    rule = "F[+X][-X]FX" 
    system = LSystem3D("FFFFX", {"X": rule, "F": "F"})
    sentence = system.generate(6)
    
    # 2. Placement des arbres
    tree_positions = generate_layout(layout_type)
    num_trees = len(tree_positions)
    print(f"Nombre d'arbres : {num_trees}")
    
    # 3. Préparation du vent
    base_wind_dir = [1.0, 0.0, 0.0]
    wind_params = {
        "wind_speed": 35.0, # Vent fort
        "wind_dir": base_wind_dir, 
        "wind_frequency": 0.5 
    }
    
    # 4. Lancement de la physique
    engine = PhysicsEngine3D(dt=0.05)
    tree_roots = []
    for i in range(num_trees):
        # Chaque arbre est identique
        roots = parse_to_graph_3d(sentence, angle_increment=math.radians(24.0), segment_length=1.0)
        root = roots[0]
        init_physics_properties(root)
        tree_roots.append(root)
        
    sim_time = 0.0
    max_torques = [0.0] * num_trees
    rot_y_up = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
    
    for root in tree_roots:
        engine.update_kinematics(root, parent_R_abs=rot_y_up)
        
    print("Calcul du vent en cours (sans affichage)...")
    
    for step in range(80): # 4 secondes de simulation
        # Oscillation du vent (Turbulences de ±35 degrés)
        # Ceci simule les violentes rafales tournantes d'une tempête
        angle = math.sin(sim_time * 2.0) * 0.6 
        c, s = math.cos(angle), math.sin(angle)
        current_wind_dir = [base_wind_dir[0]*c - base_wind_dir[1]*s, 
                            base_wind_dir[0]*s + base_wind_dir[1]*c, 
                            0.0]
        wind_params["wind_dir"] = current_wind_dir
        
        # Le vent change, donc les couloirs Venturi et les ombres se déplacent !
        multipliers, phase_offsets = compute_wind_effects(tree_positions, current_wind_dir)
        
        # Cinématique
        for root in tree_roots:
            engine.update_kinematics(root, parent_R_abs=rot_y_up)
            
        # Dynamique avec multiplicateur de vent
        for i, root in enumerate(tree_roots):
            engine.update_segment(root, sim_time, wind_params, wind_multiplier=multipliers[i], phase_offset=0.0)
            
            if hasattr(root, 'total_torque'):
                torque = np.linalg.norm(root.total_torque)
                if torque > max_torques[i]:
                    max_torques[i] = torque
                    
        sim_time += engine.dt
        
    # 5. Récolte des données
    print("\n--- RÉSULTATS ---")
    stress_moyen = np.mean(max_torques)
    stress_max = np.max(max_torques)
    stress_min = np.min(max_torques)
    variance = np.var(max_torques)
    
    print(f"Stress Moyen par arbre : {stress_moyen:.2f} N.m")
    print(f"Stress Maximum (l'arbre qui prend le plus de vent) : {stress_max:.2f} N.m")
    print(f"Stress Minimum (l'arbre le mieux caché) : {stress_min:.2f} N.m")
    print(f"Variance (répartition de la force) : {variance:.2f}")
    
    # Trouver combien d'arbres souffrent beaucoup
    # On pourrait comparer avec un arbre tout seul pour voir
    return stress_moyen, stress_max, variance

if __name__ == "__main__":
    mean_g, max_g, var_g = evaluate_topology("grid")
    mean_q, max_q, var_q = evaluate_topology("quinconce")
    
    print("\n=============================================")
    print("RESULTAT FINAL")
    print("=============================================")
    if max_q < max_g:
        print("La plantation en QUINCONCE protège mieux l'arbre le plus exposé.")
    else:
        print("La plantation ALIGNÉE (GRILLE) protège mieux l'arbre le plus exposé.")
        
    if mean_q < mean_g:
        print("La plantation en QUINCONCE réduit le stress moyen de la forêt.")
    else:
        print("La plantation ALIGNÉE (GRILLE) réduit le stress moyen de la forêt.")
