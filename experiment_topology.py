import math
import numpy as np
from lsystem_3d import LSystem3D, parse_to_graph_3d
from physics_3d import PhysicsEngine3D

def compute_wind_effects(tree_positions, wind_dir):
    num_trees = len(tree_positions)
    multipliers = np.ones(num_trees)
    
    wd = np.array([wind_dir[0], wind_dir[1], 0.0])
    wd_norm = np.linalg.norm(wd)
    if wd_norm > 1e-6:
        wd = wd / wd_norm
        
    SHIELDING_LENGTH = 20.0 
    SHIELDING_RADIUS = 10.0 
    MAX_SHIELD = 0.85 
    
    for i, pi in enumerate(tree_positions):
        for j, pj in enumerate(tree_positions):
            if i == j: continue
            D = pi - pj
            dist_wind = np.dot(D, wd)
            
            if 0 < dist_wind < SHIELDING_LENGTH:
                D_perp = D - dist_wind * wd
                dist_perp = np.linalg.norm(D_perp)
                if dist_perp < SHIELDING_RADIUS:
                    shadow = (1.0 - dist_wind / SHIELDING_LENGTH) * (1.0 - dist_perp / SHIELDING_RADIUS)
                    multipliers[i] *= (1.0 - shadow * MAX_SHIELD)
    return multipliers

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
    wind_params = {
        "wind_speed": 35.0, # Vent fort
        "wind_dir": [1.0, 0.0, 0.0], # Vent selon l'axe X
        "wind_frequency": 0.5 
    }
    multipliers = compute_wind_effects(tree_positions, wind_params["wind_dir"])
    
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
    
    for step in range(60): # 3 secondes de simulation
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
