import math
import numpy as np
from lsystem_3d import LSystem3D, parse_to_graph_3d
from physics_3d import PhysicsEngine3D
from wind_model import compute_wind_effects

def init_physics_properties(segment, scale=1.0):
    thickness_pow = 0
    segment.inertia = 0.0
    for child in segment.children:
        init_physics_properties(child, scale)
        thickness_pow += child.thickness ** 2.5
        segment.inertia += child.inertia
        
    if not segment.children:
        segment.thickness = 0.05 * scale
    else:
        segment.thickness = (thickness_pow ** (1.0 / 2.5))

    segment.mass = (segment.thickness ** 2) * 0.2
    segment.inertia += segment.mass
    segment.stiffness = (segment.thickness ** 4) * 500.0 
    segment.damping = segment.stiffness * 0.45
    segment.is_kinematic = segment.thickness > (0.12 * scale)

def generate_layout():
    tree_positions = []
    rows, cols = 4, 6 # Forêt plus profonde pour bien voir l'effet lisière
    spacing_x = 6.0 
    spacing_y = 6.0 
    start_x = - (cols - 1) * spacing_x / 2.0
    start_y = - (rows - 1) * spacing_y / 2.0
    
    for r in range(rows):
        for c in range(cols):
            x = start_x + c * spacing_x
            y = start_y + r * spacing_y
            # Toujours en quinconce car c'est la meilleure topologie
            if c % 2 == 1:
                y += spacing_y / 2.0
            tree_positions.append(np.array([x, y, 0.0], dtype=np.float32))
    return tree_positions, cols

def evaluate_lisiere(mode):
    print(f"\n--- Évaluation du schéma : QUINCONCE {mode.upper()} ---")
    
    rule = "F[+X][-X]FX" 
    system = LSystem3D("FFFFX", {"X": rule, "F": "F"})
    sentence = system.generate(6)
    
    tree_positions, cols = generate_layout()
    num_trees = len(tree_positions)
    print(f"Nombre d'arbres : {num_trees}")
    
    # 1. Détermination des échelles
    tree_scales = np.ones(num_trees)
    if mode == "progressif":
        # La ligne de front (c=0) est composée de jeunes arbres très flexibles
        # La deuxième ligne (c=1) est composée de jeunes adultes
        for i in range(num_trees):
            c = i % cols
            if c == 0:
                tree_scales[i] = 0.5
            elif c == 1:
                tree_scales[i] = 0.75
    
    base_wind_dir = [1.0, 0.0, 0.0]
    wind_params = {
        "wind_speed": 40.0, # Très grosse tempête
        "wind_dir": base_wind_dir, 
        "wind_frequency": 0.5 
    }
    
    engine = PhysicsEngine3D(dt=0.05)
    tree_roots = []
    for i in range(num_trees):
        scale = tree_scales[i]
        # L'arbre est géométriquement plus petit
        roots = parse_to_graph_3d(sentence, angle_increment=math.radians(24.0), segment_length=1.0 * scale)
        root = roots[0]
        init_physics_properties(root, scale)
        tree_roots.append(root)
        
    sim_time = 0.0
    max_torques = [0.0] * num_trees
    rot_y_up = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
    
    for root in tree_roots:
        engine.update_kinematics(root, parent_R_abs=rot_y_up)
        
    print("Calcul du vent en cours (sans affichage)...")
    
    for step in range(80):
        # Tempête tourbillonnante
        angle = math.sin(sim_time * 2.0) * 0.6 
        c_a, s_a = math.cos(angle), math.sin(angle)
        current_wind_dir = [base_wind_dir[0]*c_a - base_wind_dir[1]*s_a, 
                            base_wind_dir[0]*s_a + base_wind_dir[1]*c_a, 
                            0.0]
        wind_params["wind_dir"] = current_wind_dir
        
        # Les ombres et venturis dépendent des tailles respectives
        multipliers, _ = compute_wind_effects(tree_positions, current_wind_dir, tree_scales)
        
        for root in tree_roots:
            engine.update_kinematics(root, parent_R_abs=rot_y_up)
            
        for i, root in enumerate(tree_roots):
            engine.update_segment(root, sim_time, wind_params, wind_multiplier=multipliers[i], phase_offset=0.0)
            
            if hasattr(root, 'total_torque'):
                torque = np.linalg.norm(root.total_torque)
                if torque > max_torques[i]:
                    max_torques[i] = torque
                    
        sim_time += engine.dt
        
    print("\n--- RÉSULTATS ---")
    
    # On isole les grands arbres (ceux qu'on veut récolter et protéger)
    adult_torques = [max_torques[i] for i in range(num_trees) if tree_scales[i] == 1.0]
    
    stress_moyen = np.mean(max_torques)
    stress_max = np.max(max_torques)
    stress_max_adult = np.max(adult_torques) if adult_torques else 0.0
    
    print(f"Stress Moyen Global : {stress_moyen:.2f} N.m")
    print(f"Stress Max Global : {stress_max:.2f} N.m")
    print(f"Stress Max sur les Adultes (Arbres à protéger) : {stress_max_adult:.2f} N.m")
    
    return stress_moyen, stress_max, stress_max_adult

if __name__ == "__main__":
    mean_u, max_u, max_adult_u = evaluate_lisiere("uniforme")
    mean_p, max_p, max_adult_p = evaluate_lisiere("progressif")
    
    print("\n=============================================")
    print("RESULTAT FINAL : LISIERE")
    print("=============================================")
    if max_adult_p < max_adult_u:
        print("La Lisière Progressive protège MIEUX les grands arbres adultes.")
    else:
        print("La Lisière Uniforme protège MIEUX les grands arbres adultes.")
