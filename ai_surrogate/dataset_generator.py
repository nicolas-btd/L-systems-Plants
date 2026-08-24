"""
Module de génération de données synthétiques physiques (Physics-to-Data).
Échantillonne l'espace des paramètres environnementaux et exécute les simulations en parallèle.
"""

import math
import time
import os
import sys
from concurrent.futures import ProcessPoolExecutor

# Assure l'accès aux modules du projet racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm

from lsystem_3d import LSystem3D, parse_to_graph_3d
from physics_3d import PhysicsEngine3D
from wind_model import compute_wind_effects


# Pré-génération de la grammaire fractale 3D de base
RULE = "F[+X][-X]FX"
BASE_SYSTEM = LSystem3D("FFFFX", {"X": RULE, "F": "F"})
CACHED_SENTENCE = BASE_SYSTEM.generate(3)


def init_physics_properties(segment, scale=1.0):
    """Calcule récursivement la masse, l'inertie et la raideur (Loi de Murray)."""
    thickness_pow = 0.0
    segment.inertia = 0.0
    for child in segment.children:
        init_physics_properties(child, scale=scale)
        thickness_pow += child.thickness ** 2.5
        segment.inertia += child.inertia
        
    if not segment.children:
        segment.thickness = 0.05 * scale
    else:
        segment.thickness = thickness_pow ** (1.0 / 2.5)
        
    segment.mass = (segment.thickness ** 2) * 0.2
    segment.inertia += segment.mass
    segment.stiffness = (segment.thickness ** 4) * 500.0
    segment.damping = segment.stiffness * 0.45
    segment.is_kinematic = segment.thickness > (0.12 * scale)


def generate_tree_positions(layout_type, rows, cols, spacing_x, spacing_y):
    """Génère les coordonnées 3D de plantation."""
    tree_positions = []
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


def run_single_simulation(
    layout_type="grid",
    rows=4,
    cols=4,
    spacing_x=6.0,
    spacing_y=6.0,
    wind_speed=35.0,
    wind_angle_deg=0.0,
    wind_oscillation_deg=35.0,
    wind_frequency=0.5,
    progressive_edge=False,
    sim_steps=30,
    dt=0.05
):
    """
    Exécute une simulation numérique 3D complète et extrait les contraintes physiques réelles.
    """
    sentence = CACHED_SENTENCE
    tree_positions = generate_tree_positions(layout_type, rows, cols, spacing_x, spacing_y)
    num_trees = len(tree_positions)
    
    scales = [1.0] * num_trees
    if progressive_edge:
        for idx in range(num_trees):
            col_idx = idx % cols
            if col_idx == 0:
                scales[idx] = 0.5
            elif col_idx == 1:
                scales[idx] = 0.75
                
    tree_roots = []
    for i in range(num_trees):
        roots = parse_to_graph_3d(sentence, angle_increment=math.radians(24.0), segment_length=1.0 * scales[i])
        root = roots[0]
        init_physics_properties(root, scale=scales[i])
        tree_roots.append(root)
        
    engine = PhysicsEngine3D(dt=dt)
    rot_y_up = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
    
    for root in tree_roots:
        engine.update_kinematics(root, parent_R_abs=rot_y_up)
        
    base_wind_rad = math.radians(wind_angle_deg)
    base_wind_dir = [math.cos(base_wind_rad), math.sin(base_wind_rad), 0.0]
    
    wind_params = {
        "wind_speed": float(wind_speed),
        "wind_dir": base_wind_dir,
        "wind_frequency": float(wind_frequency)
    }
    
    max_torques = [0.0] * num_trees
    sim_time = 0.0
    osc_amp_rad = math.radians(wind_oscillation_deg)
    
    for _ in range(sim_steps):
        turb_angle = math.sin(sim_time * 2.0 * math.pi * wind_frequency) * osc_amp_rad
        c_turb, s_turb = math.cos(turb_angle), math.sin(turb_angle)
        
        current_wind_dir = [
            base_wind_dir[0] * c_turb - base_wind_dir[1] * s_turb,
            base_wind_dir[0] * s_turb + base_wind_dir[1] * c_turb,
            0.0
        ]
        wind_params["wind_dir"] = current_wind_dir
        
        multipliers, _ = compute_wind_effects(tree_positions, current_wind_dir)
        
        for root in tree_roots:
            engine.update_kinematics(root, parent_R_abs=rot_y_up)
            
        for i, root in enumerate(tree_roots):
            engine.update_segment(root, sim_time, wind_params, wind_multiplier=multipliers[i], phase_offset=0.0)
            if hasattr(root, 'total_torque'):
                torque = np.linalg.norm(root.total_torque)
                if torque > max_torques[i]:
                    max_torques[i] = torque
                    
        sim_time += engine.dt
        
    return {
        "mean_stress": float(np.mean(max_torques)),
        "max_stress": float(np.max(max_torques)),
        "min_stress": float(np.min(max_torques)),
        "var_stress": float(np.var(max_torques)),
        "std_stress": float(np.std(max_torques))
    }


def _worker_sample(params):
    """Fonction exécutée en parallèle sur chaque cœur CPU."""
    t0 = time.perf_counter()
    targets = run_single_simulation(
        layout_type=params["layout_type"],
        rows=params["rows"],
        cols=params["cols"],
        spacing_x=params["spacing_x"],
        spacing_y=params["spacing_y"],
        wind_speed=params["wind_speed"],
        wind_angle_deg=params["wind_angle_deg"],
        wind_oscillation_deg=params["wind_oscillation_deg"],
        wind_frequency=params["wind_frequency"],
        progressive_edge=params["progressive_edge"],
        sim_steps=30,
        dt=0.05
    )
    sim_duration = time.perf_counter() - t0
    
    return {
        "layout_is_quinconce": 1 if params["layout_type"] == "quinconce" else 0,
        "rows": params["rows"],
        "cols": params["cols"],
        "num_trees": params["rows"] * params["cols"],
        "spacing_x": params["spacing_x"],
        "spacing_y": params["spacing_y"],
        "wind_speed": params["wind_speed"],
        "wind_angle_deg": params["wind_angle_deg"],
        "wind_oscillation_deg": params["wind_oscillation_deg"],
        "wind_frequency": params["wind_frequency"],
        "progressive_edge": 1 if params["progressive_edge"] else 0,
        "sim_duration_sec": sim_duration,
        "mean_stress": targets["mean_stress"],
        "max_stress": targets["max_stress"],
        "min_stress": targets["min_stress"],
        "var_stress": targets["var_stress"],
        "std_stress": targets["std_stress"]
    }


def generate_dataset(num_samples=400, output_path="ai_surrogate/dataset_forest_physics.csv", seed=42):
    """
    Génère un jeu de données synthétiques varié en parallèle (multi-cœurs).
    """
    np.random.seed(seed)
    print(f"=== Génération du Dataset Synthétique ({num_samples} simulations en parallèle) ===")
    
    param_list = []
    for _ in range(num_samples):
        p = {
            "layout_type": str(np.random.choice(["grid", "quinconce"])),
            "rows": int(np.random.choice([3, 4, 5])),
            "cols": int(np.random.choice([3, 4, 5])),
            "spacing_x": float(np.random.uniform(4.0, 9.0)),
            "spacing_y": float(np.random.uniform(4.0, 9.0)),
            "wind_speed": float(np.random.uniform(15.0, 45.0)),
            "wind_angle_deg": float(np.random.uniform(0.0, 360.0)),
            "wind_oscillation_deg": float(np.random.uniform(10.0, 45.0)),
            "wind_frequency": float(np.random.uniform(0.2, 0.8)),
            "progressive_edge": bool(np.random.choice([True, False]))
        }
        param_list.append(p)
        
    start_wall_time = time.time()
    max_workers = os.cpu_count() or 4
    records = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(_worker_sample, param_list), total=num_samples, desc="Simulations physiques"):
            records.append(res)
            
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    
    total_time = time.time() - start_wall_time
    print(f"\n[OK] Dataset sauvegardé avec succès dans : {output_path}")
    print(f"Nombre d'échantillons : {len(df)}")
    print(f"Temps total de calcul : {total_time:.2f} s ({total_time / len(df):.3f} s/échantillon équivalent)")
    return df


if __name__ == "__main__":
    generate_dataset(num_samples=400)
