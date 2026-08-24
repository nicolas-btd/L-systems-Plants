"""
Visualiseur 3D Comparatif : Physique Numérique Classique vs Émulateur IA (Surrogate Model).
Arbres 3D réalistes avec feuillage volumétrique.
Basculez instantanément entre le solveur physique (Euler) et l'IA en appuyant sur 'M'.
"""

import math
import time
import argparse
import random
import numpy as np
import pyvista as pv

from lsystem_3d import LSystem3D, parse_to_graph_3d
from physics_3d import PhysicsEngine3D
from wind_model import compute_wind_effects
from ai_surrogate.benchmark_inference import ForestSurrogatePredictor


def main():
    parser = argparse.ArgumentParser(description="Comparateur 3D : Physique vs IA.")
    parser.add_argument("--rows", type=int, default=5, help="Nombre de rangées d'arbres (défaut: 5)")
    parser.add_argument("--cols", type=int, default=5, help="Nombre de colonnes d'arbres (défaut: 5)")
    parser.add_argument("--layout", type=str, choices=["grid", "quinconce", "random"], default="quinconce")
    args = parser.parse_args()

    rows, cols = args.rows, args.cols
    num_trees = rows * cols
    spacing_x = 6.5
    spacing_y = 6.5
    forest_size = max(rows * spacing_y, cols * spacing_x)

    print(f"\n=======================================================")
    print(f"  VISUALISEUR 3D COMPARATIF : {num_trees} ARBRES RÉALISTES")
    print(f"  Touche 'M' : Basculer PHYSIQUE (Euler) <-> IA (Deep Learning)")
    print(f"=======================================================\n")

    # 1. Règles 3D réalistes (multiaxiales)
    AXIOM = "FFFFX"
    RULES = {
        "X": [
            "F[+X][-X]FX", # Axe 1
            "F[&X][^X]FX", # Axe 2
            "F[+X][^X]FX", # Diagonale 1
            "F[-X][&X]FX", # Diagonale 2
            "F[+X][&X]FX", # Diagonale 3
            "F[-X][^X]FX"  # Diagonale 4
        ],
        "F": "F"
    }
    system = LSystem3D(AXIOM, RULES)
    sentence = system.generate(4)

    # 2. Positions de plantation
    tree_positions = []
    start_x = - (cols - 1) * spacing_x / 2.0
    start_y = - (rows - 1) * spacing_y / 2.0

    if args.layout == "random":
        for _ in range(num_trees):
            rx = random.uniform(-forest_size / 2, forest_size / 2)
            ry = random.uniform(-forest_size / 2, forest_size / 2)
            tree_positions.append(np.array([rx, ry, 0.0], dtype=np.float32))
    else:
        for r in range(rows):
            for c in range(cols):
                x = start_x + c * spacing_x
                y = start_y + r * spacing_y
                if args.layout == "quinconce" and c % 2 == 1:
                    y += spacing_y / 2.0
                tree_positions.append(np.array([x, y, 0.0], dtype=np.float32))

    # 3. Génération des arbres 3D avec variation organique
    forest_roots = []
    for i in range(num_trees):
        scale = random.uniform(0.85, 1.15) if args.layout == "random" else 1.0
        roots = parse_to_graph_3d(
            sentence,
            angle_increment=math.radians(24.0),
            segment_length=1.0 * scale,
            noise=math.radians(10.0),
            tropism_vector=[-1.0, 0.0, 0.0],
            tropism_factor=0.05 * scale
        )
        forest_roots.extend(roots)

    # 4. Propriétés physiques (Loi de Murray)
    def init_physics_properties(segment):
        thickness_pow = 0.0
        segment.inertia = 0.0
        for child in segment.children:
            init_physics_properties(child)
            thickness_pow += child.thickness ** 2.5
            segment.inertia += child.inertia

        if not segment.children:
            segment.thickness = 0.05
            segment.has_leaf = True
        else:
            segment.thickness = thickness_pow ** (1.0 / 2.5)

        segment.mass = (segment.thickness ** 2) * 0.2
        segment.inertia += segment.mass
        # Raideur organique cubique (évite l'hyper-flexibilité des brindilles terminales)
        segment.stiffness = (segment.thickness ** 3) * 450.0
        segment.damping = segment.stiffness * 0.6
        segment.is_kinematic = segment.thickness > 0.10

    for root in forest_roots:
        init_physics_properties(root)

    engine = PhysicsEngine3D(dt=0.033)
    rot_y_up = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)

    all_segments = []
    def collect_segments(seg):
        all_segments.append(seg)
        for c in seg.children:
            collect_segments(c)
    for r in forest_roots:
        collect_segments(r)

    num_segments = len(all_segments)

    # 5. Tableaux géométriques pour le rendu VTK
    points = np.zeros((num_segments * 2, 3), dtype=np.float32)
    lines = np.zeros((num_segments, 3), dtype=np.int32)
    thicknesses = np.zeros(num_segments * 2, dtype=np.float32)

    leaf_offsets_list = []
    leaf_parent_indices_list = []
    leaf_orient_list = []

    for i, seg in enumerate(all_segments):
        lines[i] = [2, i * 2, i * 2 + 1]
        thicknesses[i * 2] = seg.parent.thickness if seg.parent else seg.thickness * 1.3
        thicknesses[i * 2 + 1] = seg.thickness

        if getattr(seg, 'has_leaf', False) or not seg.children:
            for _ in range(3):
                u = np.random.normal(0, 1, 3)
                u = u / (np.linalg.norm(u) + 1e-6) * np.random.uniform(0.1, 0.6)
                leaf_offsets_list.append(u)
                leaf_parent_indices_list.append(i)
                n = np.random.normal(0, 1, 3)
                leaf_orient_list.append(n / (np.linalg.norm(n) + 1e-6))

    leaf_offsets = np.array(leaf_offsets_list, dtype=np.float32)
    leaf_parent_indices = np.array(leaf_parent_indices_list, dtype=np.int32)
    num_leaves = len(leaf_offsets)
    leaf_points = np.zeros((num_leaves, 3), dtype=np.float32)

    def calc_absolute_positions(segment, start_pos):
        H_abs = segment.absolute_R[:, 0]
        end_pos = start_pos + segment.length * H_abs
        segment.start_pos = start_pos
        segment.end_pos = end_pos
        for child in segment.children:
            calc_absolute_positions(child, end_pos)

    # Positions initiales
    for root in forest_roots:
        engine.update_kinematics(root, parent_R_abs=rot_y_up)
    for i, root in enumerate(forest_roots):
        calc_absolute_positions(root, tree_positions[i])
    for i, seg in enumerate(all_segments):
        points[i * 2] = seg.start_pos
        points[i * 2 + 1] = seg.end_pos
    if num_leaves > 0:
        leaf_points[:] = points[leaf_parent_indices * 2 + 1] + leaf_offsets

    # 6. Scène PyVista
    plotter = pv.Plotter(title=f"Comparatif 3D Réaliste : Physique vs IA ({num_trees} arbres)")
    plotter.set_background('#FFFFFF')

    ground = pv.Plane(center=(0, 0, -0.05), direction=(0, 0, 1), i_size=forest_size * 1.3, j_size=forest_size * 1.3)
    plotter.add_mesh(ground, color='#C4A482', lighting=True)

    mesh_branches = pv.PolyData(points, lines=lines)
    mesh_branches.point_data['thickness'] = thicknesses * 0.45
    mesh_tubes = mesh_branches.tube(scalars='thickness', absolute=True, n_sides=6)
    branches_actor = plotter.add_mesh(mesh_tubes, color='#4A3320', show_scalar_bar=False, smooth_shading=True)

    # Feuillage
    base_leaf = pv.Sphere(theta_resolution=4, phi_resolution=4, radius=0.6)
    base_leaf.points[:, 2] *= 0.2
    mesh_leaves = pv.PolyData(leaf_points)
    mesh_leaves['orient'] = np.array(leaf_orient_list, dtype=np.float32)
    mesh_leaves.active_vectors_name = 'orient'
    leaf_glyphs = mesh_leaves.glyph(geom=base_leaf, orient='orient', factor=1.0)
    leaf_actor = plotter.add_mesh(leaf_glyphs, color='#2c7a26', opacity=0.95, lighting=True)

    # 7. Initialisation IA (Surrogate Model)
    try:
        predictor = ForestSurrogatePredictor()
        has_ai = True
    except Exception as e:
        print(f"[!] Erreur chargement modèle IA : {e}. Mode Physique activé.")
        has_ai = False

    state = {
        "mode": "ia" if has_ai else "physics",
        "current_time": 0.0,
        "wind_speed": 35.0,
        "last_frame_time": time.perf_counter(),
        "fps": 60.0,
        "compute_ms": 0.0
    }

    def toggle_mode():
        if not has_ai:
            return
        state["mode"] = "ia" if state["mode"] == "physics" else "physics"
        print(f"-> Basculement de mode : {state['mode'].upper()}")

    plotter.add_key_event("m", toggle_mode)
    plotter.add_key_event("M", toggle_mode)

    # Scénario IA de base
    ai_scenario = {
        "layout_type": args.layout if args.layout != "random" else "grid",
        "layout_is_quinconce": 1 if args.layout == "quinconce" else 0,
        "rows": rows,
        "cols": cols,
        "num_trees": num_trees,
        "spacing_x": spacing_x,
        "spacing_y": spacing_y,
        "wind_speed": state["wind_speed"],
        "wind_angle_deg": 0.0,
        "wind_oscillation_deg": 35.0,
        "wind_frequency": 0.5,
        "progressive_edge": 0
    }

    ai_preds = predictor.predict(ai_scenario) if has_ai else {"max_stress": 0.09, "mean_stress": 0.07}

    base_wind_dir = [1.0, 0.0, 0.0]
    multipliers, phases = compute_wind_effects(tree_positions, base_wind_dir)

    def animation_callback(step):
        t0 = time.perf_counter()
        t = state["current_time"]

        wind_angle_rad = math.sin(t * 1.5) * 0.45
        c_w, s_w = math.cos(wind_angle_rad), math.sin(wind_angle_rad)
        current_wind_dir = [c_w, s_w, 0.0]

        wind_params = {
            "wind_speed": state["wind_speed"],
            "wind_dir": current_wind_dir,
            "wind_frequency": 0.4
        }

        if state["mode"] == "physics":
            # --- 🔴 MODE PHYSIQUE NUMÉRIQUE ITÉRATIF (Euler pas à pas) ---
            cur_multipliers, _ = compute_wind_effects(tree_positions, current_wind_dir)
            for root in forest_roots:
                engine.update_kinematics(root, parent_R_abs=rot_y_up)
            for i, root in enumerate(forest_roots):
                engine.update_segment(root, t, wind_params, wind_multiplier=cur_multipliers[i], phase_offset=phases[i])
            for i, root in enumerate(forest_roots):
                calc_absolute_positions(root, tree_positions[i])

        else:
            # --- 🟢 MODE IA ÉMULATEUR (Surrogate Deep Learning) ---
            stress_level = ai_preds["max_stress"]
            
            for i, root in enumerate(forest_roots):
                tree_exposure = multipliers[i]
                # Flexion dynamique bien visible et naturelle (~11 degrés max)
                tree_sway = math.sin(t * 2.2 + phases[i]) * 0.08 + (stress_level * 1.4 * tree_exposure)
                bend_angle = min(0.19, tree_sway)
                
                c_b, s_b = math.cos(bend_angle), math.sin(bend_angle)
                R_bend = np.array([
                    [c_b, 0, s_b * c_w],
                    [0, 1, s_b * s_w],
                    [-s_b, 0, c_b]
                ])
                R_tree = rot_y_up @ R_bend
                
                def apply_ai_flex(seg, R_parent, depth=0):
                    seg.absolute_R = R_parent @ seg.R_base
                    for ch in seg.children:
                        extra_sway = math.sin(t * 3.0 + depth * 0.8) * 0.03 if not ch.children else 0.01
                        R_extra = np.eye(3)
                        if extra_sway != 0:
                            R_extra[0, 0] = math.cos(extra_sway)
                            R_extra[0, 2] = math.sin(extra_sway)
                            R_extra[2, 0] = -math.sin(extra_sway)
                            R_extra[2, 2] = math.cos(extra_sway)
                        apply_ai_flex(ch, seg.absolute_R @ R_extra, depth + 1)
                        
                apply_ai_flex(root, R_tree)
                calc_absolute_positions(root, tree_positions[i])

        # Mise à jour des coordonnées géométriques
        for i, seg in enumerate(all_segments):
            points[i * 2] = seg.start_pos
            points[i * 2 + 1] = seg.end_pos

        if num_leaves > 0:
            leaf_points[:] = points[leaf_parent_indices * 2 + 1] + leaf_offsets

        # Rendu VTK
        mesh_branches.points = points
        new_tubes = mesh_branches.tube(scalars='thickness', absolute=True, n_sides=6)
        branches_actor.mapper.dataset = new_tubes

        mesh_leaves.points = leaf_points
        new_glyphs = mesh_leaves.glyph(geom=base_leaf, orient='orient', factor=1.0)
        leaf_actor.mapper.dataset = new_glyphs

        compute_duration = (time.perf_counter() - t0) * 1000.0
        state["compute_ms"] = 0.9 * state["compute_ms"] + 0.1 * compute_duration

        state["current_time"] += 0.033

        # Calcul FPS
        now = time.perf_counter()
        dt_frame = now - state["last_frame_time"]
        state["last_frame_time"] = now
        if dt_frame > 0:
            state["fps"] = 0.9 * state["fps"] + 0.1 * (1.0 / dt_frame)

        # Affichage HUD overlay
        mode_str = "🔴 PHYSIQUE CLASSIQUE (Solveur Euler)" if state["mode"] == "physics" else "🟢 ÉMULATEUR IA (Surrogate Deep Learning)"
        hud_text = (
            f"Mode Actif : {mode_str}\n"
            f"Arbres simulés : {num_trees} ({num_segments} branches)\n"
            f"Temps de calcul : {state['compute_ms']:.1f} ms / frame\n"
            f"Fluidité globale : {state['fps']:.0f} FPS\n"
            f"\n[Appuyez sur 'M' pour basculer instantanément de mode]"
        )
        plotter.add_text(hud_text, position="upper_left", font_size=11, color="black", shadow=False, name="hud_overlay")

    plotter.camera_position = 'yz'
    plotter.camera.focal_point = (0.0, 0.0, 3.5)
    plotter.camera.elevation = 18
    plotter.camera.azimuth = -25
    plotter.camera.zoom(1.2)

    # Boucle d'animation temps réel native
    plotter.show(interactive_update=True, auto_close=False)

    step = 0
    while not getattr(plotter, '_closed', False):
        animation_callback(step)
        plotter.update()
        step += 1
        time.sleep(0.01)


if __name__ == "__main__":
    main()
