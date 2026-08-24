"""
Benchmark comparatif de performance et d'accélération :
Moteur Physique Numérique 3D (Euler) vs Émulateur IA (Surrogate Model).
"""

import os
import sys
import json
import time
import math

# Assure l'accès aux modules du projet racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from ai_surrogate.model import ForestSurrogateNet, FEATURE_COLUMNS, TARGET_COLUMNS, extract_physics_features
from ai_surrogate.dataset_generator import run_single_simulation


class ForestSurrogatePredictor:
    """
    Module d'inférence ultra-rapide pour l'évaluation en temps réel de forêts.
    """
    def __init__(self, weights_path="ai_surrogate/forest_surrogate_weights.pt", scaler_path="ai_surrogate/scaler_params.json"):
        with open(scaler_path, "r") as f:
            self.scaler_params = json.load(f)
            
        self.mean_X = np.array(self.scaler_params["mean_X"], dtype=np.float32)
        self.scale_X = np.array(self.scaler_params["scale_X"], dtype=np.float32)
        self.mean_y = np.array(self.scaler_params["mean_y"], dtype=np.float32)
        self.scale_y = np.array(self.scaler_params["scale_y"], dtype=np.float32)
        
        self.model = ForestSurrogateNet(
            in_features=len(FEATURE_COLUMNS),
            hidden_dim=128,
            out_features=len(TARGET_COLUMNS),
            num_blocks=3
        )
        self.model.load_state_dict(torch.load(weights_path, weights_only=True))
        self.model.eval()

    def predict(self, scenario_dict):
        """Prédit instantanément [mean_stress, max_stress, std_stress] à partir d'un dictionnaire."""
        df_single = pd.DataFrame([scenario_dict])
        df_feat = extract_physics_features(df_single)
        features = df_feat.values[0].astype(np.float32)
        features_norm = (features - self.mean_X) / self.scale_X
        
        with torch.no_grad():
            x_tensor = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0)
            preds_norm = self.model(x_tensor).squeeze(0).numpy()
            preds = preds_norm * self.scale_y + self.mean_y
            
        return {
            "mean_stress": float(max(0.0, preds[0])),
            "max_stress": float(max(0.0, preds[1])),
            "std_stress": float(max(0.0, preds[2]))
        }

    def predict_batch(self, df_features):
        """Prédit un lot de scénarios en une seule passe matricielle."""
        df_feat = extract_physics_features(df_features)
        features = df_feat.values.astype(np.float32)
        features_norm = (features - self.mean_X) / self.scale_X
        with torch.no_grad():
            x_tensor = torch.tensor(features_norm, dtype=torch.float32)
            preds_norm = self.model(x_tensor).numpy()
            preds = preds_norm * self.scale_y + self.mean_y
        return np.maximum(0.0, preds)


def run_speed_and_accuracy_benchmark(num_scenarios=15):
    print("\n" + "="*65)
    print(f"   BENCHMARK SCIENTIFIQUE : MOTEUR PHYSIQUE 3D vs ÉMULATEUR IA")
    print("="*65)
    
    predictor = ForestSurrogatePredictor()
    
    test_scenarios = []
    np.random.seed(123)
    
    for _ in range(num_scenarios):
        layout = np.random.choice(["grid", "quinconce"])
        sc = {
            "layout_type": layout,
            "layout_is_quinconce": 1 if layout == "quinconce" else 0,
            "rows": int(np.random.choice([3, 4, 5])),
            "cols": int(np.random.choice([3, 4, 5])),
            "spacing_x": float(np.random.uniform(4.5, 8.5)),
            "spacing_y": float(np.random.uniform(4.5, 8.5)),
            "wind_speed": float(np.random.uniform(20.0, 45.0)),
            "wind_angle_deg": float(np.random.uniform(0.0, 360.0)),
            "wind_oscillation_deg": float(np.random.uniform(15.0, 40.0)),
            "wind_frequency": float(np.random.uniform(0.3, 0.7)),
            "progressive_edge": bool(np.random.choice([True, False]))
        }
        sc["num_trees"] = sc["rows"] * sc["cols"]
        test_scenarios.append(sc)
        
    physics_results = []
    physics_times = []
    
    print(f"\n1. Exécution de {num_scenarios} simulations avec le Moteur Physique Numérique 3D...")
    for sc in test_scenarios:
        t0 = time.perf_counter()
        res = run_single_simulation(
            layout_type=sc["layout_type"],
            rows=sc["rows"],
            cols=sc["cols"],
            spacing_x=sc["spacing_x"],
            spacing_y=sc["spacing_y"],
            wind_speed=sc["wind_speed"],
            wind_angle_deg=sc["wind_angle_deg"],
            wind_oscillation_deg=sc["wind_oscillation_deg"],
            wind_frequency=sc["wind_frequency"],
            progressive_edge=sc["progressive_edge"],
            sim_steps=30,
            dt=0.05
        )
        t_elapsed = time.perf_counter() - t0
        physics_times.append(t_elapsed)
        physics_results.append(res)
        
    ai_results = []
    ai_times = []
    
    print(f"2. Exécution de {num_scenarios} inférences avec l'Émulateur IA (Surrogate Model)...")
    for sc in test_scenarios:
        sc_dict = sc.copy()
        sc_dict["progressive_edge"] = 1 if sc["progressive_edge"] else 0
        t0 = time.perf_counter()
        res = predictor.predict(sc_dict)
        t_elapsed = time.perf_counter() - t0
        ai_times.append(t_elapsed)
        ai_results.append(res)

    total_physics_time = sum(physics_times)
    total_ai_time = sum(ai_times)
    speedup = total_physics_time / (total_ai_time + 1e-9)

    print("\n" + "="*65)
    print("                     RÉSULTATS DU BENCHMARK")
    print("="*65)
    print(f"Temps moyen / simulation Physique : {np.mean(physics_times)*1000:.2f} ms")
    print(f"Temps moyen / inférence IA        : {np.mean(ai_times)*1000:.4f} ms")
    print(f"-> FACTEUR D'ACCÉLÉRATION IA      : x{speedup:.1f} PLUS RAPIDE !")
    print("="*65)

    # Affichage d'un échantillon comparatif
    print("\nÉchantillon de comparaison directe (Physique vs IA) :")
    print(f"{'Scénario':<10} | {'Vent (m/s)':<10} | {'Max Stress (Physique)':<22} | {'Max Stress (IA)':<16} | {'Erreur':<8}")
    print("-" * 75)
    for i in range(min(5, num_scenarios)):
        p_max = physics_results[i]["max_stress"]
        ai_max = ai_results[i]["max_stress"]
        rel_err = abs(p_max - ai_max) / (p_max + 1e-6) * 100
        w_spd = test_scenarios[i]["wind_speed"]
        print(f"#{i+1:<9} | {w_spd:<10.1f} | {p_max:<22.2f} | {ai_max:<16.2f} | {rel_err:.1f}%")
    print("="*75)


if __name__ == "__main__":
    run_speed_and_accuracy_benchmark(num_scenarios=15)
