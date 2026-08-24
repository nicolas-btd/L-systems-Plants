"""
Script d'entraînement et d'évaluation du modèle IA Surrogate (Deep Learning).
Calcule les métriques scientifiques (R2, RMSE, MAE) et génère les graphiques d'évaluation.
"""

import os
import sys
import json
import time

# Assure l'accès aux modules du projet racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
except ImportError as e:
    print(f"Erreur d'import : {e}. Assurez-vous d'avoir installé torch, scikit-learn et pandas.")

from ai_surrogate.model import ForestSurrogateNet, FEATURE_COLUMNS, TARGET_COLUMNS, extract_physics_features
from ai_surrogate.dataset_generator import generate_dataset


def train_surrogate(
    dataset_path="ai_surrogate/dataset_forest_physics.csv",
    epochs=150,
    batch_size=32,
    lr=1e-3,
    save_dir="ai_surrogate",
    plot_results=True
):
    """
    Entraîne le réseau de neurones Surrogate sur le dataset de physique.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Chargement ou génération des données
    if not os.path.exists(dataset_path):
        print(f"[!] Dataset non trouvé à '{dataset_path}'. Génération automatique...")
        df = generate_dataset(num_samples=400, output_path=dataset_path)
    else:
        df = pd.read_csv(dataset_path)
        print(f"[OK] Chargement du dataset existant : {len(df)} échantillons.")

    # Physics-Informed Feature Engineering
    df_feat = extract_physics_features(df)
    X_raw = df_feat.values.astype(np.float32)
    y_raw = df[TARGET_COLUMNS].values.astype(np.float32)

    # 2. Séparation Train / Test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    # 3. Normalisation des données (StandardScaler)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_norm = scaler_X.fit_transform(X_train)
    X_test_norm = scaler_X.transform(X_test)
    
    y_train_norm = scaler_y.fit_transform(y_train)
    y_test_norm = scaler_y.transform(y_test)

    # Sauvegarde des paramètres du scaler pour l'inférence
    scaler_params = {
        "mean_X": scaler_X.mean_.tolist(),
        "scale_X": scaler_X.scale_.tolist(),
        "mean_y": scaler_y.mean_.tolist(),
        "scale_y": scaler_y.scale_.tolist(),
        "feature_names": FEATURE_COLUMNS,
        "target_names": TARGET_COLUMNS
    }
    with open(os.path.join(save_dir, "scaler_params.json"), "w") as f:
        json.dump(scaler_params, f, indent=4)

    # 4. DataLoaders PyTorch
    train_dataset = TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32), torch.tensor(y_train_norm, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test_norm, dtype=torch.float32), torch.tensor(y_test_norm, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5. Instanciation du modèle, fonction de perte et optimiseur
    model = ForestSurrogateNet(
        in_features=len(FEATURE_COLUMNS),
        hidden_dim=128,
        out_features=len(TARGET_COLUMNS),
        num_blocks=3
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_losses = []

    print("\n=== Entraînement du Réseau de Neurones Surrogate (Physics-Informed) ===")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(bx)
        scheduler.step()

        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in test_loader:
                pred = model(bx)
                loss = criterion(pred, by)
                val_loss += loss.item() * len(bx)
        val_loss /= len(test_dataset)
        val_losses.append(val_loss)

        if epoch % 25 == 0 or epoch == epochs:
            print(f"Époque [{epoch:3d}/{epochs}] | Train MSE : {epoch_loss:.4f} | Val MSE : {val_loss:.4f}")

    train_duration = time.time() - t_start
    print(f"\n[OK] Entraînement achevé en {train_duration:.2f} s.")

    # 6. Évaluation finale dénormalisée
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
        y_pred_norm = model(X_test_tensor).numpy()
        y_pred = scaler_y.inverse_transform(y_pred_norm)

    print("\n" + "="*55)
    print("           PERFORMANCES SCIENTIFIQUES DU MODÈLE IA")
    print("="*55)
    
    r2_scores = []
    for i, target_name in enumerate(TARGET_COLUMNS):
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / (y_test[:, i] + 1e-6))) * 100
        r2_scores.append(r2)
        print(f"Cible : {target_name.upper():12s} | R² = {r2:.4f} | RMSE = {rmse:.2f} N.m | MAE = {mae:.2f} N.m | Erreur Moyenne = {mape:.2f}%")
    print("="*55)

    # 7. Sauvegarde des poids du modèle
    model_weights_path = os.path.join(save_dir, "forest_surrogate_weights.pt")
    torch.save(model.state_dict(), model_weights_path)
    print(f"[OK] Poids du modèle sauvegardés dans : {model_weights_path}")

    # 8. Graphiques d'évaluation scientifique (Parity Plot & Loss)
    if plot_results:
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        
        # Courbe d'apprentissage
        axs[0].plot(train_losses, label="Perte Entraînement (MSE)", color="#2563EB", lw=2)
        axs[0].plot(val_losses, label="Perte Validation", color="#DC2626", lw=2, linestyle="--")
        axs[0].set_title("Convergence de l'Apprentissage (Surrogate Model)", fontsize=12, fontweight="bold")
        axs[0].set_xlabel("Époques")
        axs[0].set_ylabel("Perte MSE (normalisée)")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()

        # Parity Plot
        idx_target = TARGET_COLUMNS.index("max_stress")
        y_true_max = y_test[:, idx_target]
        y_pred_max = y_pred[:, idx_target]

        min_val = min(y_true_max.min(), y_pred_max.min())
        max_val = max(y_true_max.max(), y_pred_max.max())

        axs[1].scatter(y_true_max, y_pred_max, color="#059669", alpha=0.7, edgecolors="none", s=40, label="Données Test")
        axs[1].plot([min_val, max_val], [min_val, max_val], "k--", lw=2, label="Prédiction Parfaite (Idéale)")
        axs[1].set_title(f"Parity Plot : Stress Maximum (R² = {r2_scores[idx_target]:.4f})", fontsize=12, fontweight="bold")
        axs[1].set_xlabel("Physique Réelle (Moteur 3D pas à pas) [N.m]")
        axs[1].set_ylabel("Prédiction IA (Surrogate Model) [N.m]")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()

        plt.tight_layout()
        metrics_plot_path = os.path.join(save_dir, "surrogate_metrics.png")
        plt.savefig(metrics_plot_path, dpi=300)
        plt.close()
        print(f"[OK] Graphique des métriques enregistré dans : {metrics_plot_path}")

    return model, scaler_params, r2_scores


if __name__ == "__main__":
    train_surrogate(epochs=150, batch_size=32)
