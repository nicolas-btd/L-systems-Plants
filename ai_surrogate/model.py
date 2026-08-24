"""
Architecture du Réseau de Neurones Surrogate (ResMLP) à guide physique (Physics-Guided ML).
Intègre l'encodage de la traînée aérodynamique quadratique (v^2), des projections trigonométriques et de la densité de peuplement.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object


if HAS_TORCH:
    class ResidualBlock(nn.Module):
        """Bloc résiduel avec Normalisation et activation non-linéaire GELU."""
        def __init__(self, dim):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim)
            )
            self.act = nn.GELU()

        def forward(self, x):
            return self.act(x + self.block(x))


    class ForestSurrogateNet(nn.Module):
        """
        Réseau Multi-Layer Perceptron (MLP) profond à connexions résiduelles.
        Prédit instantanément les contraintes mécaniques [mean_stress, max_stress, std_stress].
        """
        def __init__(self, in_features=14, hidden_dim=128, out_features=3, num_blocks=3):
            super().__init__()
            self.in_proj = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            
            self.blocks = nn.ModuleList([
                ResidualBlock(hidden_dim) for _ in range(num_blocks)
            ])
            
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, out_features)
            )

        def forward(self, x):
            h = self.in_proj(x)
            for block in self.blocks:
                h = block(h)
            return self.head(h)

else:
    class ForestSurrogateNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch n'est pas installé. Exécutez 'pip install torch'.")


# Features d'entrée et cibles physiques
RAW_FEATURE_COLUMNS = [
    "layout_is_quinconce",
    "rows",
    "cols",
    "num_trees",
    "spacing_x",
    "spacing_y",
    "wind_speed",
    "wind_angle_deg",
    "wind_oscillation_deg",
    "wind_frequency",
    "progressive_edge"
]

FEATURE_COLUMNS = [
    "layout_is_quinconce",
    "rows",
    "cols",
    "num_trees",
    "spacing_x",
    "spacing_y",
    "wind_speed",
    "wind_speed_sq",       # Traînée aérodynamique quadratique (v^2)
    "wind_cos",            # Projection vectorielle en X
    "wind_sin",            # Projection vectorielle en Y
    "wind_oscillation_deg",
    "wind_frequency",
    "progressive_edge",
    "forest_density"       # Densité surfacique de plantation
]

TARGET_COLUMNS = [
    "mean_stress",
    "max_stress",
    "std_stress"
]


def extract_physics_features(df):
    """
    Enrichit le DataFrame avec des descripteurs physiques explicites (Physics-Guided Feature Engineering).
    """
    df_feat = df.copy()
    angles_rad = np.radians(df_feat["wind_angle_deg"].values)
    df_feat["wind_speed_sq"] = df_feat["wind_speed"] ** 2
    df_feat["wind_cos"] = np.cos(angles_rad)
    df_feat["wind_sin"] = np.sin(angles_rad)
    
    # Surface totale de la parcelle
    surface = (df_feat["cols"] * df_feat["spacing_x"]) * (df_feat["rows"] * df_feat["spacing_y"])
    df_feat["forest_density"] = df_feat["num_trees"] / (surface + 1e-6)
    
    return df_feat[FEATURE_COLUMNS]
