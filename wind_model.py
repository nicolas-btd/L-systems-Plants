import numpy as np

def compute_wind_effects(tree_positions, wind_dir):
    num_trees = len(tree_positions)
    multipliers = np.ones(num_trees)
    
    wd = np.array([wind_dir[0], wind_dir[1], 0.0])
    wd_norm = np.linalg.norm(wd)
    if wd_norm > 1e-6:
        wd = wd / wd_norm
        
    SHIELDING_LENGTH = 35.0 # L'ombre porte plus loin
    SHIELDING_RADIUS = 2.5  # Ombre plus étroite (représente le tronc / cœur dense)
    MAX_SHIELD = 0.95       # Très forte protection si on est parfaitement aligné
    
    VENTURI_RADIUS = 8.0   # Zone de couloir où le vent accélère
    MAX_VENTURI = 1.6       # 60% d'accélération max dans le couloir
    
    for i, pi in enumerate(tree_positions):
        for j, pj in enumerate(tree_positions):
            if i == j: continue
            D = pi - pj
            dist_wind = np.dot(D, wd)
            
            # Si l'arbre i est DERRIÈRE l'arbre j (dans le sens du vent)
            if 0 < dist_wind < SHIELDING_LENGTH:
                D_perp = D - dist_wind * wd
                dist_perp = np.linalg.norm(D_perp)
                
                dist_decay = (1.0 - dist_wind / SHIELDING_LENGTH)
                
                # Zone 1 : Ombre aérodynamique (ralentissement)
                if dist_perp < SHIELDING_RADIUS:
                    shadow = dist_decay * (1.0 - dist_perp / SHIELDING_RADIUS)
                    multipliers[i] *= (1.0 - shadow * MAX_SHIELD)
                
                # Zone 2 : Effet Venturi (accélération dans les couloirs)
                elif SHIELDING_RADIUS <= dist_perp < VENTURI_RADIUS:
                    venturi_intensity = 1.0 - (dist_perp - SHIELDING_RADIUS) / (VENTURI_RADIUS - SHIELDING_RADIUS)
                    multipliers[i] *= (1.0 + venturi_intensity * dist_decay * (MAX_VENTURI - 1.0))
                    
    # Protection contre les cumuls aberrants
    multipliers = np.clip(multipliers, 0.1, 2.5)
    
    # Calcul des décalages de phase (effets de vague) pour la visualisation
    phase_offsets = np.zeros(num_trees)
    WIND_SPEED_M_S = 25.0 # Valeur de référence
    for i, pi in enumerate(tree_positions):
        phase_offsets[i] = np.dot(pi, wd) / WIND_SPEED_M_S
        
    return multipliers, phase_offsets
