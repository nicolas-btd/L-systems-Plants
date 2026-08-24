import numpy as np

def compute_wind_effects(tree_positions, wind_dir, tree_scales=None):
    num_trees = len(tree_positions)
    if tree_scales is None:
        tree_scales = np.ones(num_trees)
        
    multipliers = np.ones(num_trees)
    
    wd = np.array([wind_dir[0], wind_dir[1], 0.0])
    wd_norm = np.linalg.norm(wd)
    if wd_norm > 1e-6:
        wd = wd / wd_norm
        
    BASE_SHIELDING_LENGTH = 35.0 # L'ombre porte plus loin
    BASE_SHIELDING_RADIUS = 2.5  # Ombre plus étroite
    MAX_SHIELD = 0.95       # Très forte protection si on est parfaitement aligné
    
    BASE_VENTURI_RADIUS = 8.0   # Zone de couloir où le vent accélère
    MAX_VENTURI = 1.6       # 60% d'accélération max dans le couloir
    
    for i, pi in enumerate(tree_positions):
        for j, pj in enumerate(tree_positions):
            if i == j: continue
            
            # L'ombre et le couloir Venturi dépendent de la TAILLE de l'arbre qui fait face au vent
            scale_j = tree_scales[j]
            shield_len = BASE_SHIELDING_LENGTH * scale_j
            shield_rad = BASE_SHIELDING_RADIUS * scale_j
            venturi_rad = BASE_VENTURI_RADIUS * scale_j
            
            D = pi - pj
            dist_wind = np.dot(D, wd)
            
            # Si l'arbre i est DERRIÈRE l'arbre j (dans le sens du vent)
            if 0 < dist_wind < shield_len:
                D_perp = D - dist_wind * wd
                dist_perp = np.linalg.norm(D_perp)
                
                dist_decay = (1.0 - dist_wind / shield_len)
                
                # Zone 1 : Ombre aérodynamique (ralentissement)
                if dist_perp < shield_rad:
                    shadow = dist_decay * (1.0 - dist_perp / shield_rad)
                    multipliers[i] *= (1.0 - shadow * MAX_SHIELD)
                
                # Zone 2 : Effet Venturi (accélération dans les couloirs)
                elif shield_rad <= dist_perp < venturi_rad:
                    venturi_intensity = 1.0 - (dist_perp - shield_rad) / (venturi_rad - shield_rad)
                    multipliers[i] *= (1.0 + venturi_intensity * dist_decay * (MAX_VENTURI - 1.0))
                    
    # Protection contre les cumuls aberrants
    multipliers = np.clip(multipliers, 0.1, 2.5)
    
    # Calcul des décalages de phase (effets de vague) pour la visualisation
    phase_offsets = np.zeros(num_trees)
    WIND_SPEED_M_S = 25.0 # Valeur de référence
    for i, pi in enumerate(tree_positions):
        phase_offsets[i] = np.dot(pi, wd) / WIND_SPEED_M_S
        
    return multipliers, phase_offsets
