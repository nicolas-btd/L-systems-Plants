import numpy as np
import math

def get_rotation_matrix(rotvec):
    """
    Convertit un vecteur de rotation (vecteur d'Euler) en matrice de rotation 3x3
    via la formule de Rodrigues.
    """
    theta = np.linalg.norm(rotvec)
    if theta < 1e-6:
        return np.eye(3)
    axis = rotvec / theta
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

class PhysicsEngine3D:
    """
    Moteur physique 3D pour l'arbre. Utilise l'intégration d'Euler semi-implicite.
    Les calculs de dynamique (Moments) sont effectués dans le repère LOCAL de chaque segment.
    """
    def __init__(self, dt=0.01):
        self.dt = dt

    def update_kinematics(self, segment, parent_R_abs=None):
        """Met à jour les matrices de rotation absolues de la hiérarchie."""
        if parent_R_abs is None:
            parent_R_abs = np.eye(3)
            
        # R_deflect est la matrice de déformation dynamique (liée à theta)
        R_deflect = get_rotation_matrix(segment.theta)
        
        # Rotation locale totale par rapport au parent
        R_local = segment.R_base @ R_deflect
        
        # Rotation absolue
        segment.absolute_R = parent_R_abs @ R_local
        
        for child in segment.children:
            self.update_kinematics(child, segment.absolute_R)

    def compute_torque(self, segment, current_time, wind_params):
        """Calcule le couple total sur un segment, exprimé dans son REPÈRE LOCAL."""
        # 1. Couple de Rappel (Raideur locale)
        restoring_torque = -segment.stiffness * segment.theta
        
        # 2. Couple du vent
        # H_abs est la direction absolue de la branche (1ère colonne de absolute_R)
        H_abs = segment.absolute_R[:, 0]
        
        wind_speed = wind_params["wind_speed"]
        wind_dir = np.array(wind_params["wind_dir"])
        wind_dir = wind_dir / np.linalg.norm(wind_dir)
        
        # Cycle de vent naturel (vent de base + rafales superposées)
        freq = wind_params["wind_frequency"]
        t = current_time
        # Des bourrasques espacées par des moments de calme plat (vent = 0)
        # Le 'max(0.0, ...)' annule le vent lors des phases négatives du sinus.
        # L'arbre va donc naturellement revenir à sa position statique et s'y stabiliser.
        base_wind = math.sin(2 * math.pi * freq * t) + 0.5 * math.sin(2 * math.pi * (freq * 2.37) * t)
        gust = max(0.0, base_wind * 1.2)
        
        # On utilise une échelle non linéaire pour la force du vent (aéroélasticité simulée)
        # Cela évite que les brindilles (qui ont une très faible rigidité) ne soient arrachées, 
        # tout en poussant modérément le tronc
        force_mag = wind_speed * gust * (segment.thickness ** 3)
        
        # Surface de prise au vent (projetée via l'angle)
        cross_prod = np.cross(H_abs, wind_dir)
        effective_area = np.linalg.norm(cross_prod)
        if getattr(segment, 'has_leaf', False):
            effective_area *= 1.5 # Prise au vent de la feuille (réduite pour éviter les torsions)
            
        # Force absolue du vent (plus besoin d'atténuation artificielle selon la profondeur)
        F_wind_abs = force_mag * effective_area * wind_dir
        
        # Bras de levier absolu
        r_abs = (segment.length / 2.0) * H_abs
        
        # Couple absolu = r x F
        wind_torque_abs = np.cross(r_abs, F_wind_abs)
        
        # Passage du couple dans le repère local du segment
        wind_torque_local = segment.absolute_R.T @ wind_torque_abs
        
        # 3. Couplage avec les enfants (Action/Réaction)
        coupling_torque = np.zeros(3)
        for child in segment.children:
            # Le couple exercé par l'enfant est +k*theta_enfant dans le repère de l'enfant
            child_torque_local = child.stiffness * child.theta
            # On le ramène dans le repère du parent
            coupling_torque += child.R_base @ child_torque_local
            
        return restoring_torque + wind_torque_local + coupling_torque

    def update_segment(self, segment, current_time, wind_params):
        if getattr(segment, 'is_kinematic', False):
            # Le tronc est un bloc de béton : il refuse de plier (theta=0)
            segment.theta = np.zeros(3)
            segment.omega = np.zeros(3)
        else:
            total_torque = self.compute_torque(segment, current_time, wind_params)
            
            # PFD vectoriel : alpha = Tau / I
            angular_accel = total_torque / segment.inertia
            
            # Euler semi-implicite vectoriel
            damping_factor = segment.damping / segment.inertia
            segment.omega = (segment.omega + angular_accel * self.dt) / (1.0 + damping_factor * self.dt)
            segment.theta += segment.omega * self.dt
        
        for child in segment.children:
            self.update_segment(child, current_time, wind_params)

    def step(self, roots, current_time, wind_params):
        # 1. Cinématique
        for root in roots:
            self.update_kinematics(root)
            
        # 2. Dynamique
        for root in roots:
            self.update_segment(root, current_time, wind_params)
