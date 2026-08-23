"""
Moteur physique pour la simulation de l'action du vent sur un arbre.

Il utilise une approche par intégration d'Euler explicite pour calculer
les forces (raideur, frottements, vent) s'appliquant sur chaque segment (noeud)
de l'arbre généré par L-System.
"""
import math

class PhysicsEngine:
    def __init__(self, dt=0.01):
        self.dt = dt

    def get_wind_force(self, segment, current_time, wind_speed, wind_frequency, gust_duration):
        """
        Calcule la force du vent s'appliquant sur un segment donné à un instant t.
        Le vent pousse horizontalement. Plus le segment est vertical, plus il offre de prise au vent.
        """
        # Cycle de rafale (ex: actif pendant gust_duration, puis repos)
        cycle_time = current_time % (gust_duration * 3)
        if cycle_time < gust_duration:
                                                abs_angle = segment.get_absolute_angle()
            
                                    effective_area = math.cos(abs_angle)
            if getattr(segment, 'has_leaf', False):
                effective_area *= 5.0 # Les feuilles offrent une grande prise au vent
            
            # Atténuation simplifiée en fonction de la profondeur de la branche
            depth = 1
            curr = segment.parent
            while curr is not None:
                depth += 1
                curr = curr.parent
            attenuation = 1.0 / math.sqrt(depth)
            
            # Force oscillante
            force = wind_speed * effective_area * math.sin(2 * math.pi * wind_frequency * current_time) * attenuation
            return force
        return 0.0

    def compute_torque(self, segment, current_time, wind_params):
        """
        Calcule la somme des moments des forces (torques) s'appliquant sur un segment.
        """
        # 1. Force de rappel (Raideur) : tend à ramener le segment à son angle de base
        # M_rappel = - C1 * (theta - base_angle)
        restoring_torque = -segment.stiffness * (segment.theta - segment.base_angle)
        
        # 2. (L'amortissement est maintenant géré de manière implicite dans update_segment pour la stabilité)
        
        # 3. Action du vent
        wind_force = self.get_wind_force(segment, current_time, **wind_params)
        # Moment du vent = Force * Bras de levier (approximé à la longueur du segment / 2)
        wind_torque = wind_force * (segment.length / 2.0)
        
        # 4. Couplage avec les enfants (3ème loi de Newton : Action/Réaction)
        # L'enfant exerce sur le parent le couple opposé à son propre couple de rappel
        coupling_torque = 0.0
        for child in segment.children:
            coupling_torque += child.stiffness * (child.theta - child.base_angle)
            
        return restoring_torque + wind_torque + coupling_torque

    def update_segment(self, segment, current_time, wind_params):
        """Met à jour la physique d'un segment via Euler."""
        # Calcul des moments
        total_torque = self.compute_torque(segment, current_time, wind_params)
        
        # PFD (Moment Cinétique) : I * alpha = Somme(Moments)
        # alpha = accélération angulaire (sans l'amortissement)
        angular_accel = total_torque / segment.inertia
        
        # Intégration d'Euler semi-implicite avec amortissement implicite
                damping_factor = segment.damping / segment.inertia
        segment.omega = (segment.omega + angular_accel * self.dt) / (1.0 + damping_factor * self.dt)
        segment.theta += segment.omega * self.dt
        
        # Mise à jour récursive des enfants
        for child in segment.children:
            self.update_segment(child, current_time, wind_params)

    def step(self, roots, current_time, wind_params):
        """Fait avancer la simulation d'un pas de temps `dt`."""
        for root in roots:
            self.update_segment(root, current_time, wind_params)
