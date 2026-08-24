# L-systems-Plants

3D simulation of procedural forests under wind stress, combining aeroelastic physics and neural surrogate modeling for real-time performance.

---

<div align="center">
  <img src="docs/img/forest.gif" alt="3D Forest Real-Time Simulation" width="750"/>
  <p><em>Real-time 3D simulation of dynamic forest canopies under wind waves (PyVista / VTK).</em></p>
</div>

---

## Overview & Goals

- **Procedural Botany**: Generating realistic branching structures and tree geometries using stochastic Lindenmayer Systems (L-Systems) and Murray's diameter scaling law ($r^{2.5}$).
- **Aeroelastic Physics**: Modeling branch dynamics and mechanical stress under turbulent wind forces using numerical integration.
- **Canopy Fluid Dynamics**: Simulating micro-climate interactions such as wake shielding and Venturi corridor effects across different forest layouts (e.g. grid vs. staggered plantations).
- **Real-Time AI Acceleration**: Replacing heavy differential equation solvers with a lightweight neural surrogate model (`ForestSurrogateNet`) to enable 60 FPS performance in interactive applications and game engines (~275x speedup).

---

## Technical Stack

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch, Scikit-Learn
- **3D Graphics & Rendering**: PyVista, VTK
- **Scientific Computing**: NumPy, Pandas, Matplotlib
- **Parallel Computing**: ProcessPoolExecutor (multiprocessing)

---

## Key Features

- **3D Procedural L-Systems**: Biologically inspired tree morphogenesis with 3D Rodrigues rotation frames and Murray's law.
- **Aeroelastic Engine**: Semi-implicit Euler integration per segment accounting for wood elasticity, inertia, and aerodynamic drag.
- **Fluid & Forest Modeling**: Wake propagation, directional turbulence, and Venturi acceleration through forest corridors.
- **Silvicultural Analysis**: Quantitative stress evaluation showing ~26% peak stress reduction in staggered (*quinconce*) layouts compared to regular grids, as well as the sheltering impact of progressive borders.
- **Neural Surrogate Model**: Deep residual network predicting structural stress in under 3 ms, making large-scale forest simulations feasible in real time.
- **Interactive 3D Viewer**: PyVista interface with real-time switching between numerical physics and neural surrogate using the `M` key.

---

## Development Steps

1. **2D Proof of Concept (`lsystem.py`, `physics.py`, `visualize_2d.py`)**:
   - Validation of string rewriting rules and 2D angular momentum torque dynamics on individual trees.
2. **3D Aeroelastic Model (`lsystem_3d.py`, `physics_3d.py`, `visualize_3d.py`)**:
   - Extension to 3D rotation frames, Murray's diameter scaling, and VTK-based rendering.
3. **Fluid Dynamics & Canopy Experiments (`wind_model.py`, `experiment_topology.py`, `experiment_lisiere.py`)**:
   - Wind wake propagation, channel acceleration, and topology comparisons.
4. **Surrogate Model & Real-Time Viewer (`ai_surrogate/`, `visualize_forest_comparison.py`)**:
   - Neural network training on synthetic simulation data and interactive comparison tool.

---

## Benchmark: Physics Solver vs. AI Surrogate

<div align="center">
  <img src="docs/img/surrogate_metrics.png" alt="Surrogate Model Metrics" width="750"/>
</div>

| Method | Mean Computation Time / Simulation | Speedup | Target Use Case |
| :--- | :--- | :--- | :--- |
| **Numerical 3D Solver (Euler)** | ~650 ms | $1\times$ (Baseline) | Offline scientific computation |
| **Surrogate Model (`ForestSurrogateNet`)** | **~2.3 ms** | **~275x** | **Interactive 3D, Games & VFX** |

---

## Quick Start

### Installation

```bash
git clone https://github.com/nicolas-btd/L-systems-Plants.git
cd L-systems-Plants
pip install -r requirements.txt
```

### Running the Project

```bash
# 1. Launch the interactive 3D viewer (Press 'M' to toggle between Physics and AI)
python visualize_forest_comparison.py --rows 8 --cols 8 --layout quinconce

# 2. Run the performance benchmark (Numerical Physics vs. AI)
python ai_surrogate/benchmark_inference.py

# 3. Run forestry experiments (Grid vs. Staggered topologies)
python experiment_topology.py
python experiment_lisiere.py

# 4. Train the AI surrogate network from synthetic physics data
python ai_surrogate/train.py
```
