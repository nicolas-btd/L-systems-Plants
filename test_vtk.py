import pyvista as pv
import numpy as np

points = np.zeros((4, 3), dtype=np.float32)
points[:, 2] = np.arange(4)
lines = np.array([[2, 0, 1], [2, 1, 2], [2, 2, 3]])
thicknesses_point = np.array([0.1, 0.2, 0.3, 0.4])

mesh = pv.PolyData(points, lines=lines)
mesh.point_data['thickness'] = thicknesses_point

tubes = mesh.tube(scalars='thickness', radius=1.0, absolute=True)
print("Point data keys:", tubes.point_data.keys())
print("Cell data keys:", tubes.cell_data.keys())
print("Thickness array size in points:", len(tubes.point_data.get('thickness', [])))
