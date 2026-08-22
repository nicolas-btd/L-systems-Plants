import pyvista as pv
import numpy as np
import vtk

points = np.array([[0,0,0], [1,1,1]], dtype=float)
mesh = pv.PolyData(points)
mesh['orient'] = np.array([[1,0,0], [0,1,0]], dtype=float)
mesh.active_vectors_name = 'orient'

leaf = pv.Sphere(radius=0.1)

mapper = vtk.vtkGlyph3DMapper()
mapper.SetInputData(mesh)
mapper.SetSourceData(leaf)
mapper.SetOrientationModeToDirection()

actor = vtk.vtkActor()
actor.SetMapper(mapper)

print("SUCCESS")
