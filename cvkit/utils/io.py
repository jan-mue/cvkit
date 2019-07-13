from geometer import Polyhedron
import meshio
import numpy as np


def load_off(filename):
    mesh = meshio.read(filename)
    vertices = np.append(mesh.points, np.ones((len(mesh.points), 1)), axis=1)
    return Polyhedron(vertices[mesh.cells["triangle"]])
