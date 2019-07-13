import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.collections import PolyCollection


def display_polhedron(poly):
    fig = plt.figure()
    axes = mplot3d.Axes3D(fig)

    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(poly.array[:, :, :-1]))
    plt.show()


def display_polygons(polys):

    fig, ax = plt.subplots()

    patches = [(p.array[:, :-1].T / p.array[:, -1]).T for p in polys]
    patches = PolyCollection(patches)

    ax.add_collection(patches)
    ax.autoscale()
    plt.show()


def plot_conics(conics):
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    args = np.array([x, y, np.ones_like(y)])
    x, y = np.meshgrid(x, y)
    plt.axhline(0, alpha=.1)
    plt.axvline(0, alpha=.1)

    for c in conics:
        plt.contour(x, y, args.T.dot(c.array).dot(args), [0], colors='k')

    plt.show()
