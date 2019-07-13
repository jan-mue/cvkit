import numpy as np
from geometer import Point
from geometer.utils import hat_matrix
import scipy

from .projection import standard_projection


def essential_matrix(points1, points2):
    chi = [np.kron(p1, p2).flatten() for p1, p2 in zip(points1, points2)]
    _, _, vh = np.linalg.svd(chi)

    E = vh[-1, :9].reshape((3, 3))

    u, _, vh = np.linalg.svd(E)
    s = np.diag([1, 1, 0])

    if np.linalg.det(u) < 0:
        u = -u

    if np.linalg.det(vh) < 0:
        vh = -vh

    return vh.T @ s @ u.T


def decompose_essential_matrix(E):
    u, s, vh = np.linalg.svd(E)

    s = np.diag(s)

    Rz1 = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]])

    Rz2 = np.array([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])

    R1 = u @ Rz1 @ vh
    T_hat1 = u @ Rz1 @ s @ u.T

    R2 = u @ Rz2 @ vh
    T_hat2 = u @ Rz2 @ s @ u.T

    T1 = T_hat1[[2, 0, 1], [1, 2, 0]]
    T2 = T_hat2[[2, 0, 1], [1, 2, 0]]

    return R1, R2, T1, T2


def reconstruct(R, T, points1, points2):
    M = scipy.linalg.block_diag(*[np.reshape(hat_matrix(*x2) @ R @ x1, (3, 1)) for x1, x2 in zip(points1, points2)])
    M = np.append(M, np.concatenate([hat_matrix(*x) @ T for x in points2]).reshape((-1, 1)), axis=1)

    w, v = np.linalg.eigh(M.T.dot(M))

    lambda1 = v[:-1, 0]
    gamma = v[-1, 0]

    if gamma < 0:
        lambda1 *= -1
        gamma *= -1

    if np.any(lambda1 < 0):
        raise ValueError("Invalid camera parameters.")

    lambda1 /= gamma

    x1 = lambda1 * points1.T
    x2 = (R @ x1).T + T
    lambda2 = x2[:, -1]

    if np.any(lambda2 < 0):
        raise ValueError("Invalid camera parameters.")

    return x1


class Camera:

    def __init__(self, image, intrinsic_matrix=np.eye(3)):
        self.image = image
        self.intrinsic_matrix = intrinsic_matrix

    @property
    def position(self):
        return Point(0, 0, 0)

    def calibrate(self, img):
        raise NotImplemented()

    def project(self, points):
        points = np.asarray(points)
        return points @ standard_projection.T @ self.intrinsic_matrix.T


class Scene:
    """Class implementing the 8-point algorithm.

    """

    def __init__(self):
        self._cameras = []
        self._points = []

    def add(self, camera, points):
        points = np.append(points, np.ones((len(points), 1)), axis=1)
        points = points @ np.linalg.inv(camera.intrinsic_matrix).T
        self._cameras.append(camera)
        self._points.append(points)

    def reconstruct(self):
        points1, points2 = self._points
        E = essential_matrix(points1, points2)
        R1, R2, T1, T2 = decompose_essential_matrix(E)

        try:
            return reconstruct(R1, T1, points1, points2)
        except ValueError:
            pass

        try:
            return reconstruct(R1, T2, points1, points2)
        except ValueError:
            pass

        try:
            return reconstruct(R2, T1, points1, points2)
        except ValueError:
            pass

        try:
            return reconstruct(R2, T2, points1, points2)
        except ValueError:
            pass

        raise ValueError("Reconstruction failed")
