import numpy as np
from geometer import Point
from geometer.utils import hat_matrix
import scipy
from scipy.optimize import least_squares

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


class Camera:

    def __init__(self, intrinsic_matrix=np.eye(3)):
        self.intrinsic_matrix = np.asarray(intrinsic_matrix)

    @property
    def position(self):
        return Point(0, 0, 0)

    def calibrate(self, img):
        raise NotImplemented()

    def project(self, points):
        points = np.append(points, np.ones((len(points), 1)), axis=1)
        return points @ standard_projection.T @ self.intrinsic_matrix.T


class Scene:
    """Class implementing the 8-point algorithm for stereo reconstruction, a reconstruction algorithm for
    reconstruction from multiple views and a bundle adjustment algorithm.

    """

    def __init__(self):
        self._images = []
        self._points = []

    def add(self, camera, image, points=None):
        points = np.append(points, np.ones((len(points), 1)), axis=1)
        points = points @ np.linalg.inv(camera.intrinsic_matrix).T
        self._images.append((camera, image))

        # TODO: implement feature matching if no points given
        self._points.append(points)

    def _bundle_adjustment(self, camera_motion, X0):

        n = X0.shape[0]

        def E(x):
            result = []
            i = 0
            X = x[-3*n:].reshape((n, 3))

            for (c, img), img_points in zip(self._images, self._points):
                R = x[i:i+9].reshape((3, 3))
                T = x[i+9:i+12]

                weights = np.all(~np.isnan(img_points), axis=1)
                result.append(np.linalg.norm(img_points[weights] - c.project(X[weights] @ R.T + T), axis=1))

                i += 12

            return np.concatenate(result)

        x0 = np.concatenate([a for R, T in camera_motion for a in (np.ravel(R), T)] + [np.ravel(X0)])
        opt = least_squares(E, np.nan_to_num(x0), method="lm" if n*len(camera_motion) >= 12*len(camera_motion)+3*n else "trf")
        return opt.x[-3*n:].reshape((n, 3))

    @staticmethod
    def _estimate_structure_stereo(R, T, points1, points2):
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

        return x1.T

    def _reconstruct_stereo(self, points1, points2):
        E = essential_matrix(points1, points2)
        R1, R2, T1, T2 = decompose_essential_matrix(E)

        try:
            return R1, T1, self._estimate_structure_stereo(R1, T1, points1, points2)
        except ValueError:
            pass

        try:
            return R1, T2, self._estimate_structure_stereo(R1, T2, points1, points2)
        except ValueError:
            pass

        try:
            return R2, T1, self._estimate_structure_stereo(R2, T1, points1, points2)
        except ValueError:
            pass

        try:
            return R2, T2, self._estimate_structure_stereo(R2, T2, points1, points2)
        except ValueError:
            pass

        raise ValueError("Reconstruction failed")

    @staticmethod
    def _estimate_structure(camera_motion, points):
        lambda1 = []
        for j, x1j in enumerate(points[0]):
            a = 0
            b = 0
            for xi, (Ri, Ti) in zip(points, camera_motion):
                a += (hat_matrix(*xi[j]) @ Ti).T @ hat_matrix(*xi[j]) @ Ri @ x1j
                b += np.sum((hat_matrix(*xi[j]) @ Ti) ** 2)

            lambda1.append(b / a)

        lambda1 = np.asarray(lambda1)

        if np.all(lambda1 < 0):
            lambda1 *= -1

        return (lambda1 * points[0].T).T

    @staticmethod
    def _estimate_motion(X, points):
        motion = []
        for img_points in points[2:]:
            row1 = [np.kron(x1, hat_matrix(*xi)).reshape((3, 9)).T for x1, xi in zip(points[0], img_points)]
            row2 = [alpha * hat_matrix(*xi).T for alpha, xi in zip(X[:, -1], img_points)]
            P = np.block([row1, row2]).T

            u, s, vh = np.linalg.svd(P)
            R = vh[-1, :9].reshape((3, 3))
            T = vh[-1, 9:]

            motion.append((R, T))

        return motion

    def _reconstruct_multiple_views(self, points):
        R, T, X = self._reconstruct_stereo(*points[:2])

        camera_motion = [(np.eye(3), np.zeros(3)), (R, T)]

        if len(points) == 2:
            return camera_motion, X

        camera_motion.extend(self._estimate_motion(X, points))
        X = self._estimate_structure(camera_motion, points)

        return camera_motion, X

    def reconstruct(self):
        points = np.concatenate(self._points, axis=1)
        visible_count = np.sum(~np.isnan(points), axis=1)
        visible_first_image = np.isnan(points).argmin(axis=1)

        result = np.full(self._points[0].shape, np.nan)

        for i in np.unique(visible_first_image):
            for j in np.unique(visible_count):
                if j <= 3:
                    continue

                ind = (visible_first_image == i) & (visible_count == j)

                if np.sum(ind) < 8:
                    continue

                x = np.split(points[ind, i:i+j], j/3, axis=1)

                try:
                    motion_new, x_new = self._reconstruct_multiple_views(x)
                except ValueError:
                    continue

                result[ind] = x_new

                i = int(i / 3)
                if i == 0:
                    camera_motion = motion_new
                elif i + len(motion_new) >= len(camera_motion):
                    Ri, Ti = camera_motion[i] if i < len(camera_motion) else camera_motion[-1]
                    camera_motion = camera_motion[:i] + [(R @ Ri, T + Ti) for R, T in motion_new]

        # return self._bundle_adjustment(camera_motion, X)
        return result
