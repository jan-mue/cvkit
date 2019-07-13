import numpy as np


standard_projection = np.eye(3, 4)


def projection_matrix(sx=1, sy=1, s_theta=0, ox=0, oy=0, f=1):
    K = np.eye(3)
    K[0, 0] = f*sx
    K[1, 1] = f*sy
    K[0, 1] = f*s_theta
    K[:2, 2] = [ox, oy]
    return K.dot(standard_projection)
