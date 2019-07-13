import numpy as np


def gaussian2d(x, mu=0, sigma=1):
    x = np.asarray(x)
    y = x - mu
    return np.exp(-1/(2*sigma**2)*(y[0]**2 + y[1]**2)) / (2*np.pi * sigma**2)
