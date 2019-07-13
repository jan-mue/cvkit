from cvkit.utils import gaussian2d
from scipy.stats import multivariate_normal
import numpy as np


def test_gaussian():
    var = multivariate_normal([2, 2], 4*np.eye(2))

    assert np.isclose(var.pdf([3, 1]), gaussian2d([3, 1], 2, 2))
    assert np.isclose(var.pdf([4, 2]), gaussian2d([4, 2], 2, 2))
    assert np.isclose(var.pdf([5, 3]), gaussian2d([5, 3], 2, 2))
    assert np.isclose(var.pdf([-3, 3]), gaussian2d([-3, 3], 2, 2))
