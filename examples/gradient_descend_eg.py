import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np

from gradient_method.gradient_descent import GradientDescent


def main():
    # 目的関数(Objective function)
    # Booth function
    def func(x):
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

    def deriv(x):
        dx1 = 10 * x[0] + 8 * x[1] - 34
        dx2 = 8 * x[0] + 10 * x[1] - 38
        return np.array([dx1, dx2])

    x_start = np.array([10.0, 10.0])

    gd = GradientDescent(func, deriv, alpha=None)
    result = gd.minimize(x_start, verbosity=10)


if __name__ == "__main__":
    main()
