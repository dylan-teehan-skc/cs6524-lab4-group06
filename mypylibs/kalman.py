import numpy as np

# state [x, y, vx, vy] in px and px/frame, dt = 1 frame, constant velocity
F = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], float)
# we measure position and velocity, so H = I and is left out
Q_UNIT = np.array([[.25, 0, .5, 0],
                   [0, .25, 0, .5],
                   [.5, 0, 1, 0],
                   [0, .5, 0, 1]])  # white-acceleration noise, kalmanfilter.net
STD_FLOOR = np.array([2, 2, .5, .5])  # a few points can give std ~ 0
GATE = 13.3  # chi-square, 4 dof, 99 %
VELOCITY_GATE = 9.2  # chi-square, 2 dof, 99 %


def noise(std):
    return np.diag(np.maximum(std, STD_FLOOR) ** 2)


def predict(x, P, sigma_a):
    return F @ x, F @ P @ F.T + sigma_a ** 2 * Q_UNIT


def distance(x, P, z, R, part=slice(0, 4)):
    """Squared Mahalanobis distance of the innovation (or of part of it)."""
    y = (z - x)[part]
    return y @ np.linalg.solve((P + R)[part, part], y)


def update(x, P, z, R):
    K = P @ np.linalg.inv(P + R)
    I_K = np.eye(4) - K
    return x + K @ (z - x), I_K @ P @ I_K.T + K @ R @ K.T, K
