import numpy as np

from mypylibs.kalman import GATE, VELOCITY_GATE, distance, noise, predict, update

LOST_FRAMES = 10  # this many frames without a measurement = occluded
WARMUP = 3  # accept every measurement until the track has settled


def believable(x, P, z, R, settled, after_occlusion):
    if not settled:
        return True
    if after_occlusion:  # the position may be far off, so check only the motion
        return distance(x, P, z, R, slice(2, 4)) < VELOCITY_GATE
    return distance(x, P, z, R) < GATE


def follow(measurements, sigma_a):
    """Run the Kalman filter over per-frame measurements (None or (z, std)).

    Returns one step per frame and the comeback: the first measurement after
    the occlusion compared with the prediction.
    """
    steps, comeback = [], None
    x = P = ghost = None
    missed = measured = 0
    for i, measurement in enumerate(measurements):
        if ghost is not None:
            ghost = predict(*ghost, sigma_a)
        accepted, K = False, None
        if x is None:
            if measured == 0 and measurement is not None:  # start a track
                z, std = measurement
                x, P, measured, accepted = z.copy(), noise(std), 1, True
        else:
            x, P = predict(x, P, sigma_a)
            if measurement is not None:
                z, R = measurement[0], noise(measurement[1])
                accepted = believable(x, P, z, R, measured >= WARMUP, missed >= LOST_FRAMES)
            if accepted:
                if missed >= LOST_FRAMES and comeback is None:
                    ghost = x.copy(), P.copy()
                    comeback = dict(frame=i, error=float(np.linalg.norm(z[:2] - x[:2])))
                    measured = 0  # the jump also bends the velocity: let it settle again
                x, P, K = update(x, P, z, R)
                missed, measured = 0, measured + 1
            else:
                missed += 1
                if measured < WARMUP:  # broke before settling: it was noise, start again
                    x, measured, missed = None, 0, 0
        mode = None if x is None else 'measured' if accepted else 'predicted'
        steps.append(dict(state=x, mode=mode, ghost=None if ghost is None else ghost[0], gain=K, cov=P))
        if comeback is not None and missed >= LOST_FRAMES:
            x = None  # lost again after the comeback: stop the corrected track
    return steps, comeback
