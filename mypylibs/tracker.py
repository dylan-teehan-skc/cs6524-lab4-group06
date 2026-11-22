import numpy as np

from mypylibs.kalman import GATE, VELOCITY_GATE, distance, noise, predict, update

LOST_FRAMES = 10  # this many frames without a measurement = occluded
WARMUP = 3  # accept every measurement until the track has settled


def follow(measurements, sigma_a, lost_frames=LOST_FRAMES):
    """Run the Kalman filter over per-frame measurements"""
    steps, comeback = [], None
    x = P = ghost = None
    missed = measured = 0
    for i, measurement in enumerate(measurements):
        K = None
        if x is None and measured == 0:
            if measurement is not None:
                z, std = measurement
                x, P, measured = z.copy(), noise(std), 1
            steps.append(dict(state=x, mode='measured' if x is not None else None, ghost=None, gain=None))
            continue
        if ghost is not None:
            ghost = predict(*ghost, sigma_a)
        if x is None:
            steps.append(dict(state=None, mode=None, ghost=ghost and ghost[0], gain=None))
            continue
        x, P = predict(x, P, sigma_a)
        accepted = False
        if measurement is not None:
            z, std = measurement
            R = noise(std)
            if measured < WARMUP:
                accepted = True
            elif missed >= lost_frames:  # after occlusion the position may be far off, so check only the motion
                accepted = distance(x, P, z, R, slice(2, 4)) < VELOCITY_GATE
            else:
                accepted = distance(x, P, z, R) < GATE
        if accepted:
            if missed >= lost_frames and comeback is None:
                ghost = x.copy(), P.copy()
                comeback = dict(frame=i, error=float(np.linalg.norm(z[:2] - x[:2])),
                                sigma=float(np.sqrt(np.trace(P[:2, :2]))))
                measured = 0  # the jump also bends the velocity: let it settle again before gating
            x, P, K = update(x, P, z, R)
            missed, measured = 0, measured + 1
        else:
            missed += 1
        if not accepted and measured < WARMUP:  # broke before settling: it was noise, start again
            x, measured, missed = None, 0, 0
        mode = None if x is None else 'measured' if accepted else 'predicted'
        steps.append(dict(state=x, mode=mode, ghost=ghost and ghost[0], gain=K, cov=P))
        if comeback is not None and missed >= lost_frames:
            x = None  # lost again after the comeback: stop the corrected track
    return steps, comeback
