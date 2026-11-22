import cv2 as cv
import numpy as np

LK = dict(winSize=(21, 21), maxLevel=3)
MIN_SPEED = 1.5  # px/frame, the pen only moves ~2
MIN_POINTS = 3
FB_MAX = 1.0  # px, forward-backward error


def track(prev_gray, cur_gray, mask=None):
    p0 = cv.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=5, mask=mask)
    if p0 is None:
        return np.empty((0, 2)), np.empty((0, 2))
    p1, found, _ = cv.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None, **LK)
    back, found_back, _ = cv.calcOpticalFlowPyrLK(cur_gray, prev_gray, p1, None, **LK)
    fb_error = np.abs(p0 - back).reshape(-1, 2).max(axis=1)
    good = (found.ravel() == 1) & (found_back.ravel() == 1) & (fb_error < FB_MAX)
    return p0[good].reshape(-1, 2), p1[good].reshape(-1, 2)


def motion_mask(prev_gray, cur_gray, cfg):
    diff = cv.absdiff(cv.GaussianBlur(prev_gray, (0, 0), 2), cv.GaussianBlur(cur_gray, (0, 0), 2))
    mask = cv.dilate((diff > cfg['diff_threshold']).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
    mask[:, cfg['board_x0']:cfg['board_x1']] = 0
    mask[:cfg['row_min']] = 0
    mask[cfg['row_max']:] = 0
    return mask


def estimate_pos_and_velocity(p0, p1):
    velocities = p1 - p0
    is_moving = np.linalg.norm(velocities, axis=1) > MIN_SPEED
    points, velocities = p1[is_moving], velocities[is_moving]
    if len(points) < MIN_POINTS:
        return None
    # keep the points that move with the majority (drops hand, shadows, noise)
    median_velocity = np.median(velocities, axis=0)
    spread = max(2.0, 0.5 * np.linalg.norm(median_velocity))
    coherent = np.linalg.norm(velocities - median_velocity, axis=1) < spread
    points, velocities = points[coherent], velocities[coherent]
    if len(points) < MIN_POINTS:
        return None
    z = np.r_[np.median(points, axis=0), np.median(velocities, axis=0)]
    std = np.r_[points.std(axis=0), velocities.std(axis=0)]
    return z, std, points
