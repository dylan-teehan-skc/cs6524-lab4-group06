import cv2 as cv
import numpy as np

LK = dict(winSize=(21, 21), maxLevel=3)
MIN_SPEED = 2  
MIN_POINTS = 5  

def track(prev_frame, cur_frame):
    p0 = cv.goodFeaturesToTrack(prev_frame, maxCorners=500, qualityLevel=0.01, minDistance=5)
    p1, found, _ = cv.calcOpticalFlowPyrLK(prev_frame, cur_frame, p0, None, **LK)
    found = found.ravel() == 1
    return p0[found].reshape(-1, 2), p1[found].reshape(-1, 2)


def estimate_pos_and_velocity(p0, p1):
    velocities = p1 - p0
    speeds = np.linalg.norm(velocities, axis=1)
    is_moving = speeds > MIN_SPEED
    if is_moving.sum() < MIN_POINTS:
        return None  
    position = np.median(p1[is_moving], axis=0)
    velocity = np.median(velocities[is_moving], axis=0)
    return position, velocity
