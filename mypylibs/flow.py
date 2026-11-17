import cv2 as cv

LK_SETTINGS = dict(winSize=(21, 21), maxLevel=3)


def track(prev_gray, next_gray):
    old_points = cv.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=5)
    new_points, was_tracked, _ = cv.calcOpticalFlowPyrLK(prev_gray, next_gray, old_points, None, **LK_SETTINGS)
    was_tracked = was_tracked.ravel() == 1
    return old_points[was_tracked].reshape(-1, 2), new_points[was_tracked].reshape(-1, 2)
