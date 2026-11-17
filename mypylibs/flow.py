import cv2 as cv

LK = dict(winSize=(21, 21), maxLevel=3)

def track(prev_frame, cur_frame):
    p0 = cv.goodFeaturesToTrack(prev_frame, maxCorners=500, qualityLevel=0.01, minDistance=5)
    p1, found, _ = cv.calcOpticalFlowPyrLK(prev_frame, cur_frame, p0, None, **LK)
    found = found.ravel() == 1
    return p0[found].reshape(-1, 2), p1[found].reshape(-1, 2)
