import os

import cv2 as cv
import numpy as np

from mypylibs.flow import track


def to_gray(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


def run_one(name):
    video = cv.VideoCapture(f'in/{name}.mp4')
    ok, frame = video.read()
    height, width = frame.shape[:2]
    fps = video.get(cv.CAP_PROP_FPS)
    flow_video = cv.VideoWriter(f'out/{name}_of.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    prev_gray = to_gray(frame)
    ok, frame = video.read()
    while ok:
        next_gray = to_gray(frame)
        old_points, new_points = track(prev_gray, next_gray)
        for old, new in zip(old_points.astype(int), new_points.astype(int)):
            if np.linalg.norm(new - old) > 1:  # only draw points that moved
                cv.arrowedLine(frame, old, new, (0, 255, 0), 2)
        flow_video.write(frame)
        prev_gray = next_gray
        ok, frame = video.read()
    flow_video.release()
    print(f'wrote out/{name}_of.mp4')


def run():
    for name in ('01', '02', '03', '04'):
        if os.path.exists(f'in/{name}.mp4'):
            run_one(name)
