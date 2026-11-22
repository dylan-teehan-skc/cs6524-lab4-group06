import os
from collections import deque

import cv2 as cv
import numpy as np

from mypylibs.flow import MIN_SPEED, estimate_pos_and_velocity, motion_mask, track
from mypylibs.tracker import follow

DEFAULTS = dict(diff_threshold=6, board_x0=0, board_x1=0, row_min=0, row_max=10_000,
                normalise_light=0, sigma_a=0.7, start_frame=0)
GREEN, BLUE, RED, MAGENTA, YELLOW = (0, 255, 0), (255, 140, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255)
ARROW_EVERY = 10  # frames
MAX_ARROWS = 60  # flow arrows per frame in XX_of.mp4
TRAIL = 15  # measurements kept on screen, the blue line holds the rest
ARROW_LENGTH = 5  # frames of motion drawn per velocity arrow


def load_config(name):
    cfg = dict(DEFAULTS)
    store = cv.FileStorage(f'in/{name}.yaml', cv.FILE_STORAGE_READ)
    if store.isOpened():
        for key in store.root().keys():
            cfg[key] = type(DEFAULTS[key])(store.getNode(key).real())
    return cfg


def frames(name):
    video = cv.VideoCapture(f'in/{name}.mp4')
    ok, frame = video.read()
    while ok:
        yield frame
        ok, frame = video.read()


def to_gray(frame, normalise_light=False):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if normalise_light:  # same mean brightness every frame
        gray = cv.convertScaleAbs(gray, alpha=128 / gray.mean())
    return gray


def writer(path, like, fps):
    return cv.VideoWriter(path, cv.VideoWriter_fourcc(*'mp4v'), fps, like.shape[1::-1])


def measure_and_draw_flow(name, cfg, fps):
    """Measure the object in every frame and write XX_of.mp4."""
    measurements, out, prev_gray = [], None, None
    for frame in frames(name):
        gray = to_gray(frame, cfg['normalise_light'])
        out = out or writer(f'out/{name}_of.mp4', frame, fps)
        measurement = None
        if prev_gray is not None:
            old_points, new_points = track(prev_gray, gray, motion_mask(prev_gray, gray, cfg))
            moving = np.linalg.norm(new_points - old_points, axis=1) > MIN_SPEED  # not background jitter
            step = max(1, moving.sum() // MAX_ARROWS)
            for old, new in list(zip(old_points[moving], new_points[moving]))[::step]:
                cv.arrowedLine(frame, point(old), point(old + 3 * (new - old)), GREEN, 1, tipLength=0.3)
            found = estimate_pos_and_velocity(old_points, new_points)
            if found:
                z, std, _ = found
                measurement = z, std
                cv.circle(frame, point(z), 10, RED, 2)
                cv.arrowedLine(frame, point(z), point(z[:2] + ARROW_LENGTH * z[2:]), RED, 2, tipLength=0.3)
        measurements.append(measurement)
        draw_legend(frame, [('flow', GREEN), ('measured object', RED)])
        out.write(frame)
        prev_gray = gray
    out.release()
    return measurements


def point(v):
    return tuple(int(round(c)) for c in v[:2])


def draw_legend(frame, items):
    """Colour key in the bottom-left corner, one line per item."""
    if not items:
        return
    scale = max(0.45, frame.shape[1] / 2000)
    line_height = int(30 * scale) + 4
    width = max(cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for text, _ in items)
    top = frame.shape[0] - 8 - line_height * len(items)
    box = frame[top:, :width + 16]
    box[:] = box // 3  # darken behind the text so every colour is readable
    for k, (text, colour) in enumerate(items):
        y = top + line_height * (k + 1)
        cv.putText(frame, text, (8, y - 2), cv.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv.LINE_AA)


TRAJECTORY_LEGEND = [('measured', GREEN), ('corrected', BLUE), ('predicted', RED)]


def draw_trajectories(name, measurements, steps, fps):
    """Write XX_tr.mp4. Trajectories build up on a transparent layer."""
    out = layer = None
    shown = set()  # colours drawn so far: the legend only lists those
    recent = deque(maxlen=TRAIL)
    for i, frame in enumerate(frames(name)):
        if layer is None:
            out, layer = writer(f'out/{name}_tr.mp4', frame, fps), np.zeros_like(frame)
            width = max(2, frame.shape[1] // 400)
        step, prev = steps[i], steps[i - 1] if i else steps[0]
        if step['mode'] == 'measured':  # measurements the filter rejected are left out
            recent.append(point(measurements[i][0]))
            shown.add(GREEN)
        comeback = step['ghost'] is not None and prev['ghost'] is None
        if step['state'] is not None and prev['state'] is not None and not comeback:
            colour = BLUE if step['mode'] == 'measured' else RED
            cv.line(layer, point(prev['state']), point(step['state']), colour, width)
            shown.add(colour)
            if i % ARROW_EVERY == 0 and step['mode'] == 'measured':  # while predicting it never changes
                tip = step['state'][:2] + ARROW_LENGTH * step['state'][2:]
                cv.arrowedLine(layer, point(step['state']), point(tip), YELLOW, width, tipLength=0.3)
        if step['ghost'] is not None and prev['ghost'] is not None:
            cv.line(layer, point(prev['ghost']), point(step['ghost']), MAGENTA, width)
            shown.add(MAGENTA)
        drawn = layer.any(axis=2)
        frame[drawn] = layer[drawn]
        for p in recent:
            cv.circle(frame, p, width, GREEN, -1)
        if step['state'] is not None:
            cv.circle(frame, point(step['state']), 4 * width, BLUE if step['mode'] == 'measured' else RED, width)
        draw_legend(frame, [item for item in TRAJECTORY_LEGEND if item[1] in shown])
        out.write(frame)
    out.release()


def run_one(name):
    cfg = load_config(name)
    fps = cv.VideoCapture(f'in/{name}.mp4').get(cv.CAP_PROP_FPS)
    measurements = measure_and_draw_flow(name, cfg, fps)
    measurements[:cfg['start_frame']] = [None] * cfg['start_frame']
    steps, comeback = follow(measurements, cfg['sigma_a'])
    draw_trajectories(name, measurements, steps, fps)
    if comeback:
        print(f"{name}: back at frame {comeback['frame']}, prediction error {comeback['error']:.0f} px "
              f"(filter 1-sigma {comeback['sigma']:.0f} px)")
    else:
        print(f'{name}: object never came back')
    return measurements, steps, comeback


def run():
    os.makedirs('out', exist_ok=True)
    for name in ('01', '02', '03', '04'):
        if os.path.exists(f'in/{name}.mp4'):
            run_one(name)
