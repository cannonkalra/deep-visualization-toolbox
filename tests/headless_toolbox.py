'''Runs the real LiveVis main loop without a window: cv2 window calls are stubbed, a scripted
sequence of keys is fed in, and the rendered window buffer is saved after each step.

Usage: python tests/headless_toolbox.py SETTINGS_LOCAL_DIR OUTDIR
SETTINGS_LOCAL_DIR must contain a settings_local.py; it takes precedence over the one in the repo root.
'''

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
settings_dir, outdir = sys.argv[1], sys.argv[2]
sys.path[0:1] = [settings_dir, ROOT]
os.chdir(ROOT)
os.makedirs(outdir, exist_ok=True)

import cv2

# (key to press, seconds to wait afterwards, screenshot name or None)
SCRIPT = [
    (None, 6.0, 'start'),
    ('l', 1.0, None), ('l', 1.0, None), ('l', 1.0, None), ('l', 1.5, None),   # layer -> pool2
    ('k', 0.5, None), ('l', 1.0, None), ('l', 2.0, 'unit_selected'),          # select a unit
    ('b', 2.5, 'backprop'),
    ('n', 2.5, 'deconv'),
    ('a', 2.0, 'show_back_diffs'),
    ('a', 0.5, None), ('s', 3.0, 'pattern_mode'),
    ('s', 0.5, None), ('z', 2.0, 'zoom'),
    ('z', 0.5, None), ('z', 0.5, None), ('e', 3.0, 'next_image'),
    ('h', 1.0, 'help'),
    ('h', 0.5, None), ('esc', 1.5, 'reset'),
    ('q', 1.0, None),
]
KEYCODES = {'esc': 27}
state = {'idx': 0, 'next_at': time.time() + SCRIPT[0][1], 'last': None}


def fake_wait_key(ms):
    time.sleep(ms / 1000.0)
    if state['idx'] >= len(SCRIPT) or time.time() < state['next_at']:
        return -1
    shot = SCRIPT[state['idx']][2]
    if shot and state['last'] is not None:
        cv2.imwrite(os.path.join(outdir, shot + '.png'), state['last'])
        print('[headless] saved', shot)
    state['idx'] += 1
    if state['idx'] >= len(SCRIPT):
        return -1
    state['next_at'] = time.time() + SCRIPT[state['idx']][1]
    key = SCRIPT[state['idx']][0]
    return KEYCODES[key] if key in KEYCODES else ord(key)


def fake_imshow(name, img):
    state['last'] = img.copy()


cv2.namedWindow = lambda *args, **kwargs: None
cv2.imshow = fake_imshow
cv2.waitKey = cv2.waitKeyEx = fake_wait_key

import settings
settings.input_updater_capture_device = None
from live_vis import LiveVis

LiveVis(settings).run_loop()
print('[headless] finished cleanly')
