'''End-to-end smoke tests: the toolbox UI (headless) and optimize_image.py on caffenet-yos.'''

import os
import shutil
import subprocess
import sys

import cv2
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAFFENET_DIR = os.path.join(ROOT, 'models', 'caffenet-yos')


def _require_caffenet():
    for fname in ('caffenet-yos-weights', 'ilsvrc_2012_mean.npy'):
        if not os.path.exists(os.path.join(CAFFENET_DIR, fname)):
            pytest.skip('caffenet-yos files not found; run models/caffenet-yos/fetch.sh')


def test_toolbox_headless(tmp_path):
    _require_caffenet()
    shutil.copy(os.path.join(CAFFENET_DIR, 'settings_local.template-caffenet-yos.py'),
                tmp_path / 'settings_local.py')
    outdir = tmp_path / 'shots'
    proc = subprocess.run([sys.executable, os.path.join(ROOT, 'tests', 'headless_toolbox.py'), str(tmp_path), str(outdir)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert '[headless] finished cleanly' in proc.stdout
    for shot in ('start', 'backprop', 'deconv', 'pattern_mode', 'help'):
        img = cv2.imread(str(outdir / (shot + '.png')))
        assert img is not None and img.std() > 10, 'screenshot %s missing or blank' % shot


def test_optimize_image(tmp_path):
    _require_caffenet()
    proc = subprocess.run([sys.executable, 'optimize_image.py', '--push-layer', 'fc8', '--push-channel', '130',
                           '--decay', '0.0001', '--blur-radius', '1.0', '--blur-every', '4', '--max-iter', '150',
                           '--lr-policy', 'constant', '--lr-params', "{'lr': 100.0}",
                           '--output-prefix', str(tmp_path / 'opt')],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert 'Metaresult: majority success' in proc.stdout
    assert (tmp_path / 'opt_fc8_0130_0_best_Xpm.jpg').exists()
