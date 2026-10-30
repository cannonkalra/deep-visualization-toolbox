'''Checks torchcaffe against an independent Caffe implementation (OpenCV's cv2.dnn Caffe importer)
and checks backprop / deconv semantics. Models whose weights have not been fetched are skipped.

Run with:  uv run pytest
'''

import os

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import torchcaffe as caffe
from torchcaffe.net import read_prototxt, _net_layers, _lrn_across_channels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'models')
IMAGE = os.path.join(ROOT, 'input_images', 'ILSVRC2012_val_00000610.jpg')   # a jaguar
CAFFENET = (os.path.join(MODELS, 'caffenet-yos', 'caffenet-yos-deploy.prototxt'),
            os.path.join(MODELS, 'caffenet-yos', 'caffenet-yos-weights'))
CAFFENET_MEAN = os.path.join(MODELS, 'caffenet-yos', 'ilsvrc_2012_mean.npy')

MODEL_CASES = {
    'caffenet-yos': CAFFENET,
    'bvlc-googlenet': (os.path.join(MODELS, 'bvlc-googlenet', 'bvlc-googlenet-deploy.prototxt'),
                       os.path.join(MODELS, 'bvlc-googlenet', 'bvlc-googlenet.caffemodel')),
    'squeezenet': (os.path.join(MODELS, 'squeezenet', 'deploy.prototxt'),
                   os.path.join(MODELS, 'squeezenet', 'squeezenet_v1.0.caffemodel')),
}
DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []) + (['cuda'] if torch.cuda.is_available() else [])


def _require(*paths):
    for path in paths:
        if not os.path.exists(path):
            pytest.skip('%s not found; run the fetch.sh script in its model directory' % os.path.relpath(path, ROOT))


def _mean_for(name):
    if name == 'caffenet-yos':
        _require(CAFFENET_MEAN)
        return np.load(CAFFENET_MEAN)[:, 14:241, 14:241]
    return np.array([104., 117., 123.])


def _load(name, device):
    prototxt, weights = MODEL_CASES[name]
    _require(prototxt, weights)
    caffe.set_device(device)
    net = caffe.Classifier(prototxt, weights, mean=_mean_for(name), channel_swap=(2, 1, 0), raw_scale=1.0)
    hw = net.blobs['data'].data.shape[2:]
    img = cv2.resize(cv2.imread(IMAGE)[:, :, ::-1], hw[::-1]).astype(np.float32)
    x = net.transformer.preprocess('data', img)[None]
    net.forward(data=x)
    return net, x


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('name', list(MODEL_CASES))
def test_forward_matches_opencv_caffe_importer(name, device):
    net, x = _load(name, device)
    prototxt, weights = MODEL_CASES[name]

    # cv2.dnn returns the output of a layer; a blob's final value comes from the last layer writing it
    last_writer = {}
    for lp in _net_layers(read_prototxt(prototxt)):
        for top in lp.top:
            last_writer[top] = lp.name
    ref = cv2.dnn.readNetFromCaffe(prototxt, weights)
    ref.enableFusion(False)
    ref.setInput(x)
    names = [n for n in net.blobs if n != 'data']
    for blob_name, out in zip(names, ref.forward([last_writer[n] for n in names])):
        ours = net.blobs[blob_name].data
        rel_err = np.abs(ours - out.reshape(ours.shape)).max() / (np.abs(out).max() + 1e-8)
        assert rel_err < 1e-4, '%s: blob %s differs from cv2.dnn (rel err %g)' % (name, blob_name, rel_err)

    labels = open(os.path.join(MODELS, 'caffenet-yos', 'ilsvrc_2012_labels.txt')).read().split('\n')
    assert labels[net.blobs['prob'].data.argmax()] == 'jaguar'


@pytest.mark.parametrize('device', DEVICES)
def test_backward_matches_finite_differences(device):
    net, x = _load('caffenet-yos', device)
    conv5 = net.blobs['conv5'].data[0]
    unit = np.unravel_index(conv5.argmax(), conv5.shape)     # positive, so in-place ReLU is transparent
    diffs = np.zeros_like(net.blobs['conv5'].diff)
    diffs[(0,) + unit] = 1.0
    grad = net.backward_from_layer('conv5', diffs, zero_higher=True)['data'][0].copy()
    assert not net.blobs['fc6'].diff.any() and not net.blobs['prob'].diff.any(), 'diffs above start should be 0'

    # Directional derivative along the gradient: f(x + t g) - f(x - t g) ~= 2 t |g|^2
    for step in (0.2, 0.5, 1.0):        # target change in the unit's value
        t = step / (grad ** 2).sum()
        vals = []
        for sign in (1, -1):
            net.forward(data=x + sign * t * grad[None])
            vals.append(float(net.blobs['conv5'].data[(0,) + unit]))
        numeric = (vals[0] - vals[1]) / (2 * t)
        analytic = float((grad.astype(np.float64) ** 2).sum())
        assert abs(numeric - analytic) <= 2e-2 * analytic, (step, numeric, analytic)


@pytest.mark.parametrize('device', DEVICES)
def test_deconv_matches_manual_zeiler_fergus(device):
    '''Deconv through CaffeNet = backprop, except ReLUs pass only positive diffs (deconv-deep-vis-toolbox Caffe).'''
    net, x = _load('caffenet-yos', device)
    diffs = np.zeros_like(net.blobs['conv5'].diff)
    diffs[0, 7, 6, 6] = 1.0
    deconv = net.deconv_from_layer('conv5', diffs)['data'][0].copy()
    backprop = net.backward_from_layer('conv5', diffs)['data'][0].copy()
    assert not np.allclose(deconv, backprop)

    dev = net.device
    W = {k: [torch.from_numpy(b.data).to(dev) for b in v] for k, v in net.params.items()}
    conv = lambda name, stride, pad, group: (lambda t: F.conv2d(t, W[name][0], W[name][1], stride, pad, 1, group))
    pool = lambda t: F.max_pool2d(t, 3, 2, 0, ceil_mode=True)
    lrn = lambda t: _lrn_across_channels(t, 5, 1e-4, 0.75, 1.0)

    def vjp(fn, inp, grad):
        inp = inp.detach().requires_grad_(True)
        return torch.autograd.grad(fn(inp), inp, grad)[0]

    xt = torch.from_numpy(x).to(dev)
    r1 = conv('conv1', 4, 0, 1)(xt).clamp(min=0); n1 = lrn(pool(r1))
    r2 = conv('conv2', 1, 2, 2)(n1).clamp(min=0); p2 = pool(r2); n2 = lrn(p2)
    r3 = conv('conv3', 1, 1, 1)(n2).clamp(min=0)
    r4 = conv('conv4', 1, 1, 2)(r3).clamp(min=0)
    g = vjp(conv('conv5', 1, 1, 2), r4, torch.from_numpy(diffs).to(dev))
    g = vjp(conv('conv4', 1, 1, 2), r3, g.clamp(min=0))
    g = vjp(conv('conv3', 1, 1, 1), n2, g.clamp(min=0))
    g = vjp(pool, r2, vjp(lrn, p2, g))
    g = vjp(conv('conv2', 1, 2, 2), n1, g.clamp(min=0))
    g = vjp(pool, r1, vjp(lrn, pool(r1), g))
    g = vjp(conv('conv1', 4, 0, 1), xt, g.clamp(min=0))
    expected = g[0].cpu().numpy()
    assert np.abs(deconv - expected).max() <= 1e-4 * np.abs(expected).max()
