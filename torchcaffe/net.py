'''Caffe-style Net and Classifier implemented on PyTorch.

Reads Caffe .prototxt network definitions (both the current `layer`
format and the legacy V1 `layers` format) and .caffemodel weights, and
exposes the subset of the pycaffe API used by the Deep Visualization
Toolbox: net.blobs[name].data / .diff, net.params, net.inputs,
net.outputs, forward(), forward_all(), backward_from_layer() and
deconv_from_layer() (Zeiler & Fergus 2014 deconvnet, as implemented in
the deconv-deep-vis-toolbox branch of Caffe).

Semantics follow Caffe:
  * blobs are keyed by top name in order of creation; in-place layers
    (e.g. ReLU) do not create new blobs. A blob's .data is its final
    value (e.g. after in-place ReLU) and its .diff is the derivative
    with respect to the output of the layer that first produced it
    (matching Caffe, where in-place backward overwrites the shared diff).
  * backward_from_layer(name, diff) starts at the layer that produced
    blob `name`, skipping in-place layers applied after it, exactly as
    Caffe's Net.backward(start=name) does.
  * deconv_from_layer() is like backward, except ReLU passes only the
    positive part of the top diff, and LRN passes diffs through
    unchanged when its lrn_param.deconv_ignore is set.
'''

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from google.protobuf import text_format

from .proto import caffe_pb2

TRAIN = 0
TEST = 1

_device = torch.device('cpu')


def set_mode_cpu():
    global _device
    _device = torch.device('cpu')


def set_mode_gpu():
    '''Use the best available accelerator: Apple GPU (MPS), then CUDA, else CPU.'''
    global _device
    if torch.backends.mps.is_available():
        _device = torch.device('mps')
    elif torch.cuda.is_available():
        _device = torch.device('cuda')
    else:
        print('torchcaffe: no GPU (MPS or CUDA) available, using CPU')
        _device = torch.device('cpu')


def set_device(device):
    global _device
    _device = torch.device(device)


def get_device():
    return _device


class Blob(object):
    '''Holds .data and .diff as numpy float32 arrays, like a pycaffe Blob.'''

    def __init__(self, shape):
        self.data = np.zeros(shape, dtype=np.float32)
        self.diff = np.zeros(shape, dtype=np.float32)

    @property
    def shape(self):
        return self.data.shape

    @property
    def count(self):
        return self.data.size

    def reshape(self, *shape):
        self.data = np.zeros(shape, dtype=np.float32)
        self.diff = np.zeros(shape, dtype=np.float32)



# ---------------------------------------------------------------------------
# Prototxt / caffemodel parsing
# ---------------------------------------------------------------------------

# V1LayerParameter.LayerType enum name -> current layer type string
_V1_TYPE_NAMES = {
    'ABSVAL': 'AbsVal', 'ACCURACY': 'Accuracy', 'ARGMAX': 'ArgMax', 'BNLL': 'BNLL',
    'CONCAT': 'Concat', 'CONTRASTIVE_LOSS': 'ContrastiveLoss', 'CONVOLUTION': 'Convolution',
    'DATA': 'Data', 'DECONVOLUTION': 'Deconvolution', 'DROPOUT': 'Dropout',
    'DUMMY_DATA': 'DummyData', 'EUCLIDEAN_LOSS': 'EuclideanLoss', 'ELTWISE': 'Eltwise',
    'EXP': 'Exp', 'FLATTEN': 'Flatten', 'HDF5_DATA': 'HDF5Data', 'HDF5_OUTPUT': 'HDF5Output',
    'HINGE_LOSS': 'HingeLoss', 'IM2COL': 'Im2col', 'IMAGE_DATA': 'ImageData',
    'INFOGAIN_LOSS': 'InfogainLoss', 'INNER_PRODUCT': 'InnerProduct', 'LRN': 'LRN',
    'MEMORY_DATA': 'MemoryData', 'MULTINOMIAL_LOGISTIC_LOSS': 'MultinomialLogisticLoss',
    'MVN': 'MVN', 'POOLING': 'Pooling', 'POWER': 'Power', 'RELU': 'ReLU', 'SIGMOID': 'Sigmoid',
    'SIGMOID_CROSS_ENTROPY_LOSS': 'SigmoidCrossEntropyLoss', 'SILENCE': 'Silence',
    'SOFTMAX': 'Softmax', 'SOFTMAX_LOSS': 'SoftmaxWithLoss', 'SPLIT': 'Split', 'SLICE': 'Slice',
    'TANH': 'TanH', 'WINDOW_DATA': 'WindowData', 'THRESHOLD': 'Threshold',
}

# Layer types that only matter for training and are dropped from deploy nets
_SKIP_TYPES = {'Data', 'ImageData', 'HDF5Data', 'WindowData', 'MemoryData', 'DummyData',
               'Accuracy', 'SoftmaxWithLoss', 'EuclideanLoss', 'HingeLoss', 'InfogainLoss',
               'MultinomialLogisticLoss', 'SigmoidCrossEntropyLoss', 'ContrastiveLoss',
               'Silence', 'HDF5Output'}


def _layer_type(lp):
    if isinstance(lp, caffe_pb2.V1LayerParameter):
        return _V1_TYPE_NAMES.get(caffe_pb2.V1LayerParameter.LayerType.Name(lp.type), str(lp.type))
    return lp.type


def _net_layers(net_param):
    '''Returns layers of a NetParameter, whichever format (layer / V1 layers) it uses.'''
    return list(net_param.layer) if len(net_param.layer) > 0 else list(net_param.layers)


def _in_phase(lp, phase):
    if len(lp.include) > 0:
        return any((not rule.HasField('phase')) or rule.phase == phase for rule in lp.include)
    return not any(rule.HasField('phase') and rule.phase == phase for rule in lp.exclude)


def blobproto_to_array(blob, return_diff=False):
    '''Converts a BlobProto to a numpy array, as caffe.io.blobproto_to_array.'''
    if len(blob.shape.dim) > 0:
        shape = tuple(blob.shape.dim)
    else:
        shape = (blob.num, blob.channels, blob.height, blob.width)
    if return_diff:
        values = blob.double_diff if len(blob.double_diff) > 0 else blob.diff
    else:
        values = blob.double_data if len(blob.double_data) > 0 else blob.data
    return np.array(values, dtype=np.float32).reshape(shape)


def read_prototxt(filename):
    net_param = caffe_pb2.NetParameter()
    with open(filename, 'r') as ff:
        text_format.Merge(ff.read(), net_param)
    return net_param


def read_caffemodel_blobs(filename):
    '''Returns OrderedDict: layer name -> list of numpy arrays.'''
    net_param = caffe_pb2.NetParameter()
    with open(filename, 'rb') as ff:
        net_param.ParseFromString(ff.read())
    ret = OrderedDict()
    for lp in _net_layers(net_param):
        if len(lp.blobs) > 0:
            ret[lp.name] = [blobproto_to_array(bb) for bb in lp.blobs]
    return ret



# ---------------------------------------------------------------------------
# Layer implementations
# ---------------------------------------------------------------------------

class _ReLU(torch.autograd.Function):
    '''ReLU whose backward pass follows net.deconv_mode (deconv: pass positive diffs only).'''

    @staticmethod
    def forward(ctx, x, net, negative_slope):
        ctx.net = net
        ctx.negative_slope = negative_slope
        ctx.save_for_backward(x)
        if negative_slope == 0:
            return x.clamp(min=0)
        return torch.where(x > 0, x, x * negative_slope)

    @staticmethod
    def backward(ctx, grad):
        if ctx.net.deconv_mode:
            return grad.clamp(min=0), None, None
        x, = ctx.saved_tensors
        if ctx.negative_slope == 0:
            return grad * (x > 0).to(grad.dtype), None, None
        return torch.where(x > 0, grad, grad * ctx.negative_slope), None, None


def _lrn_across_channels(x, size, alpha, beta, k):
    # Caffe: y = x * (k + alpha/size * sum_{window} x^2) ^ -beta, window centered on channel
    sq = F.pad(x * x, (0, 0, 0, 0, (size - 1) // 2, size // 2))
    channels = x.shape[1]
    window_sum = sq[:, 0:channels]
    for offset in range(1, size):
        window_sum = window_sum + sq[:, offset:offset + channels]
    return x * (k + (alpha / size) * window_sum).pow(-beta)


def _lrn_within_channel(x, size, alpha, beta):
    # Caffe: average x^2 over a size x size spatial window (padded with zeros)
    pad = (size - 1) // 2
    avg = F.avg_pool2d(x * x, size, stride=1, padding=pad, count_include_pad=True)
    return x * (1 + alpha * avg).pow(-beta)


class _LRNDeconvIgnore(torch.autograd.Function):
    '''LRN that is skipped (diff passed through unchanged) during deconv.'''

    @staticmethod
    def forward(ctx, x, net, fn):
        ctx.net = net
        ctx.fn = fn
        ctx.save_for_backward(x)
        return fn(x)

    @staticmethod
    def backward(ctx, grad):
        if ctx.net.deconv_mode:
            return grad, None, None
        x, = ctx.saved_tensors
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            gx, = torch.autograd.grad(ctx.fn(xx), xx, grad)
        return gx, None, None


def _hw(param, name, default=None, repeated=True):
    '''Reads a 2D (h, w) conv/pool parameter, honoring name_h/name_w overrides.'''
    hw_name = 'kernel' if name == 'kernel_size' else name
    if param.HasField(hw_name + '_h') or param.HasField(hw_name + '_w'):
        return (getattr(param, hw_name + '_h'), getattr(param, hw_name + '_w'))
    if repeated:
        vals = list(getattr(param, name))
        if len(vals) == 0:
            return (default, default)
        if len(vals) == 1:
            return (vals[0], vals[0])
        return (vals[0], vals[1])
    if param.HasField(name) or default is None:
        val = getattr(param, name)
    else:
        val = default
    return (val, val)


class _Layer(object):
    '''A compiled layer: callable taking a list of bottom tensors, returning a list of top tensors.'''

    def __init__(self, name, type_name, bottoms, tops, fn, params):
        self.name = name
        self.type = type_name
        self.bottoms = bottoms
        self.tops = tops
        self.fn = fn
        self.params = params   # list of torch tensors (weights, bias, ...)


def _build_layer(net, lp, type_name, bottom_shapes, weights):
    '''Returns (fn, params) for one layer given the shapes of its bottoms.'''
    dev = net.device

    def param_tensor(idx, shape):
        if weights is not None and idx < len(weights):
            arr = weights[idx]
            assert arr.size == int(np.prod(shape)), (
                'Layer %s: weight blob %d has %d values but expected shape %s'
                % (lp.name, idx, arr.size, shape))
            arr = arr.reshape(shape)
        else:
            arr = np.zeros(shape, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(arr)).to(dev)

    if type_name == 'Convolution':
        cp = lp.convolution_param
        in_channels = bottom_shapes[0][1]
        kh, kw = _hw(cp, 'kernel_size')
        ph, pw = _hw(cp, 'pad', 0)
        sh, sw = _hw(cp, 'stride', 1)
        dilation = list(cp.dilation)
        dh, dw = (dilation[0], dilation[-1]) if dilation else (1, 1)
        group = cp.group
        W = param_tensor(0, (cp.num_output, in_channels // group, kh, kw))
        b = param_tensor(1, (cp.num_output,)) if cp.bias_term else None
        fn = lambda xs: [F.conv2d(xs[0], W, b, (sh, sw), (ph, pw), (dh, dw), group)]
        return fn, [W] + ([b] if b is not None else [])

    if type_name == 'InnerProduct':
        ip = lp.inner_product_param
        axis = ip.axis if ip.HasField('axis') else 1
        in_features = int(np.prod(bottom_shapes[0][axis:]))
        if ip.transpose:
            W = param_tensor(0, (in_features, ip.num_output)).t()
        else:
            W = param_tensor(0, (ip.num_output, in_features))
        b = param_tensor(1, (ip.num_output,)) if ip.bias_term else None

        def fn(xs):
            x = xs[0]
            return [F.linear(x.reshape(tuple(x.shape[:axis]) + (-1,)), W, b)]
        return fn, [W] + ([b] if b is not None else [])

    if type_name == 'ReLU':
        slope = lp.relu_param.negative_slope
        return (lambda xs: [_ReLU.apply(xs[0], net, slope)]), []

    if type_name == 'Pooling':
        pp = lp.pooling_param
        if pp.global_pooling:
            kh, kw = bottom_shapes[0][2], bottom_shapes[0][3]
            ph, pw, sh, sw = 0, 0, 1, 1
        else:
            kh, kw = _hw(pp, 'kernel_size', repeated=False)
            ph, pw = _hw(pp, 'pad', 0, repeated=False)
            sh, sw = _hw(pp, 'stride', 1, repeated=False)
        # Caffe rounds pooled output size up (ceil_mode), like PyTorch's ceil_mode
        if pp.pool == caffe_pb2.PoolingParameter.MAX:
            fn = lambda xs: [F.max_pool2d(xs[0], (kh, kw), (sh, sw), (ph, pw), ceil_mode=True)]
        elif pp.pool == caffe_pb2.PoolingParameter.AVE:
            fn = lambda xs: [F.avg_pool2d(xs[0], (kh, kw), (sh, sw), (ph, pw), ceil_mode=True,
                                          count_include_pad=True)]
        else:
            raise NotImplementedError('Layer %s: STOCHASTIC pooling is not supported' % lp.name)
        return fn, []

    if type_name == 'LRN':
        lrn = lp.lrn_param
        if lrn.norm_region == caffe_pb2.LRNParameter.ACROSS_CHANNELS:
            lrn_fn = lambda x: _lrn_across_channels(x, lrn.local_size, lrn.alpha, lrn.beta, lrn.k)
        else:
            lrn_fn = lambda x: _lrn_within_channel(x, lrn.local_size, lrn.alpha, lrn.beta)
        if lrn.deconv_ignore:
            return (lambda xs: [_LRNDeconvIgnore.apply(xs[0], net, lrn_fn)]), []
        return (lambda xs: [lrn_fn(xs[0])]), []

    if type_name == 'Dropout':
        # Deploy nets run in TEST phase, where Caffe's dropout is the identity
        return (lambda xs: [xs[0]]), []

    if type_name == 'Softmax':
        axis = lp.softmax_param.axis
        return (lambda xs: [F.softmax(xs[0], dim=axis)]), []

    if type_name == 'Concat':
        cp = lp.concat_param
        axis = cp.concat_dim if cp.HasField('concat_dim') else cp.axis
        return (lambda xs: [torch.cat(xs, dim=axis)]), []

    if type_name == 'Split':
        n_tops = len(lp.top)
        return (lambda xs: [xs[0]] * n_tops), []

    if type_name == 'Flatten':
        axis = lp.flatten_param.axis
        return (lambda xs: [torch.flatten(xs[0], start_dim=axis)]), []

    if type_name == 'Sigmoid':
        return (lambda xs: [torch.sigmoid(xs[0])]), []

    if type_name == 'TanH':
        return (lambda xs: [torch.tanh(xs[0])]), []

    if type_name == 'Eltwise':
        ep = lp.eltwise_param
        coeffs = list(ep.coeff)
        if ep.operation == caffe_pb2.EltwiseParameter.PROD:
            def fn(xs):
                out = xs[0]
                for x in xs[1:]:
                    out = out * x
                return [out]
        elif ep.operation == caffe_pb2.EltwiseParameter.MAX:
            def fn(xs):
                out = xs[0]
                for x in xs[1:]:
                    out = torch.maximum(out, x)
                return [out]
        else:
            def fn(xs):
                cc = coeffs if coeffs else [1.0] * len(xs)
                out = xs[0] * cc[0]
                for c, x in zip(cc[1:], xs[1:]):
                    out = out + x * c
                return [out]
        return fn, []

    raise NotImplementedError('Layer %s: type "%s" is not supported by torchcaffe' % (lp.name, type_name))



# ---------------------------------------------------------------------------
# Net
# ---------------------------------------------------------------------------

class Net(object):
    def __init__(self, network_file, weights=None, phase=TEST):
        self.device = _device
        self.deconv_mode = False
        net_param = read_prototxt(network_file)
        self.name = net_param.name
        weight_blobs = read_caffemodel_blobs(weights) if weights else {}

        # Inputs, declared either at net level or as Input layers
        input_shapes = OrderedDict()
        for ii, in_name in enumerate(net_param.input):
            if len(net_param.input_shape) > ii:
                input_shapes[in_name] = tuple(net_param.input_shape[ii].dim)
            else:
                input_shapes[in_name] = tuple(net_param.input_dim[4 * ii:4 * ii + 4])

        self.blobs = OrderedDict()
        self.params = OrderedDict()
        self._layers = []
        current = OrderedDict()   # blob name -> tensor, used to infer shapes while building
        for in_name, shape in input_shapes.items():
            self.blobs[in_name] = Blob(shape)
            current[in_name] = torch.zeros(shape, device=self.device)

        with torch.no_grad():
            for lp in _net_layers(net_param):
                type_name = _layer_type(lp)
                if not _in_phase(lp, phase) or type_name in _SKIP_TYPES:
                    continue
                if type_name == 'Input':
                    for top, shape in zip(lp.top, lp.input_param.shape):
                        input_shapes[top] = tuple(shape.dim)
                        self.blobs[top] = Blob(tuple(shape.dim))
                        current[top] = torch.zeros(tuple(shape.dim), device=self.device)
                    continue
                bottoms, tops = list(lp.bottom), list(lp.top)
                fn, params = _build_layer(self, lp, type_name,
                                          [tuple(current[bb].shape) for bb in bottoms],
                                          weight_blobs.get(lp.name))
                layer = _Layer(lp.name, type_name, bottoms, tops, fn, params)
                self._layers.append(layer)
                if params:
                    self.params[lp.name] = [_ParamBlob(pp) for pp in params]
                    if weights and lp.name not in weight_blobs:
                        print('torchcaffe WARNING: no weights for layer "%s" in %s; using zeros'
                              % (lp.name, weights))
                outs = fn([current[bb] for bb in bottoms])
                for top, out in zip(tops, outs):
                    current[top] = out
                    if top not in self.blobs:
                        self.blobs[top] = Blob(tuple(out.shape))

        self.inputs = list(input_shapes.keys())
        consumed = set()
        for layer in self._layers:
            consumed.update(bb for bb in layer.bottoms if bb not in layer.tops)
        self.outputs = [name for name in self.blobs if name not in consumed and name not in self.inputs]
        self._first = {}   # blob name -> tensor first produced for this blob in the last forward

    @property
    def layer_names(self):
        return [layer.name for layer in self._layers]

    def forward(self, blobs=None, start=None, end=None, **kwargs):
        '''Runs the whole net forward. kwargs map input names to arrays.
        Returns {output name: output data}.'''
        for in_name, arr in kwargs.items():
            if in_name not in self.inputs:
                raise Exception('Input blob arguments do not match net inputs: %s' % in_name)
            self.blobs[in_name].data = np.array(arr, dtype=np.float32)

        current = OrderedDict()
        self._first = OrderedDict()
        for in_name in self.inputs:
            tt = torch.from_numpy(self.blobs[in_name].data).to(self.device).requires_grad_(True)
            current[in_name] = tt
            self._first[in_name] = tt
        with torch.enable_grad():
            for layer in self._layers:
                outs = layer.fn([current[bb] for bb in layer.bottoms])
                for top, out in zip(layer.tops, outs):
                    current[top] = out
                    if top not in self._first:
                        self._first[top] = out
                        if out.requires_grad and not out.is_leaf:
                            out.retain_grad()

        for name, tt in current.items():
            blob = self.blobs[name]
            blob.data = tt.detach().cpu().numpy()
            if blob.diff.shape != blob.data.shape:
                blob.diff = np.zeros_like(blob.data)

        outputs = set(self.outputs + (blobs or []))
        return {out: self.blobs[out].data for out in outputs}

    def forward_all(self, blobs=None, **kwargs):
        '''Like Caffe's forward_all; the whole batch is run at once.'''
        outs = self.forward(blobs=blobs, **kwargs)
        return {name: arr.copy() for name, arr in outs.items()}

    def _backward_from(self, start_name, start_diff, diffs):
        if start_name not in self._first:
            raise Exception('Run forward() before backward_from_layer / deconv_from_layer')
        start_diff = np.asarray(start_diff, dtype=np.float32)
        if start_diff.shape != self.blobs[start_name].diff.shape:
            raise Exception('Expected start_diff of shape %s but got %s'
                            % (self.blobs[start_name].diff.shape, start_diff.shape))
        for tt in self._first.values():
            tt.grad = None
        start = self._first[start_name]
        if start.requires_grad:
            torch.autograd.backward(start, torch.from_numpy(start_diff).to(self.device),
                                    retain_graph=True)
        for name, tt in self._first.items():
            blob = self.blobs[name]
            if name == start_name:
                blob.diff = start_diff.copy()
            elif tt.grad is None:
                blob.diff = np.zeros_like(blob.data)
            else:
                blob.diff = tt.grad.detach().cpu().numpy()
        outputs = set(self.inputs + (diffs or []))
        return {out: self.blobs[out].diff for out in outputs}

    def backward_from_layer(self, start_name, start_diff, diffs=None, zero_higher=False):
        '''Backprop starting at blob/layer start_name with the given diff.
        Diffs of blobs above start_name are always zeroed (zero_higher is accepted for compatibility).'''
        self.deconv_mode = False
        return self._backward_from(start_name, start_diff, diffs)

    def deconv_from_layer(self, start_name, start_diff, diffs=None, zero_higher=False):
        '''Deconv (Zeiler & Fergus) pass starting at start_name with the given diff.'''
        self.deconv_mode = True
        try:
            return self._backward_from(start_name, start_diff, diffs)
        finally:
            self.deconv_mode = False


class _ParamBlob(object):
    '''Read-only view of a layer parameter, exposing .data and .diff like a pycaffe Blob.'''

    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def data(self):
        return self._tensor.detach().cpu().numpy()

    @property
    def diff(self):
        return np.zeros(tuple(self._tensor.shape), dtype=np.float32)

    @property
    def shape(self):
        return tuple(self._tensor.shape)
