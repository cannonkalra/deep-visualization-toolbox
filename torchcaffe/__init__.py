'''torchcaffe: the subset of the pycaffe API used by the Deep Visualization
Toolbox, implemented on PyTorch. Use it in place of pycaffe:

    import torchcaffe as caffe
    net = caffe.Classifier('deploy.prototxt', 'weights.caffemodel', ...)
'''

from .net import Net, Blob, TRAIN, TEST, set_mode_cpu, set_mode_gpu, set_device, get_device
from .classifier import Classifier
from . import io
from . import proto
