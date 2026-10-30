#! /usr/bin/env python3

import os
import sys
import numpy as np

# Make the toolbox root (which contains torchcaffe) importable when running from find_maxes/
dvt_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dvt_root)
import torchcaffe as caffe
caffe.set_mode_cpu()

caffenet_dir = os.path.join(dvt_root, 'models', 'caffenet-yos')



def load_labels():
    with open(os.path.join(caffenet_dir, 'ilsvrc_2012_labels.txt')) as ff:
        labels = [line.strip() for line in ff.readlines()]
    return labels



def load_trained_net(model_prototxt = None, model_weights = None):
    assert (model_prototxt is None) == (model_weights is None), 'Specify both model_prototxt and model_weights or neither'
    if model_prototxt is None:
        model_prototxt = os.path.join(caffenet_dir, 'caffenet-yos-deploy.prototxt')
        model_weights = os.path.join(caffenet_dir, 'caffenet-yos-weights')

    print('LOADER: loading net:')
    print('  ', model_prototxt)
    print('  ', model_weights)
    net = caffe.Classifier(model_prototxt, model_weights)

    return net



def load_imagenet_mean():
    imagenet_mean = np.load(os.path.join(caffenet_dir, 'ilsvrc_2012_mean.npy'))
    imagenet_mean = imagenet_mean[:, 14:14+227, 14:14+227]    # (3,256,256) -> (3,227,227) Crop to center 227x227 section
    return imagenet_mean
